#!/usr/bin/env python3
"""Prepare the public MPEA v1 workbook as an audited evaluation candidate.

No imputation, learned preprocessing, model fitting, or partitioning happens here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd


DATASET_DIR = Path(__file__).resolve().parent
DATASETS_DIR = DATASET_DIR.parent
REPO_ROOT = DATASET_DIR.parents[2]
RAW_PATH = DATASET_DIR / "raw/Mendeley_MPEA_corrosion_database.xlsx"
EPIT_PATH = DATASETS_DIR / "electrochemical_metrics_alloys/raw/CRA_database_Scientific_Data_Publication_12102020.xlsx"
SOCCOL_PATH = DATASETS_DIR / "soccol_pitting_potential/raw/Pitting_potential_dataset_2024.xlsx"
EXPECTED_RAW_SHA256 = "595200617e007db3281115d0dc7764fd86f337203b6fc1a2ab34bf3b88800e69"
RAW_TARGET = "Pitting potential (mV vs SCE)"
TARGET = "E_pit_mV_SCE"

# CIAAW abridged standard atomic weights 2024, frozen for this preparation.
# https://ciaaw.org/abridged-atomic-weights.htm
ATOMIC_WEIGHTS = {
    "Al": 26.982, "B": 10.81, "Be": 9.0122, "Co": 58.933,
    "Cr": 51.996, "Cu": 63.546, "Fe": 55.845, "Ga": 69.723,
    "Hf": 178.49, "La": 138.91, "Mg": 24.305, "Mn": 54.938,
    "Mo": 95.95, "Nb": 92.906, "Ni": 58.693, "Si": 28.085,
    "Sn": 118.71, "Ta": 180.95, "Ti": 47.867, "V": 50.942,
    "W": 183.84, "Y": 88.906, "Zn": 65.38, "Zr": 91.222,
}
ELEMENTS = list(ATOMIC_WEIGHTS)
# MPEA columns supported by either current source prior, in workbook order.
# This is NOT yet the combined model's final global feature registry.
RETAINED_ELEMENTS = [
    "Al", "B", "Co", "Cr", "Cu", "Fe", "Mg", "Mn", "Mo",
    "Nb", "Ni", "Si", "Ta", "Ti", "V", "W", "Y",
]
UNSUPPORTED_ELEMENTS = [e for e in ELEMENTS if e not in RETAINED_ELEMENTS]
ENVIRONMENT_UNITS = {
    "CP_temp": "degC", "CP_pH": "pH", "CP_Cl": "mol/L",
    "CP_OH": "mol/L", "CP_SO4": "mol/L", "CP_NO3": "mol/L",
    "scan_rate": "mV/s",
}
FEATURES = RETAINED_ELEMENTS + list(ENVIRONMENT_UNITS)
COMPOSITION_CLOSURE_ATOL = 0.001
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s<>\"?#&]+", re.I)
SIMPLE_ELECTROLYTES = ("nacl", "naoh", "koh", "h2so4", "hno3")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_dois(value: object) -> list[str]:
    """Extract DOI-like strings, not verified publication identities."""
    if pd.isna(value):
        return []
    text = unquote(str(value)).replace("\ufeff", "").lower()
    found = []
    for match in DOI_PATTERN.finditer(text):
        doi = match.group().rstrip(".,;")
        # Keep balanced parentheses in historical DOI suffixes.
        while doi.endswith(")") and doi.count(")") > doi.count("("):
            doi = doi[:-1]
        doi = re.sub(r"/(?:pdf|full|abstract)$", "", doi)
        found.append(doi)
    return sorted(set(found))


def convert_composition(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    fractions = raw[ELEMENTS].apply(pd.to_numeric, errors="coerce")
    total = fractions.sum(axis=1, min_count=len(ELEMENTS))
    valid = (
        np.isfinite(fractions).all(axis=1)
        & fractions.ge(0).all(axis=1)
        & np.isclose(total, 1.0, rtol=0.0, atol=COMPOSITION_CLOSURE_ATOL)
    )
    masses = fractions.mul(pd.Series(ATOMIC_WEIGHTS))
    mass_total = masses.sum(axis=1, min_count=len(ELEMENTS))
    weights = masses.div(mass_total.where(valid), axis=0) * 100.0
    return weights, total, valid


def _number_or_nan(value: object) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def parse_environment(row: pd.Series) -> tuple[dict, list[str], dict]:
    """Conservative, explicit-text-only condition extraction for this workbook."""
    env = "" if pd.isna(row["Test environment"]) else str(row["Test environment"])
    env = " ".join(env.split())
    electrolyte = "" if pd.isna(row["Electrolyte"]) else str(row["Electrolyte"]).strip().lower()
    concentration = _number_or_nan(row["Concentration in M"])
    result = dict.fromkeys(ENVIRONMENT_UNITS, np.nan)
    provenance = dict.fromkeys((f"{name}_provenance" for name in ENVIRONMENT_UNITS), "unreported")
    flags = []

    patterns = {
        "CP_temp": rf"(?<![\w.])({NUMBER})\s*(K\b|[°◦º]\s*C\b)",
        "CP_pH": rf"\bpH\s*(?:=|:)?\s*({NUMBER})(?![\w.])",
        "scan_rate": rf"(?<![\w.])({NUMBER})\s*mV\s*/\s*s\b",
    }
    for name, pattern in patterns.items():
        matches = list(re.finditer(pattern, env, re.I))
        if len(matches) > 1:
            flags.append(f"ambiguous_{name}")
        elif len(matches) == 1:
            value = float(matches[0].group(1))
            if name == "CP_temp" and matches[0].group(2).lower() == "k":
                value = round(value - 273.15, 8)
            result[name] = value
            provenance[f"{name}_provenance"] = "explicit_environment_text"

    named = set(re.findall(r"\b(?:NaCl|NaOH|KOH|H2SO4|HNO3)\b", env, re.I))
    named = {name.lower() for name in named}
    if named and named != {electrolyte}:
        flags.append("electrolyte_text_conflict")
        return result, flags, provenance
    if electrolyte not in SIMPLE_ELECTROLYTES:
        flags.append("mixed_or_unspecified_electrolyte_recipe")
        return result, flags, provenance
    if "%" in env:
        flags.append("mass_percent_to_molarity_unverified")
        return result, flags, provenance

    molarity_matches = list(re.finditer(
        rf"({NUMBER})\s*(?:mol\s*/\s*l|M)\s*(?:NaCl|NaOH|KOH|H2SO4|HNO3)\b",
        env, re.I,
    ))
    origin = "reported_concentration_in_M"
    if len(molarity_matches) > 1:
        flags.append("ambiguous_molarity_text")
        return result, flags, provenance
    if molarity_matches:
        text_concentration = float(molarity_matches[0].group(1))
        if np.isfinite(concentration) and not np.isclose(concentration, text_concentration, rtol=1e-6, atol=1e-9):
            flags.append("molarity_text_conflict")
            return result, flags, provenance
        concentration = text_concentration
        origin = "explicit_environment_text_and_column" if pd.notna(row["Concentration in M"]) else "explicit_environment_text"
    if not np.isfinite(concentration) or concentration < 0:
        flags.append("missing_or_invalid_molarity")
        return result, flags, provenance
    if electrolyte in ("h2so4", "hno3"):
        # Formal acid concentration is not silently treated as free-ion concentration.
        flags.append("acid_to_ion_mapping_unverified")
        return result, flags, provenance
    feature = "CP_Cl" if electrolyte == "nacl" else "CP_OH"
    result[feature] = concentration
    provenance[f"{feature}_provenance"] = origin + ";simple_salt_stoichiometry"
    return result, flags, provenance


def prepare_tables(raw: pd.DataFrame, epit_dois: set[str], soccol_dois: set[str]):
    raw = raw.reset_index(drop=True)
    target = pd.to_numeric(raw[RAW_TARGET], errors="coerce")
    has_target = np.isfinite(target)
    weights, fraction_total, composition_valid = convert_composition(raw)
    included = has_target & composition_valid
    ids = pd.Series([f"mpea_v1_r{row:06d}" for row in range(2, len(raw) + 2)])
    environment, audit_rows = [], []
    for i, row in raw.iterrows():
        env, flags, provenance = parse_environment(row)
        environment.append(env)
        dois = set(extract_dois(row["Reference"]))
        epit_matches, soccol_matches = dois & epit_dois, dois & soccol_dois
        overlap_status = (
            "shared_source_doi" if epit_matches or soccol_matches
            else "no_shared_doi_found" if dois else "unresolved_reference"
        )
        if not composition_valid.iloc[i]:
            flags.append("composition_not_convertible")
        present_unsupported = [e for e in UNSUPPORTED_ELEMENTS if _number_or_nan(row[e]) > 0]
        if present_unsupported:
            flags.append("unsupported_elements_present")
        if overlap_status != "no_shared_doi_found":
            flags.append(overlap_status)
        if not has_target.iloc[i]:
            flags.append("missing_or_nonfinite_target")
        audit_rows.append({
            "row_id": ids.iloc[i], "raw_workbook_row": i + 2,
            "has_finite_target": bool(has_target.iloc[i]),
            "included_in_candidate_table": bool(included.iloc[i]),
            "included_in_source_screened_subset": bool(included.iloc[i] and overlap_status == "no_shared_doi_found"),
            "exclusion_reason": (
                "missing_or_nonfinite_target" if not has_target.iloc[i]
                else "composition_not_convertible" if not composition_valid.iloc[i] else ""
            ),
            "alloy_name_raw": row["Alloy name"],
            "test_environment_raw": row["Test environment"],
            "electrolyte_raw": row["Electrolyte"],
            "concentration_in_M_raw": row["Concentration in M"],
            "target_raw": row[RAW_TARGET], "reference_raw": row["Reference"],
            "source_dois": ";".join(sorted(dois)),
            "epit_shared_dois": ";".join(sorted(epit_matches)),
            "soccol_shared_dois": ";".join(sorted(soccol_matches)),
            "source_overlap_status": overlap_status,
            "atomic_fraction_sum": fraction_total.iloc[i],
            "composition_conversion_valid": bool(composition_valid.iloc[i]),
            "unsupported_elements_present": ";".join(present_unsupported),
            "unsupported_elements_wt_percent": weights.loc[i, UNSUPPORTED_ELEMENTS].sum(min_count=len(UNSUPPORTED_ELEMENTS)),
            "retained_elements_wt_percent": weights.loc[i, RETAINED_ELEMENTS].sum(min_count=len(RETAINED_ELEMENTS)),
            **provenance, "review_flags": ";".join(sorted(flags)),
        })
    candidates = pd.concat([
        ids.rename("row_id"), weights[RETAINED_ELEMENTS],
        pd.DataFrame(environment), target.rename(TARGET),
    ], axis=1).loc[included].reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    sources = audit.groupby(
        ["reference_raw", "source_dois", "epit_shared_dois", "soccol_shared_dois", "source_overlap_status"],
        dropna=False, sort=True,
    ).agg(raw_rows=("row_id", "size"), target_rows=("has_finite_target", "sum"),
          candidate_rows=("included_in_candidate_table", "sum")).reset_index()
    return candidates, audit, sources


def summarize(raw, candidates, audit):
    selected = audit.loc[audit["included_in_candidate_table"]]
    numeric = pd.to_numeric(raw[RAW_TARGET], errors="coerce")
    responses = raw[[RAW_TARGET, "Corrosion potential (mV vs SCE)", "Corrosion current density (microA/cm2)"]].apply(pd.to_numeric, errors="coerce")
    flags = selected["review_flags"].str.split(";").explode()
    return {
        "raw_rows": len(raw), "candidate_rows": len(candidates),
        "finite_target_rows_before_composition_filter": int(np.isfinite(numeric).sum()),
        "excluded_missing_or_nonfinite_target": int((~np.isfinite(numeric)).sum()),
        "excluded_incomplete_composition_with_target": int(audit["exclusion_reason"].eq("composition_not_convertible").sum()),
        "excluded_composition_excel_rows": audit.loc[audit["exclusion_reason"].eq("composition_not_convertible"), "raw_workbook_row"].tolist(),
        "source_screened_subset_rows": int(selected["included_in_source_screened_subset"].sum()),
        "zero_targets_retained": int((candidates[TARGET] == 0).sum()),
        "composition_conversion_valid_candidates": int(selected["composition_conversion_valid"].sum()),
        "candidates_with_unsupported_elements": int(selected["unsupported_elements_present"].ne("").sum()),
        "candidate_source_overlap_status": selected["source_overlap_status"].value_counts().to_dict(),
        "candidate_rows_sharing_epit_doi": int(selected["epit_shared_dois"].ne("").sum()),
        "candidate_rows_sharing_soccol_doi": int(selected["soccol_shared_dois"].ne("").sum()),
        "candidate_review_flag_counts": flags[flags.ne("")].value_counts().to_dict(),
        "feature_observed_counts": candidates[FEATURES].notna().sum().to_dict(),
        "paper_count_discrepancy": {
            "paper_reported_epit_entries": 306, "workbook_numeric_epit_entries": int(np.isfinite(numeric).sum()),
            "status": "unresolved; no undocumented 29-row exclusion was applied",
            "diagnostics_not_selection_rules": {
                "nonzero_finite_target_rows": int((np.isfinite(numeric) & numeric.ne(0)).sum()),
                "all_three_responses_finite_rows": int(np.isfinite(responses).all(axis=1).sum()),
                "target_rows_complete_across_all_raw_columns": int(raw.loc[np.isfinite(numeric)].notna().all(axis=1).sum()),
                "exact_duplicate_raw_target_rows_beyond_first": int(raw.loc[np.isfinite(numeric)].duplicated().sum()),
            },
        },
    }


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR / "processed")
    parser.add_argument("--overwrite", action="store_true", help="Replace only the named generated artifacts.")
    args = parser.parse_args()
    if file_sha256(RAW_PATH) != EXPECTED_RAW_SHA256:
        raise ValueError("MPEA raw workbook differs from the inspected v1 file; review before preparing.")
    raw = pd.read_excel(RAW_PATH, sheet_name="Sheet1")
    epit = pd.read_excel(EPIT_PATH, sheet_name="References")
    soccol = pd.read_excel(SOCCOL_PATH, sheet_name="references")
    epit_dois = {doi for value in epit["DOI"] for doi in extract_dois(value)}
    soccol_dois = {doi for value in soccol["DOI"] for doi in extract_dois(value)}
    candidates, audit, sources = prepare_tables(raw, epit_dois, soccol_dois)
    screened_ids = audit.loc[audit["included_in_source_screened_subset"], "row_id"]
    screened = candidates.loc[candidates["row_id"].isin(screened_ids)].reset_index(drop=True)
    summary = summarize(raw, candidates, audit)
    names = ["mpea_evaluation_candidates.csv", "mpea_row_audit.csv", "mpea_source_audit.csv",
             "mpea_source_screened_candidates.csv",
             "feature_manifest.json", "preprocessing_manifest.json"]
    existing = [str(args.output_dir / name) for name in names if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Use --overwrite to regenerate existing outputs: " + ", ".join(existing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for table, name in zip((candidates, audit, sources, screened), names[:4]):
        table.to_csv(args.output_dir / name, index=False, na_rep="NaN", float_format="%.12g")
    feature_manifest = {
        "schema_version": "mpea_candidate_v1", "status": "candidate_only_not_final_shared_schema",
        "row_id_column": "row_id", "target_column": TARGET, "target_unit": "mV vs SCE",
        "predictor_columns_in_order": FEATURES,
        "feature_units": {**dict.fromkeys(RETAINED_ELEMENTS, "wt%"), **ENVIRONMENT_UNITS},
        "features_reserved_but_entirely_unobserved": [name for name in FEATURES if candidates[name].isna().all()],
        "excluded_mpea_elements": UNSUPPORTED_ELEMENTS,
        "excluded_raw_predictors": [name for name in raw.columns if name not in RETAINED_ELEMENTS],
        "metadata_not_predictors": ["row_id", "all fields in mpea_row_audit.csv and mpea_source_audit.csv"],
        "missing_value_encoding": "CSV NaN; no imputation or learned missingness encoding applied",
        "shared_source_aliases": {
            "CP_temp": {"epit": "Test Temp. oC", "soccol": "CP_temp"},
            "CP_pH": {"epit": "[Cl-] pH", "soccol": "CP_pH"},
            "CP_Cl": {"epit": "[Cl-] M", "soccol": "CP_Cl"},
            **{name: {"soccol": name} for name in ("CP_OH", "CP_SO4", "CP_NO3", "scan_rate")},
        },
        "target_alignment": "MPEA kept in mV vs SCE; cross-source electrode conversion not performed here",
        "scan_rate_alignment": "MPEA and Soccol use mV/s; confirmed by Soccol manuscript Table 3; no unit conversion needed",
        "soccol_units_source": "https://lirias.kuleuven.be/retrieve/f69750dc-fb1a-4962-8e5c-8982608a4efb",
    }
    write_json(args.output_dir / "feature_manifest.json", feature_manifest)
    manifest = {
        "preparation_version": "mpea_candidate_v2", "status": "audited_candidate_not_frozen_evaluation",
        "inputs": [{"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)}
                   for path in (RAW_PATH, EPIT_PATH, SOCCOL_PATH)],
        "processor": {"path": str(Path(__file__).resolve().relative_to(REPO_ROOT)), "sha256": file_sha256(Path(__file__))},
        "software": {"pandas": pd.__version__, "numpy": np.__version__},
        "composition": {
            "raw_unit": "atomic fraction", "output_unit": "wt%", "atomic_weights": ATOMIC_WEIGHTS,
            "atomic_weights_source": "https://ciaaw.org/abridged-atomic-weights.htm",
            "fraction_sum_absolute_tolerance": COMPOSITION_CLOSURE_ATOL,
            "formula": "100 * x_i * A_i / sum_over_all_24_reported_elements(x_j * A_j)",
            "invalid_composition_policy": "exclude from candidate exports; preserve in raw workbook and all-row audit",
            "unsupported_elements_policy": "included in mass denominator, omitted from predictors, never renormalize retained subset",
        },
        "dataset_source": "https://data.mendeley.com/datasets/nskb9khgsm/1",
        "dataset_license": "CC BY 4.0",
        "paper_source": "https://www.nature.com/articles/s41529-025-00700-9",
        "source_overlap_scope": "All DOI fields in the EPIT and Soccol reference sheets; shared publication, not confirmed shared measurement",
        "source_screened_subset_policy": "Eligible rows with an extracted DOI and no EPIT/Soccol DOI match; exclude shared and unresolved sources; NOT verified measurement-independent",
        "reference_doi_resolution_verified": False,
        "exact_cross_dataset_row_overlap_assessed": False,
        "frozen_partitions_created": False, "models_trained_or_evaluated": False,
        "summary": summary,
        "outputs": {name: {"sha256": file_sha256(args.output_dir / name), "size_bytes": (args.output_dir / name).stat().st_size}
                    for name in names[:-1]},
    }
    write_json(args.output_dir / "preprocessing_manifest.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
