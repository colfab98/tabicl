#!/usr/bin/env python3
"""Build shared EPIT/Soccol context and target-free MPEA query tables.

The source datasets are read only. Missing predictor values remain NaN so a
later model-side missing-value encoder can distinguish them from measured zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATASET_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATASET_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline.split_data import (  # noqa: E402
    SOURCE_FILE as EPIT_SOURCE_FILE,
    load_epit_dataset,
)


SOCCOL_PATH = (
    REPO_ROOT
    / "corrosion_datasets/datasets/soccol_pitting_potential/processed"
    / "soccol_regression_event1.csv"
)
MPEA_PATH = (
    REPO_ROOT
    / "corrosion_datasets/datasets/mpea_corrosion/processed"
    / "mpea_source_screened_candidates.csv"
)
MPEA_AUDIT_PATH = (
    REPO_ROOT
    / "corrosion_datasets/datasets/mpea_corrosion/processed"
    / "mpea_row_audit.csv"
)

CONTEXT_NAME = "epit_soccol_context.csv"
QUERY_NAME = "mpea_queries.csv"
SCORING_NAME = "mpea_scoring_targets.csv"
REGISTRY_NAME = "shared_feature_registry.json"
MISSINGNESS_PROFILE_NAME = "context_missingness_profile.json"
MANIFEST_NAME = "preparation_manifest.json"

# Atomic-number order. The complete list is frozen in the registry so later
# datasets can add chemistry without redefining element identity.
PERIODIC_ELEMENTS = (
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
    "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
    "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
    "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)

# The first combined model uses only elements supported by EPIT or Soccol.
# MPEA-only elements remain reserved but are not query predictors.
SOURCE_SUPPORTED_ELEMENTS = frozenset(
    {
        "Fe", "Cr", "Ni", "Mo", "W", "N", "Nb", "C", "Si", "Mn", "Cu",
        "P", "S", "Al", "V", "Ta", "Re", "Ce", "Ti", "Co", "B", "Mg",
        "Y", "Gd",
    }
)
ELEMENT_FEATURES = tuple(
    element for element in PERIODIC_ELEMENTS if element in SOURCE_SUPPORTED_ELEMENTS
)

NUMERIC_FEATURES = (
    "Prep_grinding_grit",
    "Prep_Ra_micron",
    "Prep_pH",
    "Prep_redox",
    "Prep_time",
    "CP_time",
    "CP_temp",
    "CP_pH",
    "Test_area_cm2",
    "scan_rate",
    "CP_Cl",
    "CP_Br",
    "CP_OH",
    "CP_SO4",
    "CP_CO3",
    "CP_NO3",
    "CP_PO4",
    "CP_MoO4",
    "CP_CrO4",
    "CP_ion_other",
)
CATEGORICAL_FEATURES = (
    "Prep_medium",
    "CP_aeration",
    "CP_agitation",
    "CP_anions_info",
    "test_method",
    "reference_type",
)
FEATURE_COLUMNS = ELEMENT_FEATURES + NUMERIC_FEATURES + CATEGORICAL_FEATURES
FEATURE_ALIASES = {
    **{
        element: [f"Composition, wt.% {element}", f"{element} (wt%)"]
        for element in ELEMENT_FEATURES
    },
    "CP_temp": ["Test Temp. oC", "Test Temp. °C", "Test temperature (°C)"],
    "CP_pH": ["[Cl-] pH", "Test solution pH"],
    "CP_Cl": ["[Cl-] M", "Chloride concentration (M)"],
    "scan_rate": ["Scan Rate mV/s", "Scan rate (mV/s)"],
    "test_method": ["[Cl-] Test Method", "Test method"],
    "reference_type": ["Reference electrode", "Reference scale"],
}
METADATA_COLUMNS = (
    "row_id",
    "source_dataset",
    "source_row_id",
    "source_reference",
    "reference_type_evidence",
)
TARGET_COLUMN = "reported_pitting_potential_mV"

SOCCOL_COMPOSITION_ELEMENTS = (
    "Fe", "C", "N", "Si", "P", "S", "Ti", "V", "Cr", "Mn", "Ni", "Nb", "Mo"
)

# Conservative primary-source audit. All other Soccol reference types remain missing.
SOCCOL_SCE_SOURCES = frozenset(
    {
        "1970Wilde",
        "1983Shibata",
        "1995Jackson",
        "1999Wegrelius",
        "2011Elsener",
        "2013Mesquita",
        "2014Zhou",
        "2016Alar",
        "2017Mameng",
        "2018Shakir",
        "2018Shi",
        "2021Vignal",
    }
)
SOCCOL_AGAGCL_SOURCES = frozenset({"2020Melia"})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return np.nan
    return converted if math.isfinite(converted) else np.nan


def category(value: Any) -> str | float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    cleaned = str(value).strip()
    return cleaned if cleaned else np.nan


def empty_feature_row() -> dict[str, Any]:
    return {name: np.nan for name in FEATURE_COLUMNS}


def build_epit_rows() -> list[dict[str, Any]]:
    dataset = load_epit_dataset()
    output: list[dict[str, Any]] = []
    for index, (source, target) in enumerate(
        zip(dataset.rows, dataset.target, strict=True), start=1
    ):
        row = empty_feature_row()
        for element in ELEMENT_FEATURES:
            row[element] = number(source.get(f"Composition, wt.% {element}"))
        row.update(
            {
                "CP_temp": number(source.get("Test Temp. oC")),
                "CP_pH": number(source.get("[Cl-] pH")),
                "CP_Cl": number(source.get("[Cl-] M")),
                "scan_rate": number(source.get("Scan Rate mV/s")),
                "test_method": category(source.get("[Cl-] Test Method")),
                "reference_type": "SCE",
                "row_id": f"epit_r{index:06d}",
                "source_dataset": "EPIT",
                "source_row_id": category(source.get("No.")),
                "source_reference": category(source.get("Reference")),
                "reference_type_evidence": "workbook_target_column_declares_SCE",
                TARGET_COLUMN: float(target),
            }
        )
        output.append(row)
    return output


def soccol_reference(source: str) -> tuple[str | float, str]:
    if source in SOCCOL_SCE_SOURCES:
        return "SCE", "primary_source_verified_SCE"
    if source in SOCCOL_AGAGCL_SOURCES:
        return "AgAgCl", "primary_source_verified_AgAgCl_saturated_KCl"
    return np.nan, "source_reference_not_verified"


def build_soccol_rows() -> list[dict[str, Any]]:
    source_table = pd.read_csv(SOCCOL_PATH, low_memory=False)
    output: list[dict[str, Any]] = []
    for source in source_table.to_dict(orient="records"):
        row = empty_feature_row()
        for element in SOCCOL_COMPOSITION_ELEMENTS:
            missing = number(source.get(f"{element}_missing")) == 1.0
            row[element] = np.nan if missing else number(source.get(element))
        for feature in NUMERIC_FEATURES:
            row[feature] = number(source.get(feature))
        for feature in CATEGORICAL_FEATURES[:-2]:
            row[feature] = category(source.get(feature))
        reference_type, evidence = soccol_reference(str(source["source"]))
        raw_row = int(number(source["raw_workbook_row"]))
        row.update(
            {
                "test_method": np.nan,
                "reference_type": reference_type,
                "row_id": f"soccol_r{raw_row:06d}",
                "source_dataset": "Soccol",
                "source_row_id": category(source.get("label")),
                "source_reference": category(source.get("source")),
                "reference_type_evidence": evidence,
                TARGET_COLUMN: number(source.get("E_pit")),
            }
        )
        output.append(row)
    return output


def build_mpea_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_table = pd.read_csv(MPEA_PATH)
    audit = pd.read_csv(MPEA_AUDIT_PATH, low_memory=False).set_index("row_id")
    queries: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for source in source_table.to_dict(orient="records"):
        source_id = str(source["row_id"])
        audit_row = audit.loc[source_id]
        row = empty_feature_row()
        for element in ELEMENT_FEATURES:
            row[element] = number(source.get(element))
        for feature in NUMERIC_FEATURES:
            row[feature] = number(source.get(feature))
        row.update(
            {
                "reference_type": "SCE",
                "row_id": source_id,
                "source_dataset": "MPEA",
                "source_row_id": source_id,
                "source_reference": category(audit_row.get("source_dois")),
                "reference_type_evidence": "workbook_target_column_declares_SCE",
            }
        )
        queries.append(row)
        targets.append(
            {
                "row_id": source_id,
                TARGET_COLUMN: number(source.get("E_pit_mV_SCE")),
                "reference_type": "SCE",
            }
        )
    return queries, targets


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    epit_rows = build_epit_rows()
    soccol_rows = build_soccol_rows()
    mpea_rows, mpea_targets = build_mpea_rows()
    context = pd.DataFrame(
        epit_rows + soccol_rows,
        columns=[*METADATA_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN],
    )
    queries = pd.DataFrame(
        mpea_rows,
        columns=[*METADATA_COLUMNS, *FEATURE_COLUMNS],
    )
    targets = pd.DataFrame(
        mpea_targets,
        columns=["row_id", TARGET_COLUMN, "reference_type"],
    )
    validate_tables(context, queries, targets)
    return context, queries, targets


def validate_tables(
    context: pd.DataFrame,
    queries: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    if len(context) != 4787:
        raise ValueError(f"Expected 4,787 context rows, found {len(context)}.")
    if context["source_dataset"].value_counts().to_dict() != {
        "Soccol": 4027,
        "EPIT": 760,
    }:
        raise ValueError("Unexpected EPIT/Soccol context row counts.")
    if context["reference_type"].value_counts().to_dict() != {
        "SCE": 1325,
        "AgAgCl": 112,
    } or int(context["reference_type"].isna().sum()) != 3350:
        raise ValueError("Unexpected context reference-type counts.")
    unresolved = context["reference_type"].isna()
    if not context.loc[unresolved, "reference_type_evidence"].eq(
        "source_reference_not_verified"
    ).all():
        raise ValueError("Missing reference types must retain their audit evidence.")
    if len(queries) != 267 or len(targets) != 267:
        raise ValueError("Expected 267 source-screened MPEA queries and targets.")
    if TARGET_COLUMN in queries:
        raise ValueError("MPEA query table must not expose scoring targets.")
    if list(context.columns[:-1]) != list(queries.columns):
        raise ValueError("Context and query schemas differ before the context target.")
    if not context[TARGET_COLUMN].map(np.isfinite).all():
        raise ValueError("Context targets must be finite.")
    if not targets[TARGET_COLUMN].map(np.isfinite).all():
        raise ValueError("MPEA scoring targets must be finite.")
    if not queries["reference_type"].eq("SCE").all():
        raise ValueError("Every MPEA query must request the SCE reporting scale.")
    if queries["row_id"].tolist() != targets["row_id"].tolist():
        raise ValueError("MPEA query and scoring-target row IDs are not aligned.")
    if context["row_id"].duplicated().any() or queries["row_id"].duplicated().any():
        raise ValueError("Derived row IDs must be unique within each table.")


def feature_registry() -> dict[str, Any]:
    units = {element: "wt%" for element in ELEMENT_FEATURES}
    units.update(
        {
            "Prep_grinding_grit": "grit_designation",
            "Prep_Ra_micron": "micrometre",
            "Prep_pH": "pH",
            "Prep_redox": "source_workbook_unit_unverified",
            "Prep_time": "source_workbook_unit_unverified",
            "CP_time": "source_workbook_unit_unverified",
            "CP_temp": "degC",
            "CP_pH": "pH",
            "Test_area_cm2": "cm2",
            "scan_rate": "mV/s",
            "CP_Cl": "mol/L",
            "CP_Br": "mol/L",
            "CP_OH": "mol/L",
            "CP_SO4": "mol/L",
            "CP_CO3": "mol/L",
            "CP_NO3": "mol/L",
            "CP_PO4": "mol/L",
            "CP_MoO4": "mol/L",
            "CP_CrO4": "mol/L",
            "CP_ion_other": "mol/L",
        }
    )
    units.update({name: None for name in CATEGORICAL_FEATURES})
    return {
        "schema_version": "combined_epit_soccol_mpea_shared_v2",
        "feature_columns_in_order": list(FEATURE_COLUMNS),
        "numeric_feature_columns": list(ELEMENT_FEATURES + NUMERIC_FEATURES),
        "categorical_feature_columns": list(CATEGORICAL_FEATURES),
        "feature_units": units,
        "feature_aliases": FEATURE_ALIASES,
        "feature_name_matching": (
            "exact canonical name or approved alias after trimming whitespace and case folding; no fuzzy matching"
        ),
        "target_column": TARGET_COLUMN,
        "target_semantics": (
            "reported pitting potential in mV; reference_type names the scale when verified"
        ),
        "reference_type_categories": ["SCE", "AgAgCl"],
        "reference_type_missing_semantics": (
            "unverified reference scale; stored as NaN, with the audit reason retained in "
            "reference_type_evidence"
        ),
        "missing_value_convention": (
            "CSV NaN; no imputation in derived tables; the missing-aware model maps NaN to "
            "a learned column-specific token"
        ),
        "element_registry_atomic_number_order": list(PERIODIC_ELEMENTS),
        "active_source_supported_elements": list(ELEMENT_FEATURES),
        "reserved_inactive_elements": [
            element for element in PERIODIC_ELEMENTS if element not in SOURCE_SUPPORTED_ELEMENTS
        ],
        "inactive_element_policy": (
            "not exported as model columns in v1 because neither EPIT nor Soccol prior supports them"
        ),
        "source_mappings": {
            "EPIT": {
                "composition": "Composition, wt.% <element>",
                "CP_temp": "Test Temp. oC",
                "CP_pH": "[Cl-] pH",
                "CP_Cl": "[Cl-] M",
                "scan_rate": "Scan Rate mV/s",
                "test_method": "[Cl-] Test Method",
                "reference_type": "constant SCE",
                "target": "Epit, mV (SCE) Avg.",
            },
            "Soccol": {
                "composition": "same element name; original missing cells restored to NaN via *_missing",
                "environment_and_procedure": "same canonical column names",
                "test_method": "unavailable",
                "reference_type": (
                    "conservative verified-source map; otherwise NaN with audit evidence"
                ),
                "target": "E_pit",
            },
            "MPEA": {
                "composition": "atomic fractions previously converted to wt% over the complete raw composition",
                "environment_and_procedure": "same canonical column names where extracted",
                "reference_type": "constant SCE",
                "target": "kept only in mpea_scoring_targets.csv",
            },
        },
        "soccol_verified_reference_sources": {
            "SCE": sorted(SOCCOL_SCE_SOURCES),
            "AgAgCl": sorted(SOCCOL_AGAGCL_SOURCES),
        },
    }


def missingness_profile(context: pd.DataFrame) -> dict[str, Any]:
    """Compress EPIT/Soccol feature-missingness masks for training augmentation."""
    patterns_by_source: dict[str, Any] = {}
    for source_dataset, source_rows in context.groupby("source_dataset", sort=True):
        masks = source_rows.loc[:, FEATURE_COLUMNS].isna()
        bitstrings = masks.astype(np.uint8).astype(str).agg("".join, axis=1)
        counts = bitstrings.value_counts(sort=False).sort_index()
        patterns_by_source[str(source_dataset)] = {
            "row_count": int(len(source_rows)),
            "patterns": [
                {"missing_mask": bitstring, "count": int(count)}
                for bitstring, count in counts.items()
            ],
        }
    return {
        "schema_version": "combined_epit_soccol_missingness_v2",
        "source": "EPIT/Soccol labeled context only; MPEA queries and scoring targets excluded",
        "feature_columns_in_order": list(FEATURE_COLUMNS),
        "composition_feature_count": len(ELEMENT_FEATURES),
        "patterns_by_source": patterns_by_source,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR / "processed")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_paths = {
        CONTEXT_NAME: args.output_dir / CONTEXT_NAME,
        QUERY_NAME: args.output_dir / QUERY_NAME,
        SCORING_NAME: args.output_dir / SCORING_NAME,
        REGISTRY_NAME: args.output_dir / REGISTRY_NAME,
        MISSINGNESS_PROFILE_NAME: args.output_dir / MISSINGNESS_PROFILE_NAME,
        MANIFEST_NAME: args.output_dir / MANIFEST_NAME,
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Use --overwrite to replace generated artifacts: " + ", ".join(existing))

    context, queries, targets = build_tables()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context.to_csv(output_paths[CONTEXT_NAME], index=False, na_rep="NaN", float_format="%.12g")
    queries.to_csv(output_paths[QUERY_NAME], index=False, na_rep="NaN", float_format="%.12g")
    targets.to_csv(output_paths[SCORING_NAME], index=False, na_rep="NaN", float_format="%.12g")
    write_json(output_paths[REGISTRY_NAME], feature_registry())
    write_json(output_paths[MISSINGNESS_PROFILE_NAME], missingness_profile(context))

    manifest = {
        "preparation_version": "combined_context_query_v2",
        "source_files_modified": False,
        "inputs": [
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)}
            for path in (EPIT_SOURCE_FILE, SOCCOL_PATH, MPEA_PATH, MPEA_AUDIT_PATH)
        ],
        "processor": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": file_sha256(Path(__file__)),
        },
        "rows": {
            "context_total": len(context),
            "context_by_source": context["source_dataset"].value_counts().to_dict(),
            "context_by_reference_type": context["reference_type"].value_counts().to_dict(),
            "context_reference_type_missing": int(context["reference_type"].isna().sum()),
            "mpea_queries": len(queries),
            "mpea_scoring_targets": len(targets),
        },
        "target_handling": {
            "context": (
                "reported target retained; verified reference scale supplied as a categorical "
                "predictor and unresolved scale stored as feature NaN"
            ),
            "mpea_query": "target column absent",
            "mpea_scoring": "separate row-ID-aligned file; SCE only",
            "electrode_conversion_applied": False,
        },
        "known_preserved_source_issue": {
            "2016Alar06": "Soccol stores 777; audited primary table reports 771 mV/SCE",
            "2016Alar12": "Soccol stores 911; audited primary table reports 977 mV/SCE",
            "policy": "preserved rather than silently altering an existing processed dataset",
        },
        "outputs": {},
    }
    for name, path in output_paths.items():
        if name == MANIFEST_NAME:
            continue
        manifest["outputs"][name] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    write_json(output_paths[MANIFEST_NAME], manifest)
    print(json.dumps(manifest["rows"], indent=2))


if __name__ == "__main__":
    main()
