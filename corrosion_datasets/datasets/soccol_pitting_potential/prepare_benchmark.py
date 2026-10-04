"""Create reproducible Soccol benchmark tables without modifying raw data.

Outputs:
- processed/soccol_processed_all_rows.csv
- processed/soccol_regression_event1.csv
- processed/soccol_breakdown_survival.csv
- processed/feature_manifest.json
- processed/preprocessing_manifest.json

Requires openpyxl.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW_WORKBOOK = HERE / "raw" / "Pitting_potential_dataset_2024.xlsx"
SOURCE_REGISTRY = ROOT / "corrosion_datasets" / "analysis" / "soccol_source_conventions" / "source_conventions.csv"
OUTPUT_DIR = HERE / "processed"

ELEMENTS = ["C", "N", "Si", "P", "S", "Ti", "V", "Cr", "Mn", "Ni", "Nb", "Mo"]
MAJOR_ELEMENTS = ["Cr", "Mn", "Ni", "Mo"]
EXPECTED_RAW_SHA256 = "cb255c5a94d90bd207248f6e25727f965d587c29a73b2ffcef6b8c0ced6da1da"

MODEL_ALLOY_SOURCES = {"1977Sugimoto", "1983Bandy", "2000Russell", "2006Muwila"}
TAXONOMY_CAVEAT_SOURCES = {"1994Malik", "1995Malik"}


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clean(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.replace("\ufeff", "").strip()
        return stripped if stripped else None
    return value


def load_workbook_rows() -> tuple[list[str], list[dict[str, Any]]]:
    ws = load_workbook(RAW_WORKBOOK, read_only=True, data_only=True)["pitting_potentials"]
    iterator = ws.iter_rows(values_only=True)
    headers = [clean(value) for value in next(iterator)]
    rows = [dict(zip(headers, (clean(value) for value in row))) for row in iterator]
    return headers, rows


def load_registry() -> dict[str, dict[str, str]]:
    with SOURCE_REGISTRY.open(newline="", encoding="utf-8") as handle:
        return {row["source"]: row for row in csv.DictReader(handle)}


def target_eligibility(row: dict[str, Any]) -> tuple[bool, bool]:
    target_numeric = is_number(row.get("E_pit"))
    regression = target_numeric and row.get("event") == 1
    survival = target_numeric and row.get("event") in (0, 1)
    return regression, survival


def derive_row_material_family(row: dict[str, Any]) -> tuple[str, str]:
    source = str(row["source"])
    designation = str(row.get("alloy_designation") or "").strip().upper()
    chromium = float(row["Cr"]) if is_number(row.get("Cr")) else None
    nickel = float(row["Ni"]) if is_number(row.get("Ni")) else None

    if source == "2009Wong":
        return "ni_based_model_alloy", "exclude"
    if source == "1968Horvath" and nickel is not None and nickel > 50:
        return "high_Ni_Cr_Fe_uncertain", "review"
    if source == "1988Roberge":
        if nickel is not None and nickel > 50:
            return "ni_based_alloy_600", "exclude"
        if nickel is not None and nickel > 25:
            return "Fe_Ni_Cr_alloy_800", "review"
        return "fe_based_stainless", "include"
    if source == "1991Cortest" and designation == "G3":
        return "ni_based_G3", "exclude"
    if source == "1994Carroll" and chromium == 20 and nickel == 80:
        return "ni_based_20Cr80Ni_model", "exclude"
    if source == "1997Stellwag":
        return "Fe_Ni_Cr_alloy_800", "review"
    if source in TAXONOMY_CAVEAT_SOURCES:
        return "fe_based_high_alloy_taxonomy_caveat", "review"
    if source in MODEL_ALLOY_SOURCES:
        return "fe_based_model_alloy", "include"
    return "fe_based_stainless", "include"


def repaired_composition(row: dict[str, Any]) -> tuple[dict[str, float], dict[str, int], int]:
    values: dict[str, float] = {}
    missing: dict[str, int] = {}
    repair_applied = int(row["source"] == "2000Russell")
    for element in ELEMENTS:
        value = row.get(element)
        if repair_applied and element == "Ti":
            value = None
        missing[element] = int(not is_number(value))
        values[element] = float(value) if is_number(value) else 0.0
    return values, missing, repair_applied


def reconstruct_fe(
    row: dict[str, Any],
    composition: dict[str, float],
    family: str,
) -> tuple[float, int, int, str]:
    source = str(row["source"])
    major_reported = any(is_number(row.get(element)) for element in MAJOR_ELEMENTS)
    remainder = 100.0 - sum(composition.values())

    if source == "2009Wong":
        return 0.0, 0, 0, "explicit_Ni_Cr_Mo_ternary_sums_to_100"
    if source == "1994Carroll" and family == "ni_based_20Cr80Ni_model":
        return 0.0, 0, 0, "explicit_20Cr80Ni_binary_sums_to_100"
    if source == "1991Cortest" and family == "ni_based_G3":
        return 0.0, 1, 0, "not_reconstructed_for_G3_with_omitted_elements"
    if source == "1968Horvath" and family == "high_Ni_Cr_Fe_uncertain":
        return 0.0, 1, 0, "not_reconstructed_for_source-ambiguous_high-Ni_row"
    if source == "1988Roberge" and major_reported:
        value = 0.0 if -0.1 <= remainder < 0 else remainder
        return max(value, 0.0), 0, 1, "approximate_known_grade_remainder"
    if family.startswith("fe_based") or family.startswith("Fe_Ni_Cr"):
        if not major_reported or not is_number(row.get("Cr")):
            return 0.0, 1, 0, "not_reconstructed_when_major_composition_is_unreported"
        if remainder < -0.1:
            return 0.0, 1, 0, "not_reconstructed_negative_remainder"
        value = 0.0 if remainder < 0 else remainder
        return value, 0, 1, "approximate_100_minus_zero-filled_reported_elements"
    return 0.0, 1, 0, "not_applicable"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assert sha256(RAW_WORKBOOK) == EXPECTED_RAW_SHA256, "Raw workbook differs from benchmark input"
    raw_headers, raw_rows = load_workbook_rows()
    registry = load_registry()
    assert set(registry) == {str(row["source"]) for row in raw_rows}

    processed: list[dict[str, Any]] = []
    for raw_row_number, row in enumerate(raw_rows, start=2):
        source = str(row["source"])
        source_info = registry[source]
        family, material_scope = derive_row_material_family(row)
        composition, missing, ti_repair = repaired_composition(row)
        fe, fe_missing, fe_approximate, fe_basis = reconstruct_fe(row, composition, family)
        regression_eligible, survival_eligible = target_eligibility(row)

        output = dict(row)
        output.update(composition)
        output["Fe"] = round(fe, 12)
        for element in ["Fe", *ELEMENTS]:
            output[f"{element}_missing"] = fe_missing if element == "Fe" else missing[element]
        output.update(
            {
                "raw_workbook_row": raw_row_number,
                "Ti_repair_applied": ti_repair,
                "composition_reported_count": sum(1 - value for value in missing.values()),
                "composition_sum_without_Fe_wt_pct": round(sum(composition.values()), 12),
                "Fe_is_approximate": fe_approximate,
                "Fe_balance_basis": fe_basis,
                "Fe_is_largest_component": int(
                    not fe_missing and fe >= max(composition.values(), default=0.0)
                ),
                "row_material_family": family,
                "material_scope_primary": material_scope,
                "material_is_Fe_based": int(family.startswith("fe_based") or family.startswith("Fe_Ni_Cr")),
                "material_is_Ni_based": int(family.startswith("ni_based")),
                "material_scope_needs_review": int(material_scope == "review"),
                "source_review_status": source_info["review_status"],
                "source_classification_confidence": source_info["classification_confidence"],
                "source_has_composition_caveat": int(bool(source_info["source_specific_issue"])),
                "regression_eligible": int(regression_eligible),
                "survival_eligible": int(survival_eligible),
            }
        )
        processed.append(output)

    identifier_fields = ["label", "source", "ID", "alloy_designation", "raw_workbook_row"]
    composition_fields = ["Fe", *ELEMENTS]
    raw_other_fields = [
        field for field in raw_headers
        if field not in {"label", "source", "ID", "alloy_designation", *ELEMENTS}
    ]
    missing_fields = [f"{element}_missing" for element in composition_fields]
    audit_fields = [
        "Ti_repair_applied",
        "composition_reported_count",
        "composition_sum_without_Fe_wt_pct",
        "Fe_is_approximate",
        "Fe_balance_basis",
        "Fe_is_largest_component",
        "row_material_family",
        "material_scope_primary",
        "material_is_Fe_based",
        "material_is_Ni_based",
        "material_scope_needs_review",
        "source_review_status",
        "source_classification_confidence",
        "source_has_composition_caveat",
        "regression_eligible",
        "survival_eligible",
    ]
    output_fields = identifier_fields + composition_fields + raw_other_fields + missing_fields + audit_fields

    all_path = OUTPUT_DIR / "soccol_processed_all_rows.csv"
    regression_path = OUTPUT_DIR / "soccol_regression_event1.csv"
    survival_path = OUTPUT_DIR / "soccol_breakdown_survival.csv"
    feature_path = OUTPUT_DIR / "feature_manifest.json"
    manifest_path = OUTPUT_DIR / "preprocessing_manifest.json"

    regression_rows = [row for row in processed if row["regression_eligible"] == 1]
    survival_rows = [row for row in processed if row["survival_eligible"] == 1]
    write_csv(all_path, processed, output_fields)
    write_csv(regression_path, regression_rows, output_fields)
    write_csv(survival_path, survival_rows, output_fields)


    continuous_features = [
        "Prep_grinding_grit", "Prep_Ra_micron", "Prep_pH", "Prep_redox",
        "Prep_time", "CP_time", "CP_temp", "CP_pH", "Test_area_cm2",
        "scan_rate",
    ]
    ion_features = [
        "CP_Cl", "CP_Br", "CP_OH", "CP_SO4", "CP_CO3", "CP_NO3",
        "CP_PO4", "CP_MoO4", "CP_CrO4", "CP_ion_other",
    ]
    categorical_features = [
        "Prep_medium", "CP_aeration", "CP_agitation", "CP_anions_info",
    ]
    primary_numeric_features = composition_fields + continuous_features + ion_features
    feature_manifest = {
        "primary_regression_target": "E_pit",
        "target_unit": "mV_vs_AgAgCl_3M_KCl",
        "primary_regression_row_rule": "regression_eligible == 1 (numeric E_pit and event == 1)",
        "survival_time_or_threshold": "E_pit",
        "survival_event": "event",
        "composition_features_zero_filled": composition_fields,
        "composition_missingness_features": [],
        "composition_missingness_columns_retained_for_audit": missing_fields,
        "primary_numeric_features": primary_numeric_features,
        "primary_categorical_features": categorical_features,
        "model_input_columns": primary_numeric_features + categorical_features,
        "model_input_count": len(primary_numeric_features) + len(categorical_features),
        "train_fold_numeric_imputation_required": continuous_features,
        "zero_fill_if_blank": composition_fields + ion_features,
        "train_fold_categorical_encoding_required": categorical_features,
        "categorical_missing_and_unseen_code": -1,
        "audit_only_not_predictors": [
            "label", "source", "ID", "alloy_designation", "raw_workbook_row",
            "source_review_status", "source_classification_confidence",
            "source_has_composition_caveat", "material_scope_primary",
            "regression_eligible", "survival_eligible", "Ti_repair_applied",
            "composition_reported_count", "composition_sum_without_Fe_wt_pct",
            "Fe_is_approximate", "Fe_balance_basis", "Fe_is_largest_component",
            "row_material_family", "material_is_Fe_based", "material_is_Ni_based",
            "material_scope_needs_review", *missing_fields,
        ],
        "excluded_from_primary_predictors": {
            "E_corr": "same-experiment electrochemical response; exclude from primary ex-ante prediction task",
            "event": "target/censoring status",
        },
        "missing_environment_policy": "zero-fill ion blanks; fit continuous means and categorical mappings on context rows only",
        "real_compositions_renormalized": False,
        "missingness_indicator_predictors": False,
    }
    feature_path.write_text(json.dumps(feature_manifest, indent=2) + "\n", encoding="utf-8")

    for feature in primary_numeric_features:
        assert all(row.get(feature) is None or is_number(row.get(feature)) for row in processed), feature
    assert all(all(row[element] >= 0 for element in composition_fields) for row in processed)
    assert all(
        abs(sum(row[element] for element in composition_fields) - 100.0) < 1e-8
        for row in processed if row["Fe_is_approximate"] == 1
    )

    assert len(processed) == 4460
    assert len(regression_rows) == 4027
    assert len(survival_rows) == 4384
    assert sum(row["Ti_repair_applied"] for row in processed) == 24
    assert all(row["Ti"] == 0 and row["Ti_missing"] == 1 for row in processed if row["source"] == "2000Russell")
    assert all(row["Fe"] == 0 for row in processed if row["source"] == "2009Wong")

    family_counts = Counter(row["row_material_family"] for row in processed)
    scope_counts = Counter(row["material_scope_primary"] for row in processed)
    manifest = {
        "version": "soccol_benchmark_v1",
        "input_workbook": str(RAW_WORKBOOK.relative_to(ROOT)),
        "input_sha256": sha256(RAW_WORKBOOK),
        "source_registry": str(SOURCE_REGISTRY.relative_to(ROOT)),
        "source_registry_sha256": sha256(SOURCE_REGISTRY),
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "transformations": {
            "2000Russell_Ti": "24 PRE values removed, then zero-filled with Ti_missing=1",
            "composition": "C,N,Si,P,S,Ti,V,Cr,Mn,Ni,Nb,Mo blanks zero-filled with per-element missingness indicators",
            "Fe": "approximate remainder only where the row-level material rule and reported major composition support it; otherwise Fe=0 with Fe_missing=1",
            "material": "row-level family and include/exclude/review flags added; no rows removed by material family",
            "environment": "left in raw units with blanks preserved",
        },
        "targets": {
            "all_rows": len(processed),
            "uncensored_regression_event1": len(regression_rows),
            "censor_aware_defined_event": len(survival_rows),
        },
        "material_family_rows": dict(sorted(family_counts.items())),
        "material_scope_rows": dict(sorted(scope_counts.items())),
        "outputs": {},
    }
    for path, role in [
        (all_path, "all processed rows"),
        (regression_path, "numeric event=1 regression rows"),
        (survival_path, "numeric target with event in {0,1}"),
        (feature_path, "feature and target roles"),
    ]:
        manifest["outputs"][path.name] = {
            "role": role,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
