#!/usr/bin/env python
"""Build immutable Soccol feature-profile and target-rule pipeline assets."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline.artifact_hashes import sha256_file  # noqa: E402
from tabicl.prior.soccol_schema import (  # noqa: E402
    SOCCOL_CATEGORICAL_COLUMNS,
    SOCCOL_COMPOSITION_COLUMNS,
    SOCCOL_CONTINUOUS_COLUMNS,
    SOCCOL_FEATURE_COLUMNS,
    SOCCOL_FEATURE_PROFILE,
    SOCCOL_ION_COLUMNS,
    SOCCOL_SCHEMA_VERSION,
)

from compare_target_rules import RULES  # noqa: E402


DATA = Path(__file__).resolve().parent / "processed" / "soccol_regression_event1.csv"
SPLIT_DIR = Path(__file__).resolve().parent / "processed" / "splits_v1"
SPLIT_MANIFEST = SPLIT_DIR / "split_manifest.json"
SPLIT_LOCK = SPLIT_DIR / "split_lock.json"
COMPARISON = REPO_ROOT / "corrosion_datasets" / "analysis" / "soccol_target_rules_v1" / "comparison_summary.json"
ASSET_DIR = REPO_ROOT / "src" / "tabicl" / "prior" / "assets"
RULE_DIR = REPO_ROOT / "corrosion_datasets" / "analysis" / "soccol_pipeline" / "target_rules_v1"

RULE_SELECTION = (
    ("pren_n_linear", "old_pren_n_linear"),
    ("cr_mow_n_synergy", "old_cr_mo_n_synergy"),
    ("threshold_saturation", "old_threshold_saturation"),
    ("pren_n_improved_environment", "old_pren_n_improved_environment"),
    ("mo_n_acid_repassivation", "old_mo_n_acid_repassivation"),
    ("mns_inclusion_penalty", "old_mns_inclusion_penalty"),
    ("coupled_breakdown", "old_coupled_breakdown"),
    ("pren_n_coupled_mns", "new_pren_n_coupled_mns"),
    ("pren_n_coupled_mns_weak_anions", "new_pren_n_coupled_mns_weak_anions"),
)

TERM_RENAMES = {
    "log_halide_aggressiveness": "log_chloride_aggressiveness",
    "temperature_halide_interaction": "temperature_chloride_interaction",
}


def _normalize_category(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def build_feature_asset() -> tuple[Path, Path]:
    data = pd.read_csv(DATA)
    if len(data) != 4027:
        raise RuntimeError(f"Expected 4027 Soccol rows, found {len(data)}.")
    missing = sorted(set(SOCCOL_FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise RuntimeError(f"Soccol feature columns are missing: {missing}")

    output = pd.DataFrame(index=data.index)
    output["profile_row_id"] = [f"soccol_{int(value)}" for value in data["raw_workbook_row"]]
    output["task_row_index"] = np.arange(len(data), dtype=int)
    for column in SOCCOL_COMPOSITION_COLUMNS:
        output[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)

    imputation_means: dict[str, float] = {}
    for column in SOCCOL_CONTINUOUS_COLUMNS:
        values = pd.to_numeric(data[column], errors="coerce")
        mean = float(values.mean())
        if not np.isfinite(mean):
            raise RuntimeError(f"Cannot fit a finite synthetic-profile mean for {column}.")
        imputation_means[column] = mean
        output[column] = values.fillna(mean)
    for column in SOCCOL_ION_COLUMNS:
        output[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)

    category_mappings: dict[str, dict[str, int]] = {}
    for column in SOCCOL_CATEGORICAL_COLUMNS:
        normalized = data[column].map(_normalize_category)
        categories = sorted(value for value in normalized.dropna().unique())
        mapping = {value: index for index, value in enumerate(categories)}
        category_mappings[column] = mapping
        output[column] = normalized.map(mapping).fillna(-1).astype(int)

    csv_path = ASSET_DIR / f"{SOCCOL_FEATURE_PROFILE}.csv"
    json_path = ASSET_DIR / f"{SOCCOL_FEATURE_PROFILE}.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    metadata = {
        "profile_name": SOCCOL_FEATURE_PROFILE,
        "schema_version": SOCCOL_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(output)),
        "feature_columns": list(SOCCOL_FEATURE_COLUMNS),
        "source_file": str(DATA.relative_to(REPO_ROOT)),
        "source_sha256": sha256_file(DATA),
        "csv_sha256": sha256_file(csv_path),
        "target_columns_used": [],
        "sampling": {
            "composition_and_context_rows": "sampled independently with replacement",
            "composition_templates": "all rows",
            "composition_perturbation": "positive entries receive multiplicative log-normal noise; zeros stay zero; result closes to 100 wt.%",
        },
        "preprocessing": {
            "composition_blanks": 0,
            "ion_blanks": 0,
            "continuous_numeric_imputation_means": imputation_means,
            "categorical_missing_code": -1,
            "categorical_mappings": category_mappings,
            "no_missingness_indicators": True,
        },
        "max_process_category_count": max(len(values) for values in category_mappings.values()),
    }
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def _renamed(mapping: dict[str, float]) -> dict[str, float]:
    return {TERM_RENAMES.get(name, name): float(value) for name, value in mapping.items()}


def build_rule_assets() -> Path:
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    if comparison.get("method", {}).get("final_test_targets_used") is not False:
        raise RuntimeError("Soccol rule comparison did not keep final-test targets masked.")
    split_manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    split_lock = json.loads(SPLIT_LOCK.read_text(encoding="utf-8"))
    source_sha256 = sha256_file(DATA)
    if source_sha256 != split_manifest["dataset"]["source_sha256"] or source_sha256 != split_lock["source_sha256"]:
        raise RuntimeError("Soccol source and frozen split hashes differ.")

    result_by_name = {record["rule"]: record for record in comparison["rules"]}
    spec_by_name = {spec.name: spec for spec in RULES}
    RULE_DIR.mkdir(parents=True, exist_ok=True)
    summary_records: list[dict[str, object]] = []
    for family, comparison_name in RULE_SELECTION:
        result = result_by_name[comparison_name]
        spec = spec_by_name[comparison_name]
        coefficients = _renamed(result["all_development_calibration"]["coefficients"])
        bounds = _renamed(
            dict(zip(spec.term_names, spec.upper_bounds.tolist(), strict=True))
        )
        artifact = {
            "schema_version": "epit_target_rule_v3",
            "rule_family": family,
            "source_rule_name": comparison_name,
            "source_sha256": source_sha256,
            "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
            "split_lock_sha256": sha256_file(SPLIT_LOCK),
            "development_rows": split_manifest["split_design"]["development_rows"],
            "final_test_targets_used": False,
            "coefficient_upper_bounds": bounds,
            "final_development_calibration": {
                "coefficients": coefficients,
                "objective": float(result["all_development_calibration"]["objective"]),
            },
            "direct_evaluation": result["direct_evaluation"],
            "formula": {
                "material": result["material_formula"],
                "halide": result["halide_formula"],
                "terms": list(coefficients),
            },
        }
        artifact_path = RULE_DIR / f"{family}.json"
        artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        direct = result["direct_evaluation"]
        summary_records.append(
            {
                "rule_family": family,
                "source_rule_name": comparison_name,
                "evaluation_role": "candidate",
                "artifact": artifact_path.name,
                "artifact_sha256": sha256_file(artifact_path),
                "rows": int(direct["rows"]),
                "mean_fold_spearman": float(direct["mean_fold_spearman"]),
                "standard_deviation_fold_spearman": float(direct["std_fold_spearman"]),
                "minimum_fold_spearman": float(direct["minimum_fold_spearman"]),
                "maximum_fold_spearman": float(direct["maximum_fold_spearman"]),
                "pooled_out_of_fold_spearman": float(direct["pooled_oof_spearman"]),
            }
        )

    summary = {
        "schema_version": "epit_target_rule_calibration_summary_v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_sha256,
        "split_manifest": str(SPLIT_MANIFEST),
        "split_manifest_sha256": sha256_file(SPLIT_MANIFEST),
        "split_lock": str(SPLIT_LOCK),
        "split_lock_sha256": sha256_file(SPLIT_LOCK),
        "development_rows": split_manifest["split_design"]["development_rows"],
        "final_test_rows_excluded": split_manifest["split_design"]["final_test_rows"],
        "final_test_targets_masked_before_calibration": True,
        "final_test_targets_used": False,
        "rules": summary_records,
    }
    summary_path = RULE_DIR / "calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    csv_path, json_path = build_feature_asset()
    rules_path = build_rule_assets()
    print(csv_path)
    print(json_path)
    print(rules_path)


if __name__ == "__main__":
    main()
