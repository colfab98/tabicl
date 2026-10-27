#!/usr/bin/env python
"""Build reviewed v3 with uniform composition templates and joint context rows."""

from __future__ import annotations

import csv
import argparse
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
    SOCCOL_FEATURE_PROFILE as LEGACY_SOCCOL_FEATURE_PROFILE,
    SOCCOL_ION_COLUMNS,
    SOCCOL_SCHEMA_VERSION,
)

if __package__:
    from .compare_target_rules import RULES  # noqa: E402
else:
    from compare_target_rules import RULES  # noqa: E402


DATA = Path(__file__).resolve().parent / "processed" / "soccol_regression_event1.csv"
SPLIT_DIR = Path(__file__).resolve().parent / "processed" / "splits_v1"
SPLIT_MANIFEST = SPLIT_DIR / "split_manifest.json"
SPLIT_LOCK = SPLIT_DIR / "split_lock.json"
COMPARISON = REPO_ROOT / "corrosion_datasets" / "analysis" / "soccol_target_rules_v1" / "comparison_summary.json"
ASSET_DIR = REPO_ROOT / "src" / "tabicl" / "prior" / "assets"
RULE_DIR = REPO_ROOT / "corrosion_datasets" / "analysis" / "soccol_pipeline" / "target_rules_v1"

from scripts.soccol_pipeline.target_rules import RULE_SELECTION, TERM_RENAMES  # noqa: E402


def _normalize_category(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def load_reviewed_feature_source(dataset=None) -> pd.DataFrame:
    """Target-free accepted rows for the reviewed sampling asset.

    Select before computing any sampling probabilities, numeric filling values
    or categories. Historical 37-column assets remain reproducible separately.
    """
    from scripts.soccol_pipeline.data import select_accepted_development

    development = select_accepted_development(dataset)
    output = development.features.copy()
    output.insert(0, "task_row_index", development.task_row_indices)
    output.insert(0, "profile_row_id", development.rows.row_id.to_numpy())
    output.attrs["soccol_development_selection"] = development.identity
    return output


def build_reviewed_feature_assets(dataset=None) -> dict[str, bytes]:
    """Freeze both sampling banks and their statistics from accepted development.

    Keep every accepted observation for context, and identify one representative
    per distinct composition for uniform composition draws. No family filter.
    """
    from scripts.soccol_pipeline.data import load_soccol_dataset
    from tabicl.prior.soccol_reviewed_schema import (
        SOCCOL_REVIEWED_FEATURE_PROFILE as name,
        SOCCOL_REVIEWED_SCHEMA_VERSION as schema,
        SOCCOL_REVIEWED_FEATURE_COLUMNS as columns,
        SOCCOL_REVIEWED_COMPOSITION_COLUMNS as composition,
        SOCCOL_REVIEWED_CONTEXT_COLUMNS as context,
        SOCCOL_REVIEWED_CATEGORICAL_COLUMNS as categorical,
        SOCCOL_COMPOSITION_SAMPLING_POLICY,
    )

    dataset = load_soccol_dataset() if dataset is None else dataset
    if tuple(dataset.registry["feature_columns"]) != columns or tuple(
        c for c in columns if dataset.registry["columns"][c]["dtype"] == "categorical"
    ) != categorical:
        raise RuntimeError("Reviewed Soccol sampling schema differs from the saved registry.")
    source = load_reviewed_feature_source(dataset)
    selection = source.attrs["soccol_development_selection"]
    output = source[["profile_row_id", "task_row_index"]].copy()
    mappings, fill_values = {}, {}
    for column in columns:
        values = source[column]
        if column in categorical:
            categories = sorted(values.dropna().astype(str).unique())
            mapping = {category: index for index, category in enumerate(categories)}
            mappings[column] = mapping
            output[column] = values.map(mapping).astype(float)
            modes = output[column].mode()
            fill_values[column] = float(modes.iloc[0]) if len(modes) else 0.0
        else:
            output[column] = pd.to_numeric(values, errors="raise")
            if column in context:
                observed = output[column].dropna()
                fill_values[column] = float(observed.mean()) if len(observed) else 0.0
    if np.isinf(output[list(columns)].to_numpy()).any():
        raise RuntimeError("Reviewed Soccol sampling inputs contain infinite values.")
    csv_bytes = output.to_csv(index=False, lineterminator="\n").encode("utf-8")
    distinct = ~source[list(composition)].duplicated()
    metadata = {
        "profile_name": name, "schema_version": schema,
        "rows": len(source), "feature_columns": list(columns),
        "composition_columns": list(composition), "context_columns": list(context),
        "categorical_columns": list(categorical),
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "development_selection": selection,
        "target_columns_used": [],
        "sampling_scope": {
            "outer_partition": "development", "composition_status": "accepted",
            "material_scope": "all accepted Soccol materials; no family restriction",
            "final_query_rows_used": 0, "withheld_development_rows_used": 0,
        },
        "sampling": {
            "composition_and_context_rows": "sampled independently with replacement",
            "composition_templates": "one representative per distinct saved composition; uniform template probability; first accepted source row retained",
            "composition_sampling_policy": SOCCOL_COMPOSITION_SAMPLING_POLICY,
            "composition_template_source_task_row_indices": source.loc[distinct, "task_row_index"].astype(int).tolist(),
            "distinct_compositions": len(source[list(composition)].drop_duplicates()),
            "context_rows": "all non-composition fields and their missingness sampled together from one accepted development row",
            "composition_perturbation": "positive entries receive multiplicative log-normal noise; zeros stay zero; only synthetic compositions close to 100 wt.%",
        },
        "preprocessing": {
            "source_rows": len(source), "selection_sha256": selection["selection_sha256"],
            "category_mappings": mappings, "latent_fill_values": fill_values,
            "latent_fill_policy": "Accepted-development mean for numeric fields, mode for categories, zero only if entirely missing; internal calculations only.",
            "model_missing_policy": "Saved NaNs retained; no independent missingness mask or source-value imputation.",
        },
        "missing_counts": output[list(columns)].isna().sum().astype(int).to_dict(),
        "feature_units": {c: dataset.registry["columns"][c]["unit"] for c in columns},
        "builder": "corrosion_datasets/datasets/soccol_pitting_potential/prepare_pipeline_assets.py --reviewed-data",
    }
    return {
        f"{name}.csv": csv_bytes,
        f"{name}.json": (json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    }


def build_legacy_feature_asset() -> tuple[Path, Path]:
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

    csv_path = ASSET_DIR / f"{LEGACY_SOCCOL_FEATURE_PROFILE}.csv"
    json_path = ASSET_DIR / f"{LEGACY_SOCCOL_FEATURE_PROFILE}.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    metadata = {
        "profile_name": LEGACY_SOCCOL_FEATURE_PROFILE,
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


def build_legacy_rule_assets() -> Path:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--reviewed-data", dest="reviewed_data", action="store_true",
        help="Use the reviewed 58-column v3 sampling bank (default).",
    )
    mode.add_argument(
        "--legacy-data", dest="reviewed_data", action="store_false",
        help="Explicitly rebuild the historical 37-column v1 bank and rule artifacts.",
    )
    parser.set_defaults(reviewed_data=True)
    parser.add_argument("--inspect-inputs", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--check", action="store_true", help="Verify the reviewed bank matches its accepted development inputs.")
    args = parser.parse_args(argv)
    if not args.reviewed_data and (args.inspect_inputs or args.check or args.output_dir != ASSET_DIR):
        parser.error("--inspect-inputs, --check and --output-dir require the reviewed route; omit --legacy-data")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.reviewed_data:
        if not args.inspect_inputs:
            assets = build_reviewed_feature_assets()
            if args.check:
                for name, content in assets.items():
                    if (args.output_dir / name).read_bytes() != content:
                        raise RuntimeError(f"Reviewed Soccol bank differs from accepted development inputs: {name}")
                print("Reviewed Soccol bank matches its accepted development inputs.")
                return
            if any((args.output_dir / name).exists() for name in assets):
                raise FileExistsError("Refusing to overwrite a versioned Soccol feature asset; use --check to verify it.")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for name, content in assets.items():
                (args.output_dir / name).write_bytes(content)
                print(args.output_dir / name)
            return
        source = load_reviewed_feature_source()
        selection = source.attrs["soccol_development_selection"]
        print(json.dumps({
            "sampling_source_rows": len(source),
            "fold_counts": selection["fold_counts"],
            "selection_sha256": selection["selection_sha256"],
            "target_columns_used": [],
            "final_query_rows_used": 0,
        }, indent=2))
        return
    csv_path, json_path = build_legacy_feature_asset()
    rules_path = build_legacy_rule_assets()
    print(csv_path)
    print(json_path)
    print(rules_path)


if __name__ == "__main__":
    main()
