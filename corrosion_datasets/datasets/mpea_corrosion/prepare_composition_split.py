#!/usr/bin/env python3
"""Freeze an MPEA development/final-test split using the EPIT/Soccol method.

Run from the repository root with ``python -m
corrosion_datasets.datasets.mpea_corrosion.prepare_composition_split``.
Existing frozen splits are never overwritten. This does not fit formulas or models.
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.epit_pipeline.artifact_hashes import (
    FrozenSplit,
    build_split_lock,
    load_frozen_split,
    sha256_file,
)
from scripts.epit_pipeline.split_data import (
    BALANCE_BLOCK_WEIGHTS,
    COMPOSITION_DISTANCE_THRESHOLD,
    COMPOSITION_ROUND_DECIMALS,
    INNER_FOLD_COUNT,
    EpitDataset,
    SplitAssignment,
    Table,
    build_balance_features,
    build_balance_labels,
    build_composition_groups,
    optimize_inner_folds,
    optimize_outer_split,
    validate_split_assignment,
)
from scripts.epit_pipeline.split_refinement import refine_inner_folds


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SHARED_DIR = HERE.parent / "combined_epit_soccol_mpea" / "processed"
DEFAULT_OUTPUT_DIR = HERE / "processed" / "splits_v1"
EXPECTED_ROWS = 267
FINAL_TEST_ROWS = round(EXPECTED_ROWS * 0.20)
TARGET = "reported_pitting_potential_mV"
MANIFEST_SCHEMA = "mpea_composition_split_manifest_v1"
INPUT_FILES = {
    "queries": SHARED_DIR / "mpea_queries.csv",
    "targets": SHARED_DIR / "mpea_scoring_targets.csv",
    "registry": SHARED_DIR / "shared_feature_registry.json",
}
OMITTED_BLOCKS = {
    "material_class": "No comparable recorded material-family classification in the MPEA inputs.",
    "test_method_family": "No recorded test-method values in the MPEA inputs.",
}


def load_mpea() -> tuple[EpitDataset, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Adapt the existing shared table to the established split helpers only.

    Aliases below support the EPIT grouping/balance code; they do not change
    exported feature values, missingness, units, or column order.
    """
    registry = json.loads(INPUT_FILES["registry"].read_text())
    queries = pd.read_csv(INPUT_FILES["queries"])
    targets = pd.read_csv(INPUT_FILES["targets"])
    features = registry["feature_columns_in_order"]
    composition = [name for name in features if registry["feature_units"][name] == "wt%"]
    if len(queries) != EXPECTED_ROWS or len(targets) != EXPECTED_ROWS:
        raise ValueError(f"Expected the existing {EXPECTED_ROWS}-row MPEA subset.")
    if len(features) != 50 or len(composition) != 24:
        raise ValueError("The existing shared feature schema changed.")
    if [name for name in queries.columns if name in features] != features or TARGET in queries:
        raise ValueError("MPEA query schema is invalid.")
    for frame in (queries, targets):
        if frame.row_id.isna().any() or not frame.row_id.is_unique:
            raise ValueError("MPEA row IDs must be present and unique.")
        if not frame.reference_type.eq("SCE").all():
            raise ValueError("Expected unchanged SCE targets and reference labels.")
    if set(queries.row_id) != set(targets.row_id):
        raise ValueError("Query and target row IDs differ.")
    targets = targets.set_index("row_id").loc[queries.row_id].reset_index()
    y = pd.to_numeric(targets[TARGET], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("MPEA targets must be finite.")
    if queries.test_method.notna().any():
        raise ValueError("MPEA now has method data; review the omitted method balance block.")
    aliases = {
        "CP_temp": "Test Temp. oC",
        "CP_Cl": "[Cl-] M",
        "CP_pH": "[Cl-] pH",
        "source_reference": "Reference",
    }
    adapted = queries.copy()
    for source, alias in aliases.items():
        adapted[alias] = adapted[source]
    rows = adapted.to_dict("records")
    table = Table(
        dataset="mpea_corrosion", table="source_screened_pitting_potential",
        columns=list(adapted.columns), groups={name: "material" for name in composition}, rows=rows,
    )
    values = queries[composition].to_numpy(dtype=float)
    # These EPIT-specific eligibility fields are not used for splitting. Do not
    # claim MPEA prior-template eligibility before the separate formula audit.
    dataset = EpitDataset(
        table=table, rows=rows, target=y, composition_columns=composition,
        composition_reported_sums=np.nansum(values, axis=1),
        composition_missing_counts=np.isnan(values).sum(axis=1),
        composition_structural_zero_counts=np.zeros(len(rows), dtype=int),
        pretraining_template_eligible=np.zeros(len(rows), dtype=bool),
    )
    return dataset, queries, targets, registry


def load_frozen_mpea_split(output_dir: Path = DEFAULT_OUTPUT_DIR) -> FrozenSplit:
    """Verify the split, its input files, and every exported table before reuse."""
    frozen = load_frozen_split(
        output_dir / "split_manifest.json", expected_manifest_schema=MANIFEST_SCHEMA,
    )
    for record in frozen.manifest["inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Frozen MPEA input changed: {path}")
    for record in frozen.lock["exported_tables"].values():
        path = frozen.manifest_path.parent / record["file"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Frozen MPEA export changed: {path}")
    return frozen


def prepare_split(output_dir: Path, *, outer_time_limit: float, inner_time_limit: float) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace existing split artifacts: {output_dir}")
    dataset, queries, targets, registry = load_mpea()
    grouping = build_composition_groups(dataset)
    labels = {
        key: value for key, value in build_balance_labels(dataset, grouping).items()
        if key not in OMITTED_BLOCKS
    }
    features = build_balance_features(labels, grouping)
    print(f"Grouping: {dataset.n_rows} rows, {grouping.n_groups} composition groups.", flush=True)
    outer, outer_record = optimize_outer_split(
        grouping, features, time_limit_seconds=outer_time_limit, outer_test_rows=FINAL_TEST_ROWS,
    )
    print("Outer split assigned; creating five development folds.", flush=True)
    folds, inner_record = optimize_inner_folds(
        grouping, features, outer, time_limit_seconds=inner_time_limit, outer_test_rows=FINAL_TEST_ROWS,
    )
    assignment = refine_inner_folds(
        dataset, grouping, SplitAssignment(outer, folds, outer_record, inner_record), labels,
        outer_test_rows=FINAL_TEST_ROWS,
    )
    validate_split_assignment(dataset, grouping, assignment, outer_test_rows=FINAL_TEST_ROWS)
    rows = []
    for index, row_id in enumerate(queries.row_id):
        group = int(grouping.row_group_index[index])
        fold = int(assignment.validation_fold_by_group[group])
        rows.append({
            "task_row_index": index,
            "row_id": row_id,
            "source_reference": str(queries.iloc[index].source_reference),
            "composition_group": grouping.group_ids[group],
            "outer_split": str(assignment.outer_split_by_group[group]),
            "optuna_validation_fold": None if fold < 0 else fold + 1,
        })
    row_assignments = pd.DataFrame(rows)
    row_assignments["optuna_validation_fold"] = row_assignments.optuna_validation_fold.astype("Int64")
    development = row_assignments.outer_split.eq("development").to_numpy()
    final_test = ~development
    fold_counts = {
        str(fold + 1): int(grouping.sizes[assignment.validation_fold_by_group == fold].sum())
        for fold in range(INNER_FOLD_COUNT)
    }
    sources_dev = set(queries.loc[development, "source_reference"])
    sources_test = set(queries.loc[final_test, "source_reference"])
    checks = {
        "development_row_count": int(development.sum()),
        "final_test_row_count": int(final_test.sum()),
        "optuna_validation_row_counts": fold_counts,
        "outer_composition_overlap": 0,
        "inner_composition_overlap_each_fold": [0] * INNER_FOLD_COUNT,
        "outer_close_composition_pair_crossings": 0,
        "inner_close_composition_pair_crossings_each_fold": [0] * INNER_FOLD_COUNT,
        "source_overlap": {
            "development_sources": len(sources_dev), "final_test_sources": len(sources_test),
            "shared_sources": len(sources_dev & sources_test),
        },
    }
    inputs = {
        name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
        for name, path in INPUT_FILES.items()
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "dataset_id": "mpea_corrosion", "usable_rows": dataset.n_rows,
            "source_file": inputs["queries"]["path"], "source_sha256": inputs["queries"]["sha256"],
            "target_column": TARGET, "target_unit": "mV vs SCE",
            "feature_columns_in_order": registry["feature_columns_in_order"],
            "model_visible_composition_columns": dataset.composition_columns,
        },
        "inputs": inputs,
        "composition_grouping": {
            "normalization": "none", "round_decimal_places": COMPOSITION_ROUND_DECIMALS,
            "blank_handling_for_distance": "zero for grouping only; exported missing values unchanged",
            "distance": "sum_absolute_difference_wt_percent", "clustering": "threshold_connected_components",
            "maximum_link_distance": COMPOSITION_DISTANCE_THRESHOLD,
            "composition_groups": grouping.n_groups, "rounded_composition_keys": grouping.rounded_key_count,
            "maximum_component_diameter": float(grouping.diameters.max()),
            "minimum_between_component_distance": float(grouping.isolation_distances.min()),
        },
        "split_design": {
            "development_rows": int(development.sum()), "final_test_rows": int(final_test.sum()),
            "optuna_validation_folds": INNER_FOLD_COUNT,
            "fold_meaning": "Fold N is an Optuna/development validation query; the other development folds are context.",
            "final_evaluation": "All development rows are context; the frozen outer final-test rows are queries.",
            "balance_block_weights": {key: BALANCE_BLOCK_WEIGHTS[key] for key in labels},
            "omitted_balance_blocks": OMITTED_BLOCKS,
            "target_balance": "global target deciles, as in EPIT and Soccol",
            "source_role": "balance and audit only, not composition grouping",
            "environment_balance_bins": {"CP_temp": [10.0, 35.0, 70.0], "CP_Cl": [0.0, 0.01, 0.1, 0.6, 1.0], "CP_pH": [3.0, 6.0, 8.0]},
            "cross_dataset_rebalancing": False,
        },
        "usage": {
            "development": "Formula checks, coefficient fitting, prior construction, Optuna and checkpoint selection only on development rows.",
            "final_test": "Scoring the frozen chosen model; never used as context or for fitting or selection.",
            "prior_exposure": "These 267 rows were previously used in external and MPEA-only exploratory evaluations; this is not a newly collected untouched dataset.",
            "previous_protocols": "The original 267 external queries and exploratory fivefold assignments are separate and unchanged.",
        },
        "optimizer": {"outer": asdict(assignment.outer_optimizer), "inner": asdict(assignment.inner_optimizer)},
        "implementation_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in (
                Path(__file__), ROOT / "scripts/epit_pipeline/split_data.py",
                ROOT / "scripts/epit_pipeline/split_refinement.py",
            )
        },
        "checks": checks,
        "groups": [{
            "composition_group": group_id, "row_count": len(grouping.group_rows[group]),
            "task_row_indices": grouping.group_rows[group],
            "outer_split": str(assignment.outer_split_by_group[group]),
            "optuna_validation_fold": None if assignment.validation_fold_by_group[group] < 0 else int(assignment.validation_fold_by_group[group]) + 1,
        } for group, group_id in enumerate(grouping.group_ids)],
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = output_dir / "split_assignments.csv"
    row_assignments.to_csv(assignment_path, index=False)
    development_table = queries.loc[development].copy()
    development_table[TARGET] = dataset.target[development]
    exports = {
        "development": ("development.csv", development_table),
        "final_test_queries": ("final_test_queries.csv", queries.loc[final_test]),
        "final_test_targets": ("final_test_scoring_targets.csv", targets.loc[final_test]),
    }
    for filename, frame in exports.values():
        frame.to_csv(output_dir / filename, index=False, na_rep="NaN")
    registry_path = output_dir / "shared_feature_registry.json"
    registry_path.write_bytes(INPUT_FILES["registry"].read_bytes())
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    report_path = output_dir / "split_report.html"
    report_path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>MPEA split</title>'
        '<body><h1>Frozen MPEA split</h1><p>EPIT/Soccol composition grouping and 80/20 design.</p>'
        '<p>Final-test targets are stored separately. No composition group crosses partitions.</p>'
        f'<pre>{html.escape(json.dumps(checks, indent=2))}</pre></body></html>\n'
    )
    lock = build_split_lock(
        manifest_path=manifest_path, assignments_path=assignment_path, report_path=report_path,
        source_sha256=inputs["queries"]["sha256"],
    )
    filenames = [filename for filename, _ in exports.values()] + [registry_path.name]
    lock["exported_tables"] = {
        name: {"file": name, "sha256": sha256_file(output_dir / name)} for name in filenames
    }
    (output_dir / "split_lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    load_frozen_mpea_split(output_dir)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outer-time-limit", type=float, default=20.0)
    parser.add_argument("--inner-time-limit", type=float, default=60.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        checks = load_frozen_mpea_split(args.output_dir).manifest["checks"]
    else:
        checks = prepare_split(
            args.output_dir, outer_time_limit=args.outer_time_limit, inner_time_limit=args.inner_time_limit,
        )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
