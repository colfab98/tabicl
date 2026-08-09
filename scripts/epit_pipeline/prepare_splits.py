#!/usr/bin/env python
"""Create, validate, and report the fixed composition-separated EPIT splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scripts.epit_pipeline.artifact_hashes import (
    SPLIT_LOCK_NAME,
    build_split_lock,
    sha256_file,
)
from scripts.epit_pipeline.split_data import (
    BALANCE_BLOCK_WEIGHTS,
    COMPOSITION_DISTANCE_THRESHOLD,
    COMPOSITION_ROUND_DECIMALS,
    DATASET_ID,
    EXPECTED_ROW_COUNT,
    INNER_FOLD_COUNT,
    OUTER_TEST_ROWS,
    REPO_ROOT,
    SOURCE_FILE,
    TABLE_NAME,
    TARGET_COLUMN,
    TASK_ID,
    CompositionGrouping,
    EpitDataset,
    SplitAssignment,
    _numeric,
    build_composition_groups,
    build_split_assignment,
    load_epit_dataset,
)
from scripts.epit_pipeline.split_report import build_split_report_html
from scripts.epit_pipeline.split_refinement import refine_inner_folds


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "corrosion_datasets" / "analysis" / "epit_pipeline" / "splits_v2"
)
MANIFEST_NAME = "split_manifest.json"
CSV_NAME = "split_assignments.csv"
REPORT_NAME = "split_report.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--outer-time-limit",
        type=float,
        default=20.0,
        help="Maximum seconds for the balanced 608/152 optimization.",
    )
    parser.add_argument(
        "--inner-time-limit",
        type=float,
        default=60.0,
        help="Maximum seconds for the five-fold development optimization.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing split artifacts in the output directory.",
    )
    return parser.parse_args()


def _row_assignment(
    row_index: int,
    grouping: CompositionGrouping,
    assignment: SplitAssignment,
) -> tuple[int, str, str, int | None]:
    group_index = int(grouping.row_group_index[row_index])
    outer_split = str(assignment.outer_split_by_group[group_index])
    fold = int(assignment.validation_fold_by_group[group_index])
    return (
        group_index,
        grouping.group_ids[group_index],
        outer_split,
        None if fold < 0 else fold + 1,
    )


def build_manifest(
    dataset: EpitDataset,
    grouping: CompositionGrouping,
    assignment: SplitAssignment,
    *,
    dataset_sha256: str,
) -> dict[str, Any]:
    rows = []
    for row_index in range(dataset.n_rows):
        _, group_id, outer_split, fold = _row_assignment(row_index, grouping, assignment)
        rows.append(
            {
                "task_row_index": row_index,
                "source_no": str(dataset.rows[row_index].get("No.") or ""),
                "composition_group": group_id,
                "outer_split": outer_split,
                "optuna_validation_fold": fold,
            }
        )

    group_records = []
    for group_index, group_id in enumerate(grouping.group_ids):
        fold = int(assignment.validation_fold_by_group[group_index])
        group_records.append(
            {
                "composition_group": group_id,
                "row_count": len(grouping.group_rows[group_index]),
                "task_row_indices": grouping.group_rows[group_index],
                "outer_split": str(assignment.outer_split_by_group[group_index]),
                "optuna_validation_fold": None if fold < 0 else fold + 1,
                "maximum_within_group_l1_wt_percent": float(grouping.diameters[group_index]),
                "nearest_other_group_l1_wt_percent": float(
                    grouping.isolation_distances[group_index]
                ),
            }
        )

    fold_row_counts = {
        str(fold + 1): int(grouping.sizes[assignment.validation_fold_by_group == fold].sum())
        for fold in range(INNER_FOLD_COUNT)
    }
    return {
        "schema_version": "epit_split_manifest_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "dataset_id": DATASET_ID,
            "table": TABLE_NAME,
            "task_id": TASK_ID,
            "source_file": str(SOURCE_FILE.relative_to(REPO_ROOT)),
            "source_sha256": dataset_sha256,
            "usable_rows": dataset.n_rows,
            "target_column": TARGET_COLUMN,
            "model_visible_composition_columns": dataset.composition_columns,
        },
        "composition_grouping": {
            "normalization": "none",
            "round_decimal_places": COMPOSITION_ROUND_DECIMALS,
            "blank_handling_for_distance": "zero",
            "distance": "sum_absolute_difference_wt_percent",
            "clustering": "threshold_connected_components",
            "pair_separation_rule": (
                "Every pair at or below the distance threshold belongs to the same "
                "atomic composition component."
            ),
            "maximum_link_distance": COMPOSITION_DISTANCE_THRESHOLD,
            "maximum_component_diameter": float(grouping.diameters.max(initial=0.0)),
            "minimum_between_component_distance": float(
                grouping.isolation_distances.min(initial=np.inf)
            ),
            "rounded_composition_keys": grouping.rounded_key_count,
            "composition_groups": grouping.n_groups,
        },
        "split_design": {
            "development_rows": EXPECTED_ROW_COUNT - OUTER_TEST_ROWS,
            "final_test_rows": OUTER_TEST_ROWS,
            "optuna_validation_folds": INNER_FOLD_COUNT,
            "fold_meaning": (
                "Rows assigned to fold N are validation for Optuna fold N; all other "
                "development rows are context."
            ),
            "balance_block_weights": BALANCE_BLOCK_WEIGHTS,
            "target_balance": "global EPIT deciles",
            "environment_balance_bins": {
                "temperature_celsius": [10.0, 35.0, 70.0],
                "chloride_molar": [0.0, 0.01, 0.1, 0.6, 1.0],
                "pH": [3.0, 6.0, 8.0],
            },
        },
        "optimizer": {
            "outer": asdict(assignment.outer_optimizer),
            "inner": asdict(assignment.inner_optimizer),
        },
        "checks": {
            "outer_composition_overlap": 0,
            "inner_composition_overlap_each_fold": [0] * INNER_FOLD_COUNT,
            "outer_close_composition_pair_crossings": 0,
            "inner_close_composition_pair_crossings_each_fold": [0] * INNER_FOLD_COUNT,
            "close_composition_pair_distance_threshold": COMPOSITION_DISTANCE_THRESHOLD,
            "development_row_count": EXPECTED_ROW_COUNT - OUTER_TEST_ROWS,
            "final_test_row_count": OUTER_TEST_ROWS,
            "optuna_validation_row_counts": fold_row_counts,
        },
        "groups": group_records,
        "rows": rows,
    }


def _csv_value(dataset: EpitDataset, row: dict[str, Any], column: str) -> str:
    value = _numeric(dataset.table, row, column)
    return "" if not math.isfinite(value) else f"{value:g}"


def write_assignment_csv(
    path: Path,
    dataset: EpitDataset,
    grouping: CompositionGrouping,
    assignment: SplitAssignment,
    labels: dict[str, np.ndarray],
) -> None:
    fixed_fields = [
        "task_row_index",
        "source_no",
        "composition_group",
        "composition_group_rows",
        "outer_split",
        "optuna_validation_fold",
        "material_class",
        "reference",
        "temperature_celsius",
        "chloride_molar",
        "pH",
        "test_method",
        "target_decile",
        "target_mV_SCE_development_only",
    ]
    fieldnames = fixed_fields + dataset.composition_columns
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, row in enumerate(dataset.rows):
            group_index, group_id, outer_split, fold = _row_assignment(
                row_index, grouping, assignment
            )
            record: dict[str, Any] = {
                "task_row_index": row_index,
                "source_no": str(row.get("No.") or ""),
                "composition_group": group_id,
                "composition_group_rows": len(grouping.group_rows[group_index]),
                "outer_split": outer_split,
                "optuna_validation_fold": "" if fold is None else fold,
                "material_class": str(row.get("Material class") or ""),
                "reference": str(row.get("Reference") or ""),
                "temperature_celsius": _csv_value(dataset, row, "Test Temp. oC"),
                "chloride_molar": _csv_value(dataset, row, "[Cl-] M"),
                "pH": _csv_value(dataset, row, "[Cl-] pH"),
                "test_method": str(row.get("[Cl-] Test Method") or ""),
                "target_decile": int(labels["target_decile"][row_index]) + 1,
                "target_mV_SCE_development_only": (
                    f"{dataset.target[row_index]:g}"
                    if outer_split == "development"
                    else ""
                ),
            }
            record.update(
                {column: _csv_value(dataset, row, column) for column in dataset.composition_columns}
            )
            writer.writerow(record)


def _ensure_output_paths(
    output_dir: Path,
    force: bool,
) -> tuple[Path, Path, Path, Path]:
    manifest_path = output_dir / MANIFEST_NAME
    csv_path = output_dir / CSV_NAME
    report_path = output_dir / REPORT_NAME
    lock_path = output_dir / SPLIT_LOCK_NAME
    if lock_path.exists():
        raise SystemExit(
            f"Refusing to replace frozen split {output_dir}. Create a new versioned directory."
        )
    existing = [path for path in (manifest_path, csv_path, report_path) if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(
            f"Refusing to replace existing artifacts ({names}). Use --force intentionally."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return manifest_path, csv_path, report_path, lock_path


def main() -> None:
    args = parse_args()
    if args.outer_time_limit <= 0 or args.inner_time_limit <= 0:
        raise SystemExit("Optimizer time limits must be positive.")

    manifest_path, csv_path, report_path, lock_path = _ensure_output_paths(
        args.output_dir.resolve(), args.force
    )
    dataset = load_epit_dataset()
    grouping = build_composition_groups(dataset)
    assignment, labels = build_split_assignment(
        dataset,
        grouping,
        outer_time_limit_seconds=args.outer_time_limit,
        inner_time_limit_seconds=args.inner_time_limit,
    )
    assignment = refine_inner_folds(dataset, grouping, assignment, labels)
    dataset_sha256 = sha256_file(SOURCE_FILE)
    manifest = build_manifest(
        dataset,
        grouping,
        assignment,
        dataset_sha256=dataset_sha256,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_assignment_csv(csv_path, dataset, grouping, assignment, labels)
    report_path.write_text(
        build_split_report_html(
            dataset,
            grouping,
            assignment,
            labels,
            dataset_sha256=dataset_sha256,
            manifest_name=MANIFEST_NAME,
            csv_name=CSV_NAME,
        ),
        encoding="utf-8",
    )
    lock = build_split_lock(
        manifest_path=manifest_path,
        assignments_path=csv_path,
        report_path=report_path,
        source_sha256=dataset_sha256,
    )
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Created {manifest_path}")
    print(f"Created {csv_path}")
    print(f"Created {report_path}")
    print(f"Frozen by {lock_path}")
    print(
        f"Rows: {dataset.n_rows}; groups: {grouping.n_groups}; "
        f"development/final: {dataset.n_rows - OUTER_TEST_ROWS}/{OUTER_TEST_ROWS}"
    )


if __name__ == "__main__":
    main()
