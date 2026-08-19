#!/usr/bin/env python3
"""Build the versioned, target-free Fe/Ni--Cr--Mo EPIT feature profile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import numpy as np

from scripts.epit_pipeline.split_data import (
    SOURCE_FILE,
    TABLE_NAME,
    TARGET_COLUMN,
    _method_family,
    _numeric,
    load_epit_dataset,
)


ASSET_DIR = REPO_ROOT / "src" / "tabicl" / "prior" / "assets"
PROFILE_NAME = "epit_fe_nicrmo_features_v1"
COMPOSITION_PROFILE_NAME = "epit_dataset_v1"
FAMILY_DISPLAY_NAMES = {
    "fe_alloy": "Fe Alloy",
    "nicrmo_alloy": "NiCrMo Alloy",
}
ENVIRONMENT_SOURCE_COLUMNS = (
    "Test Temp. oC",
    "[Cl-] M",
    "[Cl-] pH",
)
ENVIRONMENT_OUTPUT_COLUMNS = (
    "temperature_celsius",
    "chloride_molar",
    "pH",
)
METHOD_COLUMN = "[Cl-] Test Method"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: float) -> str:
    return "" if not math.isfinite(value) else format(float(value), ".17g")


def _normalized_method(value: Any) -> str:
    stripped = "" if value is None else str(value).strip()
    return "<NA>" if stripped in {"", "NA", "N/A", "nan", "None"} else stripped


def build_profile_bytes() -> tuple[bytes, bytes]:
    dataset = load_epit_dataset()
    all_rows = dataset.rows
    family_by_display = {display: family for family, display in FAMILY_DISPLAY_NAMES.items()}
    selected_rows = [
        (task_row_index, row, family_by_display[str(row.get("Material class") or "").strip()])
        for task_row_index, row in enumerate(all_rows)
        if str(row.get("Material class") or "").strip() in family_by_display
    ]

    numeric_columns = [*dataset.composition_columns, *ENVIRONMENT_SOURCE_COLUMNS]
    numeric_imputation_means: dict[str, float] = {}
    for column in numeric_columns:
        values = np.asarray([_numeric(dataset.table, row, column) for row in all_rows], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise RuntimeError(f"Cannot reproduce evaluator mean imputation for {column!r}.")
        numeric_imputation_means[column] = float(finite.mean())

    all_method_categories = sorted({_normalized_method(row.get(METHOD_COLUMN)) for row in all_rows})
    category_mapping = {category: index for index, category in enumerate(all_method_categories)}

    output = io.StringIO(newline="")
    fieldnames = [
        "environment_id",
        "task_row_index",
        "source_no",
        "family",
        *ENVIRONMENT_OUTPUT_COLUMNS,
        "test_method",
        "test_method_family",
        "test_method_category",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    family_environment_counts = {family: 0 for family in FAMILY_DISPLAY_NAMES}
    missing_counts = {column: 0 for column in ENVIRONMENT_OUTPUT_COLUMNS}
    selected_methods: set[str] = set()
    for environment_index, (task_row_index, row, family) in enumerate(selected_rows, start=1):
        values = [
            _numeric(dataset.table, row, source_column)
            for source_column in ENVIRONMENT_SOURCE_COLUMNS
        ]
        method = "" if row.get(METHOD_COLUMN) is None else str(row.get(METHOD_COLUMN)).strip()
        normalized_method = _normalized_method(method)
        record = {
            "environment_id": f"epit_environment_{environment_index:04d}",
            "task_row_index": task_row_index,
            "source_no": str(row.get("No.") or ""),
            "family": family,
            **{
                output_column: _number(value)
                for output_column, value in zip(ENVIRONMENT_OUTPUT_COLUMNS, values, strict=True)
            },
            "test_method": method,
            "test_method_family": _method_family(row.get(METHOD_COLUMN)),
            "test_method_category": category_mapping[normalized_method],
        }
        writer.writerow(record)
        family_environment_counts[family] += 1
        selected_methods.add(normalized_method)
        for output_column, value in zip(ENVIRONMENT_OUTPUT_COLUMNS, values, strict=True):
            missing_counts[output_column] += int(not math.isfinite(value))

    csv_bytes = output.getvalue().encode("utf-8")
    composition_json_path = ASSET_DIR / f"{COMPOSITION_PROFILE_NAME}.json"
    composition_csv_path = ASSET_DIR / f"{COMPOSITION_PROFILE_NAME}.csv"
    composition_metadata = json.loads(composition_json_path.read_text(encoding="utf-8"))
    family_template_counts = composition_metadata["family_template_counts"]
    metadata = {
        "profile_name": PROFILE_NAME,
        "profile_version": 1,
        "description": (
            "Target-free Fe and Ni-Cr-Mo feature profile for the fixed EPIT task; "
            "environment values remain in observed row-wise tuples."
        ),
        "source_dataset": "electrochemical_metrics_alloys",
        "source_sheet": TABLE_NAME,
        "source_workbook": str(SOURCE_FILE.relative_to(REPO_ROOT)),
        "source_workbook_sha256": _sha256(SOURCE_FILE),
        "row_filter": (
            f"finite {TARGET_COLUMN}; target used only to reproduce the 760 evaluated task rows "
            "and target values are neither stored nor used in profile statistics"
        ),
        "source_rows": len(dataset.table.rows),
        "usable_task_rows": len(all_rows),
        "families": list(FAMILY_DISPLAY_NAMES),
        "family_display_names": FAMILY_DISPLAY_NAMES,
        "environment_rows": len(selected_rows),
        "family_environment_counts": family_environment_counts,
        "environment_sampling": (
            "sample complete row-wise tuples from the pooled Fe/Ni-Cr-Mo rows; do not condition "
            "environment selection on generated composition family"
        ),
        "source_environment_columns": list(ENVIRONMENT_SOURCE_COLUMNS),
        "environment_columns": list(ENVIRONMENT_OUTPUT_COLUMNS),
        "environment_missing_counts": missing_counts,
        "missing_value_encoding": "empty CSV field",
        "composition_profile": {
            "name": COMPOSITION_PROFILE_NAME,
            "json_sha256": _sha256(composition_json_path),
            "csv_sha256": _sha256(composition_csv_path),
            "reuse": "reference existing 24-element templates without copying them",
            "allowed_families": list(FAMILY_DISPLAY_NAMES),
            "allowed_family_template_counts": {
                family: int(family_template_counts[family]) for family in FAMILY_DISPLAY_NAMES
            },
        },
        "evaluator_preprocessing": {
            "numeric_columns": numeric_columns,
            "numeric_missing": (
                "mean-impute over all 760 usable task rows before feature standardization"
            ),
            "numeric_imputation_means": numeric_imputation_means,
            "categorical_column": METHOD_COLUMN,
            "categorical_missing_token": "<NA>",
            "categorical_encoding": (
                "lexicographically sort unique strings over all 760 usable task rows and assign "
                "contiguous zero-based ordinal codes before feature standardization"
            ),
            "category_count": len(category_mapping),
            "category_mapping": category_mapping,
            "profile_category_count": len(selected_methods),
        },
        "builder": "scripts/build_epit_feature_profile.py",
        "csv_file": f"{PROFILE_NAME}.csv",
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
    }
    json_bytes = (json.dumps(metadata, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return csv_bytes, json_bytes


def _write(path: Path, content: bytes, *, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Refusing to replace existing versioned asset: {path}. Use --force.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Rebuild in memory and verify that the versioned assets are byte-for-byte current.",
    )
    args = parser.parse_args()
    csv_bytes, json_bytes = build_profile_bytes()
    csv_path = args.output_dir / f"{PROFILE_NAME}.csv"
    json_path = args.output_dir / f"{PROFILE_NAME}.json"
    if args.check:
        mismatches = [
            str(path)
            for path, expected in ((csv_path, csv_bytes), (json_path, json_bytes))
            if not path.exists() or path.read_bytes() != expected
        ]
        if mismatches:
            raise SystemExit("EPIT feature profile is stale or missing: " + ", ".join(mismatches))
        print(f"Verified {PROFILE_NAME}: {len(csv_bytes)} CSV bytes, {len(json_bytes)} JSON bytes")
        return
    _write(csv_path, csv_bytes, force=args.force)
    _write(json_path, json_bytes, force=args.force)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
