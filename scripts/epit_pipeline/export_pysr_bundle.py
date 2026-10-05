#!/usr/bin/env python
"""Export the development-only EPIT data used by the standalone PySR runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from tabicl.prior.epit_schema import (
    EPIT_COMPOSITION_COLUMNS,
    EPIT_COMPOSITION_ELEMENTS,
    EPIT_ENVIRONMENT_COLUMNS,
)

from scripts.epit_pipeline.artifact_hashes import sha256_file
from scripts.epit_pipeline.run_pysr import (
    DEFAULT_SPLIT_DIR,
    REPO_ROOT,
    load_split_rows,
)
from scripts.epit_pipeline.split_data import (
    _method_family,
    _numeric,
    load_epit_dataset,
)


DEFAULT_BUNDLE_DIR = REPO_ROOT / "standalone" / "epit_pysr"
PHYSICAL_FEATURES = tuple(
    (f"{element}_wt_pct", column)
    for element, column in zip(
        EPIT_COMPOSITION_ELEMENTS,
        EPIT_COMPOSITION_COLUMNS,
        strict=True,
    )
) + (
    ("temperature_C", EPIT_ENVIRONMENT_COLUMNS[0]),
    ("chloride_M", EPIT_ENVIRONMENT_COLUMNS[1]),
    ("pH", EPIT_ENVIRONMENT_COLUMNS[2]),
)
ELIGIBLE_MATERIAL_CLASSES = ("Fe Alloy", "NiCrMo Alloy")
METHOD_CATEGORIES = ("other", "potentiodynamic", "potentiostatic", "scratch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_DIR / "split_manifest.json",
    )
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    return parser.parse_args()


def export(bundle_dir: Path, split_manifest: Path) -> tuple[Path, Path]:
    bundle_dir = bundle_dir.expanduser().resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    data_path = bundle_dir / "epit_development.csv"
    manifest_path = bundle_dir / "bundle_manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Refusing to replace an existing bundle export; remove or move it first."
        )

    dataset = load_epit_dataset()
    split = load_split_rows(split_manifest, dataset.n_rows)
    development = sorted(int(index) for index in split.development)
    fold_by_index = {
        int(index): fold
        for fold, indices in split.validation_by_fold.items()
        for index in indices
    }
    feature_names = [name for name, _ in PHYSICAL_FEATURES]

    with data_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "task_row_index",
            "source_no",
            "material_class",
            "test_method_family",
            "validation_fold",
            *feature_names,
            "epit_mV_SCE",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in development:
            record: dict[str, object] = {
                "task_row_index": row_index,
                "source_no": str(dataset.rows[row_index].get("No.") or ""),
                "material_class": str(
                    dataset.rows[row_index].get("Material class") or ""
                ).strip(),
                "test_method_family": _method_family(
                    dataset.rows[row_index].get("[Cl-] Test Method")
                ),
                "validation_fold": fold_by_index[row_index],
                "epit_mV_SCE": float(dataset.target[row_index]),
            }
            for name, column in PHYSICAL_FEATURES:
                value = _numeric(dataset.table, dataset.rows[row_index], column)
                record[name] = "" if not math.isfinite(float(value)) else float(value)
            writer.writerow(record)

    eligible_rows = sum(
        str(dataset.rows[index].get("Material class") or "").strip()
        in ELIGIBLE_MATERIAL_CLASSES
        for index in development
    )
    if eligible_rows != 452:
        raise RuntimeError(f"Expected 452 Fe/Ni development rows, found {eligible_rows}.")

    manifest = {
        "schema_version": "epit_pysr_discovery_bundle_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": data_path.name,
        "data_sha256": sha256_file(data_path),
        "development_rows": len(development),
        "final_test_rows_exported": 0,
        "final_test_targets_exported": False,
        "validation_folds": 5,
        "features": feature_names,
        "physical_features": feature_names,
        "source_columns": [column for _, column in PHYSICAL_FEATURES],
        "eligible_material_classes": list(ELIGIBLE_MATERIAL_CLASSES),
        "eligible_development_rows": eligible_rows,
        "categorical_columns": ["material_class", "test_method_family"],
        "method_categories": list(METHOD_CATEGORIES),
        "method_baseline_category": METHOD_CATEGORIES[0],
        "method_offset_categories": list(METHOD_CATEGORIES[1:]),
        "target": "Epit, mV (SCE) Avg.",
        "split_manifest_sha256": split.manifest_sha256,
        "split_lock_sha256": split.lock_sha256,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return data_path, manifest_path


def main() -> None:
    args = parse_args()
    data_path, manifest_path = export(args.bundle_dir, args.split_manifest)
    print(f"data={data_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
