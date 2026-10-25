"""Export corrected EPIT compositions without changing existing model pipelines.

The original non-composition columns are copied verbatim. Frozen EPIT row and
fold assignments are copied, not recomputed. The previous shared table is read
only to record the changes against the dataset that was audited.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.epit_pipeline.artifact_hashes import load_frozen_split, sha256_file
from scripts.epit_pipeline.split_data import (
    EXPECTED_ROW_COUNT,
    REPO_ROOT,
    SOURCE_FILE,
    TABLE_NAME,
    TARGET_COLUMN,
    _numeric,
    load_electrochemical_metrics,
)
from tabicl.prior.epit_schema import EPIT_COMPOSITION_COLUMNS, EPIT_COMPOSITION_ELEMENTS


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "processed/epit_composition_v1"
SPLIT_MANIFEST = REPO_ROOT / "corrosion_datasets/analysis/epit_pipeline/splits_v2/split_manifest.json"
PREVIOUS_SHARED = (
    REPO_ROOT
    / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed/epit_soccol_context.csv"
)
ELEMENTS = list(EPIT_COMPOSITION_ELEMENTS)
CLOSURE_TOLERANCE = 1.0
MISSING_TEXT = {"", "na", "n/a", "nan", "none", "null", "-", "--"}
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
RENAME = dict(zip(EPIT_COMPOSITION_COLUMNS, ELEMENTS, strict=True))


def parse_amount(value: Any) -> tuple[float, str]:
    """Keep numeric amounts and explicitly mark upper-limit approximations."""
    text = "" if value is None else str(value).strip()
    if text.lower() in MISSING_TEXT:
        return math.nan, "missing"
    match = re.fullmatch(rf"<\s*=?\s*({NUMBER})", text)
    if match:
        amount, action = float(match[1]), "upper_limit_as_numeric"
    elif re.fullmatch(NUMBER, text):
        amount, action = float(text), "reported_numeric"
    else:
        raise ValueError(f"Unsupported EPIT composition notation: {value!r}")
    if not math.isfinite(amount) or not 0 <= amount <= 100:
        raise ValueError(f"Invalid EPIT composition amount: {value!r}")
    return amount, action


def process_composition(
    source: dict[str, Any], *, known_omitted_elements: tuple[str, ...] = ()
) -> tuple[dict[str, float], dict[str, str], float]:
    """Apply the agreed tolerance; do not infer a balance or rescale amounts."""
    if set(known_omitted_elements) - set(ELEMENTS):
        raise ValueError("Unknown element in omission exceptions.")
    parsed = {element: parse_amount(source.get(element)) for element in ELEMENTS}
    total = math.fsum(value for value, _ in parsed.values() if math.isfinite(value))
    values = {element: value for element, (value, _) in parsed.items()}
    actions = {element: action for element, (_, action) in parsed.items()}
    if abs(total - 100.0) <= CLOSURE_TOLERANCE + 1e-12:
        for element in ELEMENTS:
            if actions[element] == "missing" and element not in known_omitted_elements:
                values[element] = 0.0
                actions[element] = "assumed_zero_within_tolerance"
    return values, actions, total


def prepare(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to replace an existing export: {output_dir}")

    frozen = load_frozen_split(SPLIT_MANIFEST)
    if sha256_file(SOURCE_FILE) != frozen.manifest["dataset"]["source_sha256"]:
        raise RuntimeError("EPIT source differs from the frozen split input.")
    split_path = SPLIT_MANIFEST.parent / frozen.lock["assignments"]["file"]
    assignments = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    assignments["task_row_index"] = assignments.task_row_index.astype(int)
    assignments = assignments.sort_values("task_row_index").reset_index(drop=True)
    if assignments.task_row_index.tolist() != list(range(EXPECTED_ROW_COUNT)):
        raise RuntimeError("Frozen EPIT row indices are incomplete or duplicated.")

    tables = [table for table in load_electrochemical_metrics() if table.table == TABLE_NAME]
    if len(tables) != 1:
        raise RuntimeError("Expected exactly one EPIT source table.")
    table = tables[0]
    raw_rows = [row for row in table.rows if math.isfinite(_numeric(table, row, TARGET_COLUMN))]
    if len(raw_rows) != EXPECTED_ROW_COUNT:
        raise RuntimeError("The existing EPIT target-eligibility filter changed.")
    if assignments.source_no.tolist() != [str(row["No."]) for row in raw_rows]:
        raise RuntimeError("Source row identifiers differ from the frozen split.")

    previous = pd.read_csv(PREVIOUS_SHARED, low_memory=False)
    previous = previous.loc[previous.source_dataset.eq("EPIT")].set_index("row_id")
    row_ids = [f"epit_r{index + 1:06d}" for index in range(EXPECTED_ROW_COUNT)]
    if previous.index.tolist() != row_ids:
        raise RuntimeError("The previous shared EPIT row order changed.")
    if previous.source_row_id.astype(str).tolist() != [str(row["No."]) for row in raw_rows]:
        raise RuntimeError("The previous shared EPIT source identifiers changed.")

    input_paths = [SOURCE_FILE, PREVIOUS_SHARED, SPLIT_MANIFEST, frozen.lock_path, split_path]
    input_hashes = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in input_paths}
    processed, cell_log, row_audit = [], [], []
    action_counts: Counter[str] = Counter()
    for index, (row_id, source) in enumerate(zip(row_ids, raw_rows, strict=True)):
        raw_composition = {element: source[column] for column, element in RENAME.items()}
        values, actions, total = process_composition(raw_composition)
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"Unresolved composition needs review: {row_id}")
        if abs(total - 100.0) > CLOSURE_TOLERANCE + 1e-12:
            raise RuntimeError(f"Composition total needs review: {row_id}: {total}")
        if actions["Fe"] != "reported_numeric":
            raise RuntimeError(f"Expected an already reported Fe amount: {row_id}")

        output = {"row_id": row_id, "task_row_index": index}
        output.update({RENAME.get(column, column): source[column] for column in table.columns})
        output.update(values)
        processed.append(output)
        action_counts.update(actions.values())
        counts = Counter(actions.values())
        row_audit.append({
            "row_id": row_id,
            "source_no": source["No."],
            "reference": source.get("Reference"),
            "composition_sum_wt_percent": total,
            "assumed_zero_count": counts["assumed_zero_within_tolerance"],
            "upper_limit_count": counts["upper_limit_as_numeric"],
            "missing_composition_count": 0,
        })
        for element, action in actions.items():
            old = float(previous.loc[row_id, element])
            changed = not np.isclose(old, values[element], rtol=0, atol=1e-10, equal_nan=True)
            if math.isfinite(old) and changed:
                raise RuntimeError(f"Previously numeric amount changed: {row_id}/{element}")
            if action != "reported_numeric":
                cell_log.append({
                    "row_id": row_id,
                    "source_no": source["No."],
                    "element": element,
                    "source_value": raw_composition[element],
                    "processed_value": values[element],
                    "action": action,
                    "previous_shared_value": old,
                    "changed_vs_previous_shared": changed,
                })

    frame = pd.DataFrame(processed)
    log = pd.DataFrame(cell_log)
    changes = log.loc[log.changed_vs_previous_shared].copy()
    if len(changes) != 33 or changes.row_id.nunique() != 19:
        raise RuntimeError("Changes differ from the reviewed 33 cells across 19 rows.")
    split_columns = ["task_row_index", "source_no", "composition_group", "outer_split", "optuna_validation_fold"]
    frozen_rows = assignments[split_columns].copy()
    frozen_rows.insert(0, "row_id", row_ids)

    manifest = {
        "schema_version": "epit_processed_composition_v1",
        "source_table": TABLE_NAME,
        "composition_columns": ELEMENTS,
        "composition_units": "wt.%",
        "source_to_processed_column_names": RENAME,
        "non_composition_columns_copied_from_source": [c for c in table.columns if c not in RENAME],
        "target_column": TARGET_COLUMN,
        "policy": {
            "accepted_composition_total_wt_percent": [99.0, 101.0],
            "upper_limits": "numeric bound retained as approximation; original notation logged",
            "missing_within_tolerance": "assumed zero unless a known omission prevents it",
            "known_omission_exceptions": {},
            "Fe": "reported source amount retained; no new balance calculation",
            "normalization": "none",
            "unresolved_composition": "stop for review rather than silently complete or drop a frozen row",
            "target_values_used_for_composition_decisions": False,
        },
        "counts": {
            "source_rows": len(table.rows),
            "rows_without_usable_average_target": len(table.rows) - len(raw_rows),
            "retained_rows": len(frame),
            "composition_exclusions": 0,
            "development_rows": int(frozen_rows.outer_split.eq("development").sum()),
            "final_test_rows": int(frozen_rows.outer_split.eq("final_test").sum()),
            "composition_cells_by_action": dict(action_counts),
            "composition_nan_cells": int(frame[ELEMENTS].isna().sum().sum()),
            "changed_cells_vs_previous_shared": len(changes),
            "changed_rows_vs_previous_shared": int(changes.row_id.nunique()),
            "changes_by_action": changes.action.value_counts().to_dict(),
        },
        "composition_sum_wt_percent": {"min": min(row["composition_sum_wt_percent"] for row in row_audit),
                                       "max": max(row["composition_sum_wt_percent"] for row in row_audit)},
        "input_sha256": input_hashes,
        "implementation_sha256": {
            str(Path(__file__).resolve().relative_to(REPO_ROOT)): sha256_file(Path(__file__)),
            "corrosion_datasets/analysis/scripts/analyze_structure.py": sha256_file(
                REPO_ROOT / "corrosion_datasets/analysis/scripts/analyze_structure.py"),
        },
        "scope": "Standalone processed reference table; existing training profiles, loaders and shared exports were not updated.",
    }
    for path in input_paths:
        if sha256_file(path) != input_hashes[str(path.relative_to(REPO_ROOT))]:
            raise RuntimeError(f"Input changed during processing: {path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in {
        "epit_processed.csv": frame,
        "composition_processing_log.csv": log,
        "composition_row_audit.csv": pd.DataFrame(row_audit),
        "changes_vs_previous_shared.csv": changes,
        "split_assignments.csv": frozen_rows,
    }.items():
        data.to_csv(output_dir / name, index=False)
    manifest["output_sha256"] = {
        path.name: sha256_file(path) for path in sorted(output_dir.glob("*.csv"))
    }
    (output_dir / "processing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = prepare(args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
