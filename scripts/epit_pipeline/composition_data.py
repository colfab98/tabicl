"""Read reviewed EPIT compositions without repeating source preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from .artifact_hashes import load_frozen_split, sha256_file
from tabicl.prior.epit_schema import EPIT_COMPOSITION_COLUMNS, EPIT_COMPOSITION_ELEMENTS


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "corrosion_datasets/datasets/electrochemical_metrics_alloys/processed/epit_reference_v1"
SPLIT_MANIFEST = REPO_ROOT / "corrosion_datasets/analysis/epit_pipeline/splits_v2/split_manifest.json"
COMPOSITION_NOTE = "Reviewed EPIT compositions: epit_reference_v1; values preserved without filling or normalization."


@dataclass(frozen=True)
class ReviewedCompositions:
    rows: pd.DataFrame
    audit: pd.DataFrame
    assignments: pd.DataFrame
    identity: dict[str, Any]


def load_reviewed_compositions(processed_dir: Path = PROCESSED_DIR) -> ReviewedCompositions:
    """Validate the saved export and its correspondence to the frozen rows."""
    processed_dir = Path(processed_dir)
    manifest = json.loads((processed_dir / "processing_manifest.json").read_text())
    filenames = ("epit_processed.csv", "composition_row_audit.csv", "split_assignments.csv", "feature_registry.json")
    hashes = {name: sha256_file(processed_dir / name) for name in filenames}
    for name, digest in hashes.items():
        if manifest["output_sha256"].get(name) != digest:
            raise RuntimeError(f"Reviewed EPIT export checksum changed: {name}")
    registry = json.loads((processed_dir / "feature_registry.json").read_text())
    if registry["composition_columns"] != list(EPIT_COMPOSITION_ELEMENTS):
        raise RuntimeError("Reviewed EPIT composition columns differ from the 24-element schema.")

    rows = pd.read_csv(processed_dir / "epit_processed.csv", dtype={"No.": str}, float_precision="round_trip")
    assignments = pd.read_csv(processed_dir / "split_assignments.csv", dtype={"source_no": str})
    audit = pd.read_csv(processed_dir / "composition_row_audit.csv", dtype={"source_no": str})
    expected_ids = [f"epit_r{i + 1:06d}" for i in range(760)]
    for frame in (rows, assignments, audit):
        if frame.row_id.duplicated().any() or set(frame.row_id) != set(expected_ids):
            raise RuntimeError("Reviewed EPIT row IDs are incomplete or duplicated.")
    rows = rows.set_index("row_id").loc[expected_ids].reset_index()
    assignments = assignments.set_index("row_id").loc[expected_ids].reset_index()
    audit = audit.set_index("row_id").loc[expected_ids].reset_index()
    if rows.task_row_index.tolist() != list(range(760)) or assignments.task_row_index.tolist() != list(range(760)):
        raise RuntimeError("Reviewed EPIT task indices differ from their stable row IDs.")
    if rows["No."].duplicated().any() or rows["No."].tolist() != assignments.source_no.tolist() or rows["No."].tolist() != audit.source_no.tolist():
        raise RuntimeError("Reviewed EPIT source identifiers disagree.")

    frozen = load_frozen_split(SPLIT_MANIFEST)
    original = pd.read_csv(SPLIT_MANIFEST.parent / frozen.lock["assignments"]["file"], dtype={"source_no": str})
    fields = ["task_row_index", "source_no", "composition_group", "outer_split", "optuna_validation_fold"]
    pd.testing.assert_frame_equal(assignments[fields], original.sort_values("task_row_index")[fields].reset_index(drop=True), check_dtype=False)
    values = rows.loc[:, list(EPIT_COMPOSITION_ELEMENTS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
        raise RuntimeError("Reviewed EPIT compositions contain missing or invalid amounts; do not impute them during loading.")
    totals = values.sum(axis=1)
    if np.any(np.abs(totals - 100.0) > 1.0 + 1e-12):
        raise RuntimeError("Reviewed EPIT compositions fall outside the accepted 99–101 wt.% interval.")
    if not np.allclose(totals, audit.composition_sum_wt_percent, rtol=0, atol=1e-10) or audit.missing_composition_count.ne(0).any():
        raise RuntimeError("Reviewed EPIT compositions disagree with their acceptance audit.")
    identity = {
        "version": "epit_reference_v1",
        "files_sha256": hashes,
        "split_manifest_sha256": frozen.manifest_sha256,
        "composition_columns": list(EPIT_COMPOSITION_ELEMENTS),
        "accepted_total_wt_percent": [99.0, 101.0],
    }
    return ReviewedCompositions(rows, audit, assignments, identity)


def apply_reviewed_compositions(table: Any, reviewed: ReviewedCompositions | None = None) -> Any:
    """Replace only composition cells, matching source IDs rather than row order.

    Legacy column names remain an adapter for existing formulas and the current
    28-column environment path. Non-composition source cells stay untouched.
    """
    reviewed = load_reviewed_compositions() if reviewed is None else reviewed
    by_source = reviewed.rows.set_index("No.")
    output = []
    seen = set()
    for source in table.rows:
        row = dict(source)
        source_no = str(row.get("No.", ""))
        if source_no in by_source.index:
            if source_no in seen:
                raise RuntimeError(f"Duplicate EPIT source identifier: {source_no}")
            seen.add(source_no)
            accepted = by_source.loc[source_no]
            for old_name, element in zip(EPIT_COMPOSITION_COLUMNS, EPIT_COMPOSITION_ELEMENTS, strict=True):
                row[old_name] = float(accepted[element])
        output.append(row)
    if seen != set(by_source.index):
        raise RuntimeError("Source table does not contain every reviewed EPIT row.")
    notes = [note for note in (table.notes or []) if note != COMPOSITION_NOTE]
    return replace(table, rows=output, notes=[*notes, COMPOSITION_NOTE])
