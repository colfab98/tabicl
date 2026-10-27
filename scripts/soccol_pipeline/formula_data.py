"""Accepted reviewed Soccol rows for development-only formula fitting."""

from __future__ import annotations

from pathlib import Path

from scripts.soccol_pipeline.data import (
    PROCESSED_DIR, SPLIT_MANIFEST, SoccolDataset,
    load_soccol_dataset, select_accepted_development,
)


def load_formula_data(
    dataset: SoccolDataset | None = None, *, processed_dir: Path = PROCESSED_DIR,
    split_manifest: Path = SPLIT_MANIFEST,
):
    """Return accepted development data only, before any formula preprocessing."""
    dataset = load_soccol_dataset(processed_dir, split_manifest=split_manifest) if dataset is None else dataset
    development = select_accepted_development(dataset)
    rows = development.rows.copy()
    rows["task_row_index"] = development.task_row_indices
    rows["composition_group"] = development.assignments.composition_group
    rows.attrs["soccol_development_selection"] = development.identity
    return rows
