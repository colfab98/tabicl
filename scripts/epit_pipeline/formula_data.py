"""Reviewed EPIT inputs adapted to the unchanged analytical formula interface."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from scripts.epit_pipeline.composition_data import ReviewedCompositions, load_reviewed_compositions
from scripts.epit_pipeline.split_data import EpitDataset, load_epit_dataset
from tabicl.prior.epit_schema import EPIT_COMPOSITION_COLUMNS, EPIT_COMPOSITION_ELEMENTS, EPIT_METHOD_COLUMN


FORMULA_ENVIRONMENT_COLUMNS = {
    "Test Temp. oC": "CP_temp",
    "[Cl-] M": "CP_Cl",
    "[Cl-] pH": "CP_pH",
}
FORMULA_INPUT_VERSION = "epit_reviewed_formula_inputs_v1"


def reviewed_formula_input_identity(reviewed: ReviewedCompositions | None = None) -> dict[str, Any]:
    reviewed = load_reviewed_compositions() if reviewed is None else reviewed
    return {
        "version": FORMULA_INPUT_VERSION,
        "reviewed_dataset": reviewed.identity,
        "composition_column_mapping": dict(zip(EPIT_COMPOSITION_COLUMNS, EPIT_COMPOSITION_ELEMENTS, strict=True)),
        "environment_column_mapping": FORMULA_ENVIRONMENT_COLUMNS.copy(),
        "auxiliary_test_method_column": EPIT_METHOD_COLUMN,
        "missing_environment_policy": "Retained as NaN; means fitted on each formula's context rows only.",
        "composition_policy": "Saved numeric values and zeros preserved without normalization.",
    }


def load_formula_dataset() -> tuple[EpitDataset, dict[str, Any]]:
    """Keep frozen row/target order while sourcing every formula input from the reviewed export.

    Legacy aliases keep existing formula definitions and fitted-state keys intact.
    This adapter does not alter the historical loader used by older sampling assets.
    No imputation, category fitting or target fitting happens during loading.
    """
    reviewed = load_reviewed_compositions()
    dataset = load_epit_dataset()
    by_source = reviewed.rows.set_index("No.")
    mapping = {
        **dict(zip(EPIT_COMPOSITION_COLUMNS, EPIT_COMPOSITION_ELEMENTS, strict=True)),
        **FORMULA_ENVIRONMENT_COLUMNS,
    }
    table_rows = []
    seen = set()
    for source in dataset.table.rows:
        row = dict(source)
        source_no = str(row.get("No.", ""))
        if source_no in by_source.index:
            if source_no in seen:
                raise RuntimeError(f"Duplicate EPIT formula source: {source_no}")
            seen.add(source_no)
            accepted = by_source.loc[source_no]
            for old_name, processed_name in mapping.items():
                value = float(accepted[processed_name])
                if np.isinf(value):
                    raise RuntimeError(f"Infinite reviewed formula input: {processed_name}")
                row[old_name] = value
            method = accepted[EPIT_METHOD_COLUMN]
            row[EPIT_METHOD_COLUMN] = None if pd.isna(method) else str(method)
        table_rows.append(row)
    if seen != set(by_source.index):
        raise RuntimeError("The formula loader did not match every reviewed EPIT row.")
    table = replace(dataset.table, rows=table_rows, notes=[*(dataset.table.notes or []), FORMULA_INPUT_VERSION])
    corrected = {str(row.get("No.", "")): row for row in table_rows}
    rows = [corrected[str(row["No."])] for row in dataset.rows]
    if [str(row["No."]) for row in rows] != reviewed.rows["No."].tolist():
        raise RuntimeError("Reviewed formula inputs changed frozen EPIT row order.")
    return replace(dataset, table=table, rows=rows), reviewed_formula_input_identity(reviewed)
