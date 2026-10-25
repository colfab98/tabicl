from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from corrosion_datasets.datasets.electrochemical_metrics_alloys.prepare_epit_composition import (
    ELEMENTS,
    PREVIOUS_SHARED,
    RENAME,
    REPO_ROOT,
    SPLIT_MANIFEST,
    TABLE_NAME,
    TARGET_COLUMN,
    _numeric,
    load_electrochemical_metrics,
    parse_amount,
    prepare,
    process_composition,
    sha256_file,
)


def test_tolerance_fills_unknowns_without_changing_reported_amounts_or_total():
    for total in (99.0, 99.5, 100.52, 101.0):
        values, actions, actual_total = process_composition({"Fe": total - 20, "Cr": 20})
        assert values["Fe"] == total - 20
        assert values["Cr"] == 20
        assert values["Ni"] == 0
        assert actions["Ni"] == "assumed_zero_within_tolerance"
        assert actual_total == pytest.approx(total)
        assert sum(values.values()) == pytest.approx(total)

    for total in (98.99, 101.01):
        values, actions, _ = process_composition({"Fe": total - 20, "Cr": 20})
        assert math.isnan(values["Ni"])
        assert actions["Ni"] == "missing"


def test_known_omission_and_missing_bulk_composition_are_not_filled():
    values, actions, _ = process_composition(
        {"Fe": 80, "Cr": 20}, known_omitted_elements=("W",)
    )
    assert math.isnan(values["W"])
    assert actions["W"] == "missing"
    assert values["Ni"] == 0
    incomplete, _, total = process_composition({"S": 0.16})
    assert total == 0.16
    assert math.isnan(incomplete["Fe"])


def test_upper_limits_are_distinct_from_zeros_and_invalid_notation_is_rejected():
    assert parse_amount("<0.003") == (0.003, "upper_limit_as_numeric")
    assert parse_amount("0") == (0.0, "reported_numeric")
    assert parse_amount("NA")[1] == "missing"
    for invalid in ("-0.01", "101", "1-2", ">0.003", "unknown concentration"):
        with pytest.raises(ValueError):
            parse_amount(invalid)


def test_export_changes_only_reviewed_compositions_and_preserves_frozen_rows(tmp_path):
    output = tmp_path / "processed"
    manifest = prepare(output)
    actual = pd.read_csv(output / "epit_processed.csv")
    strings = pd.read_csv(output / "epit_processed.csv", dtype=str, keep_default_na=False)
    previous = pd.read_csv(PREVIOUS_SHARED, low_memory=False)
    previous = previous.loc[previous.source_dataset.eq("EPIT")].reset_index(drop=True)
    assert actual.row_id.tolist() == previous.row_id.tolist()
    assert len(actual) == 760
    assert actual[ELEMENTS].notna().all().all()
    assert actual[ELEMENTS].ge(0).all().all()
    assert actual[ELEMENTS].le(100).all().all()
    assert actual[ELEMENTS].sum(axis=1).between(99, 101).all()
    original_numeric = previous[ELEMENTS].to_numpy(dtype=float)
    updated_numeric = actual[ELEMENTS].to_numpy(dtype=float)
    reported = np.isfinite(original_numeric)
    np.testing.assert_allclose(updated_numeric[reported], original_numeric[reported], rtol=0, atol=1e-10)
    changes = pd.read_csv(output / "changes_vs_previous_shared.csv")
    assert len(changes) == 33
    assert changes.row_id.nunique() == 19
    limits = changes[changes.action.eq("upper_limit_as_numeric")]
    assert len(limits) == 16
    assert limits.element.eq("S").all()
    assert limits.source_value.eq("<0.003").all()
    assert limits.processed_value.eq(0.003).all()
    zeros = changes[changes.action.eq("assumed_zero_within_tolerance")]
    assert len(zeros) == 17
    assert set(zeros.row_id) == {"epit_r000568", "epit_r000588", "epit_r000719"}

    table = next(t for t in load_electrochemical_metrics() if t.table == TABLE_NAME)
    source_rows = [r for r in table.rows if math.isfinite(_numeric(table, r, TARGET_COLUMN))]
    for index, row in enumerate(source_rows):
        for column in table.columns:
            if column not in RENAME:
                assert strings.loc[index, column] == ("" if row[column] is None else str(row[column]))
            else:
                try:
                    value = float(row[column])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    assert actual.loc[index, RENAME[column]] == pytest.approx(value, abs=1e-12)

    original_split = pd.read_csv(SPLIT_MANIFEST.parent / "split_assignments.csv", dtype=str, keep_default_na=False)
    updated_split = pd.read_csv(output / "split_assignments.csv", dtype=str, keep_default_na=False)
    assert updated_split.row_id.tolist() == actual.row_id.tolist()
    pd.testing.assert_frame_equal(updated_split.drop(columns="row_id"), original_split[updated_split.columns.drop("row_id")])
    assert updated_split.outer_split.value_counts().to_dict() == {"development": 608, "final_test": 152}
    for path, digest in manifest["input_sha256"].items():
        assert sha256_file(REPO_ROOT / path) == digest
    for path, digest in manifest["output_sha256"].items():
        assert sha256_file(output / path) == digest
    assert json.loads((output / "processing_manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        prepare(output)
