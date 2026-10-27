from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from scripts import eval_corrosion_datasets as evaluation
from scripts.soccol_pipeline import data as loader


@pytest.fixture(scope="module")
def reviewed():
    return loader.load_soccol_dataset()


@pytest.fixture
def export_copy(tmp_path):
    destination = tmp_path / "reviewed"
    destination.mkdir()
    for filename in (*loader.INPUT_FILES, "processing_manifest.json"):
        shutil.copy2(loader.PROCESSED_DIR / filename, destination / filename)
    return destination


def rewrite_csv(folder, filename, frame):
    path = folder / filename
    frame.to_csv(path, index=False)
    manifest_path = folder / "processing_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_saved_values_zeros_and_missingness_survive_exactly(reviewed):
    saved = pd.read_csv(loader.PROCESSED_DIR / "soccol_features.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(reviewed.features, saved.drop(columns="row_id"), check_exact=True)
    source = pd.read_csv(loader.PROCESSED_DIR / "soccol_processed.csv", float_precision="round_trip")
    pd.testing.assert_series_equal(reviewed.targets, source.E_pit, check_exact=True)
    rows = reviewed.rows.set_index("row_id")
    assert rows.loc["soccol_r000112", "Ti"] == pytest.approx(1.86)
    assert rows.loc["soccol_r000112", "Fe"] == pytest.approx(77.612)
    assert rows.loc["soccol_r000152", "Cu"] == pytest.approx(3.5)
    assert rows.loc["soccol_r000152", "Fe"] == pytest.approx(43.44)
    assert rows.loc[rows.source.eq("1998Ahn"), "W"].gt(0).all()
    assert rows.loc[rows.source.eq("2000Russell"), "Ti"].eq(0).all()
    assert rows.loc[rows.source.eq("1969Boehni"), "Re"].isna().all()
    unresolved = rows.composition_status.eq("unresolved")
    assert rows.loc[unresolved, "Fe"].isna().all()
    elements = reviewed.registry["composition_columns"]
    accepted = rows.loc[~unresolved, elements]
    assert np.isfinite(accepted.to_numpy()).all()
    assert (np.abs(accepted.sum(axis=1) - 100) > 0.001).any()  # No closure of real rows.
    assert accepted.Fe.eq(0).any()
    assert accepted.Gd.eq(0).all()
    assert reviewed.features.CP_SO4.isna().any()
    assert reviewed.features.reference_type.isna().any()
    assert reviewed.features.heat_treatment_2_temperature_C.isna().all()


def test_frozen_rows_and_unresolved_queries_are_preserved(reviewed):
    assert reviewed.features.shape == (4027, 58)
    assert reviewed.rows.use_as_development_context.sum() == 2950
    assert (reviewed.rows.outer_split.eq("development") & reviewed.rows.composition_status.eq("unresolved")).sum() == 272
    query = reviewed.rows.loc[reviewed.rows.frozen_final_query]
    assert len(query) == 805
    assert query.composition_status.eq("unresolved").sum() == 10
    assert not set(query.row_id) & set(reviewed.rows.loc[reviewed.rows.use_as_development_context, "row_id"])
    assert reviewed.identity["target_conversion_applied"] is False
    assert reviewed.registry["composition_columns"] == list(loader.EPIT_COMPOSITION_ELEMENTS)
    assert not {"E_pit", "E_corr", "event", "source", "row_id", "CP_ion_other", "Prep_medium", "CP_anions_info"} & set(reviewed.features)


def test_reordered_exports_are_aligned_by_frozen_ids(export_copy, reviewed):
    for filename in ("soccol_processed.csv", "soccol_features.csv", "composition_row_audit.csv"):
        frame = pd.read_csv(export_copy / filename, float_precision="round_trip")
        rewrite_csv(export_copy, filename, frame.iloc[::-1])
    loaded = loader.load_soccol_dataset(export_copy)
    pd.testing.assert_frame_equal(loaded.features, reviewed.features, check_exact=True)
    pd.testing.assert_series_equal(loaded.targets, reviewed.targets, check_exact=True)
    pd.testing.assert_series_equal(loaded.rows.row_id, reviewed.rows.row_id)


def test_checksum_change_is_rejected(export_copy):
    path = export_copy / "soccol_processed.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="checksum changed: soccol_processed.csv"):
        loader.load_soccol_dataset(export_copy)


def test_duplicate_ids_are_rejected_even_with_updated_checksum(export_copy):
    frame = pd.read_csv(export_copy / "soccol_features.csv", float_precision="round_trip")
    frame.loc[1, "row_id"] = frame.loc[0, "row_id"]
    rewrite_csv(export_copy, "soccol_features.csv", frame)
    with pytest.raises(RuntimeError, match="duplicate"):
        loader.load_soccol_dataset(export_copy)


def test_changed_partition_is_rejected_without_resplitting(export_copy):
    frame = pd.read_csv(export_copy / "soccol_processed.csv", float_precision="round_trip")
    frame.loc[0, "development_validation_fold"] = 1
    rewrite_csv(export_copy, "soccol_processed.csv", frame)
    with pytest.raises(RuntimeError, match="partitions changed"):
        loader.load_soccol_dataset(export_copy)


def test_accepted_composition_is_not_repaired_when_invalid(export_copy):
    for filename in ("soccol_processed.csv", "soccol_features.csv"):
        frame = pd.read_csv(export_copy / filename, float_precision="round_trip")
        frame.loc[frame.row_id.eq("soccol_r000112"), "Fe"] = np.nan
        rewrite_csv(export_copy, filename, frame)
    with pytest.raises(RuntimeError, match="must be complete"):
        loader.load_soccol_dataset(export_copy)


def test_feature_table_disagreement_is_rejected(export_copy):
    frame = pd.read_csv(export_copy / "soccol_features.csv", float_precision="round_trip")
    frame.loc[0, "CP_temp"] += 1
    rewrite_csv(export_copy, "soccol_features.csv", frame)
    with pytest.raises(RuntimeError, match="feature values disagree"):
        loader.load_soccol_dataset(export_copy)


def test_reviewed_task_and_frozen_selection_do_not_read_old_source_csv(monkeypatch, reviewed):
    original_read = pd.read_csv
    original_hash = evaluation._sha256_file

    def checked_read(path, *args, **kwargs):
        assert "soccol_regression_event1.csv" not in str(path)
        return original_read(path, *args, **kwargs)

    def checked_hash(path):
        assert "soccol_regression_event1.csv" not in str(path)
        return original_hash(path)

    monkeypatch.setattr(pd, "read_csv", checked_read)
    monkeypatch.setattr(evaluation, "_sha256_file", checked_hash)
    args = evaluation.parse_args([
        "--soccol-reviewed-data", "--target-binning", "continuous",
        "--epit-split-manifest", str(loader.SPLIT_MANIFEST), "--epit-final-test", "--list-tasks",
    ])
    evaluation.validate_args(args)
    task, = evaluation.make_tasks(args)
    pd.testing.assert_frame_equal(task.X, reviewed.features, check_exact=True)
    assert task.X.attrs["reviewed_soccol_data"] == reviewed.identity
    evaluation.apply_epit_pipeline_final_test([task], manifest_path=loader.SPLIT_MANIFEST)
    expected = reviewed.assignments.loc[reviewed.assignments.outer_split.eq("final_test"), "task_row_index"].to_numpy()
    np.testing.assert_array_equal(task.fixed_split.test_index, expected)
    assert len(task.fixed_split.test_index) == 805
    assert len(task.fixed_split.train_index) == 2950


def test_reviewed_inputs_cannot_enter_old_imputation(reviewed):
    train = reviewed.features.iloc[:3].copy()
    query = reviewed.features.iloc[3:5].copy()
    before = train.copy(deep=True)
    encoded_train, encoded_query = evaluation.preprocess_soccol_split(train, query)
    assert encoded_train.shape == train.shape and encoded_query.shape == query.shape
    assert encoded_train.CP_SO4.isna().equals(train.CP_SO4.isna())
    pd.testing.assert_frame_equal(train, before, check_exact=True)


def test_reviewed_task_rejects_a_different_split_identity():
    task = evaluation._make_soccol_pipeline_task(loader.SPLIT_MANIFEST, reviewed_data=True)
    task.X.attrs["reviewed_soccol_data"]["split_manifest_sha256"] = "different"
    with pytest.raises(RuntimeError, match="different frozen split"):
        evaluation.apply_epit_pipeline_final_test([task], manifest_path=loader.SPLIT_MANIFEST)


def test_reviewed_cli_accepts_the_connected_model_evaluation_path():
    args = evaluation.parse_args([
        "--soccol-reviewed-data", "--target-binning", "continuous",
        "--epit-split-manifest", str(loader.SPLIT_MANIFEST), "--epit-validation-fold", "1",
    ])
    evaluation.validate_args(args)
