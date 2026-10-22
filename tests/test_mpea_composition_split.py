from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from corrosion_datasets.datasets.mpea_corrosion.prepare_composition_split import (
    DEFAULT_OUTPUT_DIR,
    FINAL_TEST_ROWS,
    INPUT_FILES,
    ROOT,
    TARGET,
    load_frozen_mpea_split,
    prepare_split,
)
from scripts.epit_pipeline.split_data import (
    CompositionGrouping,
    SplitAssignment,
    OptimizerRecord,
    load_epit_dataset,
    validate_split_assignment,
)


def test_frozen_mpea_partition_preserves_inputs_and_keeps_final_labels_separate():
    frozen = load_frozen_mpea_split()
    assignments = pd.read_csv(DEFAULT_OUTPUT_DIR / "split_assignments.csv")
    original = pd.read_csv(INPUT_FILES["queries"])
    original_targets = pd.read_csv(INPUT_FILES["targets"]).set_index("row_id")[TARGET]
    development = pd.read_csv(DEFAULT_OUTPUT_DIR / "development.csv")
    final_queries = pd.read_csv(DEFAULT_OUTPUT_DIR / "final_test_queries.csv")
    final_targets = pd.read_csv(DEFAULT_OUTPUT_DIR / "final_test_scoring_targets.csv")

    assert assignments.row_id.tolist() == original.row_id.tolist()
    assert assignments.row_id.is_unique
    assert len(development) == 214
    assert len(final_queries) == len(final_targets) == FINAL_TEST_ROWS == 53
    assert set(development.row_id).isdisjoint(final_queries.row_id)
    assert set(development.row_id) | set(final_queries.row_id) == set(original.row_id)
    assert TARGET not in final_queries and TARGET not in assignments
    assert final_targets.row_id.tolist() == final_queries.row_id.tolist()
    for frame in (development, final_queries):
        expected = original.set_index("row_id").loc[frame.row_id].reset_index()
        pd.testing.assert_frame_equal(frame[original.columns], expected, check_dtype=False)
    for frame in (development, final_targets):
        np.testing.assert_allclose(frame[TARGET], original_targets.loc[frame.row_id], rtol=1e-12)
    assert frozen.manifest["dataset"]["feature_columns_in_order"] == json.loads(
        INPUT_FILES["registry"].read_text()
    )["feature_columns_in_order"]
    assert (DEFAULT_OUTPUT_DIR / "shared_feature_registry.json").read_bytes() == INPUT_FILES["registry"].read_bytes()


def test_close_compositions_never_cross_outer_split_or_development_folds():
    frozen = load_frozen_mpea_split()
    assignments = pd.read_csv(DEFAULT_OUTPUT_DIR / "split_assignments.csv")
    queries = pd.read_csv(INPUT_FILES["queries"]).set_index("row_id").loc[assignments.row_id]
    columns = frozen.manifest["dataset"]["model_visible_composition_columns"]
    vectors = np.round(queries[columns].fillna(0).to_numpy(dtype=float), 2)
    distances = squareform(pdist(vectors, metric="cityblock"))
    pairs = np.argwhere(np.triu(distances <= 1.0 + 1e-12, k=1))
    assert len(pairs) > 0
    for left, right in pairs:
        first, second = assignments.iloc[left], assignments.iloc[right]
        assert first.composition_group == second.composition_group
        assert first.outer_split == second.outer_split
        if first.outer_split == "development":
            assert first.optuna_validation_fold == second.optuna_validation_fold
    development = assignments[assignments.outer_split.eq("development")]
    final_test = assignments[assignments.outer_split.eq("final_test")]
    assert final_test.optuna_validation_fold.isna().all()
    assert development.optuna_validation_fold.value_counts().sort_index().tolist() == [43, 43, 43, 43, 42]
    assert assignments.groupby("composition_group").outer_split.nunique().eq(1).all()
    assert development.groupby("composition_group").optuna_validation_fold.nunique().eq(1).all()


def test_frozen_split_rejects_export_changes_and_replacement(tmp_path):
    output = tmp_path / "splits"
    shutil.copytree(DEFAULT_OUTPUT_DIR, output)
    load_frozen_mpea_split(output)
    original_lock = (output / "split_lock.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        prepare_split(output, outer_time_limit=1, inner_time_limit=1)
    assert (output / "split_lock.json").read_bytes() == original_lock
    with (output / "development.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(RuntimeError, match="Frozen MPEA export changed"):
        load_frozen_mpea_split(output)


def test_existing_epit_split_still_validates_with_default_sizes():
    dataset = load_epit_dataset()
    path = ROOT / "corrosion_datasets/analysis/epit_pipeline/splits_v2/split_manifest.json"
    manifest = json.loads(path.read_text())
    # Validate the saved EPIT assignment, without rebuilding its historical groups
    # from a newer version of the EPIT composition loader.
    records = manifest["groups"]
    row_group = np.empty(dataset.n_rows, dtype=int)
    for index, record in enumerate(records):
        row_group[record["task_row_indices"]] = index
    grouping = CompositionGrouping(
        group_ids=[record["composition_group"] for record in records],
        group_rows=[record["task_row_indices"] for record in records],
        row_group_index=row_group,
        centroids=np.zeros((len(records), len(dataset.composition_columns))),
        diameters=np.asarray([record["maximum_within_group_l1_wt_percent"] for record in records]),
        isolation_distances=np.asarray([record["nearest_other_group_l1_wt_percent"] for record in records]),
        rounded_key_count=manifest["composition_grouping"]["rounded_composition_keys"],
    )
    dummy = OptimizerRecord("frozen", 0.0, None, None, 0.0)
    assignment = SplitAssignment(
        np.asarray([record["outer_split"] for record in records]),
        np.asarray([
            -1 if record["optuna_validation_fold"] is None
            else record["optuna_validation_fold"] - 1
            for record in records
        ]), dummy, dummy,
    )
    validate_split_assignment(dataset, grouping, assignment)
