from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from scripts.eval_corrosion_datasets import DEFAULT_FEATURE_GROUPS, build_task
from scripts.epit_pipeline.split_data import (
    COMPOSITION_DISTANCE_THRESHOLD,
    build_composition_groups,
    cluster_composition_vectors,
    load_epit_dataset,
)
from tabicl.prior.epit_schema import EPIT_COMPOSITION_COLUMNS


def test_threshold_components_keep_every_close_pair_together() -> None:
    vectors = np.asarray([[0.0], [0.6], [1.2]])
    labels = cluster_composition_vectors(vectors, distance_threshold=1.0)

    assert len(set(labels)) == 1
    distances = pdist(vectors, metric="cityblock")
    assert distances.max() > 1.0


def test_current_epit_composition_groups() -> None:
    dataset = load_epit_dataset()
    grouping = build_composition_groups(dataset)

    assert dataset.n_rows == 760
    assert grouping.rounded_key_count == 396
    assert grouping.n_groups == 321
    assert grouping.sizes.sum() == 760
    assert grouping.sizes.max() == 34
    assert np.isclose(grouping.diameters.max(), 4.84)
    assert grouping.isolation_distances.min() > COMPOSITION_DISTANCE_THRESHOLD


def test_composition_completion_retains_rows_and_marks_only_three_pretraining_exclusions() -> None:
    dataset = load_epit_dataset()
    excluded = np.flatnonzero(~dataset.pretraining_template_eligible)

    assert dataset.n_rows == 760
    assert len(dataset.composition_columns) == 24
    assert excluded.tolist() == [567, 587, 718]
    assert [str(dataset.rows[index].get("No.")) for index in excluded] == [
        "581",
        "601",
        "773",
    ]
    assert np.allclose(
        dataset.composition_reported_sums[excluded],
        [100.52, 99.8679, 99.5],
    )
    assert int(dataset.pretraining_template_eligible.sum()) == 757
    assert np.all(dataset.composition_structural_zero_counts[excluded] == 0)


def test_direct_evaluator_exposes_all_24_composition_columns() -> None:
    dataset = load_epit_dataset()
    task = build_task(
        dataset.table,
        "Epit, mV (SCE) Avg.",
        target_binning="continuous",
        target_bins=4,
        datacortech_protocol="standard",
        feature_groups=DEFAULT_FEATURE_GROUPS,
        max_category_cardinality=200,
        min_numeric_finite_ratio=0.05,
        min_categorical_nonmissing_ratio=0.05,
        min_samples=20,
        min_class_count=2,
        max_samples_per_task=0,
        random_state=42,
    )

    assert task is not None
    assert len(task.X) == 760
    assert task.X.shape[1] == 28
    assert tuple(task.X.columns[:24]) == EPIT_COMPOSITION_COLUMNS
    assert not any(
        column in dropped
        for column in EPIT_COMPOSITION_COLUMNS
        for dropped in task.dropped_feature_columns
    )
