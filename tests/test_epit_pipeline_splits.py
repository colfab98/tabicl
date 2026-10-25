from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist

from scripts.eval_corrosion_datasets import (
    DEFAULT_FEATURE_GROUPS,
    build_task,
    preprocess_epit_split,
)
from scripts.epit_pipeline.split_data import (
    COMPOSITION_DISTANCE_THRESHOLD,
    build_composition_groups,
    cluster_composition_vectors,
    load_epit_dataset,
)
from tabicl.prior.epit_schema import EPIT_COMPOSITION_COLUMNS
from tabicl.sklearn.preprocessing import TransformToNumerical
from scripts.epit_pipeline.composition_data import PROCESSED_DIR, load_reviewed_compositions
from tabicl.prior.epit_schema import EPIT_COMPOSITION_ELEMENTS


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


def test_reviewed_compositions_preserve_exact_values_and_accept_all_rows() -> None:
    dataset = load_epit_dataset()
    source = pd.read_csv(PROCESSED_DIR / "epit_processed.csv", float_precision="round_trip")
    actual = np.asarray([[row[c] for c in EPIT_COMPOSITION_COLUMNS] for row in dataset.rows])
    expected = source[list(EPIT_COMPOSITION_ELEMENTS)].to_numpy()

    assert dataset.n_rows == 760
    assert len(dataset.composition_columns) == 24
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()
    assert dataset.pretraining_template_eligible.all()
    previously_excluded = [567, 587, 718]
    assert np.allclose(
        actual.sum(axis=1)[previously_excluded],
        [100.52, 99.8679, 99.5],
    )
    assert int(dataset.composition_structural_zero_counts[previously_excluded].sum()) == 17
    assert dataset.rows[358]["Composition, wt.% Ti"] == 1.86
    assert dataset.rows[358]["Composition, wt.% Fe"] == 77.612
    legacy = load_epit_dataset(reviewed_composition=False)
    np.testing.assert_array_equal(dataset.target, legacy.target)
    for old, new in zip(legacy.rows, dataset.rows, strict=True):
        assert {k:v for k,v in old.items() if k not in EPIT_COMPOSITION_COLUMNS} == {
            k:v for k,v in new.items() if k not in EPIT_COMPOSITION_COLUMNS
        }


def test_reviewed_loader_rejects_changed_export_instead_of_reprocessing(tmp_path) -> None:
    import shutil

    export = tmp_path / "reviewed"
    shutil.copytree(PROCESSED_DIR, export)
    with (export / "epit_processed.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(RuntimeError, match="checksum changed"):
        load_reviewed_compositions(export)


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
    expected = pd.read_csv(PROCESSED_DIR / "epit_processed.csv", float_precision="round_trip")
    np.testing.assert_array_equal(task.X.iloc[:, :24].to_numpy(), expected[list(EPIT_COMPOSITION_ELEMENTS)].to_numpy())
    assert not any(
        column in dropped
        for column in EPIT_COMPOSITION_COLUMNS
        for dropped in task.dropped_feature_columns
    )


def test_epit_preprocessing_normalizes_nullable_categorical_missing_values() -> None:
    train = pd.DataFrame(
        {
            "Cr": [18.0, 20.0, 22.0],
            "[Cl-] Test Method": pd.Series(
                ["potentiodynamic", pd.NA, "scratch"],
                dtype="string",
            ),
        }
    )
    test = pd.DataFrame(
        {
            "Cr": [19.0, np.nan],
            "[Cl-] Test Method": pd.Series(
                [pd.NA, "unseen"],
                dtype="string",
            ),
        }
    )

    prepared_train, prepared_test = preprocess_epit_split(train, test)
    transformer = TransformToNumerical().fit(prepared_train)
    encoded_train = transformer.transform(prepared_train)
    encoded_test = transformer.transform(prepared_test)

    assert pd.isna(prepared_train.loc[1, "[Cl-] Test Method"])
    assert pd.isna(prepared_test.loc[0, "[Cl-] Test Method"])
    assert np.isfinite(np.asarray(encoded_train, dtype=float)).all()
    assert np.isfinite(np.asarray(encoded_test, dtype=float)).all()
