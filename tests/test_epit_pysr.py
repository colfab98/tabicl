from __future__ import annotations

import numpy as np

from scripts.epit_pipeline.run_pysr import (
    BINARY_OPERATORS,
    FEATURES,
    MODEL_FEATURE_NAMES,
    NESTED_CONSTRAINTS,
    UNARY_OPERATORS,
    apply_imputation,
    fit_imputation_means,
    rough_full_run_seconds,
)


def test_pysr_feature_set_is_the_selected_nine_inputs() -> None:
    assert [name for name, _ in FEATURES] == [
        "Fe_wt_pct",
        "Cr_wt_pct",
        "Ni_wt_pct",
        "Mo_wt_pct",
        "W_wt_pct",
        "N_wt_pct",
        "temperature_C",
        "chloride_M",
        "pH",
    ]


def test_pysr_imputation_uses_only_supplied_training_values() -> None:
    train = np.asarray([[1.0, np.nan], [3.0, 8.0]])
    validation = np.asarray([[np.nan, np.nan]])

    means = fit_imputation_means(train)
    result = apply_imputation(validation, means)

    assert np.allclose(means, [2.0, 8.0])
    assert np.allclose(result, [[2.0, 8.0]])


def test_pysr_exploratory_grammar_uses_raw_features_and_no_unary_nesting() -> None:
    assert MODEL_FEATURE_NAMES == tuple(name for name, _ in FEATURES)
    assert BINARY_OPERATORS == ("+", "-", "*", "/")
    assert UNARY_OPERATORS == ("square", "sqrt", "log", "exp")
    assert all(
        limit == 0
        for inner_limits in NESTED_CONSTRAINTS.values()
        for limit in inner_limits.values()
    )


def test_pysr_runtime_projection_scales_folds_and_iterations() -> None:
    assert rough_full_run_seconds(20.0, 100, 1_000) == 1_000.0
