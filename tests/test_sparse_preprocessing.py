"""Sparse context columns must not erase known context or query values."""

import numpy as np
import pytest

from tabicl.sklearn.preprocessing import (
    CustomStandardScaler,
    OutlierRemover,
    PreprocessingPipeline,
)


pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")
METHODS = ["none", "power", "quantile", "quantile_rtdl", "robust"]


@pytest.mark.parametrize("method", METHODS)
def test_sparse_columns_pass_through_without_affecting_supported_columns(method):
    context = np.column_stack([
        np.linspace(-2., 2., 1024),
        np.full(1024, np.nan),
        np.full(1024, np.nan),
        np.linspace(1., 3., 1024) ** 2,
    ])
    context[5, 2] = 0.1
    query = np.array([
        [0.1, 0.025, 1e6, 4.],
        [np.nan, np.nan, np.nan, 5.],
        [-0.1, -1e6, 0.1, np.nan],
    ])
    original_context, original_query = context.copy(), query.copy()
    prep = PreprocessingPipeline(normalization_method=method, random_state=42).fit(context)
    fitted_context = prep.X_transformed_.copy()
    mean, scale = prep.standard_scaler_.mean_.copy(), prep.standard_scaler_.scale_.copy()
    output = prep.transform(query)

    np.testing.assert_array_equal(prep.X_transformed_[:, 1:3], context[:, 1:3])
    np.testing.assert_array_equal(output[:, 1:3], query[:, 1:3])
    np.testing.assert_array_equal(np.isnan(prep.X_transformed_), np.isnan(context))
    np.testing.assert_array_equal(np.isnan(output), np.isnan(query))
    assert output.shape == query.shape
    np.testing.assert_allclose(prep.transform(context), prep.X_transformed_)

    # Unsupported columns do not influence processing of supported columns.
    dense = PreprocessingPipeline(normalization_method=method, random_state=42).fit(context[:, [0, 3]])
    np.testing.assert_allclose(prep.X_transformed_[:, [0, 3]], dense.X_transformed_)
    np.testing.assert_allclose(output[:, [0, 3]], dense.transform(query[:, [0, 3]]))

    # Query batches cannot refit support, change statistics, or mutate inputs.
    separate = np.concatenate([prep.transform(row[None, :]) for row in query])
    np.testing.assert_allclose(output, separate)
    np.testing.assert_array_equal(prep.X_transformed_, fitted_context)
    np.testing.assert_array_equal(prep.standard_scaler_.mean_, mean)
    np.testing.assert_array_equal(prep.standard_scaler_.scale_, scale)
    np.testing.assert_array_equal(context, original_context)
    np.testing.assert_array_equal(query, original_query)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_rows", [1, 4])
def test_all_columns_sparse_or_single_context_row(method, n_rows):
    context = np.full((n_rows, 2), np.nan)
    context[0, 1] = 0.1
    query = np.array([[1e6, -1e6], [np.nan, 0.1]])
    prep = PreprocessingPipeline(normalization_method=method).fit(context)
    np.testing.assert_array_equal(prep.X_transformed_, context)
    np.testing.assert_array_equal(prep.transform(query), query)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("known_count", [0, 1])
def test_scaler_sparse_vectors_preserve_values_including_outside_clip_limits(dtype, known_count):
    context = np.full(4, np.nan, dtype=dtype)
    if known_count:
        context[1] = 0.1
    query = np.array([1e6, -1e6, np.nan], dtype=dtype)
    scaler = CustomStandardScaler().fit(context)
    np.testing.assert_array_equal(scaler.transform(context), context)
    np.testing.assert_array_equal(scaler.transform(query), query)
    np.testing.assert_array_equal(scaler.inverse_transform(query), query)


def test_outlier_removal_keeps_values_when_second_stage_has_zero_or_one_observation():
    # First column retains just 0; second retains nothing after preliminary rejection.
    context = np.array([[-1., 10.], [0., 20.], [1., np.nan]])
    remover = OutlierRemover(threshold=0.2).fit(context)
    query = np.array([[1e6, -1e6], [np.nan, 15.]])
    np.testing.assert_array_equal(remover.transform(context), context)
    np.testing.assert_array_equal(remover.transform(query), query)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scaler_retains_existing_statistics_for_supported_columns(dtype):
    context = np.array([[1., np.nan, 7.], [3., 20., 7.], [5., 40., 7.]], dtype=dtype)
    scaler = CustomStandardScaler().fit(context)
    np.testing.assert_array_equal(scaler.mean_, np.nanmean(context, axis=0))
    np.testing.assert_array_equal(scaler.scale_, np.nanstd(context, axis=0) + scaler.epsilon)
    expected = np.clip((context - scaler.mean_) / scaler.scale_, -100., 100.)
    np.testing.assert_array_equal(scaler.transform(context), expected)


def test_outlier_removal_still_clips_extremes_in_supported_columns():
    context = np.linspace(-1., 1., 100)[:, None]
    remover = OutlierRemover().fit(context)
    query = np.array([[1e6], [-1e6], [np.nan]])
    output = remover.transform(query)
    assert 0 < output[0, 0] < 100
    assert -100 < output[1, 0] < 0
    assert np.isnan(output[2, 0])
