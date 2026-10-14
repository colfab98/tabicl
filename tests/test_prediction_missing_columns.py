"""Missing query measurements must not implicitly change the feature set."""

import numpy as np
import pandas as pd
import pytest
import torch

from tabicl import TabICLClassifier, TabICLRegressor
from tabicl.model.tabicl import TabICL
from tabicl.shap._masking import nan_mask_predict


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("missing_columns_checkpoints")
    paths = {}
    for classification in (False, True):
        for missing_aware in (False, True):
            config = dict(
                max_classes=2 if classification else 0,
                num_quantiles=9,
                embed_dim=16,
                col_num_blocks=1,
                col_nhead=2,
                col_num_inds=4,
                col_feature_group=False,
                col_missing_value_embedding=missing_aware,
                row_num_blocks=1,
                row_nhead=2,
                row_num_cls=2,
                icl_num_blocks=1,
                icl_nhead=2,
                ff_factor=2,
            )
            with torch.random.fork_rng():
                torch.manual_seed(42)
                model = TabICL(**config)
            path = root / f"{classification}_{missing_aware}.ckpt"
            torch.save({"config": config, "state_dict": model.state_dict()}, path)
            paths[classification, missing_aware] = path
    return paths


def fitted_estimator(checkpoints, classification, missing_aware=False, kv_cache=False):
    cls = TabICLClassifier if classification else TabICLRegressor
    estimator = cls(
        model_path=checkpoints[classification, missing_aware],
        allow_auto_download=False,
        device="cpu",
        use_amp=False,
        n_estimators=2,
        norm_methods=["none"],
        feat_shuffle_method="none" if missing_aware else "latin",
        kv_cache=kv_cache,
        random_state=42,
        **({} if classification else {"remove_constant_features": False}),
    )
    train = pd.DataFrame(
        {
            "ph": [5., 6., 7., 8., 9., 10.],
            "method": ["a", "b", "a", "b", "a", "b"],
            "chloride": [0.1, 0.5, 1.0, 0.3, 0.8, 1.5],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1]) if classification else np.arange(6.)
    return estimator.fit(train, y)


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("missing_aware", [False, True])
@pytest.mark.parametrize("kv_cache", [False, "kv", "repr"])
def test_missing_columns_preserve_context_and_batch_invariance(
    checkpoints, classification, missing_aware, kv_cache
):
    estimator = fitted_estimator(checkpoints, classification, missing_aware, kv_cache)
    predict = estimator.predict_proba if classification else estimator.predict
    query = pd.DataFrame(
        {"ph": [np.nan, 7.], "method": [np.nan, "a"], "chloride": [0.4, 0.7]}
    )
    original = query.copy(deep=True)
    context = estimator.ensemble_generator_.X_.copy()

    together = predict(query)
    separately = np.concatenate([predict(query.iloc[[i]]) for i in range(len(query))])

    np.testing.assert_allclose(together, separately, rtol=1e-5, atol=1e-5)
    assert estimator.n_features_in_prediction_ == 3
    np.testing.assert_array_equal(estimator.ensemble_generator_.X_, context)
    pd.testing.assert_frame_equal(query, original)

    # A new table can infer float dtype for an entirely absent string column.
    absent = pd.DataFrame({"ph": [np.nan], "method": [np.nan], "chloride": [0.4]})
    np.testing.assert_allclose(predict(absent), together[:1], rtol=1e-5, atol=1e-5)
    assert estimator.n_features_in_prediction_ == 3
    with pytest.warns(UserWarning, match="does not have valid feature names"):
        array_prediction = predict(absent.to_numpy())
    np.testing.assert_allclose(array_prediction, together[:1], rtol=1e-5, atol=1e-5)
    encoded = estimator.X_encoder_.transform(absent)
    if missing_aware:
        assert np.isnan(encoded[0, :2]).all()
    else:
        np.testing.assert_allclose(encoded[0, :2], [7.5, -1.])


@pytest.mark.parametrize("classification", [False, True])
def test_explicit_exclusion_is_temporary_and_does_not_mutate_input(checkpoints, classification):
    estimator = fitted_estimator(checkpoints, classification, kv_cache="kv")
    predict = estimator.predict_proba if classification else estimator.predict
    query = pd.DataFrame({"ph": [np.nan], "method": [np.nan], "chloride": [0.4]})
    original = query.copy(deep=True)
    baseline = predict(query)
    masked = predict(query, feature_mask=[True, True, False])

    assert np.isfinite(masked).all()
    assert estimator.n_features_in_prediction_ == 1
    pd.testing.assert_frame_equal(query, original)
    np.testing.assert_allclose(predict(query), baseline)
    assert estimator.n_features_in_prediction_ == 3
    if classification:
        assert estimator.predict(query, feature_mask=[True, True, False]).shape == (1,)


@pytest.mark.parametrize("mask", [[True], [0, 1, 0], [[True, False, False]]])
def test_feature_mask_requires_boolean_fitted_schema(checkpoints, mask):
    estimator = fitted_estimator(checkpoints, False)
    query = pd.DataFrame({"ph": [np.nan], "method": [np.nan], "chloride": [0.4]})
    with pytest.raises(ValueError, match="feature_mask must be a boolean vector"):
        estimator.predict(query, feature_mask=mask)


@pytest.mark.parametrize("classification", [False, True])
def test_explanation_coalitions_are_independent_of_batching(checkpoints, classification):
    estimator = fitted_estimator(checkpoints, classification)
    predict = estimator.predict_proba if classification else estimator.predict
    query = pd.DataFrame(
        {"ph": [np.nan, 7.], "method": [np.nan, np.nan], "chloride": [0.4, 0.7]}
    )
    # Explanation helpers use numeric inputs, as do their existing APIs.
    encoded = np.array([[np.nan, np.nan, 0.4], [7., np.nan, 0.7], [np.nan, np.nan, np.nan]])
    calls = []

    def named_predict(X, *, feature_mask):
        calls.append(feature_mask.copy())
        return predict(pd.DataFrame(X, columns=query.columns), feature_mask=feature_mask)

    explain_predict = nan_mask_predict(named_predict)
    together = explain_predict(encoded)
    separately = np.concatenate([explain_predict(encoded[[i]]) for i in range(len(encoded))])
    np.testing.assert_allclose(together, separately, rtol=1e-5, atol=1e-5)
    with pytest.warns(UserWarning, match="does not have valid feature names"):
        direct = nan_mask_predict(predict)(encoded)
    np.testing.assert_allclose(direct, together, rtol=1e-5, atol=1e-5)
    assert {tuple(mask) for mask in calls} == {
        (True, True, False), (False, True, False), (True, True, True)
    }
