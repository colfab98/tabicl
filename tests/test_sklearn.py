import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.utils.estimator_checks import parametrize_with_checks

from tabicl import TabICLClassifier, TabICLRegressor
from tabicl.sklearn.preprocessing import TransformToNumerical



def test_transform_to_numerical_preserves_dataframe_column_order():
    pd = pytest.importorskip("pandas")
    train = pd.DataFrame(
        {
            "fe": [69.7, 70.1, 71.0],
            "cr": [18.0, 19.0, 17.5],
            "method": ["b", "a", "b"],
            "ph": [7.1, np.nan, 6.8],
        }
    )
    test = pd.DataFrame({"fe": [68.5], "cr": [20.0], "method": ["unknown"], "ph": [7.0]})

    transformer = TransformToNumerical().fit(train)
    train_out = transformer.transform(train)
    test_out = transformer.transform(test)

    assert train_out.shape == (3, 4)
    assert test_out.shape == (1, 4)
    np.testing.assert_allclose(train_out[:, 0], train["fe"])
    np.testing.assert_allclose(train_out[:, 1], train["cr"])
    np.testing.assert_allclose(train_out[:, 2], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(train_out[[0, 2], 3], [7.1, 6.8])
    np.testing.assert_allclose(test_out[0, [0, 1, 3]], [68.5, 20.0, 7.0])
    assert test_out[0, 2] == -1.0


# n_estimators=2 ensures the full preprocessing and ensembling pipeline is tested:
# n_estimators=1 skips shuffling and uses only one norm method, while n_estimators=2
# exercises feature/class shuffling, multiple normalization methods, and ensemble averaging.
@parametrize_with_checks([TabICLClassifier(n_estimators=2), TabICLRegressor(n_estimators=2)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


class TestClassifierKVCache:
    @pytest.mark.parametrize("kv_cache", ["kv", "repr"])
    def test_kv_cache(self, kv_cache):
        """Predictions with kv cache should match predictions without cache."""
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train = y[:40]
        clf = TabICLClassifier(n_estimators=2)
        clf.fit(X_train, y_train)
        pred_no_cache = clf.predict_proba(X_test)

        clf_cached = TabICLClassifier(n_estimators=2, kv_cache=kv_cache)
        clf_cached.fit(X_train, y_train)
        pred_cached = clf_cached.predict_proba(X_test)

        np.testing.assert_allclose(pred_no_cache, pred_cached, rtol=1e-4, atol=1e-4)


class TestRegressorKVCache:
    @pytest.mark.parametrize("kv_cache", ["kv", "repr"])
    def test_kv_cache(self, kv_cache):
        """Predictions with kv cache should match predictions without cache."""
        X, y = make_regression(n_samples=50, n_features=5, random_state=42)
        X_train, X_test = X[:40], X[40:]
        y_train = y[:40]
        reg = TabICLRegressor(n_estimators=2)
        reg.fit(X_train, y_train)
        pred_no_cache = reg.predict(X_test)

        reg_cached = TabICLRegressor(n_estimators=2, kv_cache=kv_cache)
        reg_cached.fit(X_train, y_train)
        pred_cached = reg_cached.predict(X_test)

        # Relaxed tolerance: kv cache changes float32 computation order
        np.testing.assert_allclose(pred_no_cache, pred_cached, rtol=1e-4, atol=1e-4)
