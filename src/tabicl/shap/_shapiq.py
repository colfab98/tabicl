"""ShapIQ explanations for TabICL.

`shapiq <https://github.com/mmschlk/shapiq>`_ computes Shapley interaction
indices — richer than plain SHAP values because they also capture feature
interactions.

The custom :class:`_NaNImputer` uses NaN to represent excluded features
and explicitly requests their exclusion during explanations. Ordinary
predictions continue to treat NaN as missing data.

Example::

    from tabicl import TabICLClassifier
    from tabicl.shap import get_shapiq_explainer

    clf = TabICLClassifier().fit(X_train, y_train)

    # default: NaN imputation (recommended for TabICL)
    explainer = get_shapiq_explainer(clf, X_train, imputer="nan")
    sv = explainer.explain(X_test[0])

    # alternative: marginal imputation (model-agnostic)
    explainer = get_shapiq_explainer(clf, X_train, imputer="marginal")
"""

from __future__ import annotations

from copy import copy
from typing import Any

import numpy as np
import shapiq
from shapiq.imputer import MarginalImputer
from sklearn.base import BaseEstimator

from ._masking import nan_mask_predict


class _NaNImputer(MarginalImputer):
    """Explicitly exclude coalition features using a NaN-background adapter.

    When shapiq evaluates a coalition (subset of features), absent features are
    set to NaN. The explanation adapter passes an explicit feature mask for
    each coalition, without changing the caller's estimator or its ordinary
    missing-value handling.

    Parameters
    ----------
    model : estimator object
        A fitted estimator with ``predict`` or ``predict_proba``.

    data : ndarray of shape (n_samples, n_features)
        Background data (only its shape is used).

    class_index : int or None
        For classifiers, which class probability to return.
    """

    def __init__(self, model, data: np.ndarray, *, class_index: int | None = None, **kwargs) -> None:
        from shapiq.explainer.utils import get_predict_function_and_model_type

        model = copy(model)
        if hasattr(model, "predict_proba"):
            model.predict_proba = nan_mask_predict(model.predict_proba)
        else:
            model.predict = nan_mask_predict(model.predict)
        resolved_predict, _ = get_predict_function_and_model_type(model, class_index=class_index)
        model._shapiq_predict_function = resolved_predict
        super().__init__(model, data, sample_size=1, normalize=False, **kwargs)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return model predictions with absent features set to NaN.

        Parameters
        ----------
        coalitions : ndarray of shape (n_subsets, n_features)
            Boolean mask — ``True`` means the feature is present.

        Returns
        -------
        ndarray of shape (n_subsets,)
        """
        x_masked = np.tile(self.x, (coalitions.shape[0], 1)).astype(float)
        x_masked[~coalitions] = np.nan
        return self.predict(x_masked)


def get_shapiq_explainer(
    estimator: BaseEstimator,
    data: np.ndarray,
    *,
    imputer: str | Any = "nan",
    index: str = "k-SII",
    max_order: int = 2,
    class_index: int | None = None,
    **kwargs: Any,
) -> shapiq.TabularExplainer:
    """Create a shapiq explainer tuned for TabICL.

    Parameters
    ----------
    estimator : estimator object
        A fitted TabICL estimator (or any sklearn-compatible estimator when
        using ``imputer="marginal"``).

    data : array-like
        Background / reference data.

    imputer : str or shapiq.Imputer instance, default="nan"
        How absent features are handled when evaluating coalitions:

        - ``"nan"`` (default) — uses :class:`_NaNImputer` so that absent
          features become NaN and are explicitly excluded from context and
          queries. Deterministic, no sampling noise.
        - ``"marginal"`` — standard marginal-sampling imputation.
        - ``"baseline"`` — replace absent features with a fixed baseline
          value (typically the mean of `data`).
        - ``"conditional"`` — conditional-sampling imputation.
        - Any :class:`shapiq.Imputer` instance — forwarded directly to
          ``shapiq.TabularExplainer``.

        See the `shapiq imputer documentation
        <https://shapiq.readthedocs.io/en/latest/api/shapiq.imputer.html>`_
        for details.

    index : str, default="k-SII"
        Interaction index to compute.  Common choices:

        - ``"SV"`` — Shapley values (set ``max_order=1``).
        - ``"k-SII"`` — k-Shapley Interaction Index (default).
        - ``"SII"`` — Shapley Interaction Index.
        - ``"STII"`` — Shapley Taylor Interaction Index.
        - ``"FSII"`` — Faithful Shapley Interaction Index.
        - ``"FBII"`` — Faithful Banzhaf Interaction Index.

        See the `shapiq index documentation
        <https://shapiq.readthedocs.io/en/latest/api/shapiq.interaction_values.html>`_
        for the full list.

    max_order : int, default=2
        Maximum interaction order.

    class_index : int or None, default=None
        For classifiers, which class probability to explain.

    **kwargs
        Forwarded to ``shapiq.TabularExplainer``.

    Returns
    -------
    shapiq.TabularExplainer
    """
    data = np.asarray(data)

    if imputer == "nan":
        from tabicl.sklearn.base import TabICLBaseEstimator

        if not isinstance(estimator, TabICLBaseEstimator):
            raise TypeError(
                "imputer='nan' requires a TabICL estimator (TabICLClassifier or "
                "TabICLRegressor). For other models, use other options."
            )
        imputer = _NaNImputer(estimator, data, class_index=class_index)

    return shapiq.TabularExplainer(
        model=estimator,
        data=data,
        imputer=imputer,
        index=index,
        max_order=max_order,
        class_index=class_index,
        **kwargs,
    )
