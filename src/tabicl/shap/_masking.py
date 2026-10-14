"""Explicit feature exclusion for NaN-background explanations only."""

import numpy as np


def nan_mask_predict(predict_fn):
    """Adapt a prediction method to the NaN-background explanation convention.

    Each row may describe a different coalition. Group equal masks so a
    coalition never depends on the other rows in an explanation batch.
    Ordinary estimator predictions do not use this adapter.
    """
    def predict(X):
        X = np.asarray(X)
        masks, groups = np.unique(np.isnan(X), axis=0, return_inverse=True)
        result = None
        for index, mask in enumerate(masks):
            rows = np.flatnonzero(groups == index)
            values = np.asarray(predict_fn(X[rows], feature_mask=mask))
            if result is None:
                result = np.empty((len(X), *values.shape[1:]), dtype=values.dtype)
            result[rows] = values
        return result

    return predict
