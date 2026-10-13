import numpy as np
import pytest
import torch

from tabicl import TabICLRegressor
from tabicl.model.embedding import ColEmbedding
from tabicl.sklearn.preprocessing import EnsembleGenerator, TransformToNumerical


def make_embedder(**kwargs):
    config = dict(
        embed_dim=8,
        num_blocks=1,
        nhead=2,
        dim_feedforward=16,
        num_inds=4,
        affine=False,
        feature_group=False,
        target_aware=False,
        reserve_cls_tokens=0,
        missing_value_embedding=True,
    )
    config.update(kwargs)
    return ColEmbedding(**config)


def test_missing_cells_use_a_trainable_embedding():
    embedder = make_embedder()
    features = torch.tensor([[[1.0], [float("nan")], [3.0]]])

    safe_features, projected = embedder._project_features(features)

    assert torch.isfinite(safe_features).all()
    assert torch.isfinite(projected).all()
    torch.testing.assert_close(projected[0, 1], embedder.missing_embedding)

    projected.sum().backward()
    assert embedder.missing_embedding.grad is not None
    assert torch.count_nonzero(embedder.missing_embedding.grad) > 0


def test_missing_embedding_produces_finite_column_embeddings():
    embedder = make_embedder()
    X = torch.tensor(
        [[[1.0, float("nan")], [2.0, 4.0], [float("nan"), 5.0], [3.0, 6.0]]]
    )
    y_train = torch.tensor([[0.0, 1.0, 2.0]])

    output = embedder(X, y_train=y_train)

    assert output.shape == (1, 4, 2, 8)
    assert torch.isfinite(output).all()


def test_missing_embedding_rejects_feature_grouping():
    with pytest.raises(ValueError, match="requires feature_group=False"):
        make_embedder(feature_group="same")


def test_missing_aware_preprocessing_preserves_nan_and_fixed_columns():
    pd = pytest.importorskip("pandas")
    train = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0],
            "numeric": [2.0, np.nan, 4.0],
            "category": ["SCE", np.nan, "AgAgCl"],
        }
    )
    encoder = TransformToNumerical(preserve_missing_values=True).fit(train)
    encoded = encoder.transform(train)
    query_encoded = encoder.transform(
        pd.DataFrame({"constant": [1.0], "numeric": [np.nan], "category": [np.nan]})
    )

    assert encoded.shape == (3, 3)
    assert np.isnan(encoded[1, 1])
    assert np.isnan(encoded[1, 2])
    assert np.isnan(query_encoded[0, 1:]).all()

    generator = EnsembleGenerator(
        classification=False,
        n_estimators=1,
        norm_methods="none",
        feat_shuffle_method="none",
        remove_constant_features=False,
        random_state=42,
    ).fit(encoded, np.asarray([0.0, 1.0, 2.0]))
    (X_both, _), = generator.transform(np.asarray([[1.0, np.nan, np.nan]]), mode="both").values()

    assert X_both.shape == (1, 4, 3)
    assert np.isnan(X_both[0, 1, 1])
    assert np.isnan(X_both[0, -1, 1:]).all()
    assert generator.unique_filter_.features_to_keep_.tolist() == [True, True, True]


def test_missing_aware_estimator_aligns_named_query_columns():
    pd = pytest.importorskip("pandas")
    estimator = TabICLRegressor()
    estimator.preserve_missing_values_ = True
    estimator.feature_names_in_ = np.asarray(["Cr", "Ni", "CP_temp"], dtype=object)

    aligned = estimator._align_missing_aware_dataframe(
        pd.DataFrame({"CP_temp": [25.0], "Cr": [18.0]})
    )

    assert aligned.columns.tolist() == ["Cr", "Ni", "CP_temp"]
    assert aligned.loc[0, ["Cr", "CP_temp"]].tolist() == [18.0, 25.0]
    assert np.isnan(aligned.loc[0, "Ni"])

    with pytest.raises(ValueError, match="not present during fit"):
        estimator._align_missing_aware_dataframe(pd.DataFrame({"Cr": [18.0], "unknown": [1.0]}))

    generic_estimator = TabICLRegressor()
    generic_estimator.preserve_missing_values_ = False
    generic_estimator.feature_names_in_ = estimator.feature_names_in_
    generic_input = pd.DataFrame({"CP_temp": [25.0], "Cr": [18.0]})
    assert generic_estimator._align_missing_aware_dataframe(generic_input) is generic_input

