import torch

from tabicl.prior.dataset import PriorDataset
from tabicl.prior.prior_config import DEFAULT_FIXED_HP


def test_scm_prior_generates_continuous_targets_when_max_classes_zero():
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["cat_prob"] = 0.0

    dataset = PriorDataset(
        batch_size=4,
        batch_size_per_gp=1,
        min_features=2,
        max_features=4,
        max_classes=0,
        min_seq_len=24,
        max_seq_len=32,
        min_train_size=0.4,
        max_train_size=0.6,
        prior_type="mlp_scm",
        scm_fixed_hp=fixed_hp,
        scm_sampled_hp={},
        n_jobs=1,
        device="cpu",
    )

    X, y, d, _, train_sizes = dataset.prior.get_batch()

    assert X.shape[:2] == y.shape
    assert y.dtype.is_floating_point
    assert torch.isfinite(y).all()
    assert (d > 0).all()

    for yi, train_size in zip(y, train_sizes):
        train_size = int(train_size.item())
        assert torch.std(yi[:train_size], unbiased=False) > 0
        assert torch.std(yi[train_size:], unbiased=False) > 0
