import torch

from tabicl.prior.dataset import PriorDataset, SCMPrior
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


def test_molecular_descriptor_marginals_are_finite_and_nonconstant():
    values = torch.linspace(-3.0, 3.0, steps=128)

    for family in [
        "descriptor_standard",
        "descriptor_positive",
        "descriptor_count",
        "descriptor_bounded",
    ]:
        transformed = SCMPrior._corrosion_marginal_values(values, family)
        assert transformed.shape == values.shape
        assert torch.isfinite(transformed).all()
        assert torch.std(transformed, unbiased=False) > 0

    counts = SCMPrior._corrosion_marginal_values(values, "descriptor_count")
    assert torch.allclose(counts, torch.floor(counts))


def test_audit_v2_block_marginals_are_finite_and_nonconstant():
    values = torch.linspace(-3.0, 3.0, steps=128)
    families = [
        "concentration",
        "material_bounded",
        "material_property",
        "environment_bounded",
        "process_score",
        "intervention_dose",
    ]

    for family in families:
        transformed = SCMPrior._corrosion_marginal_values(values, family)
        assert transformed.shape == values.shape
        assert torch.isfinite(transformed).all()
        assert torch.std(transformed, unbiased=False) > 0

    for family in [
        "material_category",
        "environment_category",
        "process_category",
        "process_binary",
        "intervention_category",
        "intervention_binary",
    ]:
        transformed = SCMPrior._corrosion_marginal_values(values, family)
        assert torch.isfinite(transformed).all()
        assert torch.std(transformed, unbiased=False) > 0
        assert torch.allclose(transformed, torch.floor(transformed))


def test_informed_corrosion_mechanism_changes_normal_targets():
    torch.manual_seed(0)
    fixed_hp = dict(DEFAULT_FIXED_HP)
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.randn(96, 8)
    y = torch.randn(96)
    blocks = {
        "material": slice(0, 3),
        "environment": slice(3, 6),
        "process_history": slice(6, 7),
        "exposure_duration": slice(7, 8),
    }

    X_new, y_new = prior._apply_informed_corrosion_mechanism(
        X.clone(),
        y.clone(),
        blocks,
        "normal_corrosion",
        interaction_strength=0.35,
        intervention_strength=0.20,
    )

    assert X_new.shape == X.shape
    assert y_new.shape == y.shape
    assert torch.isfinite(X_new).all()
    assert torch.isfinite(y_new).all()
    assert torch.std(y_new - y, unbiased=False) > 0


def test_informed_corrosion_mechanism_changes_inhibitor_targets_from_descriptors():
    torch.manual_seed(1)
    fixed_hp = dict(DEFAULT_FIXED_HP)
    prior = SCMPrior(batch_size=1, fixed_hp=fixed_hp, sampled_hp={}, n_jobs=1, device="cpu")
    X = torch.randn(96, 10)
    y = torch.randn(96)
    blocks = {
        "environment": slice(0, 2),
        "direct_intervention": slice(2, 3),
        "molecular_descriptor": slice(3, 10),
    }

    _, y_new = prior._apply_informed_corrosion_mechanism(
        X.clone(),
        y.clone(),
        blocks,
        "inhibitor_agent",
        interaction_strength=0.35,
        intervention_strength=0.50,
    )

    assert y_new.shape == y.shape
    assert torch.isfinite(y_new).all()
    assert torch.std(y_new - y, unbiased=False) > 0
