"""Behavioral checks for the baseline-compatible EPIT/Soccol experiment."""
import copy

import numpy as np
import pytest
import torch

from tabicl.prior.dataset import context_mean_impute
from tabicl.prior.original_mixture import OriginalPriorMixture
from tabicl.prior.paired_streams import keyed_seed, seeded
from tabicl.train.sampling import CheckpointablePriorLoader
from tests.test_paired_prior_streams import make_prior, assert_batch_equal


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def make(family="epit", ratio=.5, **kwargs):
    return make_prior(family, ratio, dataset_cls=OriginalPriorMixture, **kwargs)


@pytest.mark.parametrize("mode", ["same", "valid"])
def test_grouped_padding_matches_separate_unpadded_tables_and_gradients(mode):
    from tabicl.model.tabicl import TabICL

    torch.manual_seed(42)
    model = TabICL(max_classes=0, num_quantiles=9, embed_dim=16, ff_factor=2,
        col_num_blocks=1, col_nhead=2, col_num_inds=4, col_feature_group=mode,
        row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2)
    # Initial attention outputs are zero: exercise weights that actually use X.
    for name, param in model.named_parameters():
        if name.endswith("attn.out_proj.weight"):
            torch.nn.init.xavier_uniform_(param)
    reference = copy.deepcopy(model)
    widths = torch.tensor([1, 5, 9])
    X = torch.randn(3, 16, 9)
    for i, width in enumerate(widths):
        X[i, :, width:] = float("nan")  # Padding must never enter the groups.
    X.requires_grad_()
    y = torch.randn(3, 8)
    actual = model(X, y, widths)
    expected = torch.cat([
        reference(X[i:i+1, :, :width], y[i:i+1]) for i, width in enumerate(widths)
    ])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for (name, param), (_, ref) in zip(model.named_parameters(), reference.named_parameters()):
        if param.grad is None:
            assert ref.grad is None, name
        else:
            torch.testing.assert_close(param.grad, ref.grad, rtol=1e-4, atol=1e-5, msg=name)
    for i, width in enumerate(widths):
        assert not X.grad[i, :, width:].any()


def test_grouping_without_padding_keeps_original_circular_rule():
    from tabicl.model.embedding import ColEmbedding

    model = ColEmbedding(embed_dim=8, num_blocks=1, nhead=2, dim_feedforward=16,
                         num_inds=4, feature_group="same", feature_group_size=3)
    X = torch.arange(30.).reshape(2, 3, 5)
    indices = torch.arange(5)
    expected = torch.stack([X[:, :, (indices + offset) % 5] for offset in (1, 2, 4)], dim=-1)
    torch.testing.assert_close(model.feature_grouping(X), expected, rtol=0, atol=0)
    torch.testing.assert_close(model.feature_grouping(X, torch.tensor([5,5])), expected, rtol=0, atol=0)


def test_zero_ratio_is_exact_original_generic_batch():
    prior = make(ratio=0.)
    for step in range(2):
        got = prior.get_batch()
        with seeded(keyed_seed(prior.seed, step, 11)):
            expected = prior.priors[0].get_batch(prior.batch_size)
        assert_batch_equal(got[:5], expected)
        assert got[4].unique().numel() == 1
        assert not got[5].any()


def test_empty_generic_table_rejected_before_split_repair(monkeypatch):
    from tabicl.prior import dataset
    generated = iter([torch.zeros(6,2), torch.arange(12.).reshape(6,2)])
    monkeypatch.setattr(dataset, "MLPSCM", lambda **kw: lambda: (next(generated), torch.arange(6.)))
    monkeypatch.setattr(dataset, "Reg2Cls", lambda params: lambda X,y: (X,y))
    prior = dataset.SCMPrior(n_jobs=1)
    checked = []
    def sanity(X, y, split):
        checked.append(X.clone())
        assert X.count_nonzero() > 0
        return True
    monkeypatch.setattr(prior, "sanity_check", sanity)
    prior.generate_dataset(dict(prior_type="mlp_scm", num_features=2, max_features=2,
                                num_classes=2, seq_len=6, train_size=3, device="cpu"))
    assert len(checked) == 1


@pytest.mark.parametrize("family", ["epit", "soccol"])
@pytest.mark.parametrize("ratio", [0., .25, .5, .75, 1.])
def test_real_tables_finite_and_batch_split_shared(family, ratio):
    prior = make(family, ratio)
    X, y, d, length, split, source = prior.get_batch()
    assert split.unique().numel() == length.unique().numel() == 1
    assert int(source.sum()) == round(8 * ratio)
    assert torch.isfinite(y).all()
    assert all(torch.isfinite(table).all() for table in X.unbind())
    assert (d > 0).all()


def test_imputation_uses_context_only_and_keeps_observed_values():
    X = torch.tensor([[2., float('nan')], [4., float('nan')],
                      [float('nan'), 100.], [999., float('nan')]])
    got = context_mean_impute(X, 2)
    torch.testing.assert_close(got, torch.tensor([[2.,0.],[4.,0.],[3.,100.],[999.,0.]]))
    changed = X.clone()
    changed[3, 0] = -999.
    assert context_mean_impute(changed, 2)[2, 0] == got[2, 0]


def test_generic_tables_identical_in_mixture_and_independent_of_informed_hps():
    pure = make(ratio=0.)
    mixed = make("soccol", .5, hp={"informed_mix_probs": (0.,1.), "informed_target_mix_weight": .9})
    expected, got = pure.get_batch(), mixed.get_batch()
    for i, (source, task) in enumerate(mixed.last_batch_ids):
        if source == 0:
            assert_batch_equal(tuple(part[i] for part in got[:5]), tuple(part[task] for part in expected[:5]))


def test_informed_preprocessing_does_not_change_targets_or_physical_samples():
    current, old = make(ratio=1.), make(ratio=1.)
    old.priors[1].fixed_hp['original_tabicl_input'] = False
    new_batch, old_batch = current.get_batch(), old.get_batch()
    torch.testing.assert_close(new_batch[1], old_batch[1], rtol=0, atol=0)
    np.testing.assert_array_equal(current.priors[1].last_pitting_feature_batch.features,
                                  old.priors[1].last_pitting_feature_batch.features)
    assert torch.isfinite(new_batch[0]).all()
    assert torch.isnan(old_batch[0]).any()


def test_informed_feature_shuffle_changes_columns_not_targets():
    shuffled, ordered = make(ratio=1.), make(ratio=1.)
    ordered.priors[1].fixed_hp['permute_features'] = False
    a, b = shuffled.get_batch(), ordered.get_batch()
    torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
    torch.testing.assert_close(a[0].sort(dim=-1).values, b[0].sort(dim=-1).values, rtol=0, atol=0)
    assert not torch.equal(a[0], b[0])


def test_two_ranks_partition_same_global_batch():
    # make() has eight local tables: compare two ranks with sixteen on one rank.
    single = make(ratio=.5)
    single.batch_size = 16
    single.group_counts = (4,4)
    left, right = make(ratio=.5, rank=0, world_size=2), make(ratio=.5, rank=1, world_size=2)
    whole, a, b = single.get_batch(), left.get_batch(), right.get_batch()
    for index in range(16):
        half, j = (a,index) if index < 8 else (b,index-8)
        assert_batch_equal(tuple(part[index] for part in whole), tuple(part[j] for part in half))


@pytest.mark.parametrize("workers", [0,2])
def test_resume_restores_exact_next_batch(workers):
    options = dict(batch_size=None, num_workers=workers)
    if workers:
        options['prefetch_factor'] = 2
    loader = CheckpointablePriorLoader(dataset=make(), **options)
    iterator = iter(loader)
    next(iterator)
    state = copy.deepcopy(loader.state_dict())
    expected = next(iterator)
    resumed = CheckpointablePriorLoader(dataset=make(), **options)
    resumed.load_state_dict(state)
    assert_batch_equal(expected, next(iter(resumed)))


def test_original_ensemble_really_has_eight_views():
    from tabicl.sklearn.preprocessing import EnsembleGenerator
    rng = np.random.default_rng(42)
    X = rng.normal(size=(32, 8))
    generator = EnsembleGenerator(n_estimators=8, norm_methods=['none','power'],
        feat_shuffle_method='latin', classification=False, random_state=42)
    generator.fit(X, rng.normal(size=32))
    views = generator.transform(X[:3])
    assert set(views) == {'none', 'power'}
    assert [value[0].shape[0] for value in views.values()] == [4,4]


@pytest.mark.parametrize("family,workers", [("epit",0), ("soccol",2)])
def test_grouped_training_resume_exact(tmp_path, monkeypatch, family, workers):
    from tests import test_paired_prior_streams as existing
    original = existing.small_training_config
    def config(*args, **kwargs):
        result = original(*args, **kwargs)
        result.original_prior_mix = True
        result.col_missing_value_embedding = False
        result.permute_features = True
        return result
    monkeypatch.setattr(existing, "small_training_config", config)
    existing.test_real_training_resume_reproduces_batches_weights_and_optimizer(tmp_path, family, workers)


@pytest.mark.parametrize("family", ["epit", "soccol"])
def test_grouped_two_process_training_matches_single(tmp_path, monkeypatch, family):
    from tests import test_paired_prior_streams as existing
    original = existing.small_training_config
    def config(*args, **kwargs):
        result = original(*args, **kwargs)
        result.original_prior_mix = True
        result.col_missing_value_embedding = False
        result.permute_features = True
        return result
    monkeypatch.setattr(existing, "small_training_config", config)
    existing.test_two_process_training_matches_single_and_resumes(tmp_path, family)


def test_grouped_checkpoint_predicts_with_imputation_and_eight_views(tmp_path):
    from tests.test_paired_prior_streams import small_training_config
    from tabicl.train.run import Trainer
    from tabicl import TabICLRegressor
    config = small_training_config(tmp_path, "epit")
    config.original_prior_mix = True
    config.col_missing_value_embedding = False
    config.permute_features = True
    trainer = Trainer(config)
    trainer.run_batch(next(trainer.dataloader))
    trainer.save_checkpoint("step-1.ckpt")
    model = TabICLRegressor(model_path=tmp_path / "step-1.ckpt", allow_auto_download=False,
        device="cpu", n_estimators=8, norm_methods=['none','power'], feat_shuffle_method='latin',
        n_jobs=1, use_amp=False)
    rng = np.random.default_rng(12)
    X, y = rng.normal(size=(28, 12)), rng.normal(size=28)
    X[:,0] = 1.  # Constant filtering must work with grouped checkpoints.
    X[0,2] = X[25,3] = np.nan
    model.fit(X[:24], y[:24])
    pred = model.predict(X[24:])
    assert pred.shape == (4,) and np.isfinite(pred).all()
    assert not model.preserve_missing_values_
    assert model.ensemble_generator_.n_features_in_ == 11
    assert sum(len(configs) for configs in model.ensemble_generator_.ensemble_configs_.values()) == 8


@pytest.mark.parametrize("family", ["epit", "soccol"])
def test_informed_optuna_parameters_keep_the_same_base_samples(family):
    a = make(family, 1., hp={"informed_mix_probs": (1.,0.), "informed_target_mix_weight": .1})
    b = make(family, 1., hp={"informed_mix_probs": (0.,1.), "informed_target_mix_weight": .9})
    first, second = a.get_batch(), b.get_batch()
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    assert not torch.equal(first[1], second[1])
