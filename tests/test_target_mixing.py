"""Verify shared EPIT/Soccol target weights and preservation of other streams."""
import numpy as np
import pytest
import torch

from tabicl.prior.epit_target_v5 import EpitV5PriorMixture
from tabicl.prior.original_mixture import OriginalPriorMixture
from tabicl.prior.target_mixing import (
    VariableTargetMixture, _VariableTargetMixPrior, mix_policy_identity, table_mix_weight,
)
from tests.test_paired_prior_streams import assert_batch_equal, loader, make_prior


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_epit_and_soccol_use_the_same_per_table_weight_and_target_equation(monkeypatch):
    observed = {}
    generate = _VariableTargetMixPrior._generate_scm_epit_target_from_physical_features

    def record(self, X, params, prior_cls):
        target = generate(self, X, params, prior_cls)
        rule = self.last_pitting_target_rule
        weight = rule['target_mix_weight']
        assert weight == table_mix_weight(.28, params['paired_task_seed'])
        assert 0. < weight < .56
        expected = self._standardize_signal(
            (1 - weight) * self._standardize_signal(self._scm_target_for_mix)
            + weight * self._standardize_signal(self.last_pitting_target_drive))
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
        observed[params['paired_task_seed']] = weight
        return target

    monkeypatch.setattr(_VariableTargetMixPrior, '_generate_scm_epit_target_from_physical_features', record)
    weights = []
    for family, cls in [('epit', EpitV5PriorMixture), ('soccol', VariableTargetMixture)]:
        observed.clear()
        prior = make_prior(family, 1., dataset_cls=cls, hp={'informed_target_mix_weight': .28})
        for _ in range(3):
            assert torch.isfinite(prior.get_batch()[1]).all()
        weights.append(observed.copy())
    assert weights[0] == weights[1]
    assert len(set(weights[0].values())) == 24


@pytest.mark.parametrize('family', ['epit', 'soccol'])
@pytest.mark.parametrize('ratio', [0., .5, 1.])
def test_variable_weight_changes_only_informed_targets(family, ratio):
    before = make_prior(family, ratio, dataset_cls=OriginalPriorMixture)
    after = make_prior(family, ratio, dataset_cls=VariableTargetMixture)
    for _ in range(2):
        fixed, varying = before.get_batch(), after.get_batch()
        assert_batch_equal((fixed[0], *fixed[2:]), (varying[0], *varying[2:]))
        generic = varying[5] == 0
        torch.testing.assert_close(fixed[1][generic], varying[1][generic], rtol=0, atol=0)
        if ratio:
            assert not torch.equal(fixed[1][~generic], varying[1][~generic])
            np.testing.assert_array_equal(before.priors[1].last_pitting_feature_batch.features,
                                          after.priors[1].last_pitting_feature_batch.features)
            for key in ['target_rule_family', 'target_rule_coefficients']:
                assert before.priors[1].last_pitting_target_rule[key] == after.priors[1].last_pitting_target_rule[key]


@pytest.mark.parametrize('mean', [0., 1.])
def test_endpoint_means_match_fixed_weight_targets(mean):
    settings = {'informed_target_mix_weight': mean}
    fixed = make_prior('soccol', 1., dataset_cls=OriginalPriorMixture, hp=settings)
    varying = make_prior('soccol', 1., dataset_cls=VariableTargetMixture, hp=settings)
    assert_batch_equal(fixed.get_batch(), varying.get_batch())


def test_changed_target_policy_cannot_resume_fixed_weight_loader():
    fixed = loader(make_prior('soccol', .5, dataset_cls=OriginalPriorMixture), 0)
    next(fixed)
    varying = loader(make_prior('soccol', .5, dataset_cls=VariableTargetMixture), 0)
    assert varying.dataset.checkpoint_identity()['informed_target_mix_policy'] == mix_policy_identity(.4)
    with pytest.raises(ValueError, match='same worker count and dataset settings'):
        varying.load_state_dict(fixed.state_dict())


@pytest.fixture
def variable_training(monkeypatch):
    from tests import test_paired_prior_streams as existing

    original = existing.small_training_config
    def config(*args, **kwargs):
        result = original(*args, **kwargs)
        result.original_prior_mix = True
        result.col_missing_value_embedding = False
        result.permute_features = True
        result.informed_target_mix_policy = 'uniform_per_table'
        result.informed_target_mix_weight = .28
        return result
    monkeypatch.setattr(existing, 'small_training_config', config)
    return existing


@pytest.mark.parametrize('workers', [0, 2])
def test_soccol_training_resumes_with_identical_batches_and_weights(tmp_path, variable_training, workers):
    variable_training.test_real_training_resume_reproduces_batches_weights_and_optimizer(tmp_path, 'soccol', workers)


def test_soccol_two_process_training_matches_single_and_resumes(tmp_path, variable_training):
    variable_training.test_two_process_training_matches_single_and_resumes(tmp_path, 'soccol')
