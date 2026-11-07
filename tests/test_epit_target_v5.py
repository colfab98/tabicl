"""End-to-end checks for the opt-in EPIT final-target recipe."""
import copy

import numpy as np
import pytest
import torch

from scripts.epit_pipeline import run_optuna as search, train_final
from scripts.epit_pipeline.formula_extension_study import load_study
from tabicl.prior.epit_expanded_schema import EPIT_EXPANDED_FEATURE_COLUMNS
from tabicl.prior.epit_feature_profile import load_epit_feature_profile
from tabicl.prior.epit_target_v5 import (
    EpitV5PriorMixture, _EpitV5Prior, heat_formula_drive, load_recipe,
    mix_bounds, recipe_identity, table_mix_weight,
)
from tabicl.prior.original_mixture import OriginalPriorMixture
from tests.test_paired_prior_streams import make_prior, assert_batch_equal


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make(ratio=1.0, **kwargs):
    return make_prior(ratio=ratio, dataset_cls=EpitV5PriorMixture, **kwargs)


def test_runtime_equation_matches_development_fit_and_excludes_final_rows():
    recipe = load_recipe()
    study, frozen, _ = load_study(search.DEFAULT_SPLIT_MANIFEST)
    assert recipe['split_manifest_sha256'] == frozen.manifest_sha256
    assert recipe['formula']['fitting_task_row_indices'] == study.rows.tolist()
    assert recipe['final_test_targets_used'] is False
    frame = study.frame.copy()
    for name, mapping in recipe['category_mappings'].items():
        frame[name] = frame[name].map(mapping)
    X = torch.as_tensor(frame.loc[:, list(EPIT_EXPANDED_FEATURE_COLUMNS)].to_numpy(dtype=float))
    profile = load_epit_feature_profile()
    method_codes = torch.tensor([
        profile.test_method_mapping[str(study.dataset.rows[i].get('[Cl-] Test Method') or '<NA>').strip()]
        for i in study.rows])
    actual = heat_formula_drive(X, method_codes, recipe).numpy()
    np.testing.assert_allclose(actual, study.predict_artifact(recipe['formula']), rtol=1e-12, atol=1e-12)


def test_heat_labels_affect_target_and_unknown_labels_are_neutral():
    recipe = load_recipe()
    X = torch.zeros(4, 70, dtype=torch.float64)
    X[:, EPIT_EXPANDED_FEATURE_COLUMNS.index('Cr')] = 18.
    X[:, EPIT_EXPANDED_FEATURE_COLUMNS.index('CP_temp')] = 25.
    X[:, EPIT_EXPANDED_FEATURE_COLUMNS.index('CP_Cl')] = .1
    X[:, EPIT_EXPANDED_FEATURE_COLUMNS.index('CP_pH')] = 7.
    heat_col = EPIT_EXPANDED_FEATURE_COLUMNS.index('heat_treatment_type')
    cooling_col = EPIT_EXPANDED_FEATURE_COLUMNS.index('heat_treatment_cooling')
    X[:, cooling_col] = float('nan')
    X[:, heat_col] = torch.tensor([recipe['category_mappings']['heat_treatment_type'][level]
                                  for level in ['annealing', 'soaking']] + [float('nan'), 999.])
    score = heat_formula_drive(X, torch.zeros(4, dtype=torch.long), recipe)
    assert score[0] != score[1]
    assert score[2] == score[3]
    neutral = copy.deepcopy(recipe)
    neutral['formula']['coefficients'][1:] = [0.] * (len(neutral['formula']['coefficients']) - 1)
    assert score[2] == heat_formula_drive(X, torch.zeros(4, dtype=torch.long), neutral)[2]


def test_variable_mix_preserves_mean_and_rng_states():
    mean = .28659758621052106
    before = np.random.get_state()
    values = np.array([table_mix_weight(mean, seed) for seed in range(10000)])
    assert mix_bounds(mean) == (0., 2 * mean)
    assert 0 < values.min() < values.max() < 2 * mean
    assert values.mean() == pytest.approx(mean, abs=.004)
    assert values.std() > .1
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert table_mix_weight(mean, 42) == table_mix_weight(mean, 42)
    assert table_mix_weight(0., 42) == 0.
    assert table_mix_weight(1., 42) == 1.
    for value in [-.1, 1.1, float('nan')]:
        with pytest.raises(ValueError):
            table_mix_weight(value, 42)


@pytest.mark.parametrize('ratio', [0., .5, 1.])
def test_v5_changes_only_informed_targets_at_matching_hyperparameters(ratio):
    original = make_prior(ratio=ratio, dataset_cls=OriginalPriorMixture)
    current = make(ratio=ratio)
    for _ in range(2):
        before, after = original.get_batch(), current.get_batch()
        assert_batch_equal((before[0], *before[2:]), (after[0], *after[2:]))
        generic = after[5] == 0
        torch.testing.assert_close(before[1][generic], after[1][generic], rtol=0, atol=0)
        if ratio > 0:
            assert not torch.equal(before[1][~generic], after[1][~generic])
            np.testing.assert_array_equal(original.priors[1].last_pitting_feature_batch.features,
                                          current.priors[1].last_pitting_feature_batch.features)


def test_actual_target_uses_one_weight_per_table_and_both_formula_sources(monkeypatch):
    observed = []
    original = _EpitV5Prior._generate_scm_epit_target_from_physical_features
    def record(self, X, params, cls):
        target = original(self, X, params, cls)
        weight = self.last_pitting_target_rule['target_mix_weight']
        expected = self._standardize_signal((1 - weight) * self._standardize_signal(self._scm_target_for_mix)
                   + weight * self._standardize_signal(self.last_pitting_target_drive))
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
        observed.append((self.last_pitting_target_rule['target_rule_family'], weight))
        return target
    monkeypatch.setattr(_EpitV5Prior, '_generate_scm_epit_target_from_physical_features', record)
    prior = make()
    for _ in range(3):
        assert torch.isfinite(prior.get_batch()[1]).all()
    assert {family for family, _ in observed} == {'pren_n_linear', 'extend_heat_categories'}
    assert len(set(weight for _, weight in observed)) > 1


def test_v5_distributed_partition_and_resume_match_single_process():
    full = make(ratio=.5)
    full.get_batch()
    ranks = [make(ratio=.5, rank=rank, world_size=2) for rank in range(2)]
    # make_prior has 8 tables/rank; compare with a 16-table global batch.
    global_prior = EpitV5PriorMixture(batch_size=16, batch_size_per_gp=2,
        min_features=70, max_features=70, max_seq_len=32, min_train_size=.5, max_train_size=.8,
        scm_fixed_hp=ranks[0].priors[1].fixed_hp, scm_sampled_hp={},
        informed_prior_ratio=.5, return_source_ids=True)
    global_batch = global_prior.get_batch()
    for rank, prior in enumerate(ranks):
        local = prior.get_batch()
        for index in range(8):
            assert_batch_equal(tuple(t[index] for t in local),
                               tuple(t[rank * 8 + index] for t in global_batch))
    state = full.sampling_state_dict()
    resumed = make(ratio=.5)
    resumed.load_sampling_state_dict(state)
    assert_batch_equal(full.get_batch(), resumed.get_batch())
    assert full.checkpoint_identity() == resumed.checkpoint_identity()
    assert full.checkpoint_identity() != make_prior(ratio=.5, dataset_cls=OriginalPriorMixture).checkpoint_identity()
    assert full.checkpoint_identity()['epit_target_recipe'] == recipe_identity(.4)


def test_final_v5_overrides_are_explicit_and_do_not_change_selected_trial():
    args = train_final.parse_args(['--storage', 'journal:///unused', '--epit-target-version', 'v5'])
    args.pitting_composition_mode = 'empirical_features_scm_target'
    selected = search.trial_params_from_mapping({
        'mlp_prob': .6887744348882755, 'informed_target_mix_weight': .28659758621052106,
        'pitting_composition_perturb_strength': .29595367291630664,
    }, composition_mode=args.pitting_composition_mode, fixed_informed_prior_ratio=1.)
    effective, provenance = train_final.apply_parameter_overrides(args, selected, composition_mode=args.pitting_composition_mode)
    assert effective.mlp_prob == .70
    assert effective.pitting_composition_perturb_strength == .30
    assert effective.informed_target_mix_weight == selected.informed_target_mix_weight
    assert selected.mlp_prob == .6887744348882755
    assert provenance['trial_parameter_overrides'] == {'mlp_prob': .70, 'pitting_composition_perturb_strength': .30}
    with pytest.raises(ValueError, match='another dataset'):
        make_prior('soccol', 1., dataset_cls=EpitV5PriorMixture)


@pytest.mark.parametrize('workers', [0, 2])
def test_v5_training_checkpoint_resumes_identically(tmp_path, monkeypatch, workers):
    from tests import test_paired_prior_streams as existing
    original = existing.small_training_config
    def config(*args, **kwargs):
        result = original(*args, **kwargs)
        result.original_prior_mix = True
        result.col_missing_value_embedding = False
        result.permute_features = True
        result.epit_target_version = 'v5'
        return result
    monkeypatch.setattr(existing, 'small_training_config', config)
    existing.test_real_training_resume_reproduces_batches_weights_and_optimizer(tmp_path, 'epit', workers)
