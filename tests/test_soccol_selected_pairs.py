"""Selected-pair integration and EPIT parity; inference fixtures are not results."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.soccol_pipeline import prepare_selected_pairs as prepare
from scripts.soccol_pipeline import train_selected_pairs as launch
from scripts.soccol_pipeline.selected_pair_formulas import load_inputs, predict_formulas
from tabicl.prior.soccol_selected_pairs import SelectedPairSampler, SelectedPairPriorMixture
from tabicl.prior.epit_schema import EPIT_COMPOSITION_ELEMENTS
from tabicl.train.train_config import build_parser
from tests.test_paired_prior_streams import make_prior, assert_batch_equal, small_training_config


@pytest.fixture(scope='module')
def recipe(tmp_path_factory):
    directory = tmp_path_factory.mktemp('soccol_selected_pair_test_only')
    checkpoint = directory / 'fake-checkpoint.ckpt'
    checkpoint.write_bytes(b'Test only: inference is mocked, not model results.')
    output = directory / 'recipe'
    seen = []
    dataset, accepted, _ = load_inputs()

    def predictions(**kwargs):
        split = kwargs['task_split']
        assert not set(split.train_index) & set(split.test_index)
        assert kwargs['native_tabicl_baseline']
        assert kwargs['regression_output'] == 'median'
        assert not kwargs['expected_preserve_missing_values']
        query = split.test_index
        seen.extend(query.tolist())
        target = dataset.targets.iloc[query].to_numpy()
        kwargs['prediction_sink'](pd.DataFrame({'task_row_index': query, 'observed_mV': target,
                                              'prediction_mV': target + 180.}))
        return {'test_mae': 180.}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(prepare.ev, 'evaluate_estimator', predictions)
        prepare.run(Namespace(checkpoint=checkpoint, output_dir=output, dry_run=False))
    assert len(seen) == len(set(seen)) == 3755
    assert set(accepted.task_row_index) <= set(seen)
    assert len(pd.read_csv(output / 'excluded_rows.csv')) == 282
    assert len(pd.read_csv(output / 'all_pairs.csv')) == 37450
    return output


@pytest.fixture(scope='module')
def sampler(recipe):
    return SelectedPairSampler(recipe)


def test_complete_scope_and_frozen_predictions(sampler):
    assert len(sampler.rows) == 3745
    assert sampler.rows.outer_split.eq('final_test').sum() == 795
    assert sampler.rows.composition_status.eq('accepted').all()
    assert len(sampler.formulas) == 10
    assert 'extend_surface_numeric' in sampler.formulas
    batch = sampler.sample(1500, perturb_strength=0, random_state=42)
    expected = sampler.pairs.iloc[batch.pair_indices]
    np.testing.assert_allclose(batch.target_mV, expected.prediction_mV, rtol=1e-10, atol=1e-8)
    assert len(set(batch.formulas)) == 10
    parents = sampler.rows.iloc[sampler.parent_positions.loc[batch.parent_indices].to_numpy()].reset_index(drop=True)
    pd.testing.assert_frame_equal(batch.physical, parents)
    # Runtime formula evaluation must never depend on parent labels.
    unlabeled = batch.physical.copy()
    unlabeled['E_pit'] = np.nan
    predictions = predict_formulas(unlabeled, sampler.formulas)
    for name in sampler.formulas:
        mask = batch.formulas == name
        np.testing.assert_allclose(predictions[name][mask], batch.target_mV[mask])


def test_parent_perturbation_and_pair_ratio(sampler):
    a = sampler.sample(100, perturb_strength=.3, random_state=43)
    b = sampler.sample(100, perturb_strength=.3, random_state=43)
    np.testing.assert_array_equal(a.target_mV, b.target_mV)
    parents = sampler.rows.iloc[sampler.parent_positions.loc[a.parent_indices].to_numpy()].reset_index(drop=True)
    columns = list(EPIT_COMPOSITION_ELEMENTS)
    perturbed = a.physical[columns].to_numpy()
    np.testing.assert_allclose(perturbed.sum(axis=1), 100)
    assert (perturbed[parents[columns].to_numpy() == 0] == 0).all()
    pd.testing.assert_frame_equal(a.physical.drop(columns=columns), parents.drop(columns=columns))
    assert sampler.ratio == len(sampler.pairs) / (len(sampler.rows) * len(sampler.formulas))


def prior_kwargs(recipe):
    return dict(family='soccol', dataset_cls=SelectedPairPriorMixture, selected_pair_dir=recipe,
                hp={'pitting_coefficient_variation': 0., 'pitting_composition_perturb_strength': .3})


def test_generic_stream_and_formula_only_targets(recipe, monkeypatch):
    from tabicl.prior.epit_selected_pairs import _SelectedPrior
    from tabicl.prior.original_mixture import OriginalPriorMixture
    torch.set_num_threads(1)
    def forbidden(*args, **kwargs):
        pytest.fail('Selected informed target called SCM target generation')
    monkeypatch.setattr(_SelectedPrior, '_generate_scm_target_from_physical_features', forbidden)
    selected = make_prior(**prior_kwargs(recipe))
    reference = make_prior('soccol', dataset_cls=OriginalPriorMixture, ratio=0.)
    for _ in range(3):
        batch, generic = selected.get_batch(), reference.get_batch()
        lookup = {key: i for i, key in enumerate(reference.last_batch_ids)}
        for i, key in enumerate(selected.last_batch_ids):
            if key[0] == 0:
                j = lookup[key]
                assert_batch_equal(tuple(x[i] for x in batch[:5]), tuple(x[j] for x in generic[:5]))
        assert torch.isfinite(batch[1]).all()
        assert all(torch.isfinite(x).all() for x in batch[0].unbind())


def test_global_allocation_and_resume(recipe):
    torch.set_num_threads(1)
    whole = make_prior(**prior_kwargs(recipe))
    whole.batch_size = 16
    ranks = [make_prior(**prior_kwargs(recipe), rank=i, world_size=2) for i in range(2)]
    for _ in range(3):
        full = whole.get_batch()
        parts = [prior.get_batch() for prior in ranks]
        assert whole.last_batch_ids == [key for prior in ranks for key in prior.last_batch_ids]
        for rank, part in enumerate(parts):
            for row in range(8):
                assert_batch_equal(tuple(x[rank * 8 + row] for x in full), tuple(x[row] for x in part))
    state = whole.sampling_state_dict()
    expected = whole.get_batch()
    restored = make_prior(**prior_kwargs(recipe))
    restored.batch_size = 16
    restored.load_sampling_state_dict(state)
    assert_batch_equal(expected, restored.get_batch())


@pytest.mark.parametrize('workers', [0, 2])
def test_training_checkpoint_restores_stream_and_optimizer(recipe, tmp_path, workers):
    from tabicl.train.run import Trainer
    torch.set_num_threads(1)
    config = small_training_config(tmp_path, 'soccol', workers)
    config.original_prior_mix = True
    config.col_missing_value_embedding = False
    config.soccol_selected_pair_dir = str(recipe)
    config.pitting_coefficient_variation = 0.
    config.pitting_composition_perturb_strength = .3
    config.informed_target_mix_policy = 'fixed'
    config.informed_target_mix_weight = 1.
    config.permute_features = True
    first = Trainer(config)
    if workers:
        first.dataloader.kwargs['multiprocessing_context'] = 'spawn'
    first.run_batch(next(first.dataloader))
    first.curr_step = 1
    first.save_checkpoint('step-1.ckpt')
    expected = next(first.dataloader)
    first.run_batch(expected)
    restored = Trainer(deepcopy(config))
    if workers:
        restored.dataloader.kwargs['multiprocessing_context'] = 'spawn'
    assert restored.curr_step == 1
    batch = next(restored.dataloader)
    assert_batch_equal(batch, expected)
    restored.run_batch(batch)
    for key, value in first.raw_model.state_dict().items():
        torch.testing.assert_close(value, restored.raw_model.state_dict()[key], rtol=0, atol=0)
    assert first.scheduler.state_dict() == restored.scheduler.state_dict()
    for key, state in first.optimizer.state_dict()['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, restored.optimizer.state_dict()['state'][key][name], rtol=0, atol=0)


def test_launcher_epit_parity_and_global_restoration(recipe):
    from scripts.epit_pipeline import train_selected_pairs as epit
    epit_recipe = epit.ROOT / 'corrosion_datasets/analysis/epit_pipeline/selected_pairs_all760_generic10k'
    args = Namespace(recipe_dir=epit_recipe, run_name='selected_pair_parity_test', device='cpu',
                     max_steps=10000, scheduler_total_steps=10000, nproc_per_node=1, workers=4)
    before = epit.build_plan(args)
    soccol_args = deepcopy(args)
    soccol_args.recipe_dir = recipe
    plan = launch.build_plan(soccol_args)
    after = epit.build_plan(args)
    assert before == after
    left, right = [vars(build_parser().parse_args(p['command'][2:])) for p in (before, plan)]
    differences = {key for key in left if left[key] != right[key]}
    # Dataset-specific inputs/formulas/paths only; architecture, optimizer,
    # batch settings, scheduler and all training seeds must agree exactly.
    allowed = {'checkpoint_dir', 'wandb_name', 'epit_selected_pair_dir', 'soccol_selected_pair_dir',
        'informed_prior_ratio', 'min_features', 'max_features', 'pitting_feature_profile',
        'pitting_target_rule_scores', 'pitting_target_rule_coefficients', 'pitting_coefficient_upper_bounds',
        'informed_normal_block_allocation', 'informed_normal_block_allocation_min_counts',
        'pitting_composition_family_probs', 'pitting_process_role', 'pitting_process_category_count'}
    assert differences <= allowed, {key: (left[key], right[key]) for key in differences - allowed}
    assert right['soccol_selected_pair_dir'] == str(recipe)
    assert right['informed_target_mix_weight'] == 1.
    assert right['informed_target_mix_policy'] == 'fixed'
    assert right['pitting_coefficient_variation'] == 0.
    assert not plan['automatic_evaluation']


@pytest.mark.parametrize('checkpoint,maximum', [('latest', None), ('500', None), ('all', 2000)])
def test_evaluation_controls_match_epit(checkpoint, maximum):
    from scripts.soccol_pipeline import evaluate_checkpoint_comparison as soccol
    from scripts.epit_pipeline import evaluate_checkpoint_comparison as epit
    argv = ['--candidate-run', '/tmp/test_candidate', '--candidate-label', 'selected', '--checkpoint', checkpoint,
            '--checkpoint-step-interval', '0', '--exploratory-cv-folds', '5', '--device', 'cpu']
    if maximum is not None:
        argv += ['--max-checkpoint-step', str(maximum)]
    a, b = epit.evaluation_command(epit.parse_args(argv)), soccol.evaluation_command(soccol.parse_args(argv))
    for flag in ('--checkpoint', '--checkpoint-step-interval', '--exploratory-cv-folds', '--device', '--n-estimators',
                 '--random-state', '--regression-output'):
        assert a[a.index(flag) + 1] == b[b.index(flag) + 1]
    assert '--epit-final-test' not in b
    assert '--native-tabicl-baselines' in b
    if maximum is not None:
        assert b[b.index('--max-checkpoint-step') + 1] == str(maximum)
