"""S2 initialization, actual source allocation, source parity and exact replay."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts.combined_corrosion import finetune_single_rule as launcher
from scripts.epit_pipeline import finetune
from tabicl.train.run import Trainer
from tabicl.train.run_single_rule_finetuning import SingleRuleFinetuningTrainer
from tabicl.train.run_selected_pairs_single_rule import SingleRuleTrainer
from tests.test_combined_selected_pairs import config as combined_config
from tests.test_selected_pairs_single_rule import config_for
from tests.test_paired_prior_streams import assert_batch_equal
from tests.test_epit_finetuning import legacy_parent, stop_workers


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(tmp_path, prior, workers=0):
    c = combined_config(tmp_path, workers) if prior == 'combined' else config_for(
        tmp_path, 'soccol' if prior == 'soccol' else 'epit', workers)
    c.single_rule_s2_prior = prior
    c.informed_prior_ratio = 0. if prior == 'generic' else 1.
    if prior != 'combined':
        c.combined_soccol_overrides = None
    else:
        c.combined_soccol_overrides['informed_prior_ratio'] = 1.
    if prior == 'generic':
        c.epit_selected_pair_dir = None
    c.batch_size = 16
    c.batch_size_per_gp = 4
    c.np_seed = c.python_seed = c.torch_seed = c.prior_stream_seed = 43
    return c


def dataset(c, rank=0, world=1, cls=SingleRuleFinetuningTrainer):
    t = cls.__new__(cls)
    t.config = deepcopy(c)
    t.config.batch_size //= world
    t.ddp_rank, t.ddp_world_size, t.master_process = rank, world, False
    t.missingness_augmenter = None
    t.configure_prior()
    return t.dataloader.dataset


@pytest.mark.parametrize('prior', ['epit', 'soccol', 'combined', 'generic'])
def test_actual_allocation_and_distributed_tables(tmp_path, prior):
    c = config(tmp_path, prior)
    single = dataset(c)
    ranks = [dataset(c, rank, 2) for rank in range(2)]
    expected_counts = {'epit': (0, 4), 'soccol': (0, 4), 'combined': (0, 2, 2), 'generic': (4, 0)}[prior]
    for step in range(3):
        assert single._global_group_counts(step) == expected_counts
        batch = single.get_batch()
        expected = {key: tuple(t[i] for t in batch) for i, key in enumerate(single.last_batch_ids)}
        seen = set()
        for rank in ranks:
            actual = rank.get_batch()
            for i, key in enumerate(rank.last_batch_ids):
                assert key not in seen
                seen.add(key)
                assert_batch_equal(tuple(t[i] for t in actual), expected[key])
        assert seen == set(expected)


def test_combined_keeps_standalone_tables_and_generic_control_unchanged(tmp_path):
    combo = dataset(config(tmp_path, 'combined'))
    sources = [dataset(config(tmp_path, p)) for p in ('epit', 'soccol')]
    for _ in range(3):
        expected = {}
        for source, p in enumerate(sources, 1):
            b = p.get_batch()
            expected.update({(source, key[1]): tuple(t[i] for t in b) for i, key in enumerate(p.last_batch_ids)})
        b = combo.get_batch()
        for i, key in enumerate(combo.last_batch_ids):
            assert_batch_equal(tuple(t[i] for t in b), expected[key])
    c = config(tmp_path, 'generic')
    new, original = dataset(c), dataset(c, cls=Trainer)
    for _ in range(3):
        assert_batch_equal(new.get_batch(), original.get_batch())


@pytest.mark.parametrize('prior', ['epit', 'soccol', 'combined'])
@pytest.mark.parametrize('workers', [0, 2])
def test_parent_weights_fresh_optimizer_and_exact_resume(tmp_path, prior, workers):
    parent_config = config(tmp_path/'parent', 'generic')
    parent = Trainer(parent_config)
    parent.run_batch(next(parent.dataloader))
    parent.curr_step = 1
    parent.save_checkpoint('step-1.ckpt')
    parent_state = deepcopy(parent.raw_model.state_dict())
    c = config(tmp_path/'stage', prior, workers)
    c.checkpoint_path = str(tmp_path/'parent/step-1.ckpt')
    c.only_load_model = c.save_initial_checkpoint = True
    for key, value in finetune.S2_SETTINGS.items():
        if key not in ('batch_size', 'micro_batch_size', 'max_steps'):
            setattr(c, key, value)
    c.max_steps = 2
    c.save_perm_every = c.save_temp_every = 1
    trained = SingleRuleFinetuningTrainer(c)
    assert trained.curr_step == 0 and not trained.optimizer.state
    assert trained.optimizer.param_groups[0]['lr'] == 2e-5
    for k, v in parent_state.items():
        torch.testing.assert_close(trained.raw_model.state_dict()[k], v, atol=0, rtol=0)
    if workers:
        trained.dataloader.kwargs['multiprocessing_context'] = 'spawn'
    try:
        trained.train()
    finally:
        stop_workers(trained)
    zero = torch.load(tmp_path/'stage/step-0.ckpt', weights_only=True)
    assert zero['curr_step'] == 0 and not zero['optimizer_state']['state']
    for k, v in parent_state.items():
        torch.testing.assert_close(zero['state_dict'][k], v, atol=0, rtol=0)
    resumed_config = deepcopy(c)
    resumed_config.checkpoint_path = str(tmp_path/'stage/step-1.ckpt')
    resumed_config.only_load_model = False
    resumed = SingleRuleFinetuningTrainer(resumed_config)
    assert resumed.curr_step == 1
    if workers:
        resumed.dataloader.kwargs['multiprocessing_context'] = 'spawn'
    try:
        resumed.train()
    finally:
        stop_workers(resumed)
    for k, v in trained.raw_model.state_dict().items():
        torch.testing.assert_close(resumed.raw_model.state_dict()[k], v, atol=0, rtol=0)
    assert resumed.scheduler.state_dict() == trained.scheduler.state_dict()
    for name, state in trained.optimizer.state_dict()['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(resumed.optimizer.state_dict()['state'][name][key], value, atol=0, rtol=0)
    with pytest.raises(ValueError, match='matching recipe/profile/seed'):
        SingleRuleTrainer(resumed_config)


@pytest.mark.parametrize('prior', ['epit', 'soccol', 'combined', 'generic'])
@pytest.mark.parametrize('processes', [1, 2])
def test_plan_and_resume(tmp_path, legacy_parent, prior, processes):
    args = launcher.parse_args(['--prior', prior, '--finetune-from', str(legacy_parent),
        '--checkpoint-root', str(tmp_path/'checkpoints'), '--output-dir', str(tmp_path/'output'),
        '--nproc-per-node', str(processes)])
    plan = launcher.build_plan(args)
    c = launcher.validate_plan(plan)
    for key, value in {**finetune.S2_SETTINGS, **finetune.RETAINED_TABLE_SETTINGS}.items():
        assert getattr(c, key) == value
    assert plan['source_proportions'] == {
        'generic': float(prior == 'generic'), 'epit': .5 if prior == 'combined' else float(prior == 'epit'),
        'soccol': .5 if prior == 'combined' else float(prior == 'soccol')}
    directory, output = Path(plan['checkpoint_dir']), Path(plan['output_dir'])
    assert not directory.exists() and not output.exists()
    directory.mkdir(parents=True); output.mkdir()
    (directory/'single_rule_finetuning.json').write_text(json.dumps(plan))
    (output/'training_plan.json').write_text(json.dumps(plan))
    (directory/'step-100.ckpt').touch()
    with pytest.raises(FileExistsError):
        launcher.build_plan(args)
    args.resume = True
    assert launcher.build_plan(args) == plan
    resumed = launcher.training_config(launcher.launch_command(plan, resume=True))
    assert resumed.checkpoint_path is None and not resumed.only_load_model
    args.workers += 1
    with pytest.raises(ValueError, match='Resume requires identical'):
        launcher.build_plan(args)


@pytest.mark.parametrize('name', ['epit', 'soccol', 'mpea', 'rebar'])
def test_evaluation_uses_native_cv_and_s2_steps(tmp_path, name):
    from scripts.combined_corrosion import evaluate_finetuning_single_rule as evaluation
    args = evaluation.parse_args(['--prior', 'combined', '--dataset', name,
        '--output-dir', str(tmp_path/'results'), '--max-checkpoint-step', '1500',
        '--checkpoint-scope', 'common', '--dry-run'])
    evaluator, options = evaluation.comparison_args(args)
    command = evaluator.evaluation_command(options)
    assert command[command.index('--exploratory-cv-folds')+1] == '5'
    assert command[command.index('--min-checkpoint-step')+1] == '0'
    assert command[command.index('--max-checkpoint-step')+1] == '1500'
    assert command[command.index('--checkpoint-scope')+1] == 'common'
    assert options.output_dir.name == f'results_{name}'
    assert options._selected_model_runs == [
        ('combined_single_rule_s2', args.finetune_run), ('generic_continuation', args.control_run)]
    assert 'generic_baseline' not in command
    assert '--fixed-schema-model' not in command


def test_evaluation_rejects_unmatched_controls(tmp_path, legacy_parent):
    from scripts.combined_corrosion import evaluate_finetuning_single_rule as evaluation
    from scripts.epit_pipeline.train_final import replace_option
    plans = {}
    for prior in ('combined', 'generic'):
        args = launcher.parse_args(['--prior', prior, '--finetune-from', str(legacy_parent),
            '--checkpoint-root', str(tmp_path/'checkpoints'), '--output-dir', str(tmp_path/prior)])
        plan = launcher.build_plan(args)
        directory = Path(plan['checkpoint_dir']); directory.mkdir(parents=True)
        (directory/'single_rule_finetuning.json').write_text(json.dumps(plan))
        (directory/'step-0.ckpt').touch()
        plans[prior] = plan
    args = evaluation.parse_args(['--prior', 'combined', '--dataset', 'mpea',
        '--finetune-run', plans['combined']['checkpoint_dir'],
        '--control-run', plans['generic']['checkpoint_dir'], '--output-dir', str(tmp_path/'eval')])
    evaluation.validate_stages(args)
    control_path = Path(plans['generic']['checkpoint_dir'])/'single_rule_finetuning.json'
    for change in ('parent', 'world', 'hyperparameter', 'generator'):
        changed = deepcopy(plans['generic'])
        if change == 'parent':
            changed['finetuning']['parent_sha256'] = 'another_parent'
        elif change == 'world':
            changed['nproc_per_node'] = 2
        elif change == 'hyperparameter':
            replace_option(changed['command'], '--icl_nhead', '8')
        else:
            changed['generic_stream_identity']['implementation_sha256']['mlp_scm.py'] = 'changed'
        control_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            evaluation.validate_stages(args)


def test_actual_historical_control_remains_compatible(tmp_path):
    from scripts.combined_corrosion import evaluate_finetuning_single_rule as evaluation
    if not (evaluation.DEFAULT_CONTROL/'step-0.ckpt').is_file() or not finetune.DEFAULT_PARENT.is_file():
        pytest.skip('Local historical generic S2 control is not available.')
    for prior in ('epit', 'soccol', 'combined'):
        a = launcher.parse_args(['--prior', prior, '--checkpoint-root', str(tmp_path/'checkpoints'),
            '--output-dir', str(tmp_path/prior)])
        plan = launcher.build_plan(a)
        directory = Path(plan['checkpoint_dir']); directory.mkdir(parents=True)
        (directory/'single_rule_finetuning.json').write_text(json.dumps(plan))
        (directory/'step-0.ckpt').touch()
        args = evaluation.parse_args(['--prior', prior, '--dataset', 'epit',
            '--finetune-run', str(directory), '--output-dir', str(tmp_path/'evaluation')])
        evaluation.validate_stages(args)
