"""Early S2 replays keep the original schedule, data streams and initialization."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from scripts.combined_corrosion import finetune_single_rule as base
from scripts.combined_corrosion import finetune_single_rule_early as early
from scripts.combined_corrosion import evaluate_finetuning_single_rule as evaluation
from tests.test_epit_finetuning import legacy_parent, stop_workers
from tests.test_single_rule_finetuning import config, threads
from tabicl.train.run_single_rule_finetuning import SingleRuleFinetuningTrainer


def arguments(tmp_path, parent, prior):
    return early.parse_args(['--prior', prior, '--finetune-from', str(parent),
        '--checkpoint-root', str(tmp_path/'checkpoints'), '--output-dir', str(tmp_path/prior)])


@pytest.mark.parametrize('prior', ['epit', 'soccol', 'combined', 'generic'])
def test_only_stopping_and_saving_change(tmp_path, legacy_parent, prior):
    args = arguments(tmp_path, legacy_parent, prior)
    assert args.run_name == base.NAMES[prior] + '_early100_v1'
    expected = base.build_plan(args)
    plan = early.build_plan(args)
    before = vars(base.training_config(expected['command']))
    after = vars(base.training_config(plan['command']))
    assert {k for k in before if before[k] != after[k]} == {
        'max_steps', 'save_temp_every', 'save_perm_every'}
    assert (after['max_steps'], after['save_temp_every'], after['save_perm_every']) == (100, 10, 10)
    assert after['scheduler_total_steps'] == 2000
    assert after['save_initial_checkpoint'] and after['only_load_model']
    reference = early.reference_plan(plan)
    reference['implementation_sha256'] = expected['implementation_sha256']
    assert reference == expected
    assert 'tabicl_early_s2_plan_' not in json.dumps(plan)
    assert not Path(plan['checkpoint_dir']).exists() and not Path(plan['output_dir']).exists()

    directory, output = Path(plan['checkpoint_dir']), Path(plan['output_dir'])
    directory.mkdir(parents=True)
    output.mkdir()
    (directory/'single_rule_finetuning.json').write_text(json.dumps(plan))
    (output/'training_plan.json').write_text(json.dumps(plan))
    (directory/'step-10.ckpt').touch()
    with pytest.raises(FileExistsError):
        early.build_plan(args)
    args.resume = True
    assert early.build_plan(args) == plan
    resumed = base.training_config(base.launch_command(plan, resume=True))
    assert resumed.checkpoint_path is None and not resumed.only_load_model
    assert resumed.max_steps == 100 and resumed.scheduler_total_steps == 2000
    args.workers += 1
    with pytest.raises(ValueError, match='Resume requires identical'):
        early.build_plan(args)


@pytest.mark.parametrize('prior', ['epit', 'soccol', 'combined', 'generic'])
def test_more_frequent_saving_and_shorter_stop_preserve_training(tmp_path, prior):
    # Small real CPU models test the shared training loop, rather than merely
    # asserting command-line flags. Both runs start from the same parent weights.
    parent = SingleRuleFinetuningTrainer(config(tmp_path/'parent', 'generic'))
    parent.save_checkpoint('step-0.ckpt')
    checkpoints = []
    for name, steps, interval in [('reference', 4, 2), ('early', 2, 1)]:
        c = config(tmp_path/name, prior)
        for key, value in base.finetune.S2_SETTINGS.items():
            if key not in ('batch_size', 'micro_batch_size', 'max_steps'):
                setattr(c, key, value)
        c.max_steps = steps
        c.save_perm_every = c.save_temp_every = interval
        c.only_load_model = c.save_initial_checkpoint = True
        c.checkpoint_path = str(tmp_path/'parent/step-0.ckpt')
        trainer = SingleRuleFinetuningTrainer(c)
        try:
            trainer.train()
        finally:
            stop_workers(trainer)
        checkpoints.append(torch.load(tmp_path/name/'step-2.ckpt', weights_only=True))
    assert {p.name for p in (tmp_path/'early').glob('step-*.ckpt')} == {
        'step-0.ckpt', 'step-1.ckpt', 'step-2.ckpt'}
    left, right = checkpoints
    for key, value in left['state_dict'].items():
        torch.testing.assert_close(right['state_dict'][key], value, atol=0, rtol=0)
    assert left['scheduler_state'] == right['scheduler_state']
    assert left['optimizer_state']['param_groups'] == right['optimizer_state']['param_groups']
    for name, state in left['optimizer_state']['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(right['optimizer_state']['state'][name][key], value, atol=0, rtol=0)


@pytest.mark.parametrize('prior,dataset', [('epit', 'epit'), ('soccol', 'soccol'),
                                         ('combined', 'mpea'), ('combined', 'rebar')])
def test_early_evaluation_validates_schedule_and_uses_native_cv(tmp_path, legacy_parent, prior, dataset):
    plans = {}
    for source in (prior, 'generic'):
        plan = early.build_plan(arguments(tmp_path, legacy_parent, source))
        directory = Path(plan['checkpoint_dir'])
        directory.mkdir(parents=True)
        (directory/'single_rule_finetuning.json').write_text(json.dumps(plan))
        (directory/'step-0.ckpt').touch()
        plans[source] = plan
    args = evaluation.parse_args(['--prior', prior, '--dataset', dataset,
        '--finetune-run', plans[prior]['checkpoint_dir'],
        '--control-run', plans['generic']['checkpoint_dir'], '--output-dir', str(tmp_path/'eval'),
        '--max-checkpoint-step', '100', '--checkpoint-step-interval', '0',
        '--checkpoint-scope', 'common'])
    stages = evaluation.validate_stages(args)
    assert stages['informed']['finetuning']['additional_steps'] == 100
    assert stages['informed']['finetuning']['primary_checkpoint_step'] is None
    evaluator, options = evaluation.comparison_args(args)
    command = evaluator.evaluation_command(options)
    for key, value in {'exploratory-cv-folds': '5', 'min-checkpoint-step': '0',
                       'max-checkpoint-step': '100', 'checkpoint-step-interval': '0',
                       'checkpoint-scope': 'common'}.items():
        assert command[command.index('--'+key)+1] == value
    assert options.output_dir.name == 'eval_' + dataset
    control_path = Path(plans['generic']['checkpoint_dir'])/'single_rule_finetuning.json'
    for key, value in [('scheduler_total_steps', 100), ('lr', 1e-4), ('informed_prior_ratio', .3)]:
        changed = deepcopy(plans['generic'])
        base.train_final.replace_option(changed['command'], '--'+key, value)
        control_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            evaluation.validate_stages(args)
