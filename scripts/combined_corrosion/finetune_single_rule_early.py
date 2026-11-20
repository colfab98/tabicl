"""Replay the first 100 S2 steps, saving every 10 without shortening the LR schedule."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scripts.combined_corrosion import finetune_single_rule as base

SCHEMA = 'single_rule_finetuning_early_v1'
STOP = 100
SAVE_EVERY = 10
NAMES = {key: value + '_early100_v1' for key, value in base.NAMES.items()}
DIAGNOSTIC = {'stop_after_steps': STOP, 'save_every_steps': SAVE_EVERY,
              'reference_max_steps': 2000, 'purpose': 'early trajectory; no final-model selection'}


def parse_args(argv=None):
    args = base.parse_args(argv)
    # The base parser supplies the normal default when no name was passed.
    if args.run_name == base.NAMES[args.prior]:
        args.run_name = NAMES[args.prior]
    return args


def reference_plan(plan):
    """Validate the diagnostic differences before comparing the original S2 settings."""
    if plan.get('schema_version') != SCHEMA or plan.get('early_checkpoint_diagnostic') != DIAGNOSTIC:
        raise ValueError('Unsupported early S2 diagnostic.')
    config = base.training_config(plan['command'])
    for key, value in {'max_steps': STOP, 'save_temp_every': SAVE_EVERY,
                       'save_perm_every': SAVE_EVERY}.items():
        if getattr(config, key) != value:
            raise ValueError(f'Early S2 requires {key}={value}.')
    identity = plan['finetuning']
    if (identity['optimization'] != {**base.finetune.S2_SETTINGS, 'max_steps': STOP}
            or identity['additional_steps'] != STOP or identity['primary_checkpoint_step'] is not None):
        raise ValueError('Early stopping must preserve the original S2 optimization schedule.')
    reference = deepcopy(plan)
    reference['schema_version'] = base.SCHEMA
    del reference['early_checkpoint_diagnostic']
    for key, value in {'max_steps': 2000, 'save_temp_every': 100, 'save_perm_every': 100}.items():
        base.train_final.replace_option(reference['command'], '--' + key, value)
    reference['finetuning']['optimization'] = dict(base.finetune.S2_SETTINGS)
    reference['finetuning'].update(additional_steps=2000, primary_checkpoint_step=2000)
    base.validate_plan(reference)
    return reference


def build_plan(args):
    family = args.prior if args.prior != 'generic' else 'combined'
    directory = ((args.checkpoint_root or base.ROOT/f'checkpoints/{family}_finetuning')/args.run_name).resolve()
    analysis = 'combined_corrosion' if family == 'combined' else f'{family}_pipeline'
    output = (args.output_dir or base.ROOT/f'corrosion_datasets/analysis/{analysis}/single_rule_finetuning'/args.run_name).resolve()
    if not args.resume:
        for path in (directory, output):
            if path.exists() and (not path.is_dir() or any(path.iterdir())):
                raise FileExistsError(f'Run already exists: {path}. Use a new --run-name or --resume.')
    # Build the unchanged 2000-step reference in disposable, empty planning paths.
    # This also permits validating a resume without asking the old launcher to
    # interpret diagnostic metadata. No trainer or data generation is started.
    with TemporaryDirectory(prefix='tabicl_early_s2_plan_') as temporary:
        reference_args = deepcopy(args)
        reference_args.resume = False
        reference_args.checkpoint_root = Path(temporary)/'checkpoints'
        reference_args.output_dir = Path(temporary)/'output'
        plan = base.build_plan(reference_args)
    plan.update(schema_version=SCHEMA, checkpoint_dir=str(directory), output_dir=str(output),
                early_checkpoint_diagnostic=dict(DIAGNOSTIC))
    for key, value in {'checkpoint_dir': str(directory), 'max_steps': STOP,
                       'save_temp_every': SAVE_EVERY, 'save_perm_every': SAVE_EVERY}.items():
        base.train_final.replace_option(plan['command'], '--' + key, value)
    plan['finetuning']['optimization']['max_steps'] = STOP
    plan['finetuning'].update(additional_steps=STOP, primary_checkpoint_step=None)
    plan['implementation_sha256'][str(Path(__file__).relative_to(base.ROOT))] = base.sha256_file(Path(__file__))
    reference_plan(plan)
    if args.resume:
        if not list(directory.glob('step-*.ckpt')):
            raise FileNotFoundError('Resume requires a saved early S2 checkpoint.')
        for path in (directory/'single_rule_finetuning.json', output/'training_plan.json'):
            if not path.is_file() or json.loads(path.read_text()) != plan:
                raise ValueError('Resume requires identical early S2 plans, settings and implementation.')
    return plan


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    command = base.launch_command(plan, resume=args.resume)
    print(json.dumps({key: plan[key] for key in ('prior', 'source_proportions', 'checkpoint_dir',
                     'output_dir', 'early_checkpoint_diagnostic')}, indent=2), flush=True)
    if args.dry_run:
        print('Validated early S2 command; no training started.\n' + shlex.join(command))
        return
    directory, output = Path(plan['checkpoint_dir']), Path(plan['output_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        base.train_final.write_json(directory/'single_rule_finetuning.json', plan, exclusive=True)
        base.train_final.write_json(output/'training_plan.json', plan, exclusive=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(base.ROOT/'src'), str(base.ROOT), env.get('PYTHONPATH', '')])
    with (output/'train.log').open('a' if args.resume else 'w') as log:
        process = subprocess.Popen(command, cwd=base.ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise SystemExit(code)
    endpoint = base.finetune.terminal_checkpoint(SimpleNamespace(max_steps=STOP), directory)
    base.train_final.write_json(output/'diagnostic_endpoint.json', endpoint)
    print('Early S2 diagnostic finished at step 100. No evaluation or final-model selection launched.')


if __name__ == '__main__':
    main()
