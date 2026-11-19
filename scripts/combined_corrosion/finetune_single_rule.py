"""Generic 10k -> single-rule informed S2, sharing the established fine-tuning settings."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

from scripts.epit_pipeline import finetune, train_final
from scripts.epit_pipeline.artifact_hashes import sha256_file
from scripts.epit_pipeline import train_selected_pairs_single_rule as epit
from scripts.soccol_pipeline import train_selected_pairs_single_rule as soccol
from scripts.combined_corrosion import train_selected_pairs_single_rule as combined
from tabicl.train.run_combined_selected_pairs import SOCCOL_OVERRIDE_KEYS
from tabicl.train.run_single_rule_finetuning import build_parser
from tabicl.prior.original_mixture import stream_policy

ROOT = train_final.REPO_ROOT
TRAINER = ROOT/'src/tabicl/train/run_single_rule_finetuning.py'
SCHEMA = 'single_rule_finetuning_s2_v1'
NAMES = {p: f'generic10k_{p}_single_rule100_s2' for p in ('epit', 'soccol', 'combined')}
NAMES['generic'] = 'generic10k_generic_single_rule_control_s2'


def parse_args(argv=None, *, default_prior='combined'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', choices=list(NAMES), default=default_prior)
    parser.add_argument('--run-name')
    parser.add_argument('--finetune-from', type=Path, default=finetune.DEFAULT_PARENT)
    parser.add_argument('--epit-recipe-dir', type=Path, default=combined.DEFAULT_EPIT)
    parser.add_argument('--soccol-recipe-dir', type=Path, default=combined.DEFAULT_SOCCOL)
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--nproc-per-node', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    args.run_name = args.run_name or NAMES[args.prior]
    if Path(args.run_name).name != args.run_name or args.run_name in ('', '.', '..'):
        parser.error('--run-name must be a single directory name.')
    if args.nproc_per_node < 1 or args.workers < 0 or 512 % (args.nproc_per_node*4):
        parser.error('Processes must divide the global 512-table batch into whole groups of four; workers must be nonnegative.')
    if args.device == 'cpu' and args.nproc_per_node != 1:
        parser.error('CPU training uses one process.')
    return args


def training_config(command):
    return build_parser().parse_args(command[command.index(str(TRAINER))+1:])


def build_plan(args):
    family = args.prior if args.prior != 'generic' else 'combined'
    directory = ((args.checkpoint_root or ROOT/f'checkpoints/{family}_finetuning')/args.run_name).resolve()
    output = (args.output_dir or ROOT/f'corrosion_datasets/analysis/{family}_pipeline/single_rule_finetuning'/args.run_name).resolve()
    if family == 'combined' and args.output_dir is None:
        output = ROOT/'corrosion_datasets/analysis/combined_corrosion/single_rule_finetuning'/args.run_name
    if not args.resume:
        for path in (directory, output):
            if path.exists() and (not path.is_dir() or any(path.iterdir())):
                raise FileExistsError(f'Run already exists: {path}. Use a new --run-name or --resume.')
    # Read-only planning of the current standalone priors; no fitting or evaluation.
    plans, configs = {}, {}
    needed = ['epit', 'soccol'] if args.prior == 'combined' else ['soccol' if args.prior == 'soccol' else 'epit']
    for name in needed:
        a = SimpleNamespace(recipe_dir=getattr(args, f'{name}_recipe_dir'), run_name=args.run_name,
            device=args.device, nproc_per_node=args.nproc_per_node, workers=args.workers,
            max_steps=2000, scheduler_total_steps=2000)
        plan = {'epit': epit, 'soccol': soccol}[name].build_plan(a)
        command = list(plan['command'])
        command[command.index(str(epit.TRAINER))] = str(ROOT/'src/tabicl/train/run.py')
        configs[name] = vars(finetune.training_config(command))
        plans[name] = plan
    name = needed[0]
    command = list(plans[name]['command'])
    command[command.index(str(epit.TRAINER))] = str(ROOT/'src/tabicl/train/run.py')
    overrides = None
    if args.prior == 'combined':
        overrides = {k: deepcopy(v) for k,v in configs['soccol'].items()
                     if v != configs['epit'][k] and k != 'checkpoint_dir'}
        if set(overrides)-SOCCOL_OVERRIDE_KEYS:
            raise ValueError('Standalone settings differ outside dataset-specific fields.')
        overrides.update(epit_selected_pair_dir=None, soccol_selected_pair_dir=str(args.soccol_recipe_dir.resolve()),
                         informed_prior_ratio=1.)
    for key,value in {'checkpoint_dir': str(directory), 'wandb_name': args.run_name,
        'informed_prior_ratio': 0 if args.prior == 'generic' else 1,
        'np_seed': 43, 'torch_seed': 43, 'python_seed': 43, 'prior_stream_seed': 43,
        'save_temp_every': 100, 'save_perm_every': 100}.items():
        train_final.replace_option(command, f'--{key}', value)
    if args.prior == 'generic':
        index = command.index('--epit_selected_pair_dir')
        del command[index:index+2]
    template = finetune.parse_args(['--finetune-from', str(args.finetune_from), '--override-informed-prior-ratio', '0'])
    identity = finetune.parent_identity(template)
    template.finetuning_identity = identity
    finetune.configure_initialization(template, command)
    command[command.index(str(ROOT/'src/tabicl/train/run.py'))] = str(TRAINER)
    command.extend(['--single_rule_s2_prior', args.prior])
    if overrides is not None:
        command.extend(['--combined_soccol_overrides', json.dumps(overrides, sort_keys=True)])
    proportions = {'generic': 0., 'epit': 0., 'soccol': 0.}
    if args.prior == 'combined':
        proportions.update(epit=.5, soccol=.5)
    else:
        proportions[args.prior] = 1.
    plan = dict(schema_version=SCHEMA, prior=args.prior, command=command,
        checkpoint_dir=str(directory), output_dir=str(output), source_proportions=proportions,
        finetuning=identity, nproc_per_node=args.nproc_per_node,
        generic_stream_identity=stream_policy(43),
        sources={} if args.prior == 'generic' else {
            k:{'recipe_identity': p['recipe_identity'], 'table_sampling_policy': p['table_sampling_policy']}
            for k,p in plans.items()},
        automatic_evaluation=False, all_data_recipe=args.prior != 'generic',
        implementation_sha256={str(p.relative_to(ROOT)):sha256_file(p) for p in [Path(__file__), TRAINER,
            ROOT/'src/tabicl/train/run_combined_selected_pairs.py', ROOT/'src/tabicl/prior/combined_selected_pairs.py']})
    validate_plan(plan)
    if args.resume:
        saved_path = directory/'single_rule_finetuning.json'
        if not saved_path.is_file() or not list(directory.glob('step-*.ckpt')):
            raise FileNotFoundError('Resume requires the saved fine-tuning plan and a checkpoint.')
        if json.loads(saved_path.read_text()) != plan:
            raise ValueError('Resume requires identical parent, recipes, settings, world size, paths and implementation.')
        if json.loads((output/'training_plan.json').read_text()) != plan:
            raise ValueError('Checkpoint and output training plans differ.')
    return plan


def validate_plan(plan):
    if plan['schema_version'] != SCHEMA:
        raise ValueError('Unsupported single-rule S2 plan.')
    config = training_config(plan['command'])
    expected = {**finetune.S2_SETTINGS, **finetune.RETAINED_TABLE_SETTINGS,
        'np_seed':43, 'torch_seed':43, 'python_seed':43, 'prior_stream_seed':43,
        'save_temp_every':100, 'save_perm_every':100, 'only_load_model':True, 'save_initial_checkpoint':True,
        'checkpoint_path':plan['finetuning']['parent_checkpoint'], 'checkpoint_dir':plan['checkpoint_dir'],
        'single_rule_s2_prior':plan['prior'], 'informed_prior_ratio':0. if plan['prior']=='generic' else 1.,
        'original_prior_mix':True, 'paired_prior_streams':True,
        'informed_target_mix_weight':1., 'informed_target_mix_policy':'fixed',
        'pitting_composition_perturb_strength':.3, 'pitting_coefficient_variation':0.,
        'col_missing_value_embedding':False, 'epit_selected_pair_informed_only':False}
    for key,value in expected.items():
        if getattr(config,key) != value:
            raise ValueError(f'Single-rule S2 requires {key}={value!r}.')
    prior = plan['prior']
    proportions = {'generic': 0., 'epit': 0., 'soccol': 0.}
    proportions.update({'epit': .5, 'soccol': .5} if prior == 'combined' else {prior: 1.})
    if plan['source_proportions'] != proportions:
        raise ValueError('S2 source proportions do not match the selected experiment.')
    if prior == 'combined':
        overrides = config.combined_soccol_overrides
        if (not isinstance(overrides, dict) or set(overrides)-SOCCOL_OVERRIDE_KEYS
                or overrides.get('epit_selected_pair_dir', 'missing') is not None
                or not overrides.get('soccol_selected_pair_dir')
                or overrides.get('informed_prior_ratio') != 1.):
            raise ValueError('Combined S2 needs separate single-rule recipes and 50/50 allocation.')
    elif config.combined_soccol_overrides is not None:
        raise ValueError('Only combined S2 accepts Soccol overrides.')
    recipes = (bool(config.epit_selected_pair_dir), bool(config.soccol_selected_pair_dir))
    if recipes != {'epit':(True, False), 'soccol':(False, True),
                   'combined':(True, False), 'generic':(False, False)}[prior]:
        raise ValueError('Selected S2 prior and recipe paths differ.')
    return config


def launch_command(plan, *, resume=False):
    command = list(plan['command'])
    if resume:
        i = command.index('--checkpoint_path'); del command[i:i+2]
        train_final.replace_option(command, '--only_load_model', 'false')
    return command


def main(argv=None, *, default_prior='combined'):
    args = parse_args(argv, default_prior=default_prior)
    plan = build_plan(args)
    command = launch_command(plan, resume=args.resume)
    print(json.dumps({k:plan[k] for k in ('prior','source_proportions','checkpoint_dir','output_dir')}, indent=2), flush=True)
    if args.dry_run:
        print('Validated S2 command; no training started.\n'+shlex.join(command))
        return
    directory, output = Path(plan['checkpoint_dir']), Path(plan['output_dir'])
    directory.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        train_final.write_json(directory/'single_rule_finetuning.json', plan, exclusive=True)
        train_final.write_json(output/'training_plan.json', plan, exclusive=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'),str(ROOT),env.get('PYTHONPATH','')])
    with (output/'train.log').open('a' if args.resume else 'w') as log:
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            code=process.wait()
        except BaseException:
            process.terminate();process.wait();raise
    if code:
        raise SystemExit(code)
    endpoint=finetune.terminal_checkpoint(SimpleNamespace(max_steps=2000),directory)
    train_final.write_json(output/'terminal_checkpoint.json',endpoint)
    print('Fine-tuning finished. No evaluation was launched.')


if __name__ == '__main__':
    main()
