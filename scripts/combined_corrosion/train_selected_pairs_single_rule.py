"""Launch the combined native-input single-rule model; never evaluate automatically."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from scripts.epit_pipeline import train_selected_pairs_single_rule as epit
from scripts.soccol_pipeline import train_selected_pairs_single_rule as soccol
from scripts.epit_pipeline.train_final import replace_option
from tabicl.train.train_config import build_parser
from tabicl.train.run_combined_selected_pairs import SOCCOL_OVERRIDE_KEYS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NAME = 'combined_selected_pairs_single_rule_v1'
DEFAULT_EPIT = ROOT/'corrosion_datasets/analysis/epit_pipeline/selected_pairs_all760_generic10k'
DEFAULT_SOCCOL = ROOT/'corrosion_datasets/analysis/soccol_pipeline/selected_pairs_allaccepted_generic10k'
TRAINER = ROOT/'src/tabicl/train/run_combined_selected_pairs.py'
CHECKPOINT_ROOT = ROOT/'checkpoints/combined_selected_pairs'
OUTPUT_ROOT = ROOT/'corrosion_datasets/analysis/combined_corrosion/selected_pair_training'


def build_plan(args):
    checkpoint = CHECKPOINT_ROOT/args.run_name
    output = OUTPUT_ROOT/args.run_name
    if not getattr(args, 'resume', False):
        for path in (checkpoint, output):
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(f'Run already exists: {path}. Choose a new --run-name or use --resume.')
    plans, configs = {}, {}
    for name, module, recipe in [('epit', epit, args.recipe_dir), ('soccol', soccol, args.soccol_recipe_dir)]:
        source_args = deepcopy(args)
        source_args.recipe_dir = recipe
        plan = module.build_plan(source_args)
        plans[name] = plan
        index = plan['command'].index(str(ROOT/'src/tabicl/train/run_selected_pairs_single_rule.py'))
        configs[name] = vars(build_parser().parse_args(plan['command'][index+1:]))
    differences = {key: value for key, value in configs['soccol'].items()
                   if value != configs['epit'][key] and key != 'checkpoint_dir'}
    unexpected = set(differences)-SOCCOL_OVERRIDE_KEYS
    if unexpected:
        raise ValueError(f'Standalone scientific settings differ outside their datasets: {sorted(unexpected)}')
    # Include explicit recipe switching even if the directory defaults change later.
    differences.update(epit_selected_pair_dir=None, soccol_selected_pair_dir=str(args.soccol_recipe_dir.resolve()))
    ratios = {name: plans[name]['informed_proportion']/2 for name in ('epit', 'soccol')}
    ratios['generic'] = 1-sum(ratios.values())
    command = list(plans['epit']['command'])
    command[command.index(str(ROOT/'src/tabicl/train/run_selected_pairs_single_rule.py'))] = str(TRAINER)
    replace_option(command, '--checkpoint_dir', str(checkpoint))
    replace_option(command, '--informed_prior_ratio', repr(1-ratios['generic']))
    command.extend(['--combined_soccol_overrides', json.dumps(differences, sort_keys=True)])
    plan = {
        'command': command, 'checkpoint_dir': str(checkpoint), 'output_dir': str(output),
        'source_proportions': ratios, 'informed_proportion': 1-ratios['generic'],
        'mixture_policy': 'equal weighting of the two standalone recipes; generic mass pooled',
        'sources': {name: {'recipe_identity': plans[name]['recipe_identity'],
                          'table_sampling_policy': plans[name]['table_sampling_policy']} for name in plans},
        'max_steps': args.max_steps, 'scheduler_total_steps': args.scheduler_total_steps,
        'initialization': 'from scratch', 'automatic_evaluation': False, 'all_data_recipe': True,
    }
    import hashlib
    plan['implementation_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__), TRAINER, ROOT/'src/tabicl/prior/combined_selected_pairs.py')}
    if getattr(args, 'resume', False):
        manifest = output/'training_plan.json'
        if not manifest.is_file() or not any(checkpoint.glob('step-*.ckpt')):
            raise FileNotFoundError('Resume requires the original training_plan.json and a saved checkpoint.')
        saved = json.loads(manifest.read_text())
        def identity(p):
            result = deepcopy(p)
            result.pop('max_steps')
            replace_option(result['command'], '--max_steps', '<stopping point>')
            return result
        if identity(saved) != identity(plan):
            raise ValueError('Resume requires identical recipes, training settings, schedule, and implementation.')
    return plan


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe-dir', '--epit-recipe-dir', type=Path, default=DEFAULT_EPIT)
    parser.add_argument('--soccol-recipe-dir', type=Path, default=DEFAULT_SOCCOL)
    parser.add_argument('--run-name', default=DEFAULT_NAME)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--nproc-per-node', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--scheduler-total-steps', type=int, default=10000,
                        help='LR horizon remains independent of the stopping step.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in ('.','..'):
        parser.error('--run-name must be a single directory name.')
    if min(args.max_steps,args.scheduler_total_steps,args.nproc_per_node)<1 or args.workers<0:
        parser.error('Steps/processes must be positive; workers must be nonnegative.')
    if args.device=='cpu' and args.nproc_per_node!=1:
        parser.error('CPU training uses one process.')
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    print(json.dumps({k:v for k,v in plan.items() if k in ('checkpoint_dir','output_dir','source_proportions','max_steps','scheduler_total_steps')}, indent=2),flush=True)
    if args.dry_run:
        print('Validated command; no training started.\n'+shlex.join(plan['command']))
        return
    output = Path(plan['output_dir'])
    output.mkdir(parents=True, exist_ok=args.resume)
    if not args.resume:
        (output/'training_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'),str(ROOT),env.get('PYTHONPATH','')])
    with (output/'train.log').open('a' if args.resume else 'w') as log:
        process = subprocess.Popen(plan['command'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            code = process.wait()
        except BaseException:
            process.terminate();process.wait();raise
    if code:raise SystemExit(code)
    print('Training finished. No evaluation was launched.')


if __name__=='__main__':
    main()
