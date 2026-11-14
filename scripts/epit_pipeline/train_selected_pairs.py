"""Launch the separate all-data selected-pair experiment, without automatic evaluation."""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from scripts.epit_pipeline import train_final
from tabicl.prior.epit_selected_pairs import SelectedPairSampler
from tabicl.train.train_config import build_parser

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_NAME='epit_selected_pairs_all760_generic10k_v1'


def build_plan(args):
    sampler=SelectedPairSampler(args.recipe_dir)
    checkpoint_dir=ROOT/'checkpoints/epit_selected_pairs'/args.run_name
    output_dir=ROOT/'corrosion_datasets/analysis/epit_pipeline/selected_pair_training'/args.run_name
    for path in (checkpoint_dir,output_dir):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f'Run already exists: {path}. Choose a new --run-name.')
    # Reuse the established architecture, optimizer and batch settings, but call
    # only its planning path. Never enter its checkpoint selection/evaluation.
    template=train_final.parse_args([
        '--manual-hyperparameters','--override-mlp-prob','0.70',
        '--override-informed-prior-ratio','1','--override-informed-target-mix-weight','1',
        '--override-pitting-composition-perturb-strength','0.30',
        '--informed-target-mix-policy','fixed','--final-run-name',args.run_name,
        '--checkpoint-root',str(checkpoint_dir.parent),'--output-dir',str(output_dir),
        '--device',args.device,'--max-steps',str(args.max_steps),
        '--scheduler-total-steps',str(args.scheduler_total_steps),'--nproc-per-node',str(args.nproc_per_node),
        '--dataloader-num-workers',str(args.workers),'--dry-run'])
    captured=io.StringIO()
    with redirect_stdout(captured):
        train_final.run(template)
    command=json.loads(captured.getvalue())['training']['train_command']
    # Use the interpreter that launched this script, including its environment.
    if args.nproc_per_node==1:
        command=[sys.executable,*command[command.index(str(ROOT/'src/tabicl/train/run.py')):]]
    elif command[0]=='torchrun':
        command=[sys.executable,'-m','torch.distributed.run',*command[1:]]
    else:
        command[0]=sys.executable
    train_final.replace_option(command,'--pitting_coefficient_variation','0')
    train_final.replace_option(command,'--informed_prior_ratio',repr(sampler.ratio))
    train_final.replace_option(command,'--wandb_name',args.run_name)
    command.extend(['--epit_selected_pair_dir',str(sampler.path),
                    '--epit_target_version','v4','--informed_target_mix_policy','fixed'])
    start=command.index(str(ROOT/'src/tabicl/train/run.py'))+1
    parsed=build_parser().parse_args(command[start:])
    if (parsed.pitting_coefficient_variation!=0 or parsed.informed_target_mix_weight!=1
            or parsed.pitting_composition_perturb_strength!=.30
            or parsed.checkpoint_path is not None):
        raise ValueError('Selected-pair command changed the agreed recipe or fresh initialization.')
    return {'command':command,'checkpoint_dir':str(checkpoint_dir),'output_dir':str(output_dir),
            'recipe_identity':sampler.identity,'informed_proportion':sampler.ratio,
            'initialization':'from scratch','automatic_evaluation':False,
            'max_steps':args.max_steps,'scheduler_total_steps':args.scheduler_total_steps,
            'device':args.device,'all_data_recipe':True}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe-dir',type=Path,default=ROOT/'corrosion_datasets/analysis/epit_pipeline/selected_pairs_all760_generic10k')
    parser.add_argument('--run-name',default=DEFAULT_NAME)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--nproc-per-node',type=int,default=1)
    parser.add_argument('--max-steps',type=int,default=10000)
    parser.add_argument('--scheduler-total-steps',type=int,default=10000,
                        help='LR schedule horizon, independent of --max-steps.')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    if not args.run_name or Path(args.run_name).name!=args.run_name or args.run_name in ('.','..'):
        parser.error('--run-name must be a single directory name.')
    if args.max_steps<1 or args.scheduler_total_steps<1 or args.nproc_per_node<1 or args.workers<0:
        parser.error('Steps/processes must be positive; workers must be nonnegative.')
    if args.device=='cpu' and args.nproc_per_node!=1:
        parser.error('CPU training uses one process.')
    plan=build_plan(args)
    print(f"Training from scratch: {args.max_steps} steps on {args.device}; {plan['informed_proportion']:.2%} informed.",flush=True)
    print('Checkpoints:',plan['checkpoint_dir'],flush=True)
    print('Log:',str(Path(plan['output_dir'])/'train.log'),flush=True)
    if args.dry_run:
        print('Validated command; no training started.\n'+shlex.join(plan['command']))
        return
    output=Path(plan['output_dir'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'training_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'src'),str(ROOT),env.get('PYTHONPATH','')])
    with (output/'train.log').open('w') as log:
        process=subprocess.Popen(plan['command'],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:
                print(line,end='',flush=True)
                log.write(line);log.flush()
            code=process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise SystemExit(code)
    print('Training finished. No evaluation was launched.')


if __name__=='__main__':
    main()
