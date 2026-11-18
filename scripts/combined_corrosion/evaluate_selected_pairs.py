"""Evaluate the combined model on one dataset at a time, with exploratory CV5.

MPEA is the default; EPIT and Soccol use their existing evaluators unchanged.
The mean across folds is exploratory, not the frozen final-test score.
"""
import argparse
from pathlib import Path
import subprocess
import sys
from scripts.combined_corrosion.train_selected_pairs_single_rule import ROOT, DEFAULT_NAME


def commands(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Additional options are passed to the dataset evaluator, e.g. --checkpoint all --checkpoint-scope common --max-checkpoint-step 3000 --device cpu.')
    parser.add_argument('--dataset', choices=['mpea', 'epit', 'soccol'], default='mpea')
    parser.add_argument('--candidate-run', type=Path, default=ROOT/'checkpoints/combined_selected_pairs'/DEFAULT_NAME)
    parser.add_argument('--candidate-label', default='combined_selected_pairs')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--exploratory-cv-folds', type=int, choices=[5], default=5)
    args, remaining = parser.parse_known_args(argv)
    forbidden = {'--ratio-runs', '--final-model-manifest', '--legacy-data', '--legacy-model-inputs',
                 '--trial-run', '--context-folds'}
    if any(item.split('=', 1)[0] in forbidden for item in remaining):
        parser.error('Use the native candidate checkpoint and exploratory CV; alternative input/context modes are not supported.')
    return [[sys.executable, '-m', f'scripts.{args.dataset}_pipeline.evaluate_checkpoint_comparison',
        '--candidate-run', str(args.candidate_run), '--candidate-label', args.candidate_label,
        '--output-dir', str(args.output_dir / args.dataset), '--exploratory-cv-folds', '5', *remaining]]


def main(argv=None):
    for command in commands(argv):
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
