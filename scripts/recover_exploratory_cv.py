"""Rebuild exploratory CV summaries from five saved folds without model inference."""
import argparse
import json
from pathlib import Path

from scripts import eval_corrosion_datasets as ev
from scripts.epit_pipeline.exploratory_cv import run_cv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='Existing directory containing cv_fold_1 through cv_fold_5.')
    parser.add_argument('--split-manifest', type=Path)
    parser.add_argument('--checkpoint-step-interval', type=int, help='Strict interval; defaults to the saved interval. Use 0 for all complete checkpoints.')
    options = parser.parse_args()
    directory = options.output_dir.expanduser().resolve()
    first = json.loads((directory / 'cv_fold_1/results.json').read_text())
    if 'soccol_exploratory_cv_split' in first:
        from scripts.soccol_pipeline.exploratory_cv import fold_membership
        from scripts.soccol_pipeline.data import SPLIT_MANIFEST
        settings = dict(membership_fn=fold_membership, membership_key='soccol_exploratory_cv_split',
                        dataset_label='Soccol', split_strategy='soccol_pipeline_exploratory_cv')
    elif 'epit_reviewed_split' in first:
        from scripts.epit_pipeline.run_optuna import DEFAULT_SPLIT_MANIFEST as SPLIT_MANIFEST
        settings = {}
    else:
        raise ValueError('Saved results do not identify a supported CV dataset.')
    # Match the saved scoring/plot configuration; no models or checkpoints are loaded.
    argv = ['--checkpoint', first.get('checkpoint', 'all'),
            '--target-binning', first.get('target_binning', 'continuous'),
            '--regression-output', first.get('regression_output', 'median'),
            '--epit-split-manifest', str(options.split_manifest or SPLIT_MANIFEST)]
    for flag, name in (('--output-json', 'results.json'), ('--output-csv', 'rows.csv'),
                       ('--output-wide-csv', 'wide.csv'), ('--output-summary-csv', 'summary.csv'),
                       ('--output-plot-dir', 'plots')):
        argv.extend([flag, str(directory / name)])
    interval = options.checkpoint_step_interval
    if interval is None:
        interval = first.get('checkpoint_step_interval', 500)
    if interval < 0:
        parser.error('--checkpoint-step-interval must be nonnegative.')
    argv.extend(['--checkpoint-step-interval', str(interval)])
    args = ev.parse_args(argv)
    run_cv(args, ev, recover=True, **settings)


if __name__ == '__main__':
    main()
