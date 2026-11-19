"""Compare native checkpoints on reviewed Rebar alone, using the shared five-fold CV convention."""
import argparse
from copy import copy
import json
from pathlib import Path
import shlex
import subprocess

from scripts.combined_corrosion.evaluate_final import add_evaluation_arguments, comparison_command
from scripts.epit_pipeline.evaluate_checkpoint_comparison import select_available_models, REQUIRED_PLOTS
from scripts.eval_corrosion_datasets import slugify
from scripts.rebar_pipeline.reviewed_evaluation_data import ROOT, DEFAULT_SPLIT, TASK_ID, CATEGORICAL_COLUMNS, MEMBERSHIP_KEY, fold_membership, frozen_split


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser, default_output_dir=None, include_combined_data=False, default_device='cpu')
    parser.add_argument('--candidate-run', type=Path, required=True)
    parser.add_argument('--candidate-label', default='combined_selected_pairs')
    parser.add_argument('--split-manifest', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--exploratory-cv-folds', type=int, choices=[5], default=5)
    parser.add_argument('--include-latest-checkpoint', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.output_dir is None:
        parser.error('--output-dir is required.')
    if args.checkpoint_scope == 'common' and args.checkpoint != 'all':
        parser.error('--checkpoint-scope common requires --checkpoint all.')
    if args.n_estimators <= 0 or args.checkpoint_step_interval < 0 or args.catboost_thread_count == 0:
        parser.error('Estimators must be positive, checkpoint interval nonnegative, and CatBoost threads nonzero.')
    if args.max_checkpoint_step is not None and (args.checkpoint != 'all'
            or args.max_checkpoint_step < max(0, args.min_checkpoint_step)):
        parser.error('--max-checkpoint-step requires --checkpoint all and a valid step range.')
    args.candidate_label = slugify(args.candidate_label)
    if args.candidate_label in {'generic_baseline', 'pretrained_tabicl_v2', 'catboost'}:
        parser.error('Candidate label must differ from the three baseline labels.')
    return args


def model_runs(args):
    return getattr(args, '_selected_model_runs', [
        (args.candidate_label, args.candidate_run), ('generic_baseline', args.generic_run)])


def evaluation_command(args):
    command = comparison_command(args, model_runs(args), missing_aware_models=(), task_id=TASK_ID,
        task_arguments=['--rebar-split-manifest', str(args.split_manifest.expanduser().resolve()),
                        '--exploratory-cv-folds', '5'], categorical_columns=CATEGORICAL_COLUMNS)
    command[1] = str(ROOT / 'scripts/rebar_pipeline/evaluate_native.py')
    if not args.include_latest_checkpoint:
        command[command.index('--include-latest-common-checkpoint')] = '--no-include-latest-checkpoint'
    return command


def validate_results(args, payload):
    if not payload.get('exploratory_cv') or payload.get('cv_folds') != [1, 2, 3, 4, 5] or payload.get('errors'):
        raise RuntimeError('Rebar comparison requires all five complete exploratory CV folds.')
    if len(payload.get('cv_fold_outputs', [])) != 5 or not payload.get('rows'):
        raise RuntimeError('Rebar comparison has missing folds or results.')
    for output in payload['cv_fold_outputs']:
        fold = output['cv_fold']
        saved = json.loads(Path(output['json']).read_text())
        expected = fold_membership(args.split_manifest, fold)
        if saved.get(MEMBERSHIP_KEY) != expected or saved.get('errors'):
            raise RuntimeError(f'Rebar fold {fold} has incorrect membership or model errors.')
        if saved.get('missing_aware_models') or saved.get('fixed_schema_models'):
            raise RuntimeError('Rebar must use native preprocessing for every model.')
        for row in saved['rows']:
            if (row['task_id'] != TASK_ID or row['n_train'] != len(expected['context_task_row_indices'])
                    or row['n_test'] != len(expected['query_task_row_indices']) or row['n_features'] != len(frozen_split(args.split_manifest)['dataset']['feature_columns_in_order'])
                    or row['split_strategy'] != f'rebar_pipeline_exploratory_cv_fold_{fold}'):
                raise RuntimeError(f'Rebar fold {fold} used the wrong dataset or split.')
    if args.checkpoint == 'all':
        for filename in REQUIRED_PLOTS:
            path = args.output_dir / 'plots' / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f'Required plot missing: {path}')


def run(args):
    args = copy(args)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.dry_run:
        print(shlex.join(evaluation_command(args)))
        return args.output_dir
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        raise FileExistsError(f'Comparison output is not empty: {args.output_dir}')
    args._selected_model_runs, skipped = select_available_models(args, runs=model_runs(args))
    subprocess.run(evaluation_command(args), cwd=ROOT, check=True)
    result_path = args.output_dir / 'results.json'
    payload = json.loads(result_path.read_text())
    payload['skipped_models'] = skipped
    result_path.write_text(json.dumps(payload, indent=2) + '\n')
    validate_results(args, payload)
    return args.output_dir


def main():
    run(parse_args())


if __name__ == '__main__':
    main()
