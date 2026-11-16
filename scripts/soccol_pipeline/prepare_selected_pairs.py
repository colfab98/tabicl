"""Build Soccol's exploratory selected-pair recipe with EPIT's comparison policy.

Fit all composition-accepted rows, including former final rows. Generic query
predictions use the existing five-fold CV. No training/evaluation split changes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import eval_corrosion_datasets as ev
from scripts.epit_pipeline.artifact_hashes import sha256_file
from scripts.epit_pipeline.prepare_selected_pairs import compare_pairs
from scripts.soccol_pipeline import exploratory_cv as cv
from scripts.soccol_pipeline import evaluate_checkpoint_comparison as comparison
from scripts.soccol_pipeline.selected_pair_formulas import (
    REPO_ROOT as ROOT, EXTENSION_REPORT, load_inputs, extension_spec, fit_all, source_identity,
)

DEFAULT_OUTPUT = ROOT / 'corrosion_datasets/analysis/soccol_pipeline/selected_pairs_allaccepted_generic10k'


def run(args):
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    dataset, accepted, input_identity = load_inputs()
    spec, penalty = extension_spec()
    memberships = {fold: cv.fold_membership(comparison.DEFAULT_SPLIT_MANIFEST, fold) for fold in cv.FOLDS}
    query = [i for member in memberships.values() for i in member['query_task_row_indices']]
    eligible = set(accepted.task_row_index)
    if len(query) != len(set(query)) or not eligible <= set(query):
        raise ValueError('CV must predict every eligible parent exactly once.')
    for member in memberships.values():
        if set(member['query_task_row_indices']) & set(member['context_task_row_indices']):
            raise ValueError('Query/context overlap.')
    if args.dry_run:
        print(f'Validated {len(accepted)} eligible parents, 10 formulas and five disjoint query folds ({len(query)} queries).')
        print(f'Extension: {spec.name}; fixed penalty: {penalty}')
        print('Generic checkpoint:', checkpoint)
        print('Output:', args.output_dir.resolve())
        print('No fitting or inference performed.')
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = {**source_identity(), **{str(p.relative_to(ROOT)): sha256_file(p) for p in
        (Path(__file__), Path(ev.__file__), Path(cv.__file__), ROOT / 'scripts/epit_pipeline/prepare_selected_pairs.py')}}
    identity = {'version': 1, 'all_eligible_rows_used_for_fitting': True, 'independent_evaluation': False,
        'data': input_identity, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(checkpoint),
        'source_sha256': sources, 'extension': spec.name, 'extension_penalty': penalty,
        'extension_report_sha256': sha256_file(EXTENSION_REPORT),
        'n_estimators': 8, 'random_state': 42, 'device': 'cpu', 'regression_output': 'median',
        'memberships': {str(k): v for k, v in memberships.items()}}
    manifest = args.output_dir / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Existing output has different settings; choose a new --output-dir.')
    if not manifest.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Nonempty output without a manifest; choose a new --output-dir.')
    manifest.write_text(json.dumps(identity, indent=2) + '\n')
    artifacts, formula = fit_all(accepted)
    (args.output_dir / 'fitted_formulas.json').write_text(json.dumps(artifacts, indent=2, allow_nan=False) + '\n')
    formula.to_csv(args.output_dir / 'formula_predictions.csv', index=False)
    excluded = dataset.rows.loc[~dataset.rows.composition_status.eq('accepted')].copy()
    excluded.insert(0, 'task_row_index', excluded.index)
    excluded.to_csv(args.output_dir / 'excluded_rows.csv', index=False)
    config = ev.parse_args(['--task', comparison.SOCCOL_TASK_ID, '--target-mode', 'primary', '--target-binning', 'continuous',
        '--soccol-reviewed-data', '--epit-split-manifest', str(comparison.DEFAULT_SPLIT_MANIFEST),
        '--epit-exploratory-cv-fold', '1', '--native-tabicl-baselines', '--n-estimators', '8',
        '--random-state', '42', '--device', 'cpu', '--regression-output', 'median',
        '--no-regression-uncertainty', '--max-samples-per-task', '0'])
    task, _, _, _ = ev._load_epit_pipeline_indices(ev.make_tasks(config), manifest_path=comparison.DEFAULT_SPLIT_MANIFEST)
    shuffle, norms = ev.tabicl_inference_settings(config, 'generic_baseline')
    predictions = []
    for fold, member in memberships.items():
        folder = args.output_dir / f'cv_fold_{fold}'
        folder.mkdir(exist_ok=True)
        path, meta_path = folder / 'generic_predictions.csv', folder / 'metrics.json'
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if sha256_file(path) != meta['prediction_sha256']:
                raise ValueError('Cached predictions changed.')
            frame = pd.read_csv(path)
            print(f'Reusing generic fold {fold}', flush=True)
        else:
            print(f'Predicting generic: fold {fold}/5', flush=True)
            context = np.asarray(member['context_task_row_indices'])
            split = ev.EvalSplit(train_index=context, test_index=np.asarray(member['query_task_row_indices']),
                preprocessing_index=context.copy(), split_strategy=f'soccol_pipeline_exploratory_cv_fold_{fold}')
            captured = []
            metrics = ev.evaluate_estimator(model_label='generic_baseline', model_kind='local_tabicl_regressor',
                estimator_factory=lambda: ev.make_tabicl_regressor(model_path=str(checkpoint),
                    checkpoint_version='tabicl-regressor-v2-20260212.ckpt', device='cpu', n_estimators=8,
                    random_state=42, allow_auto_download=False, feat_shuffle_method=shuffle, norm_methods=norms),
                task=task, task_split=split, test_size=config.test_size, random_state=42, regression_output='median',
                regression_uncertainty=False, expected_preserve_missing_values=False, native_tabicl_baseline=True,
                prediction_sink=captured.append)
            if len(captured) != 1:
                raise ValueError('Expected exactly one prediction batch.')
            frame = captured[0]
            frame['cv_fold'] = fold
            frame.to_csv(path, index=False)
            meta_path.write_text(json.dumps({'prediction_sha256': sha256_file(path),
                'metrics': {k: float(v) for k, v in metrics.items() if k in ('test_mae', 'test_rmse', 'test_spearman', 'test_r2')}}, indent=2) + '\n')
        if frame.task_row_index.tolist() != member['query_task_row_indices'] or not frame.cv_fold.eq(fold).all():
            raise ValueError('Cached prediction fold mismatch.')
        predictions.append(frame)
    generic = pd.concat(predictions, ignore_index=True)
    generic.to_csv(args.output_dir / 'generic_cv_predictions.csv', index=False)
    pairs = compare_pairs(formula, generic[generic.task_row_index.isin(eligible)]).merge(
        accepted[['task_row_index', 'row_id', 'raw_workbook_row']], on='task_row_index', validate='many_to_one')
    pairs.to_csv(args.output_dir / 'all_pairs.csv', index=False)
    pairs[pairs.retained].to_csv(args.output_dir / 'retained_pairs.csv', index=False)
    summary = {'samples': len(accepted), 'generic_cv_samples': len(generic), 'excluded_compositions': len(excluded),
        'formulas': pairs.formula.nunique(), 'tested_pairs': len(pairs), 'retained_pairs': int(pairs.retained.sum()),
        'informed_proportion': float(pairs.retained.mean()),
        'selection': 'strictly lower absolute error than generic; ties discarded',
        'scope': 'all-accepted-data recipe construction, not independent formula evaluation'}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('Saved:', args.output_dir.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/tabicl_s1_regression_baseline/step-10000.ckpt')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dry-run', action='store_true')
    run(parser.parse_args())
