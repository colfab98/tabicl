"""Reproduce saved exploratory CV predictions and compare retained sample groups."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import eval_corrosion_datasets as ev
from scripts.epit_pipeline.diagnose_cv_samples import digest, verify_metrics


def run():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['epit', 'soccol'], required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--baseline-step', type=int)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--baseline', default='generic_baseline')
    p.add_argument('--name', required=True)
    args = p.parse_args()
    args.metric_atol = args.metric_rtol = 1e-5
    torch.set_num_threads(8)
    if args.dataset == 'epit':
        from scripts.epit_pipeline import evaluate_checkpoint_comparison as comp, exploratory_cv as cv
        task_id, flag, key = comp.EPIT_TASK_ID, '--epit-expanded-features', 'epit_reviewed_split'
        recipe = Path('corrosion_datasets/analysis/epit_pipeline/selected_pairs_all760_generic10k')
    else:
        from scripts.soccol_pipeline import evaluate_checkpoint_comparison as comp, exploratory_cv as cv
        task_id, flag, key = comp.SOCCOL_TASK_ID, '--soccol-reviewed-data', 'soccol_exploratory_cv_split'
        recipe = Path('corrosion_datasets/analysis/soccol_pipeline/selected_pairs_allaccepted_generic10k')
    root = Path(__file__).resolve().parent / args.name
    root.mkdir(exist_ok=True)
    top = json.loads((args.results / 'results.json').read_text())
    assert top['exploratory_cv'] and top['cv_folds'] == [1, 2, 3, 4, 5]
    frames, deltas, hashes = [], [], {}
    for fold in range(1, 6):
        source = args.results / f'cv_fold_{fold}/results.json'
        saved = json.loads(source.read_text())
        membership = cv.fold_membership(comp.DEFAULT_SPLIT_MANIFEST, fold)
        assert saved[key] == membership == top['fold_membership'][str(fold)]
        assert not saved['fixed_schema_models'] and not saved['missing_aware_models']
        assert not saved['pitting_magpie_models'] and not saved['regression_uncertainty']
        assert saved['regression_output'] == 'median' and saved['target_binning'] == 'continuous'
        config = ev.parse_args(['--task', task_id, flag, '--target-mode', 'primary',
            '--target-binning', 'continuous', '--max-samples-per-task', '0',
            '--epit-split-manifest', str(comp.DEFAULT_SPLIT_MANIFEST),
            '--epit-exploratory-cv-fold', str(fold), '--native-tabicl-baselines',
            '--n-estimators', str(saved['n_estimators']), '--random-state', str(saved['random_state']),
            '--device', 'cpu', '--regression-output', 'median', '--no-regression-uncertainty'])
        tasks = ev.make_tasks(config)
        cv.apply_fold(tasks, config, ev)
        task = next(t for t in tasks if t.task_id == task_id)
        for label in [args.baseline, *args.models]:
            model_step = (args.baseline_step or args.step) if label == args.baseline else args.step
            refs = [r for r in saved['rows'] if r['model'] == label and r['checkpoint_step'] == model_step]
            assert len(refs) == 1, (fold, label, args.step)
            ref = refs[0]
            assert len(task.X.columns) == ref['n_features']
            assert len(task.fixed_split.train_index) == ref['n_train']
            assert len(task.fixed_split.test_index) == ref['n_test']
            checkpoint = Path(ref['model_source'])
            if checkpoint not in hashes:
                hashes[checkpoint] = digest(checkpoint)
            csv_path = root / f'fold_{fold}_{label}.csv'
            meta_path = csv_path.with_suffix('.json')
            identity = {'checkpoint_sha256': hashes[checkpoint], 'source_sha256': digest(source),
                        'split_sha256': digest(comp.DEFAULT_SPLIT_MANIFEST), 'step': model_step}
            if csv_path.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text())
                assert meta['identity'] == identity and meta['csv_sha256'] == digest(csv_path)
                frame, metrics = pd.read_csv(csv_path), meta['metrics']
                print(f'Reuse {args.name} fold {fold} {label}', flush=True)
            else:
                print(f'Extract {args.name} fold {fold} {label}', flush=True)
                captured = []
                shuffle, norms = ev.tabicl_inference_settings(config, label)
                metrics = ev.evaluate_estimator(model_label=label, model_kind=ref['model_kind'],
                    estimator_factory=lambda: ev.make_tabicl_regressor(model_path=str(checkpoint),
                        checkpoint_version=saved['pretrained_checkpoint_version'], device='cpu',
                        n_estimators=config.n_estimators, random_state=config.random_state,
                        allow_auto_download=False, feat_shuffle_method=shuffle, norm_methods=norms),
                    task=task, task_split=task.fixed_split, test_size=config.test_size,
                    random_state=config.random_state, regression_output='median', regression_uncertainty=False,
                    expected_preserve_missing_values=False, native_tabicl_baseline=True,
                    prediction_sink=captured.append)
                assert len(captured) == 1
                frame = captured[0]
                verify_metrics(metrics, ref, args)
                frame.to_csv(csv_path, index=False)
                meta_path.write_text(json.dumps({'identity': identity, 'csv_sha256': digest(csv_path),
                    'metrics': {k: float(metrics[k]) for k in ('test_mae', 'test_rmse', 'test_spearman', 'test_pearson', 'test_r2')}}, indent=2))
            verify_metrics(metrics, ref, args)
            assert frame.task_row_index.tolist() == membership['query_task_row_indices']
            for metric in ('test_mae', 'test_rmse', 'test_spearman', 'test_pearson', 'test_r2'):
                deltas.append({'fold': fold, 'model': label, 'metric': metric, 'delta': metrics[metric] - ref[metric]})
            frame['model'], frame['cv_fold'] = label, fold
            frames.append(frame)
            gc.collect()
    predictions = pd.concat(frames, ignore_index=True)
    assert not predictions.duplicated(['model', 'task_row_index']).any()
    counts = pd.read_csv(recipe / 'retained_pairs.csv').groupby('task_row_index').size()
    eligible = set(pd.read_csv(recipe / 'all_pairs.csv').task_row_index)
    baseline = predictions[predictions.model == args.baseline].drop(columns='model')
    summaries, samples = [], []
    for label in args.models:
        model = predictions[predictions.model == label].drop(columns='model')
        pair = baseline.merge(model, on=['task_row_index', 'cv_fold'], suffixes=('_generic', '_informed'), validate='one_to_one')
        assert len(pair) == len(baseline) == len(model)
        assert np.allclose(pair.observed_mV_generic, pair.observed_mV_informed)
        pair['retained_formulas'] = pair.task_row_index.map(counts).fillna(0).astype(int)
        pair['group'] = np.where(pair.retained_formulas > 0, 'retained', 'not_retained')
        pair['recipe_eligible'] = pair.task_row_index.isin(eligible)
        pair['generic_error_mV'] = abs(pair.prediction_mV_generic - pair.observed_mV_generic)
        pair['informed_error_mV'] = abs(pair.prediction_mV_informed - pair.observed_mV_generic)
        pair['delta_error_mV'] = pair.informed_error_mV - pair.generic_error_mV
        pair['model'] = label
        samples.append(pair)
        for fold, view in [('all', pair), *list(pair.groupby('cv_fold'))]:
            for group, part in [('all', view), *list(view.groupby('group'))]:
                summaries.append({'model': label, 'checkpoint_step': args.step, 'baseline_step': args.baseline_step or args.step, 'fold': fold,
                    'group': group, 'n': len(part), 'recipe_ineligible_n': int((~part.recipe_eligible).sum()),
                    'generic_mae_mV': part.generic_error_mV.mean(),
                    'informed_mae_mV': part.informed_error_mV.mean(), 'delta_mae_mV': part.delta_error_mV.mean(),
                    'improved_pct': 100 * (part.delta_error_mV < -1).mean(),
                    'worsened_pct': 100 * (part.delta_error_mV > 1).mean(),
                    'tied_pct': 100 * (part.delta_error_mV.abs() <= 1).mean()})
    summary = pd.DataFrame(summaries)
    summary.to_csv(root / 'retention_summary.csv', index=False)
    pd.concat(samples).to_csv(root / 'samples.csv', index=False)
    pd.DataFrame(deltas).to_csv(root / 'metric_reproduction_deltas.csv', index=False)
    print(summary[summary.fold == 'all'].to_string(index=False), flush=True)


if __name__ == '__main__':
    run()
