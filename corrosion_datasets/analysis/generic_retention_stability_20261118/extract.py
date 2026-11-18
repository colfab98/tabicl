"""Exploratory recipe sensitivity, not independent final-model evaluation.
Reproduce the recipe's five query folds with fixed inference settings and formulas.
"""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts import eval_corrosion_datasets as ev
from scripts.epit_pipeline.artifact_hashes import sha256_file as digest

ROOT = Path(__file__).resolve().parent

def run():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['epit', 'soccol'], required=True)
    p.add_argument('--step', type=int, required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    if args.dataset == 'epit':
        from scripts.epit_pipeline import exploratory_cv as cv, evaluate_checkpoint_comparison as comp
        task_id, flag, recipe_name = comp.EPIT_TASK_ID, '--epit-expanded-features', 'selected_pairs_all760_generic10k'
    else:
        from scripts.soccol_pipeline import exploratory_cv as cv, evaluate_checkpoint_comparison as comp
        task_id, flag, recipe_name = comp.SOCCOL_TASK_ID, '--soccol-reviewed-data', 'selected_pairs_allaccepted_generic10k'
    recipe = Path('corrosion_datasets/analysis') / f'{args.dataset}_pipeline' / recipe_name
    manifest = json.loads((recipe / 'manifest.json').read_text())
    checkpoint = Path(f'checkpoints/tabicl_s1_regression_baseline/step-{args.step}.ckpt').resolve()
    ckpt_hash = digest(checkpoint)
    assert digest(Path(manifest['checkpoint'])) == manifest['checkpoint_sha256']
    config = ev.parse_args(['--task', task_id, '--target-mode', 'primary', '--target-binning', 'continuous',
        flag, '--epit-split-manifest', str(comp.DEFAULT_SPLIT_MANIFEST), '--epit-exploratory-cv-fold', '1',
        '--native-tabicl-baselines', '--n-estimators', str(manifest['n_estimators']),
        '--random-state', str(manifest['random_state']), '--device', 'cpu', '--regression-output', 'median',
        '--no-regression-uncertainty', '--max-samples-per-task', '0'])
    task, _, _, _ = ev._load_epit_pipeline_indices(ev.make_tasks(config), manifest_path=comp.DEFAULT_SPLIT_MANIFEST)
    shuffle, norms = ev.tabicl_inference_settings(config, 'generic_baseline')
    output = ROOT / args.dataset / f'step_{args.step}'
    output.mkdir(parents=True, exist_ok=True)
    frames, metrics_rows = [], []
    for fold in range(1, 6):
        membership = cv.fold_membership(comp.DEFAULT_SPLIT_MANIFEST, fold)
        assert membership == manifest['memberships'][str(fold)]
        assert not set(membership['context_task_row_indices']) & set(membership['query_task_row_indices'])
        path = output / f'fold_{fold}.csv'
        meta_path = path.with_suffix('.json')
        identity = {'checkpoint': str(checkpoint), 'checkpoint_sha256': ckpt_hash,
            'recipe_manifest_sha256': digest(recipe / 'manifest.json'),
            'split_sha256': digest(comp.DEFAULT_SPLIT_MANIFEST), 'extractor_sha256': digest(Path(__file__)),
            'evaluation_source_sha256': digest(Path(ev.__file__)), 'n_estimators': config.n_estimators,
            'random_state': config.random_state, 'regression_output': 'median', 'device': 'cpu',
            'feat_shuffle_method': shuffle, 'norm_methods': norms, 'membership': membership}
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            assert meta['identity'] == identity and meta['csv_sha256'] == digest(path)
            frame, metrics = pd.read_csv(path), meta['metrics']
            print(f'Reuse {args.dataset} {args.step} fold {fold}', flush=True)
        elif args.step == 10000 and fold != 1:
            old = recipe / f'cv_fold_{fold}'
            oldmeta = json.loads((old / 'metrics.json').read_text())
            assert digest(old / 'generic_predictions.csv') == oldmeta['prediction_sha256']
            assert ckpt_hash == manifest['checkpoint_sha256']
            frame = pd.read_csv(old / 'generic_predictions.csv')
            metrics = oldmeta['metrics']
            frame.to_csv(path, index=False)
            meta_path.write_text(json.dumps({'identity': identity, 'csv_sha256': digest(path),
                'metrics': metrics, 'reused_recipe_predictions': str(old)}, indent=2))
            print(f'Verified recipe cache {args.dataset} {args.step} fold {fold}', flush=True)
        else:
            print(f'Predict {args.dataset} {args.step} fold {fold}', flush=True)
            context = np.asarray(membership['context_task_row_indices'])
            split = ev.EvalSplit(train_index=context, test_index=np.asarray(membership['query_task_row_indices']),
                preprocessing_index=context.copy(), split_strategy=f'{args.dataset}_pipeline_exploratory_cv_fold_{fold}')
            captured = []
            metrics = ev.evaluate_estimator(model_label='generic_baseline', model_kind='local_tabicl_regressor',
                estimator_factory=lambda: ev.make_tabicl_regressor(model_path=str(checkpoint),
                    checkpoint_version='tabicl-regressor-v2-20260212.ckpt', device='cpu',
                    n_estimators=config.n_estimators, random_state=config.random_state, allow_auto_download=False,
                    feat_shuffle_method=shuffle, norm_methods=norms), task=task, task_split=split,
                test_size=config.test_size, random_state=config.random_state, regression_output='median',
                regression_uncertainty=False, expected_preserve_missing_values=False, native_tabicl_baseline=True,
                prediction_sink=captured.append)
            assert len(captured) == 1
            frame = captured[0]
            frame['cv_fold'] = fold
            metrics = {k: float(metrics[k]) for k in ('test_mae', 'test_rmse', 'test_spearman', 'test_r2')}
            if args.step == 10000:
                old = pd.read_csv(recipe / f'cv_fold_{fold}/generic_predictions.csv')
                assert frame.task_row_index.tolist() == old.task_row_index.tolist()
                np.testing.assert_allclose(frame.observed_mV, old.observed_mV, atol=1e-8, rtol=0)
                np.testing.assert_allclose(frame.prediction_mV, old.prediction_mV, atol=0.01, rtol=0)
                print(f'10k reproduction max delta: {abs(frame.prediction_mV-old.prediction_mV).max():.8f} mV', flush=True)
                # Use original predictions for the reference, preserving the exact existing decisions.
                frame = old
                metrics = json.loads((recipe / f'cv_fold_{fold}/metrics.json').read_text())['metrics']
            frame.to_csv(path, index=False)
            meta_path.write_text(json.dumps({'identity': identity, 'csv_sha256': digest(path), 'metrics': metrics}, indent=2))
        assert frame.task_row_index.tolist() == membership['query_task_row_indices']
        frame['cv_fold'] = fold
        frames.append(frame)
        metrics_rows.append({'step': args.step, 'fold': fold, **metrics})
        gc.collect()
    frame = pd.concat(frames, ignore_index=True)
    assert not frame.task_row_index.duplicated().any()
    frame.to_csv(output / 'predictions.csv', index=False)
    pd.DataFrame(metrics_rows).to_csv(output / 'metrics.csv', index=False)
    print(f'DONE {args.dataset} {args.step}', flush=True)

if __name__ == '__main__':
    run()
