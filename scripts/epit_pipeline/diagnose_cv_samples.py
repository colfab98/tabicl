"""Export verified sample predictions for saved exploratory EPIT CV, then compare errors.

Never refits physical formulas or changes split membership. --dry-run loads only
data/metadata. Actual extraction is an evaluation and must be launched explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import eval_corrosion_datasets as ev
from scripts.epit_pipeline import evaluate_checkpoint_comparison as comparison
from scripts.epit_pipeline import exploratory_cv as cv
from scripts.epit_pipeline.composition_data import load_reviewed_compositions
from scripts.epit_pipeline.sample_error_analysis import analyze

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / 'corrosion_datasets/analysis/epit_pipeline/targetv5_four_ratios_through_8000_cv5_cpu'
DEFAULT_OUTPUT = ROOT / 'corrosion_datasets/analysis/epit_pipeline/targetv5_cv5_sample_diagnostic'
METRICS = ('test_mae', 'test_rmse', 'test_spearman', 'test_pearson', 'test_r2')
SETTINGS = ('n_estimators', 'random_state', 'regression_output', 'regression_uncertainty',
            'target_mode', 'target_binning', 'pretrained_checkpoint_version',
            'fixed_schema_models', 'missing_aware_models', 'pitting_magpie_models')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--split-manifest', type=Path, default=comparison.DEFAULT_SPLIT_MANIFEST)
    p.add_argument('--models', nargs='+', choices=['epit_25pct','epit_50pct','epit_75pct','epit_100pct'], default=['epit_100pct'])
    p.add_argument('--steps', nargs='+', type=int, help='Default: all steps shared by requested models and generic.')
    p.add_argument('--device', default='cpu')
    p.add_argument('--metric-atol', type=float, default=1e-5)
    p.add_argument('--metric-rtol', type=float, default=1e-5)
    p.add_argument('--tie-mV', type=float, default=1., dest='tie_mv', help='Practical tie band for absolute-error differences.')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--analyze-only', action='store_true', help='Require completed verified prediction exports; never load models.')
    args = p.parse_args(argv)
    if args.metric_atol < 0 or args.metric_rtol < 0 or args.tie_mv < 0:
        p.error('Tolerances must be nonnegative.')
    return args


def resolve_checkpoint(source):
    path = Path(source)
    if path.is_file():
        return path.resolve()
    if 'checkpoints' in path.parts:
        local = ROOT.joinpath(*path.parts[path.parts.index('checkpoints'):])
        if local.is_file():
            return local.resolve()
    raise FileNotFoundError(f'Checkpoint required for prediction export: {source}')


def prepare(args):
    root = args.results_dir.resolve()
    top = json.loads((root / 'results.json').read_text())
    if not top.get('exploratory_cv') or top.get('cv_folds') != cv.FOLDS or top.get('errors'):
        raise ValueError('Requires a complete, error-free exploratory five-fold result.')
    labels = ['generic_baseline', *dict.fromkeys(args.models)]
    frames, settings, memberships = [], None, {}
    for fold in cv.FOLDS:
        payload = json.loads((root / f'cv_fold_{fold}/results.json').read_text())
        membership = cv.fold_membership(args.split_manifest, fold)
        if payload.get('epit_reviewed_split') != membership or top['fold_membership'][str(fold)] != membership:
            raise ValueError(f'Fold {fold} differs from the saved membership.')
        current = {key: payload.get(key) for key in SETTINGS}
        if settings is not None and settings != current:
            raise ValueError('Saved folds disagree on inference settings.')
        settings = current
        if (payload.get('errors') or current['target_binning'] != 'continuous'
                or current['target_mode'] != 'primary' or current['regression_output'] != 'median'
                or current['regression_uncertainty'] or current['fixed_schema_models']
                or current['missing_aware_models'] or current['pitting_magpie_models']):
            raise ValueError('This exporter requires the reviewed native-input median-regression protocol.')
        frame = pd.DataFrame(payload['rows'])
        frame = frame[frame.model.isin(labels)].copy()
        if set(frame.model) != set(labels) or frame.duplicated(['model','checkpoint_step']).any():
            raise ValueError(f'Missing or duplicate model rows in fold {fold}.')
        if (not frame.n_features.eq(70).all() or frame.preserve_feature_schema.any()
                or frame.preserve_missing_values.any()
                or not frame.n_train.eq(len(membership['context_task_row_indices'])).all()
                or not frame.n_test.eq(len(membership['query_task_row_indices'])).all()
                or not frame.task_id.eq(comparison.EPIT_TASK_ID).all()
                or not frame.split_strategy.eq(f'epit_pipeline_exploratory_cv_fold_{fold}').all()):
            raise ValueError('Saved model rows use an incompatible input schema.')
        frame['cv_fold'] = fold
        frames.append(frame)
        memberships[fold] = membership
    references = pd.concat(frames, ignore_index=True)
    common = set.intersection(*[set(g.checkpoint_step.astype(int))
                               for _, g in references.groupby(['model','cv_fold'])])
    steps = sorted(set(args.steps) if args.steps else common)
    if not steps or not set(steps) <= common:
        raise ValueError(f'Requested steps must be shared by all models/folds; available: {sorted(common)}')
    references = references[references.checkpoint_step.isin(steps)].copy()
    if not args.analyze_only:
        references['checkpoint_path'] = references.model_source.map(resolve_checkpoint)
        snapshot = json.loads((root/'cv_checkpoints.json').read_text())
        sizes = {(record['step'], model['label']): model['size']
                 for record in snapshot['checkpoints'] for model in record['models']}
        for row in references.itertuples():
            if row.checkpoint_path.stat().st_size != sizes[(row.checkpoint_step,row.model)]:
                raise ValueError(f'Checkpoint size differs from the saved snapshot: {row.checkpoint_path}')
    config = ev.parse_args([
        '--task', comparison.EPIT_TASK_ID, '--target-mode','primary', '--target-binning','continuous',
        '--epit-expanded-features', '--epit-split-manifest',str(args.split_manifest),
        '--epit-exploratory-cv-fold','1', '--native-tabicl-baselines',
        '--n-estimators',str(settings['n_estimators']), '--random-state',str(settings['random_state']),
        '--device',args.device, '--regression-output','median', '--no-regression-uncertainty',
        '--max-samples-per-task','0'])
    tasks = ev.make_tasks(config)
    task, _, _, _ = ev._load_epit_pipeline_indices(tasks, manifest_path=args.split_manifest)
    identity = {
        'version': 1, 'source_results':str(root), 'steps':steps, 'models':labels,
        'settings':settings, 'device':args.device,
        'metric_atol':args.metric_atol, 'metric_rtol':args.metric_rtol,
        'split_sha256':digest(args.split_manifest),
        'source_sha256':{str(f):digest(root/f'cv_fold_{f}/results.json') for f in cv.FOLDS},
        'implementation_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [
            Path(__file__), Path(ev.__file__), Path(cv.__file__),
            ROOT/'src/tabicl/sklearn/regressor.py',
        ]},
    }
    return references, memberships, task, config, identity


def verify_metrics(actual, reference, args):
    differences = {name: {'actual':float(actual[name]), 'saved':float(reference[name])}
                   for name in METRICS if not np.isclose(actual[name],reference[name],
                       atol=args.metric_atol,rtol=args.metric_rtol,equal_nan=False)}
    if differences:
        raise ValueError(f'Prediction extraction did not reproduce saved metrics: {differences}')


def run(args):
    references, memberships, task, config, identity = prepare(args)
    print(f"{len(references)} model/fold/checkpoint jobs; steps {identity['steps']}; models {identity['models']}",flush=True)
    if args.dry_run:
        print('Verified data membership, settings and checkpoint availability/sizes. No models loaded or outputs written.')
        return
    output = args.output_dir.resolve()
    manifest = output/'extraction_manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('Output belongs to a different extraction configuration; choose a new output directory.')
    elif args.analyze_only:
        raise FileNotFoundError(manifest)
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Output is nonempty without an extraction manifest: {output}')
        output.mkdir(parents=True,exist_ok=True)
        manifest.write_text(json.dumps(identity,indent=2)+'\n')
    hashes, predictions, verification = {}, [], []
    for row in references.sort_values(['cv_fold','checkpoint_step','model']).to_dict('records'):
        fold, step, label = int(row['cv_fold']), int(row['checkpoint_step']), row['model']
        folder = output/f'cv_fold_{fold}'/f'step-{step}'
        csv_path, meta_path = folder/f'{label}.csv',folder/f'{label}.json'
        checkpoint_hash = None
        if not args.analyze_only:
            path = row['checkpoint_path']
            if path not in hashes:
                hashes[path] = digest(path)
            checkpoint_hash = hashes[path]
        if csv_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if digest(csv_path) != meta['predictions_sha256']:
                raise ValueError(f'Cached predictions changed: {csv_path}')
            if checkpoint_hash is not None and checkpoint_hash != meta['checkpoint_sha256']:
                raise ValueError(f'Checkpoint changed since cached export: {path}')
            verify_metrics(meta['metrics'],row,args)
            prediction = pd.read_csv(csv_path)
            print(f'Reusing verified {label}, step {step}, fold {fold}',flush=True)
        elif args.analyze_only:
            raise FileNotFoundError(f'Missing completed prediction export: {csv_path}')
        else:
            membership = memberships[fold]
            context=np.asarray(membership['context_task_row_indices'])
            query=np.asarray(membership['query_task_row_indices'])
            split=ev.EvalSplit(train_index=context,test_index=query,preprocessing_index=context.copy(),
                              split_strategy=f'epit_pipeline_exploratory_cv_fold_{fold}')
            shuffle,norms=ev.tabicl_inference_settings(config,label)
            captured=[]
            print(f'Extracting {label}, step {step}, fold {fold}',flush=True)
            metrics=ev.evaluate_estimator(model_label=label,model_kind='local_tabicl_regressor',
                estimator_factory=lambda:ev.make_tabicl_regressor(model_path=str(path),
                    checkpoint_version=identity['settings']['pretrained_checkpoint_version'],
                    device=args.device,n_estimators=config.n_estimators,random_state=config.random_state,
                    allow_auto_download=False,feat_shuffle_method=shuffle,norm_methods=norms),
                task=task,task_split=split,test_size=config.test_size,random_state=config.random_state,
                regression_output='median',regression_uncertainty=False,
                expected_preserve_missing_values=False,native_tabicl_baseline=True,
                prediction_sink=captured.append)
            if len(captured)!=1:
                raise RuntimeError('Expected one prediction batch.')
            verify_metrics(metrics,row,args)
            prediction=captured[0]
            prediction['model']=label;prediction['checkpoint_step']=step;prediction['cv_fold']=fold
            folder.mkdir(parents=True,exist_ok=True)
            prediction.to_csv(csv_path,index=False)
            meta={'checkpoint_sha256':checkpoint_hash,'checkpoint_path':str(path),
                  'metrics':{name:float(metrics[name]) for name in METRICS},
                  'predictions_sha256':digest(csv_path)}
            meta_path.write_text(json.dumps(meta,indent=2)+'\n')
        if (prediction.task_row_index.tolist()!=memberships[fold]['query_task_row_indices']
                or not prediction.model.eq(label).all() or not prediction.checkpoint_step.eq(step).all()
                or not prediction.cv_fold.eq(fold).all()):
            raise ValueError(f'Wrong prediction identities: {csv_path}')
        verification.append({'model':label,'checkpoint_step':step,'cv_fold':fold,
                             **{name:meta['metrics'][name]-row[name] for name in METRICS}})
        predictions.append(prediction)
    all_predictions=pd.concat(predictions,ignore_index=True)
    all_predictions.to_csv(output/'predictions.csv',index=False)
    pd.DataFrame(verification).to_csv(output/'metric_reproduction_deltas.csv',index=False)
    metadata=load_reviewed_compositions().rows
    analyze(all_predictions,metadata,output,tie_mv=args.tie_mv)
    print(f'Finished. Sample comparisons and category tables: {output}',flush=True)


if __name__=='__main__':
    run(parse_args())
