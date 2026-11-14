"""Compare saved formula draws and model predictions on five exploratory folds.

Default is analysis only. --export-pretrained runs five CPU prediction exports,
verifies them against saved metrics, then completes the comparison.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.epit_pipeline import diagnose_cv_samples as export
from scripts.epit_pipeline import exploratory_cv as cv
from scripts import eval_corrosion_datasets as ev

BASE = export.ROOT / 'corrosion_datasets/analysis/epit_pipeline'


def references(results, fold):
    payload = json.loads((results / f'cv_fold_{fold}/results.json').read_text())
    rows = pd.DataFrame(payload['rows'])
    rows = rows[rows.model.eq('pretrained_tabicl_v2')]
    if rows.empty or len(rows[list(export.METRICS)].drop_duplicates()) != 1:
        raise ValueError(f'Missing or inconsistent pretrained baseline in fold {fold}.')
    return rows.iloc[0]


def prediction_metrics(frame):
    y = frame.observed_mV.to_numpy()
    p = frame.prediction_mV.to_numpy()
    residual = y - p
    return {'test_mae': np.abs(residual).mean(), 'test_rmse': np.sqrt(np.mean(residual**2)),
            'test_r2': 1 - np.sum(residual**2) / np.sum((y-y.mean())**2),
            'test_spearman': spearmanr(y,p).statistic, 'test_pearson': np.corrcoef(y,p)[0,1]}


def verify_prediction(frame, membership, reference, tolerance):
    if frame.task_row_index.tolist() != membership['query_task_row_indices']:
        raise ValueError('Prediction rows differ from frozen query order.')
    if not np.isfinite(frame[['observed_mV','prediction_mV']].to_numpy()).all():
        raise ValueError('Nonfinite predictions or targets.')
    export.verify_metrics(prediction_metrics(frame), reference, tolerance)


def export_pretrained(args):
    config_args = export.parse_args(['--results-dir',str(args.results_dir),'--analyze-only','--device','cpu'])
    _, memberships, task, config, identity = export.prepare(config_args)
    folder = args.output_dir / 'pretrained_exports'
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / 'manifest.json'
    identity['exporter_sha256'] = export.digest(Path(__file__))
    identity['pretrained_checkpoint'] = str(args.pretrained_checkpoint) if args.pretrained_checkpoint else None
    identity['checkpoint_sha256'] = export.digest(args.pretrained_checkpoint) if args.pretrained_checkpoint else None
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('Pretrained cache configuration changed; choose another output directory.')
    manifest.write_text(json.dumps(identity,indent=2)+'\n')
    for fold in cv.FOLDS:
        path = folder / f'fold_{fold}.csv'
        member = memberships[fold]
        ref = references(args.results_dir, fold)
        if path.exists():
            frame = pd.read_csv(path)
        else:
            print(f'Exporting pretrained predictions: fold {fold}/5',flush=True)
            context = np.asarray(member['context_task_row_indices'])
            split = ev.EvalSplit(train_index=context, test_index=np.asarray(member['query_task_row_indices']),
                                preprocessing_index=context.copy(), split_strategy=f'epit_pipeline_exploratory_cv_fold_{fold}')
            shuffle,norms = ev.tabicl_inference_settings(config,'pretrained_tabicl_v2',pretrained=True)
            captured=[]
            ev.evaluate_estimator(model_label='pretrained_tabicl_v2',model_kind='pretrained_tabicl_regressor',
                estimator_factory=lambda: ev.make_tabicl_regressor(
                    model_path=str(args.pretrained_checkpoint) if args.pretrained_checkpoint else None,
                    checkpoint_version=identity['settings']['pretrained_checkpoint_version'],device='cpu',
                    n_estimators=config.n_estimators,random_state=config.random_state,allow_auto_download=False,
                    feat_shuffle_method=shuffle,norm_methods=norms),
                task=task,task_split=split,test_size=config.test_size,random_state=config.random_state,
                regression_output='median',regression_uncertainty=False,expected_preserve_missing_values=False,
                native_tabicl_baseline=True,prediction_sink=captured.append)
            if len(captured)!=1:
                raise ValueError('Expected one prediction batch.')
            frame=captured[0]
        verify_prediction(frame,member,ref,config_args)
        frame.to_csv(path,index=False)
        print(f'Fold {fold}: verified against saved metrics.',flush=True)


def analyze(args):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    f = pd.read_csv(args.formula_run / 'assigned_predictions.csv')
    if f.duplicated(['repeat','task_row_index']).any():
        raise ValueError('Duplicate formula predictions.')
    repeat_count = json.loads((args.formula_run / 'config.json').read_text())['repeats']
    if f.repeat.nunique()!=repeat_count or not f.groupby('task_row_index').size().eq(repeat_count).all():
        raise ValueError('Incomplete formula draws.')
    model = pd.read_csv(args.model_predictions / 'sample_consistency.csv')
    model = model[model.model.eq('epit_100pct')].copy()
    if model.task_row_index.duplicated().any() or set(model.cv_fold)!=set(cv.FOLDS):
        raise ValueError('Expected one row per sample, spanning all five folds.')
    formula = f.groupby('task_row_index').agg(formula_error=('absolute_error_mV','mean'),target=('observed_mV','first'))
    if set(formula.index)!=set(model.task_row_index):
        raise ValueError('Formula and model samples differ.')
    m = model[['task_row_index','No.','Material class','cv_fold','n_checkpoints','observed_mV',
               'generic_abs_error_mV','informed_abs_error_mV']].merge(formula,on='task_row_index',validate='one_to_one')
    np.testing.assert_allclose(m.observed_mV,m.target,rtol=0,atol=1e-8)
    tolerance = argparse.Namespace(metric_atol=1e-5,metric_rtol=1e-5)
    m['pretrained_error']=np.nan
    available=[]
    for fold in cv.FOLDS:
        membership=cv.fold_membership(export.comparison.DEFAULT_SPLIT_MANIFEST,fold)
        if set(m.loc[m.cv_fold.eq(fold),'task_row_index']) != set(membership['query_task_row_indices']):
            raise ValueError('Model query fold mismatch.')
        path=args.output_dir/'pretrained_exports'/f'fold_{fold}.csv'
        if path.exists():
            frame=pd.read_csv(path)
        elif fold==1:
            old=BASE/'v14_trial27_28_vs_70_columns_step5500_est8_20261106/predictions.csv'
            frame=pd.read_csv(old)[['task_row_index','target','pretrained_tabicl_v2_70cols']].rename(
                columns={'target':'observed_mV','pretrained_tabicl_v2_70cols':'prediction_mV'})
        else:
            continue
        verify_prediction(frame,membership,references(args.results_dir,fold),tolerance)
        truth=m.set_index('task_row_index').loc[frame.task_row_index,'observed_mV'].to_numpy()
        np.testing.assert_allclose(truth,frame.observed_mV,rtol=0,atol=1e-8)
        errors=pd.Series(np.abs(frame.observed_mV-frame.prediction_mV).to_numpy(),index=frame.task_row_index)
        mask=m.cv_fold.eq(fold)
        m.loc[mask,'pretrained_error']=m.loc[mask,'task_row_index'].map(errors)
        available.append(fold)
    m['gain_vs_generic']=m.generic_abs_error_mV-m.formula_error
    m['gain_vs_informed']=m.informed_abs_error_mV-m.formula_error
    m['gain_vs_pretrained']=m.pretrained_error-m.formula_error
    m['gain_vs_both']=np.minimum(m.gain_vs_generic,m.gain_vs_pretrained)
    m.to_csv(args.output_dir/'per_sample.csv',index=False)
    summaries=[]
    for fold,frame in m.groupby('cv_fold'):
        for group,part in [('all',frame),*list(frame.groupby('Material class'))]:
            complete=part.pretrained_error.notna().all()
            summaries.append({'cv_fold':fold,'group':group,'n':len(part),'formula_mae':part.formula_error.mean(),
                'generic_mae':part.generic_abs_error_mV.mean(),'informed_mae':part.informed_abs_error_mV.mean(),
                'pretrained_mae':part.pretrained_error.mean(),
                'beats_generic':int(part.gain_vs_generic.gt(1).sum()),
                'beats_informed':int(part.gain_vs_informed.gt(1).sum()),
                'beats_pretrained':int(part.gain_vs_pretrained.gt(1).sum()) if complete else np.nan,
                'beats_both':int(part.gain_vs_both.gt(1).sum()) if complete else np.nan})
    summary=pd.DataFrame(summaries)
    summary.to_csv(args.output_dir/'per_fold.csv',index=False)
    all_folds=summary[summary.group.eq('all')]
    # Do not silently average pretrained over only the available fold(s).
    means=all_folds.drop(columns=['cv_fold','group']).mean().to_dict()
    if len(available)!=5:
        for name in ('pretrained_mae','beats_pretrained','beats_both'):
            means[name]=None
    status={'pretrained_folds_available':available,'missing_pretrained_folds':sorted(set(cv.FOLDS)-set(available)),
            'five_fold_mean':means,'formula_run':str(args.formula_run.resolve()),
            'model_predictions':str(args.model_predictions.resolve()),
            'definition':'Formula MAE averaged over random assignments; generic/informed MAE averaged over saved checkpoints. Wins require >1 mV. Development folds are exploratory/in-sample for formulas, not held-out formula CV.'}
    (args.output_dir/'summary.json').write_text(json.dumps(status,indent=2)+'\n')
    print(all_folds.round(3).to_string(index=False))
    print('Five-fold means:',means)
    print('Missing pretrained prediction folds:',status['missing_pretrained_folds'])
    print('Saved:',args.output_dir.resolve())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formula-run',type=Path)
    parser.add_argument('--model-predictions',type=Path,default=BASE/'targetv5_cv5_sample_diagnostic')
    parser.add_argument('--results-dir',type=Path,default=export.DEFAULT_RESULTS)
    parser.add_argument('--output-dir',type=Path,default=BASE/'formula_mixture_real_features/cv5_comparison')
    parser.add_argument('--export-pretrained',action='store_true')
    parser.add_argument('--pretrained-checkpoint',type=Path)
    args=parser.parse_args()
    if args.formula_run is None:
        runs=sorted((BASE/'formula_mixture_real_features').glob('run_*/assigned_predictions.csv'))
        if not runs:
            parser.error('No formula run found. Supply --formula-run.')
        args.formula_run=runs[-1].parent
    if args.export_pretrained:
        export_pretrained(args)
    analyze(args)


if __name__=='__main__':
    main()
