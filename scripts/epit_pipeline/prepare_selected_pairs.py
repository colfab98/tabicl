"""Build an exploratory all-data EPIT sample/formula recipe. CPU only.

Explicitly fits on ALL real rows, including former final-test rows. Outputs
must not be described as an independent evaluation or replace frozen recipes.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from scripts import eval_corrosion_datasets as ev
from scripts.epit_pipeline import exploratory_cv as cv
from scripts.epit_pipeline import evaluate_checkpoint_comparison as comparison
from scripts.epit_pipeline.artifact_hashes import sha256_file
from scripts.epit_pipeline.composition_data import load_reviewed_compositions
from scripts.epit_pipeline.formula_data import load_formula_dataset
from scripts.epit_pipeline.formula_extension_study import FormulaStudy
from scripts.epit_pipeline.target_rules import PROMOTED_TARGET_RULE_FAMILIES
from scripts.epit_pipeline.evaluate_formula_mixture import calibrate_mv
from tabicl.prior.epit_expanded_schema import EPIT_EXPANDED_FEATURE_COLUMNS
from tabicl.prior.epit_target_v5 import load_recipe

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'corrosion_datasets/analysis/epit_pipeline/selected_pairs_all760_generic10k'


def compare_pairs(formula, generic):
    if generic.task_row_index.duplicated().any() or formula.duplicated(['task_row_index','formula']).any():
        raise ValueError('Duplicate sample predictions.')
    if set(formula.task_row_index) != set(generic.task_row_index):
        raise ValueError('Formula and generic sample sets differ.')
    result = formula.merge(generic[['task_row_index','observed_mV','prediction_mV','cv_fold']],
                           on='task_row_index',validate='many_to_one',suffixes=('','_generic'))
    np.testing.assert_allclose(result.observed_mV,result.observed_mV_generic,rtol=0,atol=1e-8)
    if not np.isfinite(result[['prediction_mV','prediction_mV_generic','observed_mV']]).all().all():
        raise ValueError('Nonfinite targets or predictions.')
    result['formula_error_mV'] = np.abs(result.prediction_mV-result.observed_mV)
    result['generic_error_mV'] = np.abs(result.prediction_mV_generic-result.observed_mV)
    result['gain_mV'] = result.generic_error_mV-result.formula_error_mV
    result['retained'] = result.gain_mV > 0
    return result


def fit_all(dataset, reviewed):
    indices=np.arange(len(dataset.target))
    study=FormulaStudy(dataset,reviewed.rows.loc[:,list(EPIT_EXPANDED_FEATURE_COLUMNS)],indices,
                       np.zeros(len(indices),dtype=int),reviewed.assignments.composition_group.to_numpy())
    artifacts,frames={},[]
    for name in PROMOTED_TARGET_RULE_FAMILIES:
        print('Fitting all rows:',name,flush=True)
        score,artifact=study.old_model(indices,name)
        prediction,calibration=calibrate_mv(score,dataset.target,indices)
        artifact.update(name=name,fitting_task_row_indices=indices.tolist(),mV_calibration=calibration)
        artifacts[name]=artifact
        frames.append(pd.DataFrame({'task_row_index':indices,'formula':name,'observed_mV':dataset.target,
                                    'raw_score':score,'prediction_mV':prediction}))
    name='extend_heat_categories'
    # Keep the existing penalty rather than adding a hyperparameter search.
    print('Fitting all rows:',name,flush=True)
    score,artifact=study.fit(indices,study.specs[name],penalty=load_recipe()['formula']['penalty'])
    calibration=artifact['mV_calibration']
    prediction=calibration['intercept']+calibration['slope']*score
    artifacts[name]=artifact
    frames.append(pd.DataFrame({'task_row_index':indices,'formula':name,'observed_mV':dataset.target,
                                'raw_score':score,'prediction_mV':prediction}))
    return artifacts,pd.concat(frames,ignore_index=True)


def run(args):
    checkpoint=args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    reviewed=load_reviewed_compositions()
    dataset,input_identity=load_formula_dataset()
    memberships={fold:cv.fold_membership(comparison.DEFAULT_SPLIT_MANIFEST,fold) for fold in cv.FOLDS}
    query=[i for member in memberships.values() for i in member['query_task_row_indices']]
    if sorted(query)!=list(range(len(dataset.target))):
        raise ValueError('CV queries must cover each sample exactly once.')
    for member in memberships.values():
        if set(member['query_task_row_indices']) & set(member['context_task_row_indices']):
            raise ValueError('Query/context overlap.')
    if args.dry_run:
        print(f'Validated {len(dataset.target)} rows, 10 formulas and five disjoint query folds.')
        print('Generic checkpoint:',checkpoint)
        print('Output:',args.output_dir.resolve())
        print('No fitting or inference performed.')
        return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),Path(ev.__file__),ROOT/'scripts/epit_pipeline/target_rules.py',
             ROOT/'scripts/epit_pipeline/formula_extension_study.py',Path(cv.__file__),
             ROOT/'scripts/epit_pipeline/evaluate_formula_mixture.py']
    identity={'version':1,'all_rows_used_for_fitting':True,'independent_evaluation':False,
              'data':input_identity,'checkpoint':str(checkpoint),'checkpoint_sha256':sha256_file(checkpoint),
              'source_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in sources},
              'heat_penalty':load_recipe()['formula']['penalty'],
              'n_estimators':8,'random_state':42,'device':'cpu','regression_output':'median',
              'memberships':{str(k):v for k,v in memberships.items()}}
    manifest=args.output_dir/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:
        raise ValueError('Existing output has different settings; choose a new --output-dir.')
    if not manifest.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Nonempty output without a manifest; choose a new --output-dir.')
    manifest.write_text(json.dumps(identity,indent=2)+'\n')
    artifacts,formula=fit_all(dataset,reviewed)
    (args.output_dir/'fitted_formulas.json').write_text(json.dumps(artifacts,indent=2,allow_nan=False)+'\n')
    formula.to_csv(args.output_dir/'formula_predictions.csv',index=False)
    config=ev.parse_args(['--task',comparison.EPIT_TASK_ID,'--target-mode','primary','--target-binning','continuous',
                         '--epit-expanded-features','--epit-split-manifest',str(comparison.DEFAULT_SPLIT_MANIFEST),
                         '--epit-exploratory-cv-fold','1','--native-tabicl-baselines','--n-estimators','8',
                         '--random-state','42','--device','cpu','--regression-output','median',
                         '--no-regression-uncertainty','--max-samples-per-task','0'])
    task,_,_,_=ev._load_epit_pipeline_indices(ev.make_tasks(config),manifest_path=comparison.DEFAULT_SPLIT_MANIFEST)
    shuffle,norms=ev.tabicl_inference_settings(config,'generic_baseline')
    predictions=[]
    for fold,member in memberships.items():
        folder=args.output_dir/f'cv_fold_{fold}'
        folder.mkdir(exist_ok=True)
        path=folder/'generic_predictions.csv'
        meta_path=folder/'metrics.json'
        if path.exists() and meta_path.exists():
            meta=json.loads(meta_path.read_text())
            if sha256_file(path)!=meta['prediction_sha256']:
                raise ValueError('Cached predictions changed.')
            frame=pd.read_csv(path)
            print(f'Reusing generic fold {fold}',flush=True)
        else:
            print(f'Predicting generic 10000: fold {fold}/5',flush=True)
            context=np.asarray(member['context_task_row_indices'])
            split=ev.EvalSplit(train_index=context,test_index=np.asarray(member['query_task_row_indices']),
                              preprocessing_index=context.copy(),split_strategy=f'epit_pipeline_exploratory_cv_fold_{fold}')
            captured=[]
            metrics=ev.evaluate_estimator(model_label='generic_baseline',model_kind='local_tabicl_regressor',
                estimator_factory=lambda:ev.make_tabicl_regressor(model_path=str(checkpoint),
                    checkpoint_version='tabicl-regressor-v2-20260212.ckpt',device='cpu',n_estimators=8,
                    random_state=42,allow_auto_download=False,feat_shuffle_method=shuffle,norm_methods=norms),
                task=task,task_split=split,test_size=config.test_size,random_state=42,regression_output='median',
                regression_uncertainty=False,expected_preserve_missing_values=False,native_tabicl_baseline=True,
                prediction_sink=captured.append)
            if len(captured)!=1:
                raise ValueError('Expected exactly one prediction batch.')
            frame=captured[0]
            frame['cv_fold']=fold
            frame.to_csv(path,index=False)
            meta_path.write_text(json.dumps({'prediction_sha256':sha256_file(path),
                'metrics':{k:float(v) for k,v in metrics.items() if k in ('test_mae','test_rmse','test_spearman','test_r2')}},indent=2)+'\n')
        if frame.task_row_index.tolist()!=member['query_task_row_indices'] or not frame.cv_fold.eq(fold).all():
            raise ValueError('Cached prediction fold mismatch.')
        predictions.append(frame)
    generic=pd.concat(predictions,ignore_index=True)
    generic.to_csv(args.output_dir/'generic_cv_predictions.csv',index=False)
    pairs=compare_pairs(formula,generic).merge(reviewed.rows[['task_row_index','row_id','No.','Material class']],
                                             on='task_row_index',validate='many_to_one')
    pairs.to_csv(args.output_dir/'all_pairs.csv',index=False)
    pairs[pairs.retained].to_csv(args.output_dir/'retained_pairs.csv',index=False)
    summary={'samples':len(generic),'formulas':pairs.formula.nunique(),'tested_pairs':len(pairs),
             'retained_pairs':int(pairs.retained.sum()),'informed_proportion':float(pairs.retained.mean()),
             'selection':'strictly lower absolute error than generic step 10000; ties discarded',
             'scope':'all-data recipe construction, not independent formula evaluation'}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print('Saved:',args.output_dir.resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/tabicl_s1_regression_baseline/step-10000.ckpt')
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--dry-run',action='store_true')
    run(parser.parse_args())
