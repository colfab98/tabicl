from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from scripts import eval_corrosion_datasets as ev
from scripts.epit_pipeline import diagnose_cv_samples as export
from scripts.epit_pipeline import sample_error_analysis as analysis


def small_task():
    return ev.EvalTask(task_id='test_regression',dataset='test',table='test',target='y',
        threshold=0.,target_binning='continuous',target_bins=0,bin_edges=[],
        class_labels=[],class_counts={},X=pd.DataFrame({'x':np.arange(12,dtype=float)}),
        y=pd.Series(np.arange(12,dtype=float)*10),y_ordinal=pd.Series(np.arange(12)),
        task_family='test',feature_groups_used=[],feature_group_counts={},
        dropped_feature_columns=[],quality_flags=[],split_groups=None,split_strategy='test',
        include_in_summary=True,summary_exclusion_reason='')


class FakeRegressor:
    preserve_missing_values_=False
    def fit(self,X,y):
        return self
    def predict(self,X,**kwargs):
        return X.x.to_numpy()*10+2.


def test_prediction_sink_uses_exact_query_order_without_changing_scores():
    task=small_task()
    split=ev.EvalSplit(train_index=np.arange(8),test_index=np.array([11,8,10,9]),split_strategy='test')
    captured=[]
    def sink(frame):
        captured.append(frame.copy())
        frame.loc[:,'observed_mV']=9999.
        frame.loc[:,'prediction_mV']=0.
    result=ev.evaluate_estimator(model_label='generic_baseline',model_kind='local_tabicl_regressor',
        estimator_factory=FakeRegressor,task=task,task_split=split,test_size=.2,random_state=42,
        regression_uncertainty=False,prediction_sink=sink)
    assert result['test_mae']==2.
    assert captured[0].task_row_index.tolist()==[11,8,10,9]
    np.testing.assert_allclose(captured[0].prediction_mV,[112,82,102,92])


def fake_predictions():
    records=[]
    for step in [1000,2000]:
        for fold in [1,2]:
            for i in range(6):
                index=(fold-1)*6+i
                for model,error in [('generic_baseline',10.),('epit_100pct',20. if i<3 else 5.)]:
                    records.append(dict(model=model,checkpoint_step=step,cv_fold=fold,
                                        task_row_index=index,observed_mV=index*10.,prediction_mV=index*10.+error))
    return pd.DataFrame(records)


def metadata():
    data=pd.DataFrame({'task_row_index':range(12),'Reference':list(range(6))*2,
                       'Material class':['Fe Alloy']*12,'row_id':[f'r{i}' for i in range(12)],
                       'Epit, mV (SCE) Avg.':np.arange(12)*10})
    for col in ['Cr','Mo','N','CP_pH','CP_temp','CP_Cl','phase_FCC','phase_BCC','phase_HCP','phase_IM']:
        data[col]=np.nan
    for col in ['heat_treatment_type','heat_treatment_cooling','surface_preparation_method','test_solution_type']:
        data[col]=pd.Series(['annealing']*12,dtype='str')
    return data


def test_analysis_pairs_by_identity_and_does_not_count_steps_as_samples(tmp_path):
    predictions=fake_predictions().sample(frac=1,random_state=42)
    pairs,samples,groups=analysis.analyze(predictions,metadata(),tmp_path)
    assert len(pairs)==24 and len(samples)==12
    assert set(samples.n_checkpoints)=={2}
    assert set(samples.error_penalty_mV)=={-5.,10.}
    families=groups[groups.category_type.eq('Material class')]
    assert families.n_samples.tolist()==[6,6]
    assert set(families.mean_error_penalty_mV)=={2.5}
    persistent=pd.read_csv(tmp_path/'persistently_worse_samples.csv')
    assert set(persistent.task_row_index)=={0,1,2,6,7,8}
    assert set(groups.scope)=={'outer_fold_1','exploratory_development_folds'}


@pytest.mark.parametrize('change',['missing','duplicate','wrong_target','duplicate_query_fold'])
def test_bad_pairings_fail(change):
    frame=fake_predictions()
    if change=='missing':
        frame=frame.iloc[1:]
    elif change=='duplicate':
        frame=pd.concat([frame,frame.iloc[:1]])
    elif change=='duplicate_query_fold':
        duplicate=frame.iloc[:1].copy();duplicate['cv_fold']=5
        frame=pd.concat([frame,duplicate])
    else:
        frame.loc[1,'observed_mV']=999.
    with pytest.raises(ValueError):
        analysis.pair_predictions(frame)


def test_metric_reproduction_rejects_drift():
    args=SimpleNamespace(metric_atol=1e-5,metric_rtol=1e-5)
    saved={m:1. for m in export.METRICS}
    export.verify_metrics(saved,saved,args)
    with pytest.raises(ValueError,match='did not reproduce'):
        export.verify_metrics({**saved,'test_spearman':.9},saved,args)


def test_export_can_resume_and_analyze_without_loading_models(tmp_path,monkeypatch):
    task=small_task()
    checkpoint=tmp_path/'model.ckpt';checkpoint.write_bytes(b'fake weights never loaded')
    args=export.parse_args(['--output-dir',str(tmp_path/'output'),'--steps','1000'])
    config=ev.parse_args(['--native-tabicl-baselines'])
    membership={'context_task_row_indices':list(range(6)),'query_task_row_indices':list(range(6,12))}
    split=ev.EvalSplit(train_index=np.arange(6),test_index=np.arange(6,12),split_strategy='test')
    metrics=ev.evaluate_estimator(model_label='test',model_kind='local_tabicl_regressor',
        estimator_factory=FakeRegressor,task=task,task_split=split,test_size=.2,random_state=42,regression_uncertainty=False)
    references=pd.DataFrame([{'model':label,'cv_fold':fold,'checkpoint_step':1000,
                              'checkpoint_path':checkpoint,**{m:metrics[m] for m in export.METRICS}}
                             for fold in range(1,6) for label in ['generic_baseline','epit_100pct']])
    identity={'steps':[1000],'models':['generic_baseline','epit_100pct'],
              'settings':{'pretrained_checkpoint_version':'fake'}}
    monkeypatch.setattr(export,'prepare',lambda a:(references,{f:membership for f in range(1,6)},task,config,identity))
    calls=[]
    monkeypatch.setattr(ev,'make_tabicl_regressor',lambda **kw:(calls.append(kw) or FakeRegressor()))
    monkeypatch.setattr(export,'load_reviewed_compositions',lambda:SimpleNamespace(rows=metadata()))
    # Here test extraction/resumption only; pairing and grouping tested separately.
    monkeypatch.setattr(export,'analyze',lambda *a,**kw:None)
    export.run(args)
    assert len(calls)==10
    export.run(args)
    assert len(calls)==10
    checkpoint.unlink()
    args.analyze_only=True
    export.run(args)
    assert len(calls)==10
    p=args.output_dir/'cv_fold_1/step-1000/generic_baseline.csv'
    p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='Cached predictions changed'):
        export.run(args)
