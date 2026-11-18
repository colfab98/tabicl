"""MPEA isolation, native preprocessing, shared CV and live-checkpoint parity."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import eval_corrosion_datasets as ev
from scripts.mpea_pipeline import evaluate_native as worker
from scripts.mpea_pipeline import evaluate_checkpoint_comparison as comparison
from scripts.mpea_pipeline.evaluation_data import DEFAULT_SPLIT, ROOT, TASK_ID, fold_membership, load_task


def test_frozen_mpea_inputs_missingness_and_complete_cv_partition():
    task, first, provenance = load_task()
    original = pd.read_csv(ROOT / provenance['inputs']['queries']['path'])
    pd.testing.assert_frame_equal(task.X, original[provenance['feature_columns']])
    assert task.X.isna().any().any()
    assert len(first['context_task_row_indices']) == 214
    assert len(first['query_task_row_indices']) == 53
    frozen = json.loads(DEFAULT_SPLIT.read_text())
    assert first['query_task_row_indices'] == [r['task_row_index'] for r in frozen['rows'] if r['outer_split']=='final_test']
    all_queries = []
    for fold in range(1, 6):
        membership = fold_membership(DEFAULT_SPLIT, fold)
        context, query = (membership[key] for key in ('context_task_row_indices','query_task_row_indices'))
        assert not set(context) & set(query)
        assert set(context) | set(query) == set(range(267))
        all_queries.extend(query)
    assert sorted(all_queries) == list(range(267))
    with pytest.raises(ValueError, match='frozen outer'):
        load_task(fold=2, exploratory=False)
    final, _, _ = load_task(exploratory=False)
    np.testing.assert_array_equal(final.fixed_split.test_index, task.fixed_split.test_index)


def test_same_native_inference_settings_as_standalones(tmp_path):
    from scripts.epit_pipeline import evaluate_checkpoint_comparison as epit
    from scripts.soccol_pipeline import evaluate_checkpoint_comparison as soccol
    commands = []
    for module in (comparison, epit, soccol):
        args = module.parse_args(['--candidate-run', str(tmp_path/'candidate'), '--candidate-label', 'combined',
            '--output-dir', str(tmp_path/'output'), '--exploratory-cv-folds', '5'])
        command = module.evaluation_command(args)
        if module is comparison:
            i=command.index('--mpea-split-manifest');del command[i:i+2]
            i=command.index('--exploratory-cv-folds');del command[i:i+2]
        parsed=ev.parse_args(command[2:])
        for label in ('combined','generic_baseline','pretrained_tabicl_v2'):
            assert ev.tabicl_inference_settings(parsed,label)==('latin',None)
            assert ev.expected_missing_value_mode(parsed,label) is False
        commands.append(parsed)
    for key in ('n_estimators','random_state','regression_output','regression_uncertainty',
                'catboost_iterations','catboost_depth','catboost_learning_rate','catboost_l2_leaf_reg',
                'checkpoint_scope','checkpoint_step_interval'):
        assert len({str(getattr(p,key)) for p in commands})==1


@pytest.mark.parametrize('scope',['all','common'])
@pytest.mark.parametrize('deletion',[False,True])
def test_mpea_cv_shared_driver_and_deleted_checkpoint(tmp_path,monkeypatch,scope,deletion):
    candidate,generic=tmp_path/'candidate',tmp_path/'generic'
    for directory in (candidate,generic):
        directory.mkdir()
        for step in (100,500):
            (directory/f'step-{step}.ckpt').write_bytes(b'mock')
    calls=[]
    original_task,_,_=load_task()
    def evaluate(**kwargs):
        task,split=kwargs['task'],kwargs['task_split']
        pd.testing.assert_frame_equal(task.X,original_task.X)
        assert task.task_id==TASK_ID
        fold=int(split.split_strategy.rsplit('_',1)[1])
        if 'tabicl' in kwargs['model_kind']:
            assert kwargs['native_tabicl_baseline']
            assert not kwargs['preserve_feature_schema']
            assert kwargs['expected_preserve_missing_values'] is False
        calls.append((fold,kwargs['model_label']))
        return {'model':kwargs['model_label'],'model_kind':kwargs['model_kind'],
                'task_id':task.task_id,'target_binning':'continuous','n_samples':len(task.y),
                'n_features':len(task.X.columns),'n_train':len(split.train_index),'n_test':len(split.test_index),
                'split_strategy':split.split_strategy,
                **{metric:float(fold) for metric in ('test_mae','test_rmse','test_spearman',
                    'test_pearson','test_r2','test_nmae_iqr','test_nrmse_iqr')}}
    def invoke(command,**kwargs):
        worker.main(command[2:])
        if deletion and '--epit-exploratory-cv-fold' in command:
            if command[command.index('--epit-exploratory-cv-fold')+1]=='1':
                (candidate/'step-100.ckpt').unlink()
                (candidate/'step-750.ckpt').write_bytes(b'new checkpoint must not enter snapshot')
    monkeypatch.setattr(ev,'evaluate_estimator',evaluate)
    monkeypatch.setattr(comparison.subprocess,'run',invoke)
    args=comparison.parse_args(['--candidate-run',str(candidate),'--generic-run',str(generic),
        '--checkpoint-step-interval','0','--checkpoint-scope',scope,'--output-dir',str(tmp_path/'results')])
    output=comparison.run(args)
    payload=json.loads((output/'results.json').read_text())
    assert not payload['errors']
    assert {f for f,_ in calls}==set(range(1,6))
    assert all(r['n_cv_folds']==5 and r['mean_test_mae']==pytest.approx(3) for r in payload['summary'])
    assert {r['checkpoint_step'] for r in payload['rows'] if r['model']==args.candidate_label}==({500} if deletion else {100,500})
    assert {r['checkpoint_step'] for r in payload['rows'] if r['model']=='generic_baseline'}==({500} if deletion and scope=='common' else {100,500})
    assert (output/'plots/trend_mean_test_mae.svg').is_file()


def test_real_combined_checkpoint_predicts_mpea_with_native_missing_values(tmp_path):
    import torch
    from tests.test_combined_selected_pairs import config
    from tabicl.train.run_combined_selected_pairs import CombinedSelectedPairTrainer

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trainer = CombinedSelectedPairTrainer(config(tmp_path))
        trainer.run_batch(next(trainer.dataloader))
        trainer.curr_step = 1
        trainer.save_checkpoint('step-1.ckpt')
        checkpoint = Path(trainer.config.checkpoint_dir) / 'step-1.ckpt'
        task, membership, _ = load_task(exploratory=False)
        predictions = []
        row = ev.evaluate_estimator(model_label='combined', model_kind='tabicl',
            estimator_factory=lambda: ev.make_tabicl_regressor(model_path=str(checkpoint),
                checkpoint_version='', device='cpu', n_estimators=2, random_state=42,
                allow_auto_download=False, feat_shuffle_method='latin', norm_methods=None),
            task=task, task_split=task.fixed_split, test_size=.2, random_state=42,
            regression_uncertainty=False, expected_preserve_missing_values=False,
            native_tabicl_baseline=True, prediction_sink=predictions.append)
        assert row['n_train']==214 and row['n_test']==53
        assert np.isfinite(row['test_mae'])
        assert np.isfinite(predictions[0]['prediction_mV']).all()
        assert predictions[0]['task_row_index'].tolist()==membership['query_task_row_indices']
    finally:
        torch.set_num_threads(previous)
