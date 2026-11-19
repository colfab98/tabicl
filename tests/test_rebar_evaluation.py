"""Rebar units, target exclusions, frozen identities, and shared evaluation behavior."""
import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

from scripts.rebar_pipeline import evaluate_checkpoint_comparison as comparison
from scripts.rebar_pipeline import evaluate_native as worker
from scripts.rebar_pipeline.evaluation_data import DEFAULT_SPLIT, TASK_ID, fold_membership, load_task
from scripts.rebar_pipeline.prepare_evaluation import RAW, prepare, source_rows
from scripts.rebar_pipeline.reviewed_evaluation_data import load_task as load_reviewed_task


def test_real_inputs_units_exclusions_and_publication_disjoint_folds():
    task, membership, provenance = load_task()
    audit = pd.read_csv(DEFAULT_SPLIT.parent/'row_audit.csv').set_index('row_id')
    wb = openpyxl.load_workbook(RAW, data_only=True)
    for i, rid in enumerate(provenance['row_ids']):
        row = audit.loc[rid]
        col = 21 if row['sheet'].endswith('1') else 32
        assert task.y.iloc[i] == pytest.approx(wb[row['sheet']].cell(int(row['excel_row']), col).value*1000)
    assert len(task.y) == 278
    assert task.X.isna().any().any()
    assert not any(x in ' '.join(task.X.columns) for x in ('Ecp', 'Epit', ')ct', 'References', 'Density', 'time'))
    assert audit.loc[provenance['row_ids'], 'status'].eq('included_workbook_reported').all()
    # Equality with Ecp is a flag, not a blanket exclusion.
    assert audit.loc[provenance['row_ids'], 'equal_to_ecp'].any()
    covered = []
    for fold in range(1, 6):
        split = fold_membership(DEFAULT_SPLIT, fold)
        a, b = split['context_task_row_indices'], split['query_task_row_indices']
        assert not set(a) & set(b)
        ids = provenance['row_ids']
        assert not set(audit.loc[[ids[i] for i in a], 'source_id']) & set(audit.loc[[ids[i] for i in b], 'source_id'])
        covered += b
    assert sorted(covered) == list(range(len(task.y)))
    with pytest.raises(ValueError, match='frozen outer'):
        load_task(fold=2, exploratory=False)
    final, _, _ = load_task(exploratory=False)
    assert final.fixed_split.test_index.tolist() == membership['query_task_row_indices']


def test_bibliographic_fill_resets_at_changed_reference():
    ws = openpyxl.Workbook().active
    for row in [(1, 'source one'), (None, None), (2, None), (2, 'source two'), (3, None)]:
        ws.append(row)
    result = list(source_rows(ws, 0, 2))
    assert [r[2] for r in result] == ['source one', 'source one', 'source two', 'source two', None]


def test_rebuild_deterministic_and_tampering_detected(tmp_path):
    out = tmp_path/'prepared'
    prepare(out)
    assert (out/'split_manifest.json').read_bytes() == DEFAULT_SPLIT.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(out)
    with (out/'features.csv').open('a') as f:
        f.write('\n')
    with pytest.raises(RuntimeError, match='export changed'):
        load_task(out/'split_manifest.json')


@pytest.mark.parametrize('scope', ['all', 'common'])
def test_shared_cv_native_settings_and_live_checkpoint_deletion(tmp_path, monkeypatch, scope):
    from scripts import eval_corrosion_datasets as ev
    candidate, generic = tmp_path/'candidate', tmp_path/'generic'
    for d in (candidate, generic):
        d.mkdir()
        for step in (100, 500):
            (d/f'step-{step}.ckpt').write_bytes(b'mock')
    original, _, _ = load_reviewed_task()
    calls = []
    def evaluate(**kw):
        task, split = kw['task'], kw['task_split']
        pd.testing.assert_frame_equal(task.X, original.X)
        fold = int(split.split_strategy.rsplit('_', 1)[1])
        if 'tabicl' in kw['model_kind']:
            assert kw['native_tabicl_baseline']
            assert not kw['preserve_feature_schema']
            assert kw['expected_preserve_missing_values'] is False
        calls.append(fold)
        return dict(model=kw['model_label'], model_kind=kw['model_kind'], task_id=TASK_ID,
            target_binning='continuous', n_samples=len(task.y), n_features=len(task.X.columns),
            n_train=len(split.train_index), n_test=len(split.test_index), split_strategy=split.split_strategy,
            **{k:float(fold) for k in ('test_mae','test_rmse','test_spearman','test_pearson','test_r2','test_nmae_iqr','test_nrmse_iqr')})
    def invoke(command, **kw):
        worker.main(command[2:])
        if '--epit-exploratory-cv-fold' in command and command[command.index('--epit-exploratory-cv-fold')+1] == '1':
            (candidate/'step-100.ckpt').unlink()
            (candidate/'step-750.ckpt').write_bytes(b'new')
    monkeypatch.setattr(ev, 'evaluate_estimator', evaluate)
    monkeypatch.setattr(comparison.subprocess, 'run', invoke)
    args = comparison.parse_args(['--candidate-run', str(candidate), '--generic-run', str(generic),
        '--checkpoint-step-interval', '0', '--checkpoint-scope', scope, '--output-dir', str(tmp_path/'results')])
    output = comparison.run(args)
    payload = json.loads((output/'results.json').read_text())
    assert set(calls) == set(range(1,6))
    assert all(r['n_cv_folds'] == 5 for r in payload['summary'])
    assert {r['checkpoint_step'] for r in payload['rows'] if r['model']==args.candidate_label} == {500}
    assert (output/'plots/trend_mean_test_mae.svg').is_file()


def test_real_checkpoint_native_prediction_on_rebar(tmp_path):
    import torch
    from scripts import eval_corrosion_datasets as ev
    from tests.test_combined_selected_pairs import config
    from tabicl.train.run_combined_selected_pairs import CombinedSelectedPairTrainer

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trainer = CombinedSelectedPairTrainer(config(tmp_path))
        trainer.save_checkpoint('step-0.ckpt')
        checkpoint = Path(trainer.config.checkpoint_dir)/'step-0.ckpt'
        task, membership, _ = load_reviewed_task(exploratory=False)
        predictions = []
        result = ev.evaluate_estimator(model_label='combined', model_kind='tabicl',
            estimator_factory=lambda: ev.make_tabicl_regressor(model_path=str(checkpoint),
                checkpoint_version='', device='cpu', n_estimators=2, random_state=42,
                allow_auto_download=False, feat_shuffle_method='latin', norm_methods=None),
            task=task, task_split=task.fixed_split, test_size=.2, random_state=42,
            regression_uncertainty=False, expected_preserve_missing_values=False,
            native_tabicl_baseline=True, prediction_sink=predictions.append)
        assert result['n_train'] == 152 and result['n_test'] == 56
        assert np.isfinite(predictions[0]['prediction_mV']).all()
        assert predictions[0]['task_row_index'].tolist() == membership['query_task_row_indices']
    finally:
        torch.set_num_threads(old_threads)
