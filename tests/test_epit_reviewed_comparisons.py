"""Exercise reviewed EPIT comparisons using real splits and mock model scores."""
from copy import deepcopy
import json
import sys

import numpy as np
import pytest

from scripts import eval_corrosion_datasets as ev
from scripts.epit_pipeline import evaluate_checkpoint_comparison as comparison
from scripts.epit_pipeline import evaluate_baseline_folds as baselines
from scripts.epit_pipeline.artifact_hashes import load_frozen_split, build_final_model_lock, sha256_file
from scripts.epit_pipeline.evaluation_identity import frozen_membership
from tabicl.prior.epit_expanded_schema import EPIT_EXPANDED_FEATURE_COLUMNS, EPIT_EXPANDED_CATEGORICAL_COLUMNS


@pytest.fixture
def mock_models(monkeypatch):
    calls = []
    def evaluate(**kwargs):
        task, split = kwargs['task'], kwargs['task_split']
        assert tuple(task.X.columns) == EPIT_EXPANDED_FEATURE_COLUMNS
        if split.split_strategy.startswith('epit_pipeline_final_test'):
            baseline = kwargs['model_label'] in {'generic_baseline', 'pretrained_tabicl_v2'}
            assert kwargs['native_tabicl_baseline'] == baseline
            assert ev.should_preserve_feature_schema(
                kwargs['model_kind'], split, kwargs['preserve_feature_schema'], native_baseline=baseline,
            ) == kwargs['model_label'].startswith('epit_')
        indices = np.arange(len(task.X))
        if split.split_strategy.startswith('epit_pipeline_development_fold_'):
            indices = np.asarray(frozen_membership(comparison.DEFAULT_SPLIT_MANIFEST)['context_task_row_indices'])
        calls.append((kwargs['model_label'], indices[split.train_index].tolist(), indices[split.test_index].tolist()))
        return {
            'model': kwargs['model_label'], 'model_kind': kwargs['model_kind'], 'task_id': task.task_id,
            'target_binning': 'continuous', 'n_samples': len(task.y), 'n_features': 70,
            'n_train': len(split.train_index), 'n_test': len(split.test_index),
            'split_strategy': split.split_strategy, 'pitting_magpie_features': False,
            **{metric: .3 for metric in ('test_spearman', 'test_mae', 'test_rmse', 'test_r2', 'test_pearson', 'test_nmae_iqr', 'test_nrmse_iqr')},
        }
    def invoke(command, **kwargs):
        with monkeypatch.context() as local:
            local.setattr(sys, 'argv', command[1:])
            ev.main()
    monkeypatch.setattr(ev, 'evaluate_estimator', evaluate)
    monkeypatch.setattr(comparison.subprocess, 'run', invoke)
    return calls


@pytest.mark.parametrize('context_folds', [None, 5])
def test_comparisons_keep_all_frozen_queries(tmp_path, mock_models, context_folds):
    candidate, generic = tmp_path / 'new_epit', tmp_path / 'generic'
    for path, steps in [(candidate, [500, 1400]), (generic, [500, 1000, 2000])]:
        path.mkdir()
        for step in steps:
            (path / f'step-{step}.ckpt').write_bytes(b'mock')
    argv = ['--candidate-run', str(candidate), '--generic-run', str(generic), '--output-dir', str(tmp_path / 'results')]
    if context_folds:
        argv += ['--context-folds', '5']
    args = comparison.parse_args(argv)
    command = comparison.evaluation_command(args)
    categories = [command[i+1] for i, x in enumerate(command) if x == '--catboost-categorical-column']
    assert categories == list(EPIT_EXPANDED_CATEGORICAL_COLUMNS)
    output = comparison.run(args)
    expected = frozen_membership(args.split_manifest)
    for _, context, queries in mock_models:
        assert queries == expected['query_task_row_indices']
        assert len(queries) == 152
        assert set(context) <= set(expected['context_task_row_indices'])
        assert len(context) in ([121, 122] if context_folds else [608])
    payload = json.loads((output / 'results.json').read_text())
    assert {r['checkpoint_step'] for r in payload['rows'] if r['model'] == 'epit_reviewed'} == {500, 1400}
    assert {r['checkpoint_step'] for r in payload['rows'] if r['model'] == 'generic_baseline'} == {500, 1000, 2000}
    if not context_folds:
        bad = deepcopy(payload)
        bad['epit_reviewed_split']['query_task_row_indices'][0] = -1
        with pytest.raises(RuntimeError, match='membership'):
            comparison.validate_results(args, bad)
    with pytest.raises(FileExistsError):
        comparison.run(args)


def test_baselines_use_only_development_queries(tmp_path, mock_models):
    checkpoint = tmp_path / 'step-1000.ckpt'
    checkpoint.write_bytes(b'mock')
    args = baselines.parse_args(['--generic-checkpoint', str(checkpoint), '--output-dir', str(tmp_path / 'baseline')])
    baselines.run(args)
    assert len(mock_models) == 15
    frozen = frozen_membership(args.split_manifest)
    for model, context, queries in mock_models:
        assert len(context) + len(queries) == 608
        assert not set(queries) & set(frozen['query_task_row_indices'])
        assert set(context) | set(queries) == set(frozen['context_task_row_indices'])


def test_manifest_selection_and_preview(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / 'new_run' / 'step-1000.ckpt'
    checkpoint.parent.mkdir(); checkpoint.write_bytes(b'mock')
    split = load_frozen_split(comparison.DEFAULT_SPLIT_MANIFEST)
    manifest = {
        'schema_version': 'epit_final_model_manifest_v1', 'final_test_rows_used': False,
        'split': {'manifest_sha256': split.manifest_sha256, 'lock_sha256': split.lock_sha256},
        'selected_checkpoint': {'path': str(checkpoint), 'sha256': sha256_file(checkpoint)},
        'evaluation_configuration': {'task_id': comparison.EPIT_TASK_ID, 'epit_expanded_features': True,
                                     'expected_n_features': 70, 'tabicl_norm_methods': ['none']},
    }
    path = tmp_path / 'final_model_manifest.json'
    def save():
        path.write_text(json.dumps(manifest))
        lock = build_final_model_lock(manifest_path=path, immutable_artifacts={'selected_checkpoint': checkpoint})
        (tmp_path / 'final_model_lock.json').write_text(json.dumps(lock))
    save()
    args = comparison.parse_args(['--final-model-manifest', str(path)])
    assert comparison.candidate_run(args) == checkpoint.parent
    manifest['evaluation_configuration']['expected_n_features'] = 28
    save()
    with pytest.raises(RuntimeError, match='reviewed EPIT'):
        comparison.candidate_run(args)
    comparison.run(comparison.parse_args(['--candidate-run', str(tmp_path / 'future_run'), '--dry-run']))
    assert '--epit-expanded-features' in capsys.readouterr().out


@pytest.mark.parametrize('context_folds', [None, 5])
def test_four_ratios_share_one_comparison(tmp_path, mock_models, context_folds):
    steps = [(500, 1000), (500, 1400), (500, 1500, 2300), (500, 1750), (500, 1000, 2000)]
    labels = [*comparison.RATIO_LABELS, 'generic_baseline']
    directories = []
    for label, checkpoints in zip(labels, steps):
        path = tmp_path / label
        path.mkdir()
        for step in checkpoints:
            (path / f'step-{step}.ckpt').write_bytes(b'mock')
        directories.append(path)
    argv = ['--ratio-runs', *map(str, directories[:4]), '--generic-run', str(directories[4]),
            '--output-dir', str(tmp_path / 'results')]
    if context_folds:
        argv += ['--context-folds', '5']
    options = comparison.parse_args(argv)
    output = comparison.run(options)
    payload = json.loads((output / 'results.json').read_text())
    assert {row['model'] for row in payload['rows']} == {*labels, 'pretrained_tabicl_v2', 'catboost'}
    for label, expected in zip(labels, steps):
        assert {row['checkpoint_step'] for row in payload['rows'] if row['model'] == label} == set(expected)
    frozen = frozen_membership(options.split_manifest)
    assert all(queries == frozen['query_task_row_indices'] for _, _, queries in mock_models)
    assert all(len(context) in ([121, 122] if context_folds else [608]) for _, context, _ in mock_models)
    assert all((output / 'plots' / name).is_file() for name in comparison.REQUIRED_PLOTS)


def test_four_ratio_protocol_matches_combined_builder():
    from scripts.combined_corrosion import evaluate_final as combined
    options = comparison.parse_args(['--ratio-runs', '/tmp/r25', '/tmp/r50', '/tmp/r75', '/tmp/r100'])
    epit_args = ev.parse_args(comparison.evaluation_command(options)[2:])
    combined_args = ev.parse_args(combined.evaluation_command(combined.parse_args(['--device', 'cpu']))[2:])
    changed = {'run', 'local_model_label', 'fixed_schema_model', 'missing_aware_model', 'task',
               'combined_data_dir', 'epit_split_manifest', 'epit_final_test', 'epit_expanded_features',
               'catboost_categorical_column', 'output_json', 'output_csv', 'output_wide_csv',
               'output_summary_csv', 'output_plot_dir'}
    assert {k: v for k, v in vars(epit_args).items() if k not in changed} == {
        k: v for k, v in vars(combined_args).items() if k not in changed}
    for label in comparison.RATIO_LABELS:
        assert ev.tabicl_inference_settings(epit_args, label) == ('none', ['none'])
        assert ev.expected_missing_value_mode(epit_args, label) is True
    for label, pretrained in [('generic_baseline', False), ('pretrained_tabicl_v2', True)]:
        assert ev.tabicl_inference_settings(epit_args, label, pretrained=pretrained) == ('latin', None)
        assert ev.expected_missing_value_mode(epit_args, label, pretrained=pretrained) is False


def test_four_ratios_reject_duplicate_paths():
    with pytest.raises(SystemExit):
        comparison.parse_args(['--ratio-runs', '/tmp/r25', '/tmp/r25', '/tmp/r75', '/tmp/r100'])
