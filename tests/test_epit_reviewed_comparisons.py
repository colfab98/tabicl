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
            baseline = 'tabicl' in kwargs['model_kind']
            assert kwargs['native_tabicl_baseline'] == baseline
            assert ev.should_preserve_feature_schema(
                kwargs['model_kind'], split, kwargs['preserve_feature_schema'], native_baseline=baseline,
            ) is False
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
    argv = ['--include-latest-checkpoint', '--candidate-run', str(candidate), '--generic-run', str(generic), '--output-dir', str(tmp_path / 'results')]
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
@pytest.mark.parametrize('max_step', [None, 5000])
def test_five_ratios_share_one_comparison(tmp_path, mock_models, context_folds, max_step):
    steps = [(500, 1000, 5000, 5500), (500, 1400, 4800, 5500),
             (500, 1500, 2300, 5000, 6000), (500, 1750, 5200),
             (500, 1000, 2000, 5000, 7000), (500, 1000, 2000, 4500, 6000)]
    labels = [*comparison.RATIO_LABELS, 'generic_baseline']
    directories = []
    for label, checkpoints in zip(labels, steps):
        path = tmp_path / label
        path.mkdir()
        for step in checkpoints:
            (path / f'step-{step}.ckpt').write_bytes(b'mock')
        directories.append(path)
    argv = ['--ratio-runs', *map(str, directories[:5]), '--generic-run', str(directories[5]),
            '--checkpoint-step-interval', '0', '--output-dir', str(tmp_path / 'results')]
    if max_step is not None:
        argv += ['--max-checkpoint-step', str(max_step)]
    if context_folds:
        argv += ['--context-folds', '5']
    options = comparison.parse_args(argv)
    output = comparison.run(options)
    payload = json.loads((output / 'results.json').read_text())
    assert {row['model'] for row in payload['rows']} == {*labels, 'pretrained_tabicl_v2', 'catboost'}
    assert payload['max_checkpoint_step'] == max_step
    for label, expected in zip(labels, steps):
        assert {row['checkpoint_step'] for row in payload['rows'] if row['model'] == label} == {
            step for step in expected if max_step is None or step <= max_step}
    if max_step is not None:
        assert max(row['checkpoint_step'] for row in payload['rows']) == max_step
        if context_folds is None:
            bad = deepcopy(payload)
            bad['rows'][0]['checkpoint_step'] = 5500
            with pytest.raises(RuntimeError, match='maximum checkpoint step'):
                comparison.validate_results(options, bad)
    frozen = frozen_membership(options.split_manifest)
    assert all(queries == frozen['query_task_row_indices'] for _, _, queries in mock_models)
    assert all(len(context) in ([121, 122] if context_folds else [608]) for _, context, _ in mock_models)
    assert all((output / 'plots' / name).is_file() for name in comparison.REQUIRED_PLOTS)


def test_five_ratio_protocol_matches_combined_builder():
    from scripts.combined_corrosion import evaluate_final as combined
    options = comparison.parse_args(['--ratio-runs', '/tmp/r0', '/tmp/r25', '/tmp/r50', '/tmp/r75', '/tmp/r100'])
    epit_args = ev.parse_args(comparison.evaluation_command(options)[2:])
    combined_args = ev.parse_args(combined.evaluation_command(combined.parse_args(['--device', 'cpu']))[2:])
    changed = {'run', 'local_model_label', 'fixed_schema_model', 'missing_aware_model', 'task',
               'combined_data_dir', 'epit_split_manifest', 'epit_final_test', 'epit_expanded_features',
               'catboost_categorical_column', 'output_json', 'output_csv', 'output_wide_csv',
               'output_summary_csv', 'output_plot_dir', 'include_latest_common_checkpoint'}
    assert epit_args.include_latest_common_checkpoint is False
    assert combined_args.include_latest_common_checkpoint is True
    assert {k: v for k, v in vars(epit_args).items() if k not in changed} == {
        k: v for k, v in vars(combined_args).items() if k not in changed}
    for label in comparison.RATIO_LABELS:
        assert ev.tabicl_inference_settings(epit_args, label) == ('latin', None)
        assert ev.expected_missing_value_mode(epit_args, label) is False
    for label, pretrained in [('generic_baseline', False), ('pretrained_tabicl_v2', True)]:
        assert ev.tabicl_inference_settings(epit_args, label, pretrained=pretrained) == ('latin', None)
        assert ev.expected_missing_value_mode(epit_args, label, pretrained=pretrained) is False


def test_five_ratios_reject_duplicate_paths():
    with pytest.raises(SystemExit):
        comparison.parse_args(['--ratio-runs', '/tmp/r0', '/tmp/r25', '/tmp/r25', '/tmp/r75', '/tmp/r100'])


@pytest.mark.parametrize('case,context_folds', [
    ('missing', None), ('missing', 5), ('empty', None),
    ('out_of_range', None), ('explicit_checkpoint', None), ('missing_generic', None),
])
def test_unavailable_model_is_skipped_without_relabeling_or_changing_queries(
    tmp_path, mock_models, capsys, case, context_folds,
):
    labels = [*comparison.RATIO_LABELS, 'generic_baseline']
    skipped_label = 'generic_baseline' if case == 'missing_generic' else 'epit_25pct'
    directories = [tmp_path / label for label in labels]
    for label, path in zip(labels, directories):
        if label == skipped_label and case in {'missing', 'missing_generic'}:
            continue
        path.mkdir()
        steps = [500, 1000]
        if label == skipped_label:
            steps = {'empty': [], 'out_of_range': [100], 'explicit_checkpoint': [500]}[case]
        for step in steps:
            (path / f'step-{step}.ckpt').write_bytes(b'mock')
    argv = ['--ratio-runs', *map(str, directories[:5]), '--generic-run', str(directories[5]),
            '--min-checkpoint-step', '500', '--output-dir', str(tmp_path / 'results')]
    if case == 'explicit_checkpoint':
        argv += ['--checkpoint', '1000']
    else:
        argv += ['--max-checkpoint-step', '1000']
    if context_folds:
        argv += ['--context-folds', '5']
    args = comparison.parse_args(argv)
    output = comparison.run(args)
    payload = json.loads((output / 'results.json').read_text())
    expected_labels = set(labels) - {skipped_label} | {'pretrained_tabicl_v2', 'catboost'}
    assert {row['model'] for row in payload['rows']} == expected_labels
    assert [record['model'] for record in payload['skipped_models']] == [skipped_label]
    assert f'Skipping {skipped_label}:' in capsys.readouterr().out
    frozen = frozen_membership(args.split_manifest)
    assert all(queries == frozen['query_task_row_indices'] for _, _, queries in mock_models)
    assert all(len(context) in ([121, 122] if context_folds else [608]) for _, context, _ in mock_models)
    if args.checkpoint == 'all':
        assert all((output / 'plots' / name).is_file() for name in comparison.REQUIRED_PLOTS)


def test_comparison_rejects_no_available_local_models(tmp_path):
    paths = [tmp_path / label for label in comparison.RATIO_LABELS]
    output = tmp_path / 'results'
    args = comparison.parse_args(['--ratio-runs', *map(str, paths),
        '--generic-run', str(tmp_path / 'missing_generic'), '--output-dir', str(output)])
    with pytest.raises(FileNotFoundError, match='No local models have checkpoints available'):
        comparison.run(args)
    assert not output.exists()


@pytest.mark.parametrize('extra', [
    ['--max-checkpoint-step', '-1'],
    ['--min-checkpoint-step', '6000', '--max-checkpoint-step', '5000'],
    ['--checkpoint', 'latest', '--max-checkpoint-step', '5000'],
])
def test_comparison_rejects_invalid_checkpoint_caps(extra):
    with pytest.raises(SystemExit):
        comparison.parse_args(['--candidate-run', '/tmp/future_run', *extra])


def test_exploratory_cv_covers_each_row_once(tmp_path, mock_models, monkeypatch):
    from scripts.epit_pipeline.exploratory_cv import fold_membership
    original_evaluate = ev.evaluate_estimator

    def scored(**kwargs):
        split = kwargs['task_split']
        assert np.array_equal(split.preprocessing_index, split.train_index)
        assert not set(split.preprocessing_index) & set(split.test_index)
        row = original_evaluate(**kwargs)
        fold = int(row['split_strategy'].rsplit('_', 1)[1])
        row['test_mae'] = float(fold) * (2 if row['model'] == 'epit_reviewed' else 1)
        return row

    monkeypatch.setattr(ev, 'evaluate_estimator', scored)
    candidate, generic = tmp_path / 'informed', tmp_path / 'generic'
    for path in (candidate, generic):
        path.mkdir()
        (path / 'step-100.ckpt').write_bytes(b'mock')
        (path / 'step-200.ckpt').write_bytes(b'mock')
    args = comparison.parse_args([
        '--candidate-run', str(candidate), '--generic-run', str(generic),
        '--exploratory-cv-folds', '5', '--checkpoint', 'all',
        '--min-checkpoint-step', '100', '--max-checkpoint-step', '200',
        '--checkpoint-step-interval', '100', '--output-dir', str(tmp_path / 'cv'),
    ])
    command = comparison.evaluation_command(args)
    assert '--exploratory-cv-folds' in command
    assert '--epit-final-test' not in command
    output = comparison.run(args)
    payload = json.loads((output / 'results.json').read_text())
    assert payload['exploratory_cv'] is True
    assert payload['original_final_test_fold'] == 1
    memberships = [fold_membership(args.split_manifest, f) for f in range(1, 6)]
    assert memberships[0] == frozen_membership(args.split_manifest)
    assert memberships == [fold_membership(args.split_manifest, f) for f in range(1, 6)]
    queries = []
    for membership in memberships:
        context, query = membership.values()
        assert len(context) == 608 and len(query) == 152
        assert not set(context) & set(query)
        assert set(context) | set(query) == set(range(760))
        queries.extend(query)
    assert sorted(queries) == list(range(760))
    for _, context, query in mock_models:
        assert {'context_task_row_indices': context, 'query_task_row_indices': query} in memberships
    for summary in payload['summary']:
        assert summary['n_cv_folds'] == 5
        factor = 2 if summary['model'] == 'epit_reviewed' else 1
        assert summary['mean_test_mae'] == pytest.approx(3 * factor)
        assert summary['std_test_mae'] == pytest.approx(factor * np.std([1, 2, 3, 4, 5], ddof=1))
        assert summary['mean_delta_test_mae_vs_pretrained_tabicl_v2'] == pytest.approx(3 * (factor - 1))
    first = json.loads((output / 'cv_fold_1/results.json').read_text())
    assert first['epit_reviewed_split'] == frozen_membership(args.split_manifest)
    assert next(r for r in first['rows'] if r['model'] == 'pretrained_tabicl_v2')['test_mae'] == 1


def test_exploratory_cv_rejects_mixed_modes():
    with pytest.raises(SystemExit):
        comparison.parse_args(['--exploratory-cv-folds', '5', '--context-folds', '5'])
    with pytest.raises(SystemExit):
        comparison.parse_args(['--exploratory-cv-folds', '5', '--legacy-data'])
    args = ev.parse_args(['--exploratory-cv-folds', '5', '--epit-final-test'])
    with pytest.raises(ValueError, match='cannot mix split modes'):
        ev.validate_args(args)


def test_exploratory_cv_incomplete_scores_fail():
    from scripts.epit_pipeline.exploratory_cv import child_arguments
    assert child_arguments(['--exploratory-cv-folds=5', '--output-json=x',
                            '--n-estimators', '8']) == ['--n-estimators', '8']
    rows = [{'model': 'informed', 'task_id': 'epit', 'context_fold': f} for f in range(1, 5)]
    with pytest.raises(RuntimeError, match='every model'):
        ev.require_complete_context_folds(rows, [], [1, 2, 3, 4, 5])
