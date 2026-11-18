"""Regression tests for live checkpoint arrivals and saved-fold recovery."""
import json
import sys

import pytest

from scripts import eval_corrosion_datasets as ev
from scripts import recover_exploratory_cv as recovery
from scripts.epit_pipeline import evaluate_checkpoint_comparison as epit
from scripts.epit_pipeline.exploratory_cv import complete_rows_only
from scripts.soccol_pipeline import evaluate_checkpoint_comparison as soccol


@pytest.mark.parametrize('comparison', [epit, soccol], ids=['epit', 'soccol'])
@pytest.mark.parametrize('scope', ['all', 'common'])
@pytest.mark.parametrize('deletion', ['between_folds', 'during_load', 'loader_value_error', 'unrelated_error'])
def test_deleted_checkpoint_is_skipped_without_partial_cv(tmp_path, monkeypatch, comparison, scope, deletion):
    import xml.etree.ElementTree as ET

    candidate, generic = tmp_path / 'candidate', tmp_path / 'generic'
    for directory in (candidate, generic):
        directory.mkdir()
        for step in (100, 500):
            (directory / f'step-{step}.ckpt').write_bytes(b'mock')
    disappearing = candidate / 'step-100.ckpt'
    label = comparison.parse_args([]).candidate_label
    calls = {}

    def evaluate(**kwargs):
        task, split = kwargs['task'], kwargs['task_split']
        fold = int(split.split_strategy.rsplit('_', 1)[1])
        key = (fold, kwargs['model_label'])
        calls[key] = calls.get(key, 0) + 1
        if fold == 2 and kwargs['model_label'] == label and calls[key] == 1 and disappearing.exists():
            disappearing.unlink()
            if deletion == 'during_load':
                raise FileNotFoundError(2, 'No such file', str(disappearing))
            if deletion == 'unrelated_error':
                raise ValueError('Invalid prediction shape')
            raise ValueError(f"Checkpoint not found at '{disappearing}' and automatic download is disabled.")
        return {'model': kwargs['model_label'], 'model_kind': kwargs['model_kind'],
                'task_id': task.task_id, 'target_binning': 'continuous',
                'n_samples': len(task.y), 'n_features': len(task.X.columns),
                'n_train': len(split.train_index), 'n_test': len(split.test_index),
                'split_strategy': split.split_strategy,
                **{metric: float(fold) for metric in ('test_mae', 'test_rmse', 'test_spearman',
                    'test_pearson', 'test_r2', 'test_nmae_iqr', 'test_nrmse_iqr')}}

    def invoke(command, **kwargs):
        with monkeypatch.context() as local:
            local.setattr(sys, 'argv', command[1:])
            ev.main()
        if deletion == 'between_folds' and '--epit-exploratory-cv-fold' in command:
            if command[command.index('--epit-exploratory-cv-fold') + 1] == '1':
                disappearing.unlink()

    monkeypatch.setattr(ev, 'evaluate_estimator', evaluate)
    monkeypatch.setattr(comparison.subprocess, 'run', invoke)
    args = comparison.parse_args(['--candidate-run', str(candidate), '--generic-run', str(generic),
        '--exploratory-cv-folds', '5', '--checkpoint-step-interval', '0', '--checkpoint-scope', scope,
        '--output-dir', str(tmp_path / 'results')])
    if deletion == 'unrelated_error':
        with pytest.raises(RuntimeError, match='requires every model to complete'):
            comparison.run(args)
        saved = json.loads((tmp_path / 'results/cv_fold_2/results.json').read_text())
        assert 'Invalid prediction shape' in saved['errors'][0]['error']
        return
    output = comparison.run(args)
    payload = json.loads((output / 'results.json').read_text())
    assert not payload['errors']
    assert {r['cv_fold'] for r in payload['skipped_checkpoints']} == {2, 3, 4, 5}
    assert all(r['n_cv_folds'] == 5 and r['mean_test_mae'] == pytest.approx(3)
               for r in payload['summary'])
    assert {r['checkpoint_step'] for r in payload['rows'] if r['model'] == label} == {500}
    assert {r['checkpoint_step'] for r in payload['rows'] if r['model'] == 'generic_baseline'} == (
        {100, 500} if scope == 'all' else {500})
    svg = ET.parse(output / 'plots/trend_mean_test_mae.svg')
    assert len(svg.findall('.//{http://www.w3.org/2000/svg}circle')) == len(payload['summary'])


def test_snapshot_still_rejects_changed_checkpoints(tmp_path):
    from scripts.epit_pipeline.exploratory_cv import load_checkpoint_snapshot, save_checkpoint_snapshot
    checkpoint, snapshot = tmp_path / 'step-100.ckpt', tmp_path / 'snapshot.json'
    checkpoint.write_bytes(b'original')
    specs = [ev.CheckpointEvalSpec('step-100', 100, (ev.LocalModelSpec('model', checkpoint),))]
    save_checkpoint_snapshot(specs, snapshot)
    checkpoint.write_bytes(b'changed contents')
    with pytest.raises(RuntimeError, match='changed since evaluation started'):
        load_checkpoint_snapshot(snapshot, ev)


@pytest.mark.parametrize('comparison', [epit, soccol], ids=['epit', 'soccol'])
def test_new_checkpoints_do_not_change_folds_and_saved_results_recover(tmp_path, monkeypatch, comparison):
    candidate, generic = tmp_path / 'candidate', tmp_path / 'generic'
    for directory, steps in ((candidate, (500, 1200)), (generic, (500, 1500))):
        directory.mkdir()
        for step in steps:
            (directory / f'step-{step}.ckpt').write_bytes(b'mock')

    def evaluate(**kwargs):
        task, split = kwargs['task'], kwargs['task_split']
        fold = int(split.split_strategy.rsplit('_', 1)[1])
        return {'model': kwargs['model_label'], 'model_kind': kwargs['model_kind'],
                'task_id': task.task_id, 'target_binning': 'continuous',
                'n_samples': len(task.y), 'n_features': len(task.X.columns),
                'n_train': len(split.train_index), 'n_test': len(split.test_index),
                'split_strategy': split.split_strategy, 'pitting_magpie_features': False,
                **{metric: float(fold) for metric in ('test_mae', 'test_rmse', 'test_spearman',
                    'test_pearson', 'test_r2', 'test_nmae_iqr', 'test_nrmse_iqr')}}

    def invoke(command, **kwargs):
        with monkeypatch.context() as local:
            local.setattr(sys, 'argv', command[1:])
            ev.main()
        if '--epit-exploratory-cv-fold' in command:
            fold = command[command.index('--epit-exploratory-cv-fold') + 1]
            if fold == '1':
                (candidate / 'step-1300.ckpt').write_bytes(b'arrived during evaluation')

    monkeypatch.setattr(ev, 'evaluate_estimator', evaluate)
    monkeypatch.setattr(comparison.subprocess, 'run', invoke)
    args = comparison.parse_args(['--include-latest-checkpoint', '--candidate-run', str(candidate), '--generic-run', str(generic),
                                  '--exploratory-cv-folds', '5', '--output-dir', str(tmp_path / 'results')])
    output = comparison.run(args)
    payload = json.loads((output / 'results.json').read_text())
    label = args.candidate_label
    assert {r['checkpoint_step'] for r in payload['rows'] if r['model'] == label} == {500, 1200}
    assert len([r for r in payload['rows'] if r['model'] == label and r['checkpoint_step'] == 1200]) == 5
    assert (output / 'cv_checkpoints.json').is_file()
    assert all(row['n_cv_folds'] == 5 for row in payload['summary'])

    # Reproduce the reported old-run failure: one model/checkpoint lacks four folds.
    originals = {}
    for fold in range(1, 6):
        path = output / f'cv_fold_{fold}/results.json'
        saved = json.loads(path.read_text())
        if fold > 1:
            saved['rows'] = [r for r in saved['rows'] if not
                             (r['model'] == label and r['checkpoint_step'] == 1200)]
        path.write_text(json.dumps(saved))
        originals[path] = path.read_bytes()
    monkeypatch.setattr(comparison.subprocess, 'run', lambda *a, **k: pytest.fail('Recovery ran inference'))
    monkeypatch.setattr(sys, 'argv', ['recover_exploratory_cv', '--checkpoint-step-interval', '0', '--output-dir', str(output),
                                    '--split-manifest', str(args.split_manifest)])
    recovery.main()
    recovered = json.loads((output / 'results.json').read_text())
    assert recovered['recovered_from_saved_folds'] is True
    assert recovered['excluded_incomplete_model_checkpoints'] == [
        {'checkpoint_name': 'step-1200', 'model': label, 'task_id': payload['rows'][0]['task_id'], 'folds': [1]}]
    assert all(r['n_cv_folds'] == 5 for r in recovered['summary'])
    assert all(r['mean_test_mae'] == pytest.approx(3) for r in recovered['summary'])
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert (output / 'plots/trend_mean_test_mae.svg').is_file()


def test_recovery_excludes_exact_reported_partial_steps():
    rows = [{'checkpoint_name': f'step-{step}', 'model': model, 'task_id': 'task', 'cv_fold': fold}
            for step, folds in ((1000, range(1, 6)), (1200, [1]), (1300, [2, 3]), (1400, [4, 5]))
            for model in ('informed', 'pretrained') for fold in folds]
    kept, excluded = complete_rows_only(rows)
    assert len(kept) == 10 and len(excluded) == 6
    assert {row['checkpoint_name'] for row in kept} == {'step-1000'}
    with pytest.raises(RuntimeError, match='Duplicate'):
        complete_rows_only([*kept, kept[0]])


@pytest.mark.parametrize('comparison', [epit, soccol], ids=['epit', 'soccol'])
@pytest.mark.parametrize('include_latest', [False, True])
def test_checkpoint_interval_is_strict_unless_opted_in(tmp_path, comparison, include_latest):
    candidate, generic = tmp_path / 'candidate', tmp_path / 'generic'
    for directory in (candidate, generic):
        directory.mkdir()
        for step in (500, 1000, 1200, 1300, 1400):
            (directory / f'step-{step}.ckpt').write_bytes(b'mock')
    argv = ['--candidate-run', str(candidate), '--generic-run', str(generic),
            '--checkpoint-step-interval', '500', '--output-dir', str(tmp_path / 'out')]
    if include_latest:
        argv.append('--include-latest-checkpoint')
    command = comparison.evaluation_command(comparison.parse_args(argv))
    specs = ev.resolve_checkpoint_eval_specs(ev.parse_args(command[2:]))
    assert {spec.checkpoint_step for spec in specs} == ({500, 1000, 1400} if include_latest else {500, 1000})


def test_recovery_applies_strict_interval():
    rows = [{'checkpoint_name': f'step-{step}', 'checkpoint_step': step,
             'model': 'informed', 'task_id': 'task', 'cv_fold': fold}
            for step in (500, 1000, 1200, 1300, 1400) for fold in range(1, 6)]
    kept, dropped = complete_rows_only(rows, step_interval=500)
    assert {r['checkpoint_step'] for r in kept} == {500, 1000}
    assert len(dropped) == 3
    assert all(r['reason'] == 'outside_checkpoint_interval' for r in dropped)


@pytest.mark.parametrize('comparison', [epit, soccol], ids=['epit', 'soccol'])
@pytest.mark.parametrize('scope', ['all', 'common'])
def test_scope_controls_cv_rows_and_every_plotted_point(tmp_path, monkeypatch, comparison, scope):
    import xml.etree.ElementTree as ET
    candidate, generic = tmp_path/'candidate', tmp_path/'generic'
    for directory, steps in ((candidate, [100, 500]), (generic, [500, 1000])):
        directory.mkdir()
        for step in steps:
            (directory/f'step-{step}.ckpt').write_bytes(b'mock')
    def evaluate(**kwargs):
        task, split = kwargs['task'], kwargs['task_split']
        return {'model': kwargs['model_label'], 'model_kind': kwargs['model_kind'],
                'task_id': task.task_id, 'target_binning': 'continuous',
                'n_samples': len(task.y), 'n_features': len(task.X.columns),
                'n_train': len(split.train_index), 'n_test': len(split.test_index),
                'split_strategy': split.split_strategy,
                **{metric: .3 for metric in ('test_mae', 'test_rmse', 'test_spearman',
                    'test_pearson', 'test_r2', 'test_nmae_iqr', 'test_nrmse_iqr')}}
    def invoke(command, **kwargs):
        with monkeypatch.context() as local:
            local.setattr(sys, 'argv', command[1:])
            ev.main()
    monkeypatch.setattr(ev, 'evaluate_estimator', evaluate)
    monkeypatch.setattr(comparison.subprocess, 'run', invoke)
    args = comparison.parse_args(['--candidate-run', str(candidate), '--generic-run', str(generic),
        '--exploratory-cv-folds', '5', '--checkpoint-step-interval', '0', '--checkpoint-scope', scope,
        '--output-dir', str(tmp_path/'results')])
    output = comparison.run(args)
    payload = json.loads((output/'results.json').read_text())
    assert payload['checkpoint_scope'] == scope
    for label, steps in [(args.candidate_label, {100, 500}), ('generic_baseline', {500, 1000})]:
        expected = steps if scope == 'all' else {500}
        assert {row['checkpoint_step'] for row in payload['rows'] if row['model'] == label} == expected
        for step in expected:
            assert {row['cv_fold'] for row in payload['rows']
                    if row['model'] == label and row['checkpoint_step'] == step} == set(range(1, 6))
    svg = ET.parse(output/'plots/trend_mean_test_mae.svg')
    assert len(svg.findall('.//{http://www.w3.org/2000/svg}circle')) == len(payload['summary'])
    manifest = output/'cv_checkpoints.json'
    from scripts.epit_pipeline.exploratory_cv import load_checkpoint_snapshot
    with pytest.raises(ValueError, match='different checkpoint scope'):
        load_checkpoint_snapshot(manifest, ev, expected_scope='common' if scope=='all' else 'all')
    # A missing requested model must not silently narrow "common" to one run.
    if scope == 'common':
        args.generic_run = tmp_path/'missing'
        with pytest.raises(FileNotFoundError, match='every requested model'):
            epit.select_available_models(args, runs=comparison.model_runs(args))
