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
