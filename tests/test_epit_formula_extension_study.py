from dataclasses import replace
import json

import numpy as np
import pytest

from scripts.epit_pipeline import formula_extension_study as formulas
from scripts.epit_pipeline.calibrate_target_rules import DEFAULT_SPLIT_DIR


@pytest.fixture(scope='module')
def study():
    return formulas.load_study(DEFAULT_SPLIT_DIR/'split_manifest.json')[0]


def test_only_development_rows_and_allowlisted_inputs(study):
    manifest = json.loads((DEFAULT_SPLIT_DIR/'split_manifest.json').read_text())
    final = [r['task_row_index'] for r in manifest['rows'] if r['outer_split'] == 'final_test']
    assert len(study.y) == 452
    assert not np.intersect1d(study.rows, final).size
    assert np.isnan(study.dataset.target[final]).all()
    assert not any('Epit' in name or name in {'Reference', 'No.', 'task_row_index', 'row_id'} for name in study.frame)


def test_missing_categories_do_not_become_a_missingness_predictor():
    raw = {'signal': np.arange(50, dtype=float), 'constant': np.r_[np.ones(30), np.full(20, np.nan)]}
    cats = {'condition': np.array(['a']*20+['b']*20+['unseen']*5+['']*5)}
    state, excluded = formulas.fit_feature_state(raw, cats, np.arange(40), ('signal', 'constant'), ('condition',))
    assert [r['term'] for r in excluded] == ['constant']
    X = formulas.transform_features(raw, cats, state, np.arange(40, 50))
    np.testing.assert_allclose(X[:, 1:], 0., atol=1e-12)
    altered_raw = {**raw, 'signal': raw['signal'].copy()}
    altered_raw['signal'][40:] = 1e20
    again, _ = formulas.fit_feature_state(altered_raw, cats, np.arange(40), ('signal', 'constant'), ('condition',))
    assert again == state


def test_coefficient_fit_respects_declared_signs():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(120, 3))
    y = -X[:, 0]+X[:, 1]+X[:, 2]
    weights = formulas.fit_linear(X, y, .1, ['PREN', 'log_chloride', 'pH'], False)
    assert weights[0] >= 0 and weights[1] <= 0
    assert weights[2] > 0


def test_held_out_targets_cannot_change_formula_or_predictions(study):
    training = np.flatnonzero(study.folds != 1)
    held_out = np.flatnonzero(study.folds == 1)
    changed = study.dataset.target.copy()
    changed[study.rows[held_out]] = np.arange(len(held_out))*1e6
    altered = formulas.FormulaStudy(replace(study.dataset, target=changed), study.frame,
                                   study.rows, study.folds, study.groups)
    spec = study.specs['extend_measurement_surface']
    actual, model = study.fit(training, spec, .1)
    after, changed_model = altered.fit(training, spec, .1)
    assert model == changed_model
    np.testing.assert_array_equal(actual, after)


def test_equation_json_roundtrip_and_category_encoding(study):
    training = np.flatnonzero(study.folds != 3)
    for name in ('existing_method_aware_pren_n', 'extend_all', 'new_pren_interactions'):
        spec = study.specs[name]
        expected, model = study.fit(training, spec, None if spec.mode == 'existing' else .03)
        stored = json.loads(json.dumps(model))
        np.testing.assert_allclose(study.predict_artifact(stored), expected, rtol=1e-12, atol=1e-12)
        assert np.isfinite(expected).all()


def test_nested_selection_does_not_use_outer_fold_targets(study, monkeypatch):
    monkeypatch.setattr(formulas, 'PENALTIES', (.03, .3))
    original = formulas.FormulaStudy(study.dataset, study.frame, study.rows, study.folds, study.groups)
    changed = study.dataset.target.copy()
    validation = study.rows[study.folds == 1]
    changed[validation] = np.arange(len(validation))*1000.
    altered = formulas.FormulaStudy(replace(study.dataset, target=changed), study.frame,
                                   study.rows, study.folds, study.groups)
    keep = ('existing_method_aware_pren_n', 'extend_scan')
    for item in (original, altered):
        item.specs = {name: item.specs[name] for name in keep}
    before = formulas.nested_evaluation(original, study.folds, progress=lambda *a, **k: None)
    after = formulas.nested_evaluation(altered, study.folds, progress=lambda *a, **k: None)
    assert [r for r in before[2] if r['fold'] == 1] == [r for r in after[2] if r['fold'] == 1]
    for name in keep:
        assert before[3][name, 1] == after[3][name, 1]
