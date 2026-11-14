"""Combined scoring, quotas, and calibration leakage checks; no model evaluation."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.epit_pipeline import evaluate_formula_mixture as diagnostic


def test_assigns_exact_quotas_and_is_reproducible():
    p = np.array([.2, .3, .5])
    actual = diagnostic.assign_formulas(101, p, np.random.default_rng(42))
    np.testing.assert_array_equal(np.bincount(actual), [20, 30, 51])
    np.testing.assert_array_equal(actual, diagnostic.assign_formulas(101, p, np.random.default_rng(42)))
    assert not np.array_equal(actual, diagnostic.assign_formulas(101, p, np.random.default_rng(43)))
    with pytest.raises(ValueError):
        diagnostic.assign_formulas(20, [1, 1], np.random.default_rng(42))


def test_mv_calibration_cannot_use_query_labels():
    scores = np.array([0., 1., 2., 3.])
    labels = np.array([10., 20., 900., -500.])
    predictions, fit = diagnostic.calibrate_mv(scores, labels, np.array([0, 1]))
    np.testing.assert_allclose(predictions, [10, 20, 30, 40])
    labels[2:] = [np.nan, np.inf]
    other, other_fit = diagnostic.calibrate_mv(scores, labels, np.array([0, 1]))
    np.testing.assert_array_equal(other, predictions)
    assert fit == other_fit


def test_end_to_end_combines_one_formula_per_sample(tmp_path, monkeypatch):
    n = 20
    rows = pd.DataFrame({'row_id': [f'r{i}' for i in range(n)], 'task_row_index': range(n),
                         'No.': range(n), 'Material class': ['Fe'] * n, 'Reference': [1] * n})
    reviewed = SimpleNamespace(rows=rows, assignments=pd.DataFrame({'outer_split': ['development'] * 10 + ['final_test'] * 10}))
    y = np.arange(n, dtype=float)
    rules = SimpleNamespace(artifact_sha256s={})
    monkeypatch.setattr(diagnostic, 'load_inputs', lambda: (rules, {}, reviewed, SimpleNamespace(target=y), {}, np.arange(10), ['a', 'b'], np.array([.5, .5])))
    predictions = np.column_stack([y, y + 100])
    monkeypatch.setattr(diagnostic, 'formula_predictions', lambda *args: (predictions, predictions, []))
    monkeypatch.setattr(diagnostic, 'sha256_file', lambda path: 'fake')
    output = tmp_path / 'results'
    diagnostic.run(SimpleNamespace(repeats=3, seed=42, output_dir=output, dry_run=False))
    exported = pd.read_csv(output / 'assigned_predictions.csv')
    assert len(exported) == 3 * n
    assert not exported.duplicated(['repeat', 'task_row_index']).any()
    np.testing.assert_allclose(exported.prediction_mV, exported.task_row_index + 100 * exported.formula.eq('b'))
    scores = pd.read_csv(output / 'repeat_metrics.csv')
    np.testing.assert_allclose(scores.mae_mV, 50)
    # Combined ranks, not an average of the two perfect per-formula correlations.
    assert scores.spearman.lt(1).all()
    config = json.loads((output / 'config.json').read_text())
    assert config['calibration_task_row_indices'] == list(range(10))
    with pytest.raises(FileExistsError):
        diagnostic.run(SimpleNamespace(repeats=1, seed=42, output_dir=output, dry_run=False))
