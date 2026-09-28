from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from scripts.epit_pipeline import diagnose_target_variation as diagnostic
from tabicl.prior.dataset import EPIT_TARGET_RULE_COEFFICIENTS, SCMPrior
from tabicl.prior.target_variation import sample_coefficients


def test_zero_variation_is_exact_and_does_not_consume_rng():
    coefficients = {"a": .56, "b": .40, "c": .04}
    rng = np.random.default_rng(12)
    state = copy.deepcopy(rng.bit_generator.state)
    result = sample_coefficients(coefficients, 0, rng)
    assert result.coefficients == coefficients
    assert result.coefficients is not coefficients
    assert rng.bit_generator.state == state
    assert result.attempts == result.rejected_draws == 0
    assert not result.used_fallback


def test_positive_multipliers_normalization_and_zero_support():
    coefficients = {"a": .56, "b": .40, "c": .04, "zero": 0.0}
    expected = np.array(list(coefficients.values())) * np.random.default_rng(4).uniform(.8, 1.2, 4)
    expected /= expected.sum()
    result = sample_coefficients(coefficients, .2, np.random.default_rng(4))
    np.testing.assert_array_equal(list(result.coefficients.values()), expected)
    assert result.coefficients["zero"] == 0
    assert coefficients == {"a": .56, "b": .40, "c": .04, "zero": 0.0}


def test_bounds_hold_across_draws():
    rng = np.random.default_rng(9)
    for _ in range(200):
        result = sample_coefficients(
            {"a": .55, "b": .4, "c": .05}, .9, rng,
            upper_bounds={"a": .6, "b": .5, "c": .1},
        )
        assert sum(result.coefficients.values()) == pytest.approx(1)
        assert all(0 <= result.coefficients[k] <= bound for k, bound in {"a": .6, "b": .5, "c": .1}.items())


def test_retry_limit_returns_original_and_reports_fallback():
    class AlwaysInvalid:
        def uniform(self, low, high, size):
            return np.array([1.2, .8])

    result = sample_coefficients(
        {"a": .5, "b": .5}, .2, AlwaysInvalid(),
        upper_bounds={"a": .5, "b": 1}, max_attempts=3,
    )
    assert result.coefficients == {"a": .5, "b": .5}
    assert result.used_fallback
    assert result.attempts == result.rejected_draws == 3


@pytest.mark.parametrize("variation", [-.1, 1, float("nan"), float("inf")])
def test_invalid_variation_rejected(variation):
    with pytest.raises(ValueError, match="variation"):
        sample_coefficients({"a": .5, "b": .5}, variation, np.random.default_rng())


@pytest.mark.parametrize("coefficients,bounds", [
    ({}, None), ({"a": -.1, "b": 1.1}, None), ({"a": .2}, None),
    ({"a": float("nan")}, None), ({"a": 1}, {"b": 1}),
    ({"a": 1}, {"a": .9}), ({"a": 1}, {"a": float("inf")}),
])
def test_invalid_coefficients_or_bounds_rejected(coefficients, bounds):
    with pytest.raises(ValueError):
        sample_coefficients(coefficients, 0, np.random.default_rng(), upper_bounds=bounds)


def test_coefficient_rng_is_reproducible_and_independent():
    np.random.seed(77)
    expected_global = np.random.random()
    np.random.seed(77)
    first = diagnostic.variation_rng(12, 3, .2, 4).random(10)
    diagnostic.variation_rng(12, 3, .3, 7).random(100)
    second = diagnostic.variation_rng(12, 3, .2, 4).random(10)
    np.testing.assert_array_equal(first, second)
    assert np.random.random() == expected_global


def test_correlation_handles_constant_targets_without_fake_zero():
    assert diagnostic.correlation([1, 1], [1, 2]) is None
    assert diagnostic.correlation([1, np.nan], [1, 2]) is None
    assert diagnostic.correlation([1, 2, 3], [1, 2, 3], ranks=True) == 1
    metrics = diagnostic.compare_targets([1, 2, 3], [1, 2, 3])
    assert metrics["mae"] == metrics["rmse"] == metrics["max_abs_change"] == 0


@pytest.fixture(scope="module")
def saved_settings():
    return diagnostic.load_settings(
        diagnostic.DEFAULT_MANIFEST,
        diagnostic.search.DEFAULT_TARGET_RULE_SUMMARY,
        diagnostic.search.DEFAULT_SPLIT_MANIFEST,
    )


@pytest.mark.parametrize("family", sorted(EPIT_TARGET_RULE_COEFFICIENTS))
def test_paired_diagnostic_reuses_real_rule_and_mixer(monkeypatch, saved_settings, family):
    fixed, bounds, _, rules = saved_settings
    hp = copy.deepcopy(fixed)
    hp["pitting_target_rule_scores"] = {family: rules.scores[family]}
    prior = diagnostic.DiagnosticPrior(batch_size=1, fixed_hp=hp, sampled_hp={}, n_jobs=1)
    calls = []

    def scm_stub(self, X, params, prior_cls):
        calls.append(1)
        return torch.linspace(-2, 2, len(X)) ** 3

    monkeypatch.setattr(SCMPrior, "_generate_scm_target_from_physical_features", scm_stub)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    params = diagnostic.make_params(prior, "mlp_scm", 64)
    table, terms, measurements, coefficients = diagnostic.inspect_table(
        prior, params, 0, bounds[family], [0, .1, .3], 2, 23, 100,
    )
    assert len(calls) == 1
    assert table["family"] == family
    assert len(measurements) == 5
    baseline = measurements[0]
    assert baseline["rule_mae"] == baseline["mixed_mae"] == 0
    assert baseline["rule_spearman"] == baseline["mixed_spearman"] == 1
    assert all(row["sampled"] <= row["upper_bound"] for row in coefficients)
    assert len(terms) == len(bounds[family]) * (len(bounds[family]) - 1) // 2
    assert prior.replay_rule is None
    before_np = copy.deepcopy(np.random.get_state())
    before_torch = torch.random.get_rng_state().clone()
    before_python = random.getstate()
    prior.replay(copy.deepcopy(prior.last_pitting_target_rule), params)
    after_np = np.random.get_state()
    assert before_np[0] == after_np[0] and before_np[2:] == after_np[2:]
    np.testing.assert_array_equal(before_np[1], after_np[1])
    assert torch.equal(before_torch, torch.random.get_rng_state())
    assert before_python == random.getstate()


def test_failure_sentinel_kept_separate(monkeypatch, saved_settings):
    fixed, bounds, _, rules = saved_settings
    fixed = copy.deepcopy(fixed)
    fixed["pitting_target_rule_scores"] = {"pren_linear": rules.scores["pren_linear"]}
    prior = diagnostic.DiagnosticPrior(batch_size=1, fixed_hp=fixed, sampled_hp={}, n_jobs=1)
    monkeypatch.setattr(SCMPrior, "_generate_scm_target_from_physical_features",
                        lambda self, X, params, prior_cls: torch.full((len(X),), -100.0))
    table, _, rows, _ = diagnostic.inspect_table(
        prior, diagnostic.make_params(prior, "mlp_scm", 64), 0, bounds["pren_linear"], [0, .2], 1, 4, 100,
    )
    assert table["scm_status"] == "failure_sentinel"
    assert table["scm_rule_pearson"] is None
    assert all(row["scm_status"] == "failure_sentinel" for row in diagnostic.summarize(rows))


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch, saved_settings):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("unchanged")
    monkeypatch.setattr(diagnostic, "load_settings", lambda *args: saved_settings)
    with pytest.raises(FileExistsError):
        diagnostic.run(diagnostic.parse_args(["--output-dir", str(output)]))
    assert sentinel.read_text() == "unchanged"
    assert list(output.iterdir()) == [sentinel]
