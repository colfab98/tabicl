from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from corrosion_datasets.datasets.mpea_corrosion import compare_target_rules as comparison
from scripts.epit_pipeline.calibrate_target_rules import calibrate_family
from scripts.epit_pipeline.target_rules import (
    CoupledBreakdownRule, CrMoWNSynergyRule, PrenNLinearRule, ThresholdSaturationRule,
)
from scripts.mpea_pipeline.target_rules import MATERIALS, get_mpea_rules


def test_loader_only_parses_development_and_separates_responses(monkeypatch):
    original = pd.read_csv
    seen = []

    def guarded(path, *args, **kwargs):
        seen.append(path.name)
        assert path.name == "development.csv"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded)
    dataset, rows, data, frozen = comparison.load_development()
    assert seen == ["development.csv"]
    assert dataset.n_rows == 214 and not len(rows.final_test)
    assert [len(indices) for indices in rows.validation_by_fold.values()] == [43, 43, 43, 43, 42]
    assert all(comparison.TARGET not in row for row in dataset.rows)
    outer_ids = {row["row_id"] for row in frozen.manifest["rows"] if row["outer_split"] == "final_test"}
    assert set(data.row_id).isdisjoint(outer_ids)


@pytest.mark.parametrize("name,original", [
    ("epit_pren", PrenNLinearRule()),
    ("epit_synergy", CrMoWNSynergyRule()),
    ("epit_threshold", ThresholdSaturationRule()),
    ("epit_coupled_pren", CoupledBreakdownRule()),
])
def test_reduced_material_expression_preserves_existing_terms(name, original):
    # Setting N=0 here checks the algebraic reduction against the old code;
    # actual MPEA missing nitrogen is never imputed to zero by this pipeline.
    matrix = np.asarray([[0, 20, 0, 0, 0], [12, 0, 1, 0.5, 0], [30, 0, 5, 3, 0]], dtype=float)
    values = {"Cr": matrix[:, 0], "Mo": matrix[:, 2], "W": matrix[:, 3]}
    expected = original._material_score(matrix if isinstance(original, PrenNLinearRule) else matrix[:, :4])
    np.testing.assert_allclose(MATERIALS[name].score(values), expected, atol=1e-12)


def test_context_preprocessing_ignores_query_values_and_absent_effects():
    dataset, rows, _, _ = comparison.load_development()
    validation = rows.validation_by_fold[1]
    context = np.setdiff1d(rows.development, validation)
    for rule in get_mpea_rules():
        prepared = rule.fit_terms(dataset, context)
        if "CP_Cl" in rule.required_columns:
            expected = np.nanmean([dataset.rows[i]["CP_Cl"] for i in context])
            assert prepared.state["imputation_means"]["CP_Cl"] == pytest.approx(expected)
        modified_rows = copy.deepcopy(dataset.rows)
        for i in validation:
            modified_rows[i]["Cr"] = 1000000.0
            modified_rows[i]["CP_Cl"] = 1000000.0
        modified = replace(dataset, rows=modified_rows)
        assert rule.fit_terms(modified, context).state == prepared.state
        batch = rule.transform_terms(dataset, validation, prepared.state)
        single = np.vstack([rule.transform_terms(dataset, np.asarray([i]), prepared.state) for i in validation])
        np.testing.assert_allclose(batch, single, atol=1e-12)
        absent_rows = copy.deepcopy(dataset.rows)
        for row in absent_rows:
            for column in ("N", "S", "CP_pH", "CP_temp", "CP_Br", "CP_NO3", "CP_SO4"):
                row[column] = 1000000.0
        absent_dataset = replace(dataset, rows=absent_rows)
        np.testing.assert_allclose(rule.transform_terms(absent_dataset, validation, prepared.state), batch)


def test_validation_targets_cannot_change_own_fold_coefficients():
    dataset, rows, _, _ = comparison.load_development()
    rule = next(rule for rule in get_mpea_rules() if rule.name == "epit_pren_chloride")
    original, _ = calibrate_family(rule, dataset, rows, anchor_strength=0.1)
    changed = dataset.target.copy()
    indices = rows.validation_by_fold[1]
    changed[indices] = changed[indices][::-1] + 100000.0
    altered, _ = calibrate_family(rule, replace(dataset, target=changed), rows, anchor_strength=0.1)
    first, second = original["fold_calibrations"][0], altered["fold_calibrations"][0]
    assert first["coefficients"] == second["coefficients"]
    assert first["feature_state"] == second["feature_state"]


def test_saved_metrics_and_coefficients_match_development_predictions():
    output = comparison.DEFAULT_OUTPUT
    summary = json.loads((output / "comparison_summary.json").read_text())
    predictions = pd.read_csv(output / "oof_predictions.csv")
    data = pd.read_csv(comparison.SPLIT_DIR / "development.csv")
    assert len(summary["results"]) == 19
    assert len(predictions) == 19 * 214
    assert set(predictions.row_id) == set(data.row_id)
    assert not summary["final_test_rows_used"]
    targets = data.set_index("row_id")[comparison.TARGET]
    np.testing.assert_allclose(predictions.observed_epit_mV_SCE, targets.loc[predictions.row_id])
    for result in summary["results"]:
        subset = predictions[predictions.rule_family.eq(result["rule_family"])]
        assert subset.row_id.is_unique
        scores = []
        for fold in result["fold_calibrations"]:
            current = subset[subset.optuna_validation_fold.eq(fold["fold"])]
            rho = spearmanr(current.observed_epit_mV_SCE, current.out_of_fold_rule_score).statistic
            assert fold["spearman"] == pytest.approx(rho, abs=1e-12)
            scores.append(rho)
        assert result["direct_evaluation"]["mean_fold_spearman"] == pytest.approx(np.mean(scores))
        for calibration in result["fold_calibrations"] + [result["final_development_calibration"]]:
            weights = calibration["coefficients"]
            assert sum(weights.values()) == pytest.approx(1.0)
            assert all(0 <= value <= result["coefficient_upper_bounds"][name] + 1e-10 for name, value in weights.items())


def test_audit_identifies_missing_inputs_and_method_reduction():
    _, _, data, _ = comparison.load_development()
    audit = comparison.original_rule_audit(data, get_mpea_rules())
    assert len(audit) == 18
    assert all(not row["directly_evaluable_unchanged"] for row in audit)
    method = next(row for row in audit if row["original_rule"] == "method_aware_pren_n")
    assert method["reduced_candidates"] == ["epit_synergy_interaction"]
    assert "test_method" in method["entirely_unrecorded_columns"]
