"""Reviewed Soccol calibration must preserve formulas and the held-out boundary."""

from copy import deepcopy
from dataclasses import replace
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from corrosion_datasets.datasets.soccol_pitting_potential import compare_target_rules as formulas
from scripts.epit_pipeline.artifact_hashes import sha256_file
from scripts.soccol_pipeline.calibrate_target_rules import DEFAULT_OUTPUT
from scripts.soccol_pipeline.data import load_soccol_dataset, SPLIT_MANIFEST
from scripts.soccol_pipeline.formula_data import load_formula_data
from scripts.soccol_pipeline.run_optuna import load_reviewed_target_rule_config
from scripts.soccol_pipeline.target_rules import RULE_SELECTION, PROMOTED_RULES, TERM_RENAMES, rename_terms
from tabicl.prior.dataset import SCMPrior
from tabicl.prior.soccol_feature_profile import load_soccol_feature_profile
from tabicl.prior.soccol_reviewed_schema import SOCCOL_REVIEWED_FEATURE_PROFILE, SOCCOL_REVIEWED_FEATURE_COLUMNS


@pytest.fixture(scope="module")
def development():
    return load_formula_data()


@pytest.fixture(scope="module")
def summary():
    return json.loads((DEFAULT_OUTPUT / "calibration_summary.json").read_text())


def _artifact(family):
    return json.loads((DEFAULT_OUTPUT / f"{family}.json").read_text())


def test_saved_inputs_and_predictions_cover_exactly_the_accepted_frozen_folds(development, summary):
    assert summary["development_rows"] == 2950
    assert summary["withheld_development_rows_excluded"] == 272
    assert summary["final_test_rows_excluded"] == 805
    assert summary["final_test_targets_used"] is False
    assert summary["development_selection"] == development.attrs["soccol_development_selection"]
    assert [r["rule_family"] for r in summary["rules"]] == list(PROMOTED_RULES)
    oof = pd.read_csv(DEFAULT_OUTPUT / "oof_predictions.csv")
    assert len(oof) == 2950 * 9
    for family, rows in oof.groupby("rule_family"):
        assert len(rows) == 2950 and rows.row_id.is_unique
        assert set(rows.row_id) == set(development.row_id)
        expected = development.set_index("row_id").loc[rows.row_id]
        np.testing.assert_array_equal(rows.fold, expected.development_validation_fold)
        np.testing.assert_array_equal(rows.task_row_index, expected.task_row_index)
    for path, digest in summary["output_sha256"].items():
        assert sha256_file(DEFAULT_OUTPUT / path) == digest


@pytest.mark.parametrize("family,source_name", RULE_SELECTION)
def test_saved_fold_states_and_coefficients_reproduce_held_out_scores(development, family, source_name):
    artifact = _artifact(family)
    spec = next(s for s in formulas.RULES if s.name == source_name)
    assert artifact["coefficient_anchor"] == rename_terms(formulas._coefficient_dict(spec, spec.anchor))
    assert artifact["coefficient_upper_bounds"] == rename_terms(formulas._coefficient_dict(spec, spec.upper_bounds))
    predictions = pd.read_csv(DEFAULT_OUTPUT / "oof_predictions.csv", float_precision="round_trip")
    scores = []
    for fold in artifact["fold_calibrations"]:
        context = development.loc[development.development_validation_fold.ne(fold["fold"])]
        validation = development.loc[development.development_validation_fold.eq(fold["fold"])]
        assert len(context) == fold["context_rows"] and len(validation) == fold["validation_rows"]
        _, expected_state = formulas._fit_term_state(context, spec)
        assert fold["feature_state"] == expected_state
        weights = np.array([fold["coefficients"][TERM_RENAMES.get(name, name)] for name in spec.term_names])
        assert np.all(weights >= 0) and weights.sum() == pytest.approx(1.)
        assert np.all(weights <= spec.upper_bounds)
        predicted = formulas._transform_terms(validation, spec, fold["feature_state"]) @ weights
        saved = predictions.loc[(predictions.rule_family == family) & predictions.fold.eq(fold["fold"])].set_index("row_id").loc[validation.row_id]
        np.testing.assert_allclose(predicted, saved.oof_rule_score, rtol=1e-12, atol=1e-12)
        score = formulas._spearman(validation.E_pit, predicted)
        assert score == pytest.approx(fold["spearman"], abs=1e-12)
        scores.append(score)
    assert np.mean(scores) == pytest.approx(artifact["direct_evaluation"]["mean_fold_spearman"])
    assert formulas._fit_term_state(development, spec)[1] == artifact["final_development_calibration"]["feature_state"]


@pytest.mark.parametrize("family,source_name", RULE_SELECTION)
def test_refitted_formulas_match_the_named_training_implementation(development, family, source_name):
    artifact = _artifact(family)
    spec = next(s for s in formulas.RULES if s.name == source_name)
    fit = artifact["final_development_calibration"]
    terms = formulas._transform_terms(development, spec, fit["feature_state"])
    coefficients = fit["coefficients"]
    weights = [coefficients[TERM_RENAMES.get(name, name)] for name in spec.term_names]
    expected = formulas._standardize(terms @ weights)
    profile = load_soccol_feature_profile(SOCCOL_REVIEWED_FEATURE_PROFILE)
    prior = SCMPrior(batch_size=1, fixed_hp={}, sampled_hp={}, n_jobs=1, device="cpu")
    rule = {"target_rule_family": family, "target_rule_coefficients": coefficients,
            "soccol_feature_columns": list(SOCCOL_REVIEWED_FEATURE_COLUMNS), "target_mix_weight": 1.}
    target, _ = prior._evaluate_soccol_target_rule_tensor(torch.tensor(profile.latent_feature_values.copy(), dtype=torch.float64), rule)
    np.testing.assert_allclose(target.numpy(), expected, atol=2e-6, rtol=2e-6)


def test_validation_features_and_targets_cannot_change_that_folds_fit(development):
    spec = next(s for s in formulas.RULES if s.name == "new_pren_n_coupled_mns_weak_anions")
    changed = development.copy()
    validation = changed.development_validation_fold.eq(1)
    changed.loc[validation, "E_pit"] = -changed.loc[validation, "E_pit"].to_numpy()[::-1]
    changed.loc[validation, "CP_Cl"] = 17 * changed.loc[validation, "CP_Cl"] + 0.02
    changed.loc[validation, "CP_pH"] = 10.
    result, _ = formulas._evaluate_rule(changed, spec, 0.1)
    original = _artifact("pren_n_coupled_mns_weak_anions")["fold_calibrations"][0]
    actual = result["fold_results"][0]
    assert actual["feature_state"] == original["feature_state"]
    assert rename_terms(actual["coefficients"]) == pytest.approx(original["coefficients"], abs=1e-10)
    assert actual["objective"] == pytest.approx(original["objective"], abs=1e-12)


def test_excluded_rows_cannot_change_formula_inputs():
    data = load_soccol_dataset()
    rows = data.rows.copy()
    excluded = ~rows.use_as_development_context
    for column in data.registry["feature_columns"]:
        rows[column] = rows[column].astype(object)
        rows.loc[excluded, column] = "excluded-value"
    rows.loc[excluded, "E_pit"] = -123456.
    changed = replace(data, rows=rows)
    pd.testing.assert_frame_equal(load_formula_data(changed), load_formula_data(data), check_dtype=False)


def test_reviewed_search_loads_exactly_the_new_coefficients_and_score_probabilities(summary):
    config = load_reviewed_target_rule_config(summary_path=DEFAULT_OUTPUT / "calibration_summary.json", split_manifest_path=SPLIT_MANIFEST)
    assert tuple(config.scores) == PROMOTED_RULES
    assert sum(config.probabilities.values()) == pytest.approx(1.)
    for record in summary["rules"]:
        family = record["rule_family"]
        assert config.coefficients[family] == _artifact(family)["final_development_calibration"]["coefficients"]
        assert all(value <= config.upper_bounds[family][term] for term, value in config.coefficients[family].items())
        assert config.scores[family] == record["mean_fold_spearman"]
        assert config.probabilities[family] == pytest.approx(record["sampling_probability"])


@pytest.mark.parametrize("corruption", ["old_count", "fold_assignment", "artifact_checksum", "score", "row_identity"])
def test_loader_rejects_mixed_or_modified_calibration(tmp_path, summary, corruption):
    for path in DEFAULT_OUTPUT.glob("*.json"):
        shutil.copy2(path, tmp_path / path.name)
    changed = deepcopy(summary)
    if corruption == "old_count":
        changed["development_rows"] = 3222
    elif corruption == "fold_assignment":
        changed["fold_assignments"][0]["context_task_row_indices"][0] = 20
    elif corruption == "score":
        changed["rules"][0]["mean_fold_spearman"] += 0.01
    elif corruption == "row_identity":
        changed["development_selection"]["source_row_ids"][0] = "soccol_final_query"
    else:
        path = tmp_path / changed["rules"][0]["artifact"]
        path.write_text(path.read_text() + " ")
    path = tmp_path / "calibration_summary.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError):
        load_reviewed_target_rule_config(summary_path=path, split_manifest_path=SPLIT_MANIFEST)
