from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from scripts.epit_pipeline.calibrate_target_rules import (
    DEFAULT_SPLIT_DIR,
    fit_coefficients,
    calibrate_family,
    _load_manifest,
)
from scripts.epit_pipeline.split_data import EpitDataset, load_epit_dataset
from scripts.epit_pipeline.target_rules import get_rule_families, PROMOTED_TARGET_RULE_FAMILIES
from scripts.epit_pipeline.formula_data import load_formula_dataset, FORMULA_ENVIRONMENT_COLUMNS
from scripts.epit_pipeline.composition_data import load_reviewed_compositions


def test_coefficient_fit_is_nonnegative_and_normalized() -> None:
    rng = np.random.default_rng(7)
    terms = rng.normal(size=(300, 3))
    expected = np.asarray([0.65, 0.25, 0.10])
    target = terms @ expected + rng.normal(scale=0.03, size=len(terms))

    fit = fit_coefficients(
        terms,
        target,
        np.full(3, 1.0 / 3.0),
        anchor_strength=0.0,
    )

    assert np.all(fit.coefficients >= 0.0)
    assert np.isclose(fit.coefficients.sum(), 1.0)
    assert fit.coefficients[0] > fit.coefficients[1] > fit.coefficients[2]
    assert spearmanr(target, terms @ fit.coefficients).correlation > 0.98


def test_anchor_regularizes_unidentified_terms() -> None:
    terms = np.zeros((40, 3), dtype=float)
    target = np.arange(40, dtype=float)
    anchor = np.asarray([0.2, 0.3, 0.5])

    fit = fit_coefficients(
        terms,
        target,
        anchor,
        anchor_strength=1.0,
    )

    assert np.allclose(fit.coefficients, anchor, atol=1e-6)


def test_all_registered_rule_terms_are_finite_on_eligible_rows() -> None:
    dataset = load_epit_dataset()
    manifest = json.loads(
        (DEFAULT_SPLIT_DIR / "split_manifest.json").read_text(encoding="utf-8")
    )
    development = np.asarray(
        [
            int(row["task_row_index"])
            for row in manifest["rows"]
            if row["outer_split"] == "development"
        ],
        dtype=int,
    )
    expected_counts = {
        "current_pren": 608,
        "current_pren_fe_ni": 452,
        "pren_linear": 452,
        "pren_n_linear": 452,
        "cr_mow_synergy": 452,
        "cr_mow_n_synergy": 452,
        "threshold_saturation": 452,
        "improved_environment": 452,
        "pren_n_improved_environment": 452,
        "mo_n_acid_repassivation": 452,
        "mns_inclusion_penalty": 452,
        "coupled_breakdown": 452,
        "fe_ni_cr_threshold": 452,
        "method_aware": 452,
        "method_aware_pren_n": 452,
        "al_chloride_temperature": 94,
        "al_amphoteric_environment": 94,
        "al_composition_environment": 94,
    }

    families = get_rule_families()
    assert {family.name for family in families} == set(expected_counts)
    for family in families:
        eligible = family.eligible_rows(dataset, development)
        prepared = family.fit_terms(dataset, eligible)
        assert len(eligible) == expected_counts[family.name]
        assert prepared.values.shape == (len(eligible), len(family.term_names))
        assert np.isfinite(prepared.values).all()
        assert len(family.coefficient_anchor) == len(family.term_names)
        assert np.isclose(family.coefficient_anchor.sum(), 1.0)
        if family.coefficient_upper_bounds is not None:
            assert len(family.coefficient_upper_bounds) == len(family.term_names)
            assert np.all(
                family.coefficient_anchor <= family.coefficient_upper_bounds
            )


def test_method_aware_transform_does_not_read_held_out_targets() -> None:
    dataset = load_epit_dataset()
    manifest = json.loads(
        (DEFAULT_SPLIT_DIR / "split_manifest.json").read_text(encoding="utf-8")
    )
    development = np.asarray(
        [
            int(row["task_row_index"])
            for row in manifest["rows"]
            if row["outer_split"] == "development"
        ],
        dtype=int,
    )
    validation = np.asarray(
        [
            int(row["task_row_index"])
            for row in manifest["rows"]
            if row["outer_split"] == "development"
            and int(row["optuna_validation_fold"]) == 1
        ],
        dtype=int,
    )
    family = get_rule_families(["method_aware"])[0]
    eligible = family.eligible_rows(dataset, development)
    validation = family.eligible_rows(dataset, validation)
    context = np.setdiff1d(eligible, validation, assume_unique=True)
    prepared = family.fit_terms(dataset, context)

    masked_target = dataset.target.copy()
    masked_target[validation] = np.nan
    masked_dataset = EpitDataset(
        table=dataset.table,
        rows=dataset.rows,
        target=masked_target,
        composition_columns=dataset.composition_columns,
        composition_reported_sums=dataset.composition_reported_sums,
        composition_missing_counts=dataset.composition_missing_counts,
        composition_structural_zero_counts=dataset.composition_structural_zero_counts,
        pretraining_template_eligible=dataset.pretraining_template_eligible,
    )
    original = family.transform_terms(dataset, validation, prepared.state)
    masked = family.transform_terms(masked_dataset, validation, prepared.state)

    assert np.allclose(original, masked)
    assert sum(prepared.state["method_counts"].values()) == len(context)
    weighted_offset = sum(
        prepared.state["method_counts"][name] * offset
        for name, offset in prepared.state["method_offsets"].items()
    )
    assert np.isclose(weighted_offset, 0.0)


@pytest.fixture(scope="module")
def reviewed_formula_data():
    dataset, identity = load_formula_dataset()
    _, partitions = _load_manifest(DEFAULT_SPLIT_DIR / "split_manifest.json", dataset)
    return dataset, identity, partitions


def test_formula_loader_preserves_reviewed_values_without_filling(reviewed_formula_data):
    dataset, identity, _ = reviewed_formula_data
    reviewed = load_reviewed_compositions()
    legacy = load_epit_dataset()
    assert len(dataset.rows) == 760
    assert [row["No."] for row in dataset.rows] == reviewed.rows["No."].tolist()
    np.testing.assert_array_equal(dataset.target, legacy.target)
    assert identity["reviewed_dataset"] == reviewed.identity
    for old_name, column in {**identity["composition_column_mapping"], **FORMULA_ENVIRONMENT_COLUMNS}.items():
        np.testing.assert_allclose(
            [row[old_name] for row in dataset.rows], reviewed.rows[column].to_numpy(float),
            rtol=0., atol=0., equal_nan=True,
        )
    # The reviewed contradiction and unsupported chloride conversion stay unknown.
    for index in (147, 755):
        assert np.isfinite(float(legacy.rows[index]["[Cl-] M"]))
        assert np.isnan(dataset.rows[index]["[Cl-] M"])
    for column in FORMULA_ENVIRONMENT_COLUMNS:
        assert np.array_equal(
            np.isnan([row[column] for row in dataset.rows]),
            reviewed.rows[FORMULA_ENVIRONMENT_COLUMNS[column]].isna(),
        )


@pytest.mark.parametrize("name", PROMOTED_TARGET_RULE_FAMILIES)
def test_reviewed_refit_excludes_final_query_features_and_targets(reviewed_formula_data, name):
    dataset, _, partitions = reviewed_formula_data
    family = get_rule_families([name])[0]
    original, predictions = calibrate_family(family, dataset, partitions, anchor_strength=0.1)
    target = dataset.target.copy()
    target[partitions.final_test] = np.nan
    changed_rows = [dict(row) for row in dataset.rows]
    for index in partitions.final_test:
        changed_rows[index] = {column: "forbidden_final_query_value" for column in changed_rows[index]}
    changed, changed_predictions = calibrate_family(
        family, replace(dataset, rows=changed_rows, target=target), partitions, anchor_strength=0.1,
    )
    assert changed == original
    assert changed_predictions == predictions
    expected = set(family.eligible_rows(dataset, partitions.development))
    assert len(expected) == 452
    assert {row["task_row_index"] for row in predictions} == expected
    assert len(predictions) == 452
    assert expected.isdisjoint(partitions.final_test)
    # Fold means must be fitted on the applicable context rows, never on validation rows.
    for fold in original["fold_calibrations"]:
        validation = family.eligible_rows(dataset, partitions.validation_by_fold[fold["fold"]])
        context = sorted(expected.difference(validation))
        state = fold["feature_state"]
        if name == "method_aware_pren_n":
            state = state["base_rule_state"]
        for column in FORMULA_ENVIRONMENT_COLUMNS:
            assert state["imputation_means"][column] == pytest.approx(
                np.nanmean([dataset.rows[index][column] for index in context])
            )


def test_new_formula_artifacts_are_used_by_default_and_legacy_remains_loadable():
    from scripts.epit_pipeline import run_optuna as search
    args = search.parse_args([])
    assert "target_rules_v4" in args.target_rule_summary.parts
    rules = search.load_target_rule_config(summary_path=args.target_rule_summary, split_manifest_path=args.split_manifest)
    assert tuple(rules.scores) == PROMOTED_TARGET_RULE_FAMILIES
    assert np.isclose(sum(rules.probabilities.values()), 1.)
    old_path = args.target_rule_summary.parent.parent / "target_rules_v3/calibration_summary.json"
    old = search.load_target_rule_config(summary_path=old_path, split_manifest_path=args.split_manifest)
    for name in rules.scores:
        new_artifact = json.loads(Path(rules.artifacts[name]).read_text())
        old_artifact = json.loads(Path(old.artifacts[name]).read_text())
        for field in ("formula", "term_names", "coefficient_anchor", "coefficient_upper_bounds", "anchor_strength", "applicability"):
            assert new_artifact[field] == old_artifact[field]
        assert new_artifact["final_development_calibration"]["rows"] == 452
        assert new_artifact["final_test_targets_used"] is False
    assert len(set(rules.artifact_sha256s.values())) == 9
