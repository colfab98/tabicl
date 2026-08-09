from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr

from scripts.epit_pipeline.calibrate_target_rules import (
    DEFAULT_SPLIT_DIR,
    fit_coefficients,
)
from scripts.epit_pipeline.split_data import EpitDataset, load_epit_dataset
from scripts.epit_pipeline.target_rules import get_rule_families


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
        "cr_mow_synergy": 452,
        "threshold_saturation": 452,
        "improved_environment": 452,
        "coupled_breakdown": 452,
        "fe_ni_cr_threshold": 452,
        "method_aware": 452,
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
