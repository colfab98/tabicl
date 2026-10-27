"""Regression checks for the audit's missingness, dependencies and fold boundary."""

import numpy as np
import pandas as pd
import pytest

from scripts.soccol_pipeline import audit_input_changes as audit
from scripts.soccol_pipeline.formula_data import load_formula_data


@pytest.fixture(scope="module")
def development():
    return load_formula_data()


def test_comparison_ignores_roundoff_but_never_confuses_zero_with_unknown():
    assert audit.same_values([0., np.nan, 0.1 + 0.2, 1.86], [np.nan, np.nan, .3, 0.]).tolist() == [False, True, True, False]
    assert audit.same_values(["no", None, "natural"], [0., None, "naturally_aerated"], categorical=True).tolist() == [False, True, False]


def test_declared_formula_dependencies_match_actual_calculations(development):
    specs = {s.name: s for s in audit.formulas.RULES}
    for family, name in audit.RULE_SELECTION:
        spec = specs[name]
        _, state = audit.formulas._fit_term_state(development, spec)
        values = audit.formulas._inputs(development, state["imputation_means"])
        # Include newly available elements that these historical rules ignore.
        values.update({c: development[c].to_numpy() for c in ("Fe", "W", "Cu")})
        baseline = audit.formulas._raw_terms(values, spec, state)
        declared = set(audit.rule_dependencies()[family])
        actual = set()
        for column, array in values.items():
            changed = dict(values)
            changed[column] = array + np.linspace(.1, 10., len(array))
            if not np.allclose(audit.formulas._raw_terms(changed, spec, state), baseline):
                actual.add(column)
        assert actual == declared, family


def test_filling_statistics_exclude_final_and_validation_rows(development):
    old = pd.read_csv(audit.LEGACY_CSV, float_precision="round_trip")
    assignments = pd.read_csv(audit.SPLIT_MANIFEST.parent / "split_assignments.csv")
    for column in ("outer_split", "development_validation_fold"):
        old[column] = assignments[column]
    _, expected = audit.formula_filling_audit(old, development, audit.rule_dependencies())
    changed_old, changed_new = old.copy(), development.copy()
    for column in ("CP_temp", "CP_pH", "CP_SO4"):
        changed_old.loc[old.outer_split.eq("final_test") | old.development_validation_fold.eq(1), column] = 12345.
        changed_new.loc[development.development_validation_fold.eq(1), column] = 23456.
    _, actual = audit.formula_filling_audit(changed_old, changed_new, audit.rule_dependencies())
    pd.testing.assert_frame_equal(expected.loc[expected.fold.eq(1)], actual.loc[actual.fold.eq(1)])


def test_audit_reconciles_raw_corrections_with_effective_inputs():
    summary, tables = audit.build_audit()
    assert all(summary["checks"].values())
    changes = tables["input_changes_vs_legacy.csv"]
    selected = changes.loc[changes.use_as_development_context]
    assert selected.groupby("change_type").size().to_dict() == {
        "category_representation": 645, "known_to_missing": 299, "numeric_correction": 71,
    }
    assert summary["accepted_development_active_formula_raw_changes"] == {"CP_SO4": 104}
    effective = tables["effective_formula_input_changes.csv"]
    assert effective.loc[effective.changed_by_processing].groupby("column").size().to_dict() == {"CP_NO3": 3, "CP_SO4": 104}
    assert effective.loc[effective.changed_by_context_selection].groupby("column").size().to_dict() == {"CP_pH": 51, "CP_temp": 33}
    assert set(effective.task_row_index) <= set(load_formula_data().task_row_index)
