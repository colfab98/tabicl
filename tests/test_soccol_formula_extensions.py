"""Development isolation and reproducible Soccol formula research."""
import json

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from scripts.soccol_pipeline.discover_formula_extensions import (
    BASE_RULE, SoccolFormulaStudy, coverage,
)
from scripts.soccol_pipeline.formula_data import load_formula_data
from scripts.soccol_pipeline.data import REPO_ROOT
from scripts.epit_pipeline.formula_extension_study import rho


@pytest.fixture(scope="module")
def development():
    return load_formula_data()


@pytest.fixture(autouse=True)
def limit_threads():
    with threadpool_limits(limits=1):
        yield


def test_requires_exact_accepted_development_membership(development):
    changed = development.copy()
    changed.loc[changed.index[0], "outer_split"] = "final_test"
    with pytest.raises(ValueError, match="unchanged accepted"):
        SoccolFormulaStudy(changed)
    with pytest.raises(ValueError, match="unchanged accepted"):
        SoccolFormulaStudy(development.iloc[:-1])


def test_validation_targets_do_not_change_fitted_formula(development):
    first = SoccolFormulaStudy(development)
    changed = development.copy()
    held_out = changed.development_validation_fold.eq(1)
    changed.loc[held_out, "E_pit"] = changed.loc[held_out, "E_pit"].to_numpy()[::-1] + 10000
    second = SoccolFormulaStudy(changed)
    train = np.flatnonzero(first.folds != 1)
    for name in ("existing_"+BASE_RULE, "extend_surface", "updated_W_"+BASE_RULE, "new_prenW_surface"):
        penalty = None if name.startswith("existing_") else .1
        left, a = first.fit(train, first.specs[name], penalty)
        right, b = second.fit(train, second.specs[name], penalty)
        np.testing.assert_array_equal(left, right)
        assert a == b
        np.testing.assert_allclose(first.predict_artifact(a), left, rtol=1e-12, atol=1e-12)
        assert not np.intersect1d(a["fitting_task_row_indices"], first.rows[first.folds == 1]).size


def test_sparse_heat_and_missing_fields_are_not_fabricated(development):
    study = SoccolFormulaStudy(development)
    table = coverage(study).set_index("column")
    assert table.loc["H2SO4_mol_L", "observed_rows"] == 0
    assert table.loc["heat_treatment_2_type", "observed_rows"] == 0
    assert table.loc["heat_treatment_temperature_C", "observed_folds"] == 1
    assert "source" not in study.raw and "source" not in study.categories
    assert "reference_type" not in study.raw and "reference_type" not in study.categories
    train = np.flatnonzero(study.folds != 2)
    _, artifact = study.fit(train, study.specs["extend_heat_numeric"], .1)
    assert artifact["term_names"] == ["base_score"]
    assert artifact["feature_state"] == []


def test_current_nine_rules_reproduce_saved_fold_scores(development):
    study = SoccolFormulaStudy(development)
    historical = json.loads((REPO_ROOT/"corrosion_datasets/analysis/soccol_pipeline/target_rules_v2/calibration_summary.json").read_text())
    for record in historical["rules"]:
        scores = []
        spec = study.specs["existing_"+record["rule_family"]]
        for fold in range(1, 6):
            train, val = np.flatnonzero(study.folds != fold), np.flatnonzero(study.folds == fold)
            prediction, _ = study.fit(train, spec)
            scores.append(rho(study.y[val], prediction[val]))
        assert np.mean(scores) == pytest.approx(record["mean_fold_spearman"], abs=1e-10)
