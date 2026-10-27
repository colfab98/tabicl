from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts import eval_corrosion_datasets as evaluation
from scripts.epit_pipeline import evaluate_optuna_folds
from scripts.soccol_pipeline.data import PROCESSED_DIR, SPLIT_MANIFEST, load_soccol_dataset, select_accepted_development
from scripts.soccol_pipeline.formula_data import load_formula_data
from corrosion_datasets.datasets.soccol_pitting_potential import compare_target_rules as formulas
from corrosion_datasets.datasets.soccol_pitting_potential.prepare_pipeline_assets import RULE_SELECTION, load_reviewed_feature_source


EXPECTED_FOLD_COUNTS = {1: 622, 2: 615, 3: 638, 4: 445, 5: 630}


@pytest.fixture(scope="module")
def reviewed():
    return load_soccol_dataset()


def test_selection_matches_the_saved_accepted_export_exactly(reviewed):
    selected = select_accepted_development(reviewed)
    saved = pd.read_csv(PROCESSED_DIR / "soccol_development_accepted.csv", float_precision="round_trip")
    # Empty audit-text columns are inferred as float in the subset CSV.
    pd.testing.assert_frame_equal(selected.rows, saved, check_dtype=False, check_exact=True)
    expected = pd.read_csv(PROCESSED_DIR / "soccol_development_features.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(selected.features, expected.drop(columns="row_id"), check_exact=True)
    assert len(selected.rows) == 2950
    assert selected.identity["fold_counts"] == {str(k): v for k, v in EXPECTED_FOLD_COUNTS.items()}
    assert selected.identity["withheld_development_rows"] == 272
    assert selected.identity["final_query_rows_excluded"] == 805
    assert selected.assignments.task_row_index.tolist() == selected.task_row_indices.tolist()
    assert np.any(np.diff(selected.task_row_indices) > 1)  # Original IDs, not renumbered source indices.
    assert not selected.task_row_indices.flags.writeable


def test_formula_and_sampling_entry_points_use_identical_row_ids_and_folds(reviewed):
    expected = select_accepted_development(reviewed)
    formula = formulas._load_data(PROCESSED_DIR / "soccol_processed.csv", SPLIT_MANIFEST.parent / "split_assignments.csv", reviewed_data=True)
    sampling = load_reviewed_feature_source(reviewed)
    assert formula.row_id.tolist() == sampling.profile_row_id.tolist() == expected.rows.row_id.tolist()
    assert formula.task_row_index.tolist() == sampling.task_row_index.tolist() == expected.task_row_indices.tolist()
    assert formula.development_validation_fold.astype(int).tolist() == expected.assignments.development_validation_fold.astype(int).tolist()
    assert formula.attrs["soccol_development_selection"]["selection_sha256"] == sampling.attrs["soccol_development_selection"]["selection_sha256"]
    assert formula.outer_split.eq("development").all()
    assert formula.composition_status.eq("accepted").all()
    assert set(sampling.columns) == {"profile_row_id", "task_row_index", *reviewed.registry["feature_columns"]}
    assert sampling.CP_SO4.isna().any()  # Selection did not fill unknown chemistry.
    assert "CP_ion_other" not in sampling and "E_pit" not in sampling and "E_corr" not in sampling


@pytest.mark.parametrize("fold", range(1, 6))
def test_optuna_context_and_validation_are_both_accepted_and_keep_original_folds(reviewed, fold):
    selected = select_accepted_development(reviewed)
    task = evaluation._make_soccol_pipeline_task(SPLIT_MANIFEST, reviewed_data=True)
    evaluation.apply_epit_pipeline_fold([task], manifest_path=SPLIT_MANIFEST, validation_fold=fold)
    pd.testing.assert_frame_equal(task.X, selected.features, check_exact=True)
    pd.testing.assert_series_equal(task.y, selected.targets, check_exact=True)
    split = task.fixed_split
    assert len(split.test_index) == EXPECTED_FOLD_COUNTS[fold]
    assert len(split.train_index) == 2950 - EXPECTED_FOLD_COUNTS[fold]
    assert not set(split.train_index) & set(split.test_index)
    assert set(split.train_index) | set(split.test_index) == set(range(2950))
    validation = selected.assignments.iloc[split.test_index]
    context = selected.assignments.iloc[split.train_index]
    assert validation.development_validation_fold.eq(fold).all()
    assert context.development_validation_fold.ne(fold).all()
    assert not set(validation.composition_group) & set(context.composition_group)


@pytest.mark.parametrize("fold", [None, *range(1, 6)])
def test_final_queries_never_change_when_context_is_filtered(reviewed, fold):
    selected = select_accepted_development(reviewed)
    task = evaluation._make_soccol_pipeline_task(SPLIT_MANIFEST, reviewed_data=True)
    evaluation.apply_epit_pipeline_final_test([task], manifest_path=SPLIT_MANIFEST, context_fold=fold)
    split = task.fixed_split
    expected_queries = np.flatnonzero(reviewed.assignments.outer_split.eq("final_test"))
    np.testing.assert_array_equal(split.test_index, expected_queries)
    np.testing.assert_array_equal(split.preprocessing_index, selected.task_row_indices)
    assert len(split.train_index) == (2950 if fold is None else EXPECTED_FOLD_COUNTS[fold])
    assert set(split.train_index) <= set(selected.task_row_indices)
    assert len(split.test_index) == 805
    query = task.X.iloc[split.test_index]
    assert query[reviewed.registry["composition_columns"]].isna().any(axis=1).sum() == 10


def test_final_and_withheld_values_cannot_change_development_inputs(reviewed):
    expected = select_accepted_development(reviewed)
    excluded = ~reviewed.rows.use_as_development_context
    changed_rows = reviewed.rows.copy(deep=True)
    changed_features = reviewed.features.copy(deep=True)
    changed_targets = reviewed.targets.astype(float).copy()
    for column in reviewed.registry["feature_columns"]:
        value = "excluded-only-category" if reviewed.registry["columns"][column]["dtype"] == "categorical" else -999999.0
        # The reference's all-missing categorical columns can have float dtype.
        if isinstance(value, str):
            changed_rows[column] = changed_rows[column].astype(object)
            changed_features[column] = changed_features[column].astype(object)
        changed_rows.loc[excluded, column] = value
        changed_features.loc[excluded, column] = value
    changed_rows.loc[excluded, "E_pit"] = -999999.0
    changed_targets.loc[excluded] = -999999.0
    changed = replace(reviewed, rows=changed_rows, features=changed_features, targets=changed_targets)
    actual = select_accepted_development(changed)
    pd.testing.assert_frame_equal(actual.features, expected.features, check_dtype=False, check_exact=True)
    pd.testing.assert_series_equal(actual.targets, expected.targets, check_dtype=False, check_exact=True)
    pd.testing.assert_frame_equal(load_formula_data(changed), load_formula_data(reviewed), check_dtype=False, check_exact=True)
    pd.testing.assert_frame_equal(load_reviewed_feature_source(changed), load_reviewed_feature_source(reviewed), check_dtype=False, check_exact=True)
    assert actual.identity["selection_sha256"] == expected.identity["selection_sha256"]


def test_sampling_source_does_not_use_any_targets(reviewed):
    rows = reviewed.rows.copy(deep=True)
    rows["E_pit"] = -999999
    rows["E_corr"] = -999999
    altered = replace(reviewed, rows=rows, targets=reviewed.targets * -1000)
    pd.testing.assert_frame_equal(load_reviewed_feature_source(altered), load_reviewed_feature_source(reviewed), check_exact=True)


def test_formula_internal_filling_uses_context_after_selection(reviewed):
    frame = load_formula_data(reviewed)
    context = frame.loc[frame.development_validation_fold.ne(1)]
    validation = frame.loc[frame.development_validation_fold.eq(1)].copy()
    means = formulas._fit_imputation(context)
    for column in ["CP_temp", "CP_pH", "CP_SO4", "CP_NO3"]:
        assert means[column] == pytest.approx(context[column].mean())
    validation["CP_SO4"] = np.nan
    internal = formulas._inputs(validation, means)
    np.testing.assert_allclose(internal["CP_SO4"], context.CP_SO4.mean())
    assert validation.CP_SO4.isna().all()
    assert "CP_ion_other" not in internal


@pytest.mark.parametrize("name", [comparison_name for _, comparison_name in RULE_SELECTION])
def test_existing_promoted_formulas_accept_selected_context_without_refitting_coefficients(reviewed, name):
    frame = load_formula_data(reviewed)
    context = frame.loc[frame.development_validation_fold.ne(1)]
    validation = frame.loc[frame.development_validation_fold.eq(1)]
    spec = next(rule for rule in formulas.RULES if rule.name == name)
    terms, state = formulas._fit_term_state(context, spec)
    transformed = formulas._transform_terms(validation, spec, state)
    assert terms.shape == (2950 - 622, len(spec.term_names))
    assert transformed.shape == (622, len(spec.term_names))
    assert np.isfinite(terms).all() and np.isfinite(transformed).all()


def test_reviewed_formula_output_default_does_not_overwrite_historical_comparison():
    args = formulas.parse_args(["--reviewed-data", "--inspect-inputs"])
    assert args.data == PROCESSED_DIR / "soccol_processed.csv"
    assert args.output_dir != formulas.DEFAULT_OUTPUT


def test_fold_wrapper_forwards_reviewed_selection_flag(tmp_path):
    args = evaluate_optuna_folds.parse_args([
        "--local-ckpt-path", str(tmp_path / "model.ckpt"), "--soccol-reviewed-data",
        "--split-manifest", str(SPLIT_MANIFEST),
    ])
    command = evaluate_optuna_folds.fold_command(args, fold=2, fold_dir=tmp_path / "fold_2")
    assert "--soccol-reviewed-data" in command
    assert command[command.index("--epit-validation-fold") + 1] == "2"


def test_forged_final_query_cannot_enter_development_indices():
    task = evaluation._make_soccol_pipeline_task(SPLIT_MANIFEST, reviewed_data=True)
    identity = task.X.attrs["reviewed_soccol_data"]
    indices = identity["accepted_development_task_row_indices"].copy()
    indices[0] = 20  # Frozen final-test row.
    indices.sort()
    identity["accepted_development_task_row_indices"] = indices
    task.X.attrs["soccol_development_selection"]["source_task_row_indices"] = indices
    with pytest.raises(RuntimeError, match="accepted-development indices"):
        evaluation.apply_epit_pipeline_fold([task], manifest_path=SPLIT_MANIFEST, validation_fold=1)
