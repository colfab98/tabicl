import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from scripts.eval_epit_direct_prior import (
    EPIT_PROCESS_COLUMN,
    EPIT_TASK_ID,
    EXPECTED_EPIT_FEATURE_GROUP_COUNTS,
    EXPECTED_EPIT_FEATURE_COUNT,
    build_phase1_candidates,
    build_phase2_candidates,
    effective_sample_size,
    fit_and_score_theta,
    load_epit_task,
    preprocess_real_epit,
    read_phase1_summary,
    run_phase1,
    run_phase2,
    sample_and_score_eta,
    sample_synthetic_dataset,
    spearman_loss,
    weights_from_spearman,
    write_phase1_outputs,
    write_phase2_outputs,
    write_smoke_outputs,
)


@pytest.fixture(scope="module")
def epit_task():
    return load_epit_task()


def test_load_epit_task_uses_existing_evaluator_schema(epit_task):
    assert epit_task.task_id == EPIT_TASK_ID
    assert epit_task.X.shape == (760, EXPECTED_EPIT_FEATURE_COUNT)
    assert len(epit_task.y) == 760
    assert dict(epit_task.feature_group_counts) == EXPECTED_EPIT_FEATURE_GROUP_COUNTS
    assert epit_task.X.columns[-1] == EPIT_PROCESS_COLUMN



def test_build_phase1_candidates_matches_planned_grid():
    candidates = build_phase1_candidates()

    assert len(candidates) == 24
    assert candidates[0].eta_id == "plain__material_dominant"
    assert candidates[-1].eta_id == "physical_sparse_dirichlet__strong_prior"
    assert {candidate.anchored_regime for candidate in candidates} == {
        "plain",
        "physical_no_dirichlet",
        "physical_mild_dirichlet",
        "physical_sparse_dirichlet",
    }
    assert {candidate.core_anchor for candidate in candidates} == {
        "material_dominant",
        "environment_dominant",
        "interaction_dominant",
        "balanced",
        "weak_prior",
        "strong_prior",
    }
    assert candidates[0].eta_params["informed_physical_marginal_prob"] == 0.0
    assert candidates[-1].eta_params["pitting_material_dirichlet_concentration"] == 0.25
    assert candidates[-1].eta_params["informed_interaction_strength"] == 0.85


def test_preprocess_real_epit_keeps_fixed_21_column_schema(epit_task):
    processed = preprocess_real_epit(epit_task)

    assert processed.X.shape == (760, EXPECTED_EPIT_FEATURE_COUNT)
    assert processed.n_rows == 760
    assert processed.n_features == EXPECTED_EPIT_FEATURE_COUNT
    assert processed.feature_columns[-1] == EPIT_PROCESS_COLUMN
    assert processed.categorical_column == EPIT_PROCESS_COLUMN
    assert processed.category_count == len(processed.category_mapping)
    assert processed.category_count > 1
    assert np.isfinite(processed.X).all()
    assert np.isfinite(processed.y_z).all()
    assert np.allclose(processed.X.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(processed.X.std(axis=0, ddof=1), 1.0, atol=1e-12)
    assert abs(float(processed.y_z.mean())) < 1e-12
    assert abs(float(processed.y_z.std(ddof=0)) - 1.0) < 1e-12

    category_values = processed.X[:, -1]
    assert np.unique(category_values).size == processed.category_count


def test_sample_synthetic_epit_dataset_keeps_fixed_schema(epit_task):
    processed = preprocess_real_epit(epit_task)

    sample = sample_synthetic_dataset(
        category_count=processed.category_count,
        synthetic_seed=0,
        seq_len=processed.n_rows,
    )

    assert sample.X.shape == (760, EXPECTED_EPIT_FEATURE_COUNT)
    assert sample.y.shape == (760,)
    assert sample.d == EXPECTED_EPIT_FEATURE_COUNT
    assert sample.seq_len == 760
    assert sample.process_unique_count == processed.category_count
    assert np.isfinite(sample.X).all()
    assert np.isfinite(sample.y).all()
    assert float(np.std(sample.y, ddof=0)) > 0.0


def test_fit_and_score_single_synthetic_surrogate(epit_task):
    processed = preprocess_real_epit(epit_task)
    sample = sample_synthetic_dataset(
        category_count=processed.category_count,
        synthetic_seed=0,
        seq_len=processed.n_rows,
    )

    score = fit_and_score_theta(processed, sample)

    assert score.predictions.shape == (processed.n_rows,)
    assert np.isfinite(score.predictions).all()
    assert -1.0 <= score.spearman <= 1.0
    assert np.isfinite(score.standardized_mae)
    assert np.isfinite(score.standardized_rmse)
    assert score.standardized_mae >= 0.0
    assert score.standardized_rmse >= 0.0


def test_spearman_weights_normalize_and_prefer_better_scores():
    weights = weights_from_spearman(np.array([-0.5, 0.0, 0.5]), temperature=0.10)

    assert np.isclose(weights.sum(), 1.0)
    assert weights[2] > weights[1] > weights[0]
    assert spearman_loss(1.0) == 0.0
    assert spearman_loss(-1.0) == 1.0
    assert 1.0 <= effective_sample_size(weights) <= 3.0


def test_sample_and_score_one_eta_multiple_seeds(epit_task):
    processed = preprocess_real_epit(epit_task)

    samples, scores, summary = sample_and_score_eta(
        processed,
        n_synth=2,
        synthetic_seed_start=0,
        seq_len=processed.n_rows,
        temperature=0.10,
    )

    assert len(samples) == 2
    assert len(scores) == 2
    assert summary.n_synth == 2
    assert summary.ensemble_predictions.shape == (processed.n_rows,)
    assert np.isclose(summary.weights.sum(), 1.0)
    assert 1.0 <= summary.ess <= 2.0
    assert -1.0 <= summary.ensemble_spearman <= 1.0




def test_build_phase2_candidates_uses_top_phase1_regimes(tmp_path):
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "selected_rank": 1,
                "eta_id": "physical_no_dirichlet__balanced",
                "phase": "phase1",
                "anchored_regime": "physical_no_dirichlet",
                "ensemble_spearman": 0.20,
                "ESS": 8.0,
            },
            {
                "selected_rank": 2,
                "eta_id": "plain__strong_prior",
                "phase": "phase1",
                "anchored_regime": "plain",
                "ensemble_spearman": 0.18,
                "ESS": 7.0,
            },
            {
                "selected_rank": 3,
                "eta_id": "physical_no_dirichlet__weak_prior",
                "phase": "phase1",
                "anchored_regime": "physical_no_dirichlet",
                "ensemble_spearman": 0.17,
                "ESS": 6.0,
            },
        ]
    ).to_csv(summary_path, index=False)

    phase1_summary = read_phase1_summary(tmp_path)
    candidates = build_phase2_candidates(
        phase1_summary,
        top_regimes=2,
        n_core_samples=3,
        random_state=123,
    )

    assert len(candidates) == 6
    assert [candidate.anchored_regime for candidate in candidates[:3]] == ["physical_no_dirichlet"] * 3
    assert [candidate.anchored_regime for candidate in candidates[3:]] == ["plain"] * 3
    assert candidates[0].eta_id == "phase2__physical_no_dirichlet__space_filling_0000"
    for candidate in candidates:
        assert candidate.core_anchor.startswith("space_filling_")
        assert 0.45 <= candidate.eta_params["epit_material_coef"] <= 0.70
        assert 0.35 <= candidate.eta_params["epit_environment_coef"] <= 0.65
        assert 0.60 <= candidate.eta_params["epit_interaction_coef"] <= 0.95
        assert 0.00 <= candidate.eta_params["informed_feature_block_strength"] <= 0.95
        assert 0.00 <= candidate.eta_params["informed_interaction_strength"] <= 1.00


def test_limited_phase1_run_writes_grid_tables(epit_task, tmp_path):
    processed = preprocess_real_epit(epit_task)
    args = Namespace(phase="phase1", random_state=42, synthetic_seed=0, n_synth=1, temperature=0.10, max_etas=2)

    candidates, first_sample, scores_by_eta, summaries = run_phase1(
        processed,
        n_synth=1,
        synthetic_seed_start=0,
        seq_len=processed.n_rows,
        temperature=0.10,
        max_etas=2,
    )

    write_phase1_outputs(processed, candidates, first_sample, scores_by_eta, summaries, tmp_path, args)

    etas = pd.read_csv(tmp_path / "etas.csv")
    theta_scores = pd.read_csv(tmp_path / "theta_scores.csv")
    weights = pd.read_csv(tmp_path / "weights.csv")
    summary = pd.read_csv(tmp_path / "summary.csv")
    config = json.loads((tmp_path / "config.json").read_text())

    assert config["phase"] == "phase1"
    assert config["n_etas"] == 2
    assert etas.shape[0] == 2
    assert theta_scores.shape[0] == 2
    assert weights.shape[0] == 2
    assert summary.shape[0] == 2
    assert summary["selected_rank"].tolist() == [1, 2]
    assert set(summary["phase"]) == {"phase1"}
    assert np.allclose(weights.groupby("eta_id")["weight"].sum().to_numpy(), 1.0)



def test_limited_phase2_run_writes_space_filling_tables(epit_task, tmp_path):
    processed = preprocess_real_epit(epit_task)
    phase1_dir = tmp_path / "phase1"
    phase2_dir = tmp_path / "phase2"
    phase1_dir.mkdir()
    pd.DataFrame(
        [
            {
                "selected_rank": 1,
                "eta_id": "plain__balanced",
                "phase": "phase1",
                "anchored_regime": "plain",
                "ensemble_spearman": 0.10,
                "ESS": 5.0,
            },
            {
                "selected_rank": 2,
                "eta_id": "physical_no_dirichlet__balanced",
                "phase": "phase1",
                "anchored_regime": "physical_no_dirichlet",
                "ensemble_spearman": 0.08,
                "ESS": 4.0,
            },
        ]
    ).to_csv(phase1_dir / "summary.csv", index=False)
    args = Namespace(
        phase="phase2",
        random_state=42,
        synthetic_seed=0,
        n_synth=1,
        temperature=0.10,
        phase2_source_dir=phase1_dir,
        phase2_top_regimes=2,
        n_core_samples=1,
    )

    candidates, first_sample, scores_by_eta, summaries = run_phase2(
        processed,
        phase1_source_dir=phase1_dir,
        n_synth=1,
        synthetic_seed_start=0,
        seq_len=processed.n_rows,
        temperature=0.10,
        n_core_samples=1,
        top_regimes=2,
        random_state=42,
    )
    write_phase2_outputs(processed, candidates, first_sample, scores_by_eta, summaries, phase2_dir, args)

    etas = pd.read_csv(phase2_dir / "etas.csv")
    summary = pd.read_csv(phase2_dir / "summary.csv")
    weights = pd.read_csv(phase2_dir / "weights.csv")
    config = json.loads((phase2_dir / "config.json").read_text())

    assert config["phase"] == "phase2"
    assert config["n_etas"] == 2
    assert config["n_core_samples"] == 1
    assert etas.shape[0] == 2
    assert summary.shape[0] == 2
    assert set(summary["phase"]) == {"phase2"}
    assert set(etas["core_anchor"]) == {"space_filling_0000"}
    assert np.allclose(weights.groupby("eta_id")["weight"].sum().to_numpy(), 1.0)


def test_write_smoke_outputs_records_schema(epit_task, tmp_path):
    processed = preprocess_real_epit(epit_task)
    args = Namespace(phase="smoke", random_state=42, synthetic_seed=0, n_synth=2, temperature=0.10)

    samples, scores, summary = sample_and_score_eta(
        processed,
        n_synth=2,
        synthetic_seed_start=0,
        seq_len=processed.n_rows,
        temperature=0.10,
    )

    write_smoke_outputs(processed, samples, scores, summary, tmp_path, args)

    config = json.loads((tmp_path / "config.json").read_text())
    schema = json.loads((tmp_path / "real_schema.json").read_text())
    synthetic_schema = json.loads((tmp_path / "synthetic_smoke_schema.json").read_text())
    smoke_score = json.loads((tmp_path / "theta_smoke_score.json").read_text())
    summary_json = json.loads((tmp_path / "eta_smoke_summary.json").read_text())
    assert config["phase"] == "smoke"
    assert schema["n_rows"] == 760
    assert schema["n_features"] == EXPECTED_EPIT_FEATURE_COUNT
    assert schema["category_count"] == processed.category_count
    assert schema["categorical_encoding"] == "ordinal_codes_before_feature_standardization"
    assert schema["feature_scaling"] == "full_dataset_column_standardization_ddof1"
    assert synthetic_schema["n_features"] == EXPECTED_EPIT_FEATURE_COUNT
    assert synthetic_schema["process_unique_count"] == processed.category_count
    assert smoke_score["synthetic_seed"] == samples[0].synthetic_seed
    assert -1.0 <= smoke_score["spearman"] <= 1.0
    assert summary_json["n_synth"] == 2
    assert 1.0 <= summary_json["ESS"] <= 2.0
    assert (tmp_path / "theta_scores.csv").exists()
    assert (tmp_path / "weights.csv").exists()
    assert (tmp_path / "summary.csv").exists()
