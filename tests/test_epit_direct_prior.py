import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.eval_epit_direct_prior as epit_direct_prior

from scripts.eval_epit_direct_prior import (
    EPIT_PROCESS_COLUMN,
    BASELINE_PRIOR_CONTROL_FEATURE_MODE,
    BASELINE_PRIOR_CONTROL_ID,
    BASELINE_PRIOR_CONTROL_PHASE,
    EPIT_TASK_ID,
    TARGET_RULE_DIAGNOSTIC,
    THETA_ARTIFACT_FORMAT,
    EXPECTED_EPIT_FEATURE_GROUP_COUNTS,
    EXPECTED_EPIT_FEATURE_COUNT,
    baseline_prior_control_records,
    build_phase1_candidates,
    build_phase2_candidates,
    effective_sample_size,
    fit_and_score_theta,
    load_epit_task,
    preprocess_real_epit,
    read_phase1_summary,
    read_target_rule_artifact,
    read_theta_artifact,
    read_baseline_prior_artifact,
    run_baseline_prior_control_with_artifacts,
    run_candidates_with_artifacts,
    run_phase1,
    run_phase2,
    sample_and_score_baseline_prior_control,
    sample_and_score_eta,
    sample_baseline_prior_control_dataset,
    sample_synthetic_dataset,
    score_target_rule_theta,
    summarize_target_rule_eta,
    spearman_loss,
    theta_artifact_path,
    uniform_theta_weights,
    write_phase1_outputs,
    write_baseline_prior_artifact,
    write_baseline_prior_control_outputs,
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

    assert len(candidates) == 120
    assert candidates[0].eta_id == "plain__material_dominant__mlp000"
    assert candidates[-1].eta_id == "physical_sparse_dirichlet__strong_prior__mlp100"
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
    assert {candidate.eta_params["informed_mlp_prob"] for candidate in candidates} == {
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
    }
    assert candidates[0].eta_params["informed_physical_marginal_prob"] == 0.0
    assert candidates[0].eta_params["informed_mlp_prob"] == 0.0
    assert candidates[-1].eta_params["pitting_material_dirichlet_concentration"] == 0.25
    assert candidates[-1].eta_params["informed_interaction_strength"] == 0.85
    assert candidates[-1].eta_params["informed_mlp_prob"] == 1.0


def test_informed_mlp_prob_maps_to_prior_mix_probs():
    fixed_hp = epit_direct_prior.build_fixed_hp_for_eta(
        category_count=3,
        eta_params={"informed_mlp_prob": 0.25},
    )

    assert fixed_hp["mix_probs"] == (0.25, 0.75)
    assert fixed_hp["informed_mix_probs"] == (0.25, 0.75)
    assert "informed_mlp_prob" not in fixed_hp

    with pytest.raises(ValueError, match="informed_mlp_prob"):
        epit_direct_prior.build_fixed_hp_for_eta(category_count=3, eta_params={"informed_mlp_prob": 1.1})


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
    assert np.isfinite(processed.X_raw).all()
    assert processed.X_raw.shape == processed.X.shape
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
    assert sample.target_rule["rule_type"] == "fixed_epit_target_rule_v2_pren_anchor"
    assert sample.target_rule["material_cols"] == list(range(17))
    assert sample.target_rule["temperature_col"] == 17
    assert sample.target_rule["chloride_col"] == 18
    assert sample.target_rule["ph_col"] == 19
    assert sample.target_rule["process_col"] == 20
    assert sample.target_rule["material_anchor_type"] == "pren_like_cr_mo_w_weak_ni_v1"
    assert sample.target_rule["environment_rule_type"] == "chloride_dominant_weak_temp_ph_v1"
    anchor = np.asarray(sample.target_rule["material_anchor_weights"], dtype=float)
    weights = np.asarray(sample.target_rule["material_weights"], dtype=float)
    env_weights = np.asarray(sample.target_rule["environment_weights"], dtype=float)
    assert anchor.shape == (17,)
    assert np.isclose(anchor[1], 1.0)
    assert np.isclose(anchor[2], 0.25)
    assert np.isclose(anchor[3], 3.3)
    assert np.isclose(anchor[4], 1.65)
    assert weights.shape == (17,)
    assert np.isclose(np.linalg.norm(weights), 1.0)
    assert weights[1] > 0.0
    assert weights[2] > 0.0
    assert weights[3] > 0.0
    assert weights[4] > 0.0
    assert env_weights[1] > env_weights[0]
    assert env_weights[1] > env_weights[2]
    assert sample.target_rule["process_coef"] < 0.11
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


def test_score_target_rule_oracle_uses_raw_real_schema(epit_task):
    processed = preprocess_real_epit(epit_task)
    sample = sample_synthetic_dataset(
        category_count=processed.category_count,
        synthetic_seed=0,
        seq_len=processed.n_rows,
    )

    score = score_target_rule_theta(processed, sample)

    assert score.predictions.shape == (processed.n_rows,)
    assert np.isfinite(score.predictions).all()
    assert -1.0 <= score.spearman <= 1.0
    assert np.isfinite(score.standardized_mae)
    assert np.isfinite(score.standardized_rmse)
    assert score.standardized_mae >= 0.0
    assert score.standardized_rmse >= 0.0


def test_sample_synthetic_epit_dataset_retries_left_packed_schema(monkeypatch):
    class FakePriorDataset:
        calls = 0

        def __init__(self, **kwargs):
            self.seq_len = int(kwargs["max_seq_len"])
            self.prior = Namespace(
                last_pitting_target_rule={
                    "rule_type": "fixed_epit_target_rule_v2_pren_anchor",
                    "material_cols": list(range(17)),
                    "temperature_col": 17,
                    "chloride_col": 18,
                    "ph_col": 19,
                    "process_col": 20,
                    "material_anchor_type": "pren_like_cr_mo_w_weak_ni_v1",
                    "environment_rule_type": "chloride_dominant_weak_temp_ph_v1",
                    "material_anchor_weights": np.asarray([0.0, 1.0, 0.25, 3.3, 1.65] + [0.0] * 12),
                    "material_weights": np.full(17, 1.0 / np.sqrt(17.0)),
                    "environment_weights": np.asarray([0.08, 0.90, 0.10]),
                    "ph_neutral": 7.0,
                    "process_offsets": np.asarray([-0.5, 0.0, 0.5]),
                    "process_category_count": 3,
                    "material_coef": 0.575,
                    "environment_coef": 0.50,
                    "interaction_coef": 0.775,
                    "exposure_coef": 0.20,
                    "process_coef": 0.05,
                    "history_coef": 0.10,
                    "descriptor_coef": 0.05,
                    "interaction_strength": 0.35,
                    "synthetic_environment_mode": "rank_semantic",
                }
            )

        def get_batch(self):
            FakePriorDataset.calls += 1
            seq_len = self.seq_len
            X = torch.randn(1, seq_len, EXPECTED_EPIT_FEATURE_COUNT)
            X[0, :, -1] = torch.arange(seq_len) % 3
            y = torch.linspace(-1.0, 1.0, seq_len).reshape(1, seq_len)
            d_value = 18 if FakePriorDataset.calls == 1 else EXPECTED_EPIT_FEATURE_COUNT
            d = torch.tensor([d_value], dtype=torch.long)
            seq_lens = torch.tensor([seq_len], dtype=torch.long)
            train_sizes = torch.tensor([seq_len // 2], dtype=torch.long)
            return X, y, d, seq_lens, train_sizes

    monkeypatch.setattr(epit_direct_prior, "PriorDataset", FakePriorDataset)

    sample = sample_synthetic_dataset(category_count=3, synthetic_seed=100000, seq_len=32)

    assert FakePriorDataset.calls == 2
    assert sample.synthetic_seed == 100000
    assert sample.sampling_seed != sample.synthetic_seed
    assert sample.schema_attempts == 2
    assert sample.X.shape == (32, EXPECTED_EPIT_FEATURE_COUNT)
    assert sample.d == EXPECTED_EPIT_FEATURE_COUNT
    assert sample.process_unique_count == 3


def test_baseline_prior_control_conditions_default_prior_on_d21(monkeypatch):
    class FakePriorDataset:
        calls = 0
        init_kwargs = []

        def __init__(self, **kwargs):
            self.seq_len = int(kwargs["max_seq_len"])
            self.max_features = int(kwargs["max_features"])
            FakePriorDataset.init_kwargs.append(kwargs)

        def get_batch(self):
            FakePriorDataset.calls += 1
            seq_len = self.seq_len
            X = torch.randn(1, seq_len, self.max_features)
            y = torch.linspace(-1.0, 1.0, seq_len).reshape(1, seq_len)
            d_value = 18 if FakePriorDataset.calls == 1 else EXPECTED_EPIT_FEATURE_COUNT
            d = torch.tensor([d_value], dtype=torch.long)
            seq_lens = torch.tensor([seq_len], dtype=torch.long)
            train_sizes = torch.tensor([max(1, seq_len // 3)], dtype=torch.long)
            return X, y, d, seq_lens, train_sizes

    monkeypatch.setattr(epit_direct_prior, "PriorDataset", FakePriorDataset)

    sample = sample_baseline_prior_control_dataset(synthetic_seed=200000, seq_len=32, max_schema_attempts=5)

    assert FakePriorDataset.calls == 2
    first_kwargs = FakePriorDataset.init_kwargs[0]
    assert first_kwargs["prior_type"] == "mix_scm"
    assert first_kwargs["min_features"] == 2
    assert first_kwargs["max_features"] == 100
    assert first_kwargs["max_classes"] == 0
    assert first_kwargs["scm_fixed_hp"]["pitting_fixed_epit_schema"] is False
    assert sample.X.shape == (32, EXPECTED_EPIT_FEATURE_COUNT)
    assert sample.d == EXPECTED_EPIT_FEATURE_COUNT
    assert sample.schema_attempts == 2
    assert sample.sampling_seed != sample.synthetic_seed
    assert sample.target_rule == {}


def test_baseline_prior_control_scores_and_writes_separate_outputs(monkeypatch, epit_task, tmp_path):
    class FakePriorDataset:
        def __init__(self, **kwargs):
            self.seq_len = int(kwargs["max_seq_len"])
            self.max_features = int(kwargs["max_features"])

        def get_batch(self):
            seq_len = self.seq_len
            X = torch.randn(1, seq_len, self.max_features)
            y = torch.linspace(-1.0, 1.0, seq_len).reshape(1, seq_len)
            d = torch.tensor([EXPECTED_EPIT_FEATURE_COUNT], dtype=torch.long)
            seq_lens = torch.tensor([seq_len], dtype=torch.long)
            train_sizes = torch.tensor([seq_len // 2], dtype=torch.long)
            return X, y, d, seq_lens, train_sizes

    monkeypatch.setattr(epit_direct_prior, "PriorDataset", FakePriorDataset)
    processed = preprocess_real_epit(epit_task)

    samples, scores, summary = sample_and_score_baseline_prior_control(
        processed,
        n_synth=2,
        synthetic_seed_start=10,
        seq_len=32,
        temperature=0.10,
        phase="phase1",
        max_schema_attempts=3,
    )
    records = baseline_prior_control_records(samples, scores)

    assert summary.eta_id == BASELINE_PRIOR_CONTROL_ID
    assert summary.core_anchor == BASELINE_PRIOR_CONTROL_FEATURE_MODE
    assert summary.n_synth == 2
    assert len(records) == 2
    assert all(record.d == EXPECTED_EPIT_FEATURE_COUNT for record in records)
    assert all(record.raw_feature_count == 100 for record in records)

    write_baseline_prior_artifact(tmp_path, phase="phase1", score=scores[0], synthetic=samples[0])
    artifact_record = read_baseline_prior_artifact(
        tmp_path, samples[0].synthetic_seed, expected_n_rows=processed.n_rows, strict=True
    )
    assert artifact_record is not None
    assert np.isclose(artifact_record.score.spearman, scores[0].spearman)
    assert artifact_record.score.predictions.shape == (processed.n_rows,)

    write_baseline_prior_control_outputs(
        processed,
        records,
        tmp_path,
        phase="phase1",
        temperature=0.10,
    )
    baseline_summary = pd.read_csv(tmp_path / "baseline_prior_summary.csv")
    baseline_scores = pd.read_csv(tmp_path / "baseline_prior_theta_scores.csv")
    baseline_weights = pd.read_csv(tmp_path / "baseline_prior_weights.csv")

    assert baseline_summary.loc[0, "control_id"] == BASELINE_PRIOR_CONTROL_ID
    assert baseline_summary.loc[0, "target_rule_diagnostic"] == "not_applicable"
    assert baseline_summary.loc[0, "n_accepted_d21"] == 2
    assert baseline_scores.shape[0] == 2
    assert set(baseline_scores["feature_mode"]) == {BASELINE_PRIOR_CONTROL_FEATURE_MODE}
    assert np.allclose(baseline_weights["weight"].to_numpy(), np.full(2, 0.5))


def test_baseline_prior_control_artifact_runner_is_baseline_only(monkeypatch, epit_task, tmp_path):
    class FakePriorDataset:
        def __init__(self, **kwargs):
            self.seq_len = int(kwargs["max_seq_len"])
            self.max_features = int(kwargs["max_features"])

        def get_batch(self):
            seq_len = self.seq_len
            X = torch.randn(1, seq_len, self.max_features)
            y = torch.linspace(-1.0, 1.0, seq_len).reshape(1, seq_len)
            d = torch.tensor([EXPECTED_EPIT_FEATURE_COUNT], dtype=torch.long)
            seq_lens = torch.tensor([seq_len], dtype=torch.long)
            train_sizes = torch.tensor([seq_len // 2], dtype=torch.long)
            return X, y, d, seq_lens, train_sizes

    monkeypatch.setattr(epit_direct_prior, "PriorDataset", FakePriorDataset)
    processed = preprocess_real_epit(epit_task)

    records, summary = run_baseline_prior_control_with_artifacts(
        processed,
        tmp_path,
        phase=BASELINE_PRIOR_CONTROL_PHASE,
        n_synth=2,
        synthetic_seed_start=40,
        seq_len=32,
        temperature=0.10,
        n_workers=1,
        chunk_size=1,
        resume=True,
        progress_interval=0.0,
        max_schema_attempts=3,
    )

    assert summary.eta_id == BASELINE_PRIOR_CONTROL_ID
    assert summary.phase == BASELINE_PRIOR_CONTROL_PHASE
    assert len(records) == 2
    assert (tmp_path / "baseline_prior_summary.csv").exists()
    assert (tmp_path / "baseline_prior_theta_scores.csv").exists()
    assert (tmp_path / "baseline_prior_weights.csv").exists()
    assert (tmp_path / "baseline_prior_progress.json").exists()
    assert not (tmp_path / "summary.csv").exists()
    assert not (tmp_path / "theta_scores.csv").exists()
    assert not (tmp_path / "target_rule_summary.csv").exists()

    progress = json.loads((tmp_path / "baseline_prior_progress.json").read_text())
    assert progress["phase"] == BASELINE_PRIOR_CONTROL_PHASE
    assert progress["completed_theta"] == 2
    artifact_record = read_baseline_prior_artifact(tmp_path, 40, expected_n_rows=processed.n_rows, strict=True)
    assert artifact_record is not None
    assert artifact_record.d == EXPECTED_EPIT_FEATURE_COUNT


def test_baseline_prior_control_enabled_for_dedicated_phase():
    assert epit_direct_prior.baseline_prior_control_enabled(
        Namespace(phase=BASELINE_PRIOR_CONTROL_PHASE, no_baseline_prior_control=True)
    )
    assert epit_direct_prior.baseline_prior_control_enabled(Namespace(phase="phase1", no_baseline_prior_control=False))
    assert not epit_direct_prior.baseline_prior_control_enabled(Namespace(phase="phase1", no_baseline_prior_control=True))
    assert not epit_direct_prior.baseline_prior_control_enabled(Namespace(phase="smoke", no_baseline_prior_control=False))


def test_uniform_weights_match_training_aligned_eta_average():
    weights = uniform_theta_weights(3)

    assert np.allclose(weights, np.full(3, 1.0 / 3.0))
    assert np.isclose(weights.sum(), 1.0)
    assert spearman_loss(1.0) == 0.0
    assert spearman_loss(-1.0) == 1.0
    assert np.isclose(effective_sample_size(weights), 3.0)


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
    expected_ensemble = np.mean(np.vstack([score.predictions for score in scores]), axis=0)
    assert summary.ensemble_predictions.shape == (processed.n_rows,)
    assert np.allclose(summary.ensemble_predictions, expected_ensemble)
    assert np.allclose(summary.weights, np.full(2, 0.5))
    assert np.isclose(summary.weights.sum(), 1.0)
    assert np.isclose(summary.ess, 2.0)
    assert summary.weighting_scheme == "uniform_theta_average"
    assert -1.0 <= summary.ensemble_spearman <= 1.0

    target_rule_scores = [score_target_rule_theta(processed, sample) for sample in samples]
    target_rule_summary = summarize_target_rule_eta(
        processed,
        target_rule_scores,
        temperature=0.10,
        eta_id=summary.eta_id,
        phase=summary.phase,
        anchored_regime=summary.anchored_regime,
        core_anchor=summary.core_anchor,
    )
    expected_target_ensemble = np.mean(np.vstack([score.predictions for score in target_rule_scores]), axis=0)
    assert target_rule_summary.ensemble_predictions.shape == (processed.n_rows,)
    assert np.allclose(target_rule_summary.ensemble_predictions, expected_target_ensemble)
    assert np.allclose(target_rule_summary.weights, np.full(2, 0.5))
    assert -1.0 <= target_rule_summary.ensemble_spearman <= 1.0




def test_artifact_runner_writes_and_resumes_phase1(epit_task, tmp_path):
    processed = preprocess_real_epit(epit_task)
    candidates = build_phase1_candidates(max_etas=1)
    args = Namespace(
        phase="phase1",
        random_state=42,
        synthetic_seed=0,
        n_synth=2,
        temperature=0.10,
        max_etas=1,
        phase2_source_dir=None,
        phase2_top_regimes=2,
        n_core_samples=64,
        n_workers=1,
        chunk_size=1,
        progress_interval=0.0,
        no_resume=False,
    )

    _, scores_by_eta, summaries = run_candidates_with_artifacts(
        processed,
        candidates,
        tmp_path,
        args,
        phase="phase1",
        scope="phase1_anchor_regime_grid",
        n_synth=2,
        synthetic_seed_start=0,
        seq_len=processed.n_rows,
        temperature=0.10,
        n_workers=1,
        chunk_size=1,
        resume=True,
        progress_interval=0.0,
    )

    eta_id = candidates[0].eta_id
    assert len(scores_by_eta[eta_id]) == 2
    assert len(summaries) == 1
    assert (tmp_path / "manifest.csv").exists()
    assert (tmp_path / "progress.json").exists()
    assert (tmp_path / "progress.log").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "theta_scores.csv").exists()
    assert (tmp_path / "target_rule_summary.csv").exists()
    assert (tmp_path / "target_rule_theta_scores.csv").exists()
    assert (tmp_path / "target_rule_weights.csv").exists()
    assert (tmp_path / "eta_diagnostic_comparison.csv").exists()
    weights = pd.read_csv(tmp_path / "weights.csv")
    assert np.allclose(weights["weight"].to_numpy(), np.full(2, 0.5))
    assert set(weights["weighting_scheme"]) == {"uniform_theta_average"}

    artifact_path = theta_artifact_path(tmp_path, eta_id, 0)
    assert artifact_path.exists()
    artifact_score = read_theta_artifact(tmp_path, eta_id, 0, expected_n_rows=processed.n_rows, strict=True)
    assert artifact_score is not None
    assert artifact_score.predictions.shape == (processed.n_rows,)
    assert np.isclose(artifact_score.spearman, scores_by_eta[eta_id][0].spearman)
    target_artifact_score = read_target_rule_artifact(
        tmp_path, eta_id, 0, expected_n_rows=processed.n_rows, strict=True
    )
    assert target_artifact_score is not None
    assert target_artifact_score.predictions.shape == (processed.n_rows,)
    with np.load(artifact_path, allow_pickle=False) as artifact:
        assert str(artifact["artifact_format"]) == THETA_ARTIFACT_FORMAT
        assert artifact["target_rule_predictions"].shape == (processed.n_rows,)
        assert str(artifact["target_rule_type"]) == "fixed_epit_target_rule_v2_pren_anchor"

    _, resumed_scores_by_eta, resumed_summaries = run_candidates_with_artifacts(
        processed,
        candidates,
        tmp_path,
        args,
        phase="phase1",
        scope="phase1_anchor_regime_grid",
        n_synth=2,
        synthetic_seed_start=0,
        seq_len=processed.n_rows,
        temperature=0.10,
        n_workers=1,
        chunk_size=1,
        resume=True,
        progress_interval=0.0,
    )

    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["skipped_theta"] == 2
    assert len(resumed_scores_by_eta[eta_id]) == 2
    assert np.isclose(resumed_summaries[0].ensemble_spearman, summaries[0].ensemble_spearman)




def test_read_phase1_summary_rejects_stale_weighted_outputs(tmp_path):
    pd.DataFrame(
        [
            {
                "selected_rank": 1,
                "eta_id": "plain__balanced",
                "phase": "phase1",
                "anchored_regime": "plain",
                "ensemble_spearman": 0.10,
                "ESS": 5.0,
            }
        ]
    ).to_csv(tmp_path / "summary.csv", index=False)

    with pytest.raises(ValueError, match="uniform-theta scorer"):
        read_phase1_summary(tmp_path)


def test_build_phase2_candidates_uses_top_phase1_regimes(tmp_path):
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "selected_rank": 1,
                "eta_id": "physical_no_dirichlet__balanced__mlp075",
                "phase": "phase1",
                "weighting_scheme": "uniform_theta_average",
                "anchored_regime": "physical_no_dirichlet",
                "ensemble_spearman": 0.20,
                "ESS": 8.0,
            },
            {
                "selected_rank": 2,
                "eta_id": "plain__strong_prior__mlp025",
                "phase": "phase1",
                "weighting_scheme": "uniform_theta_average",
                "anchored_regime": "plain",
                "ensemble_spearman": 0.18,
                "ESS": 7.0,
            },
            {
                "selected_rank": 3,
                "eta_id": "physical_no_dirichlet__weak_prior__mlp000",
                "phase": "phase1",
                "weighting_scheme": "uniform_theta_average",
                "anchored_regime": "physical_no_dirichlet",
                "ensemble_spearman": 0.17,
                "ESS": 6.0,
            },
        ]
    ).to_csv(summary_path, index=False)
    pd.DataFrame(
        [
            {"eta_id": "physical_no_dirichlet__balanced__mlp075", "informed_mlp_prob": 0.75},
            {"eta_id": "plain__strong_prior__mlp025", "informed_mlp_prob": 0.25},
            {"eta_id": "physical_no_dirichlet__weak_prior__mlp000", "informed_mlp_prob": 0.0},
        ]
    ).to_csv(tmp_path / "etas.csv", index=False)

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
    assert all(0.50 <= candidate.eta_params["informed_mlp_prob"] <= 1.00 for candidate in candidates[:3])
    assert all(0.00 <= candidate.eta_params["informed_mlp_prob"] <= 0.50 for candidate in candidates[3:])


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
    assert config["ensemble_weighting_scheme"] == "uniform_theta_average"
    assert config["target_rule_diagnostic"] == TARGET_RULE_DIAGNOSTIC
    assert config["informed_mlp_prob_search"]["phase1_grid"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert etas.shape[0] == 2
    assert theta_scores.shape[0] == 2
    assert weights.shape[0] == 2
    assert summary.shape[0] == 2
    assert summary["selected_rank"].tolist() == [1, 2]
    assert set(summary["phase"]) == {"phase1"}
    target_rule_summary = pd.read_csv(tmp_path / "target_rule_summary.csv")
    comparison = pd.read_csv(tmp_path / "eta_diagnostic_comparison.csv")
    assert target_rule_summary.shape[0] == 2
    assert comparison.shape[0] == 2
    assert "target_rule_ensemble_spearman" in comparison.columns
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
                "eta_id": "plain__balanced__mlp050",
                "phase": "phase1",
                "weighting_scheme": "uniform_theta_average",
                "anchored_regime": "plain",
                "ensemble_spearman": 0.10,
                "ESS": 5.0,
            },
            {
                "selected_rank": 2,
                "eta_id": "physical_no_dirichlet__balanced__mlp100",
                "phase": "phase1",
                "weighting_scheme": "uniform_theta_average",
                "anchored_regime": "physical_no_dirichlet",
                "ensemble_spearman": 0.08,
                "ESS": 4.0,
            },
        ]
    ).to_csv(phase1_dir / "summary.csv", index=False)
    pd.DataFrame(
        [
            {"eta_id": "plain__balanced__mlp050", "informed_mlp_prob": 0.50},
            {"eta_id": "physical_no_dirichlet__balanced__mlp100", "informed_mlp_prob": 1.0},
        ]
    ).to_csv(phase1_dir / "etas.csv", index=False)
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
    assert config["ensemble_weighting_scheme"] == "uniform_theta_average"
    assert config["target_rule_diagnostic"] == TARGET_RULE_DIAGNOSTIC
    assert config["informed_mlp_prob_search"]["phase2_window"] == 0.25
    assert etas.shape[0] == 2
    assert summary.shape[0] == 2
    assert set(summary["phase"]) == {"phase2"}
    assert set(etas["core_anchor"]) == {"space_filling_0000"}
    assert set(etas["informed_mlp_prob"]).issubset({0.5, 0.875})
    target_rule_summary = pd.read_csv(phase2_dir / "target_rule_summary.csv")
    comparison = pd.read_csv(phase2_dir / "eta_diagnostic_comparison.csv")
    assert target_rule_summary.shape[0] == 2
    assert comparison.shape[0] == 2
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
    assert config["ensemble_weighting_scheme"] == "uniform_theta_average"
    assert config["target_rule_diagnostic"] == TARGET_RULE_DIAGNOSTIC
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
    assert summary_json["weighting_scheme"] == "uniform_theta_average"
    assert np.isclose(summary_json["ESS"], 2.0)
    assert (tmp_path / "theta_scores.csv").exists()
    assert (tmp_path / "weights.csv").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "target_rule_theta_scores.csv").exists()
    assert (tmp_path / "target_rule_weights.csv").exists()
    assert (tmp_path / "target_rule_summary.csv").exists()
    assert (tmp_path / "eta_diagnostic_comparison.csv").exists()
