#!/usr/bin/env python
"""Direct EPIT synthetic-prior calibration utilities.

This script is intentionally separate from Optuna and TabICL transformer
training. The initial smoke path loads the real EPIT task and preprocesses it
into the fixed 21-column Ridge-surrogate schema used by direct prior
calibration.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_corrosion_datasets import EvalTask, make_tasks  # noqa: E402
from tabicl.prior.dataset import PriorDataset  # noqa: E402
from tabicl.prior.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP  # noqa: E402


EPIT_TASK_ID = "electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg"
EPIT_DATASET = "electrochemical_metrics_alloys"
EPIT_TARGET_COLUMN = "Epit, mV (SCE) Avg."
EPIT_PROCESS_COLUMN = "[Cl-] Test Method"
EXPECTED_EPIT_FEATURE_GROUP_COUNTS = {
    "environment": 3,
    "material": 17,
    "process_history": 1,
}
EXPECTED_EPIT_FEATURE_COUNT = 21
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "corrosion_datasets" / "analysis" / "epit_direct_prior"
DIRECT_EPIT_BLOCK_ALLOCATION = (17, 3, 1, 0, 0, 0, 0, 0, 0)
DIRECT_EPIT_TRAIN_SIZE_RATIO = 0.5
DEFAULT_TEMPERATURE = 0.10
DEFAULT_ETA_ID = "plain_balanced"
PHASE1_DEFAULT_N_SYNTH = 32
PHASE2_DEFAULT_N_SYNTH = 64
PHASE2_DEFAULT_CORE_SAMPLES = 64
PHASE2_DEFAULT_TOP_REGIMES = 2


@dataclass(frozen=True)
class ProcessedEpitData:
    task_id: str
    X: np.ndarray
    y_z: np.ndarray
    y_mean: float
    y_std: float
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_column: str
    category_mapping: dict[str, int]
    feature_group_counts: dict[str, int]

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def category_count(self) -> int:
        return len(self.category_mapping)

    def schema_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "feature_columns": self.feature_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_column": self.categorical_column,
            "category_count": self.category_count,
            "category_mapping": self.category_mapping,
            "categorical_encoding": "ordinal_codes_before_feature_standardization",
            "feature_scaling": "full_dataset_column_standardization_ddof1",
            "feature_group_counts": self.feature_group_counts,
            "target_mean": self.y_mean,
            "target_std": self.y_std,
            "target_scaling": "full_dataset_standardization_ddof0",
        }


@dataclass(frozen=True)
class SyntheticEpitSample:
    X: np.ndarray
    y: np.ndarray
    d: int
    seq_len: int
    train_size: int
    synthetic_seed: int
    process_unique_count: int
    process_unique_values: list[float]

    def schema_dict(self) -> dict[str, Any]:
        return {
            "n_rows": int(self.X.shape[0]),
            "n_features": int(self.X.shape[1]),
            "d": self.d,
            "seq_len": self.seq_len,
            "train_size": self.train_size,
            "synthetic_seed": self.synthetic_seed,
            "process_column_index": EXPECTED_EPIT_FEATURE_COUNT - 1,
            "process_unique_count": self.process_unique_count,
            "process_unique_values": self.process_unique_values,
            "target_mean": float(np.mean(self.y)),
            "target_std": float(np.std(self.y, ddof=0)),
        }


@dataclass(frozen=True)
class ThetaSurrogateScore:
    synthetic_seed: int
    spearman: float
    standardized_mae: float
    standardized_rmse: float
    predictions: np.ndarray

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "synthetic_seed": self.synthetic_seed,
            "spearman": self.spearman,
            "standardized_mae": self.standardized_mae,
            "standardized_rmse": self.standardized_rmse,
        }


@dataclass(frozen=True)
class EtaCandidate:
    eta_id: str
    anchored_regime: str
    core_anchor: str
    eta_params: dict[str, Any]

    def row_dict(self) -> dict[str, Any]:
        row = {
            "eta_id": self.eta_id,
            "anchored_regime": self.anchored_regime,
            "core_anchor": self.core_anchor,
        }
        row.update(self.eta_params)
        return row


@dataclass(frozen=True)
class EtaSummary:
    eta_id: str
    phase: str
    anchored_regime: str
    core_anchor: str
    n_synth: int
    temperature: float
    ensemble_spearman: float
    standardized_mae: float
    standardized_rmse: float
    median_theta_spearman: float
    max_theta_spearman: float
    ess: float
    collapse_threshold: float
    collapsed: bool
    weights: np.ndarray
    ensemble_predictions: np.ndarray

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "eta_id": self.eta_id,
            "phase": self.phase,
            "anchored_regime": self.anchored_regime,
            "core_anchor": self.core_anchor,
            "n_synth": self.n_synth,
            "temperature": self.temperature,
            "ensemble_spearman": self.ensemble_spearman,
            "standardized_mae": self.standardized_mae,
            "standardized_rmse": self.standardized_rmse,
            "median_theta_spearman": self.median_theta_spearman,
            "max_theta_spearman": self.max_theta_spearman,
            "ESS": self.ess,
            "collapse_threshold": self.collapse_threshold,
            "collapsed": self.collapsed,
        }


def make_epit_task_args(random_state: int = 42) -> argparse.Namespace:
    return argparse.Namespace(
        include_electrochem_features=False,
        dataset=[EPIT_DATASET],
        task=[EPIT_TASK_ID],
        exclude_quality_flag=[],
        target_mode="primary",
        min_samples=40,
        min_class_count=10,
        target_binning="continuous",
        target_bins=2,
        datacortech_protocol="author_simple",
        max_category_cardinality=80,
        min_numeric_finite_ratio=0.80,
        min_categorical_nonmissing_ratio=0.80,
        max_samples_per_task=2000,
        random_state=random_state,
    )


def load_epit_task(random_state: int = 42) -> EvalTask:
    tasks = make_tasks(make_epit_task_args(random_state=random_state))
    if len(tasks) != 1:
        raise RuntimeError(f"Expected exactly one EPIT task, got {len(tasks)}.")

    task = tasks[0]
    if task.task_id != EPIT_TASK_ID:
        raise RuntimeError(f"Loaded unexpected task_id: {task.task_id}")
    if task.target != EPIT_TARGET_COLUMN:
        raise RuntimeError(f"Loaded unexpected EPIT target column: {task.target!r}")
    if dict(task.feature_group_counts) != EXPECTED_EPIT_FEATURE_GROUP_COUNTS:
        raise RuntimeError(
            "EPIT feature group counts changed; expected "
            f"{EXPECTED_EPIT_FEATURE_GROUP_COUNTS}, got {task.feature_group_counts}."
        )
    if task.X.shape[1] != EXPECTED_EPIT_FEATURE_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_EPIT_FEATURE_COUNT} EPIT features, got {task.X.shape[1]}.")
    if EPIT_PROCESS_COLUMN not in task.X.columns:
        raise RuntimeError(f"Missing EPIT process categorical column: {EPIT_PROCESS_COLUMN}")
    if task.X.columns[-1] != EPIT_PROCESS_COLUMN:
        raise RuntimeError(
            "EPIT process categorical column must be the final direct-eval column; got "
            f"{task.X.columns[-1]!r}."
        )
    return task


def _impute_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    means = numeric.mean(axis=0)
    if means.isna().any():
        missing = means.index[means.isna()].tolist()
        raise RuntimeError(f"Cannot mean-impute all-missing EPIT numeric columns: {missing}")
    return numeric.fillna(means).astype(float)


def _ordinal_encode_category(series: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    text = series.astype("string").fillna("<NA>")
    categories = sorted(str(value) for value in text.unique())
    if len(categories) < 2:
        raise RuntimeError("EPIT process categorical column has fewer than two categories.")
    mapping = {category: idx for idx, category in enumerate(categories)}
    codes = text.astype(str).map(mapping).to_numpy(dtype=float)
    if not np.isfinite(codes).all():
        raise RuntimeError("EPIT categorical encoding produced non-finite codes.")
    return codes, mapping


def _standardize_target(y: pd.Series) -> tuple[np.ndarray, float, float]:
    values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("EPIT target contains non-finite values after task construction.")
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 0.0:
        raise RuntimeError("EPIT target has invalid full-dataset standard deviation.")
    return (values - mean) / std, mean, std


def _standardize_features(X: np.ndarray) -> np.ndarray:
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    if not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise RuntimeError("EPIT feature standardization found non-finite column statistics.")
    if np.any(stds <= 0.0):
        raise RuntimeError("EPIT feature standardization found zero-variance columns.")
    return (X - means) / stds


def preprocess_real_epit(task: EvalTask) -> ProcessedEpitData:
    feature_columns = list(task.X.columns)
    numeric_columns = [col for col in feature_columns if col != EPIT_PROCESS_COLUMN]
    if len(numeric_columns) != EXPECTED_EPIT_FEATURE_COUNT - 1:
        raise RuntimeError(f"Expected 20 numeric EPIT columns, got {len(numeric_columns)}.")

    numeric = _impute_numeric_frame(task.X[numeric_columns])
    category_codes, category_mapping = _ordinal_encode_category(task.X[EPIT_PROCESS_COLUMN])
    X_raw = np.column_stack([numeric.to_numpy(dtype=float), category_codes]).astype(float)
    X = _standardize_features(X_raw)
    if X.shape[1] != EXPECTED_EPIT_FEATURE_COUNT:
        raise RuntimeError(f"Expected processed EPIT width 21, got {X.shape[1]}.")
    if not np.isfinite(X).all():
        raise RuntimeError("Processed EPIT features contain non-finite values.")

    y_z, y_mean, y_std = _standardize_target(task.y)
    return ProcessedEpitData(
        task_id=task.task_id,
        X=X,
        y_z=y_z,
        y_mean=y_mean,
        y_std=y_std,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_column=EPIT_PROCESS_COLUMN,
        category_mapping=category_mapping,
        feature_group_counts=dict(task.feature_group_counts),
    )



def anchored_regimes() -> dict[str, dict[str, float]]:
    return {
        "plain": {
            "informed_physical_marginal_prob": 0.0,
            "pitting_material_dirichlet_prob": 0.0,
            "pitting_material_dirichlet_concentration": 1.0,
            "pitting_material_dirichlet_active_prob": 0.45,
        },
        "physical_no_dirichlet": {
            "informed_physical_marginal_prob": 1.0,
            "pitting_material_dirichlet_prob": 0.0,
            "pitting_material_dirichlet_concentration": 1.0,
            "pitting_material_dirichlet_active_prob": 0.45,
        },
        "physical_mild_dirichlet": {
            "informed_physical_marginal_prob": 1.0,
            "pitting_material_dirichlet_prob": 0.5,
            "pitting_material_dirichlet_concentration": 1.0,
            "pitting_material_dirichlet_active_prob": 0.50,
        },
        "physical_sparse_dirichlet": {
            "informed_physical_marginal_prob": 1.0,
            "pitting_material_dirichlet_prob": 1.0,
            "pitting_material_dirichlet_concentration": 0.25,
            "pitting_material_dirichlet_active_prob": 0.35,
        },
    }


def core_anchors() -> dict[str, dict[str, float]]:
    return {
        "material_dominant": {
            "epit_material_coef": 0.70,
            "epit_environment_coef": 0.40,
            "epit_interaction_coef": 0.65,
            "informed_feature_block_strength": 0.35,
            "informed_interaction_strength": 0.50,
        },
        "environment_dominant": {
            "epit_material_coef": 0.50,
            "epit_environment_coef": 0.65,
            "epit_interaction_coef": 0.65,
            "informed_feature_block_strength": 0.35,
            "informed_interaction_strength": 0.50,
        },
        "interaction_dominant": {
            "epit_material_coef": 0.55,
            "epit_environment_coef": 0.45,
            "epit_interaction_coef": 0.95,
            "informed_feature_block_strength": 0.35,
            "informed_interaction_strength": 0.75,
        },
        "balanced": {
            "epit_material_coef": 0.575,
            "epit_environment_coef": 0.50,
            "epit_interaction_coef": 0.775,
            "informed_feature_block_strength": 0.30,
            "informed_interaction_strength": 0.35,
        },
        "weak_prior": {
            "epit_material_coef": 0.50,
            "epit_environment_coef": 0.40,
            "epit_interaction_coef": 0.65,
            "informed_feature_block_strength": 0.15,
            "informed_interaction_strength": 0.20,
        },
        "strong_prior": {
            "epit_material_coef": 0.65,
            "epit_environment_coef": 0.60,
            "epit_interaction_coef": 0.90,
            "informed_feature_block_strength": 0.60,
            "informed_interaction_strength": 0.85,
        },
    }


def build_phase1_candidates(max_etas: int = 0) -> list[EtaCandidate]:
    candidates: list[EtaCandidate] = []
    for regime_name, regime_params in anchored_regimes().items():
        for anchor_name, anchor_params in core_anchors().items():
            eta_params = {**regime_params, **anchor_params}
            candidates.append(
                EtaCandidate(
                    eta_id=f"{regime_name}__{anchor_name}",
                    anchored_regime=regime_name,
                    core_anchor=anchor_name,
                    eta_params=eta_params,
                )
            )
    if max_etas > 0:
        return candidates[: int(max_etas)]
    return candidates


def core_search_ranges() -> dict[str, tuple[float, float]]:
    return {
        "epit_material_coef": (0.45, 0.70),
        "epit_environment_coef": (0.35, 0.65),
        "epit_interaction_coef": (0.60, 0.95),
        "informed_feature_block_strength": (0.00, 0.95),
        "informed_interaction_strength": (0.00, 1.00),
    }


def read_phase1_summary(source_dir: Path) -> pd.DataFrame:
    summary_path = Path(source_dir) / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Phase 2 requires Phase 1 summary.csv at {summary_path}.")
    frame = pd.read_csv(summary_path)
    required = {"eta_id", "phase", "anchored_regime", "ensemble_spearman", "ESS"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Phase 1 summary.csv is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Phase 1 summary.csv is empty.")
    if "selected_rank" in frame.columns:
        frame = frame.sort_values("selected_rank", kind="mergesort")
    else:
        frame = frame.sort_values(["ensemble_spearman", "ESS"], ascending=[False, False], kind="mergesort")
    return frame.reset_index(drop=True)


def select_phase2_regimes(phase1_summary: pd.DataFrame, top_regimes: int) -> list[str]:
    if int(top_regimes) <= 0:
        raise ValueError("top_regimes must be positive.")
    regimes: list[str] = []
    for value in phase1_summary["anchored_regime"].tolist():
        regime = str(value)
        if regime not in anchored_regimes():
            raise ValueError(f"Unknown anchored regime in Phase 1 summary: {regime}")
        if regime not in regimes:
            regimes.append(regime)
        if len(regimes) == int(top_regimes):
            break
    if not regimes:
        raise ValueError("Could not select any Phase 2 regimes from Phase 1 summary.")
    return regimes


def latin_hypercube_core_params(n_samples: int, random_state: int) -> list[dict[str, float]]:
    if int(n_samples) <= 0:
        raise ValueError("n_core_samples must be positive.")
    n_samples = int(n_samples)
    rng = np.random.default_rng(int(random_state))
    ranges = core_search_ranges()
    params_by_sample = [dict() for _ in range(n_samples)]
    for name, (low, high) in ranges.items():
        centers = (np.arange(n_samples, dtype=float) + 0.5) / float(n_samples)
        rng.shuffle(centers)
        values = float(low) + centers * (float(high) - float(low))
        for idx, value in enumerate(values):
            params_by_sample[idx][name] = float(value)
    return params_by_sample


def build_phase2_candidates(
    phase1_summary: pd.DataFrame,
    *,
    top_regimes: int = PHASE2_DEFAULT_TOP_REGIMES,
    n_core_samples: int = PHASE2_DEFAULT_CORE_SAMPLES,
    random_state: int = 42,
) -> list[EtaCandidate]:
    selected_regimes = select_phase2_regimes(phase1_summary, top_regimes=top_regimes)
    core_samples = latin_hypercube_core_params(n_core_samples, random_state=random_state)
    regimes = anchored_regimes()
    candidates: list[EtaCandidate] = []
    for regime in selected_regimes:
        for idx, core_params in enumerate(core_samples):
            eta_params = {**regimes[regime], **core_params}
            core_anchor = f"space_filling_{idx:04d}"
            candidates.append(
                EtaCandidate(
                    eta_id=f"phase2__{regime}__{core_anchor}",
                    anchored_regime=regime,
                    core_anchor=core_anchor,
                    eta_params=eta_params,
                )
            )
    return candidates



def build_fixed_hp_for_eta(category_count: int, eta_params: dict[str, Any] | None = None) -> dict[str, Any]:
    if int(category_count) < 2:
        raise ValueError("category_count must be >= 2 for the forced EPIT process slot.")

    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp.update(
        {
            "mix_probs": (0.7, 0.3),
            "informed_mix_probs": (0.7, 0.3),
            "informed_task_family_probs": (1.0, 0.0),
            "informed_normal_block_allocation": DIRECT_EPIT_BLOCK_ALLOCATION,
            "informed_normal_block_allocation_min_counts": DIRECT_EPIT_BLOCK_ALLOCATION,
            "informed_target_family": "pitting_potential",
            "informed_physical_marginal_profile": "pitting_potential_v1",
            "informed_physical_marginal_prob": 0.0,
            "informed_history_strength": 0.0,
            "informed_intervention_strength": 0.0,
            "epit_material_coef": 0.575,
            "epit_environment_coef": 0.50,
            "epit_interaction_coef": 0.775,
            "informed_feature_block_strength": 0.30,
            "informed_interaction_strength": 0.35,
            "pitting_material_dirichlet_prob": 0.0,
            "pitting_material_dirichlet_concentration": 1.0,
            "pitting_material_dirichlet_active_prob": 0.45,
            "pitting_process_role": "test_method_category",
            "pitting_process_category_count": int(category_count),
            "cat_prob": 0.0,
            "permute_features": False,
        }
    )
    if eta_params:
        fixed_hp.update(eta_params)
    return fixed_hp


def synthetic_train_size_bounds(seq_len: int) -> tuple[int, int]:
    train_size = int(int(seq_len) * DIRECT_EPIT_TRAIN_SIZE_RATIO)
    train_size = max(1, min(train_size, int(seq_len) - 1))
    return train_size, train_size + 1


def sample_synthetic_dataset(
    *,
    category_count: int,
    synthetic_seed: int,
    seq_len: int,
    eta_params: dict[str, Any] | None = None,
) -> SyntheticEpitSample:
    if int(seq_len) < 16:
        raise ValueError("seq_len must be at least 16 for a usable direct EPIT synthetic sample.")

    np.random.seed(synthetic_seed)
    random.seed(synthetic_seed)
    torch.manual_seed(synthetic_seed)
    min_train_size, max_train_size = synthetic_train_size_bounds(int(seq_len))

    dataset = PriorDataset(
        batch_size=1,
        batch_size_per_gp=1,
        min_features=EXPECTED_EPIT_FEATURE_COUNT,
        max_features=EXPECTED_EPIT_FEATURE_COUNT,
        max_classes=0,
        min_seq_len=None,
        max_seq_len=int(seq_len),
        min_train_size=min_train_size,
        max_train_size=max_train_size,
        prior_type="informed_scm",
        scm_fixed_hp=build_fixed_hp_for_eta(category_count, eta_params=eta_params),
        scm_sampled_hp=DEFAULT_SAMPLED_HP,
        n_jobs=1,
        informed_prior_ratio=1.0,
        device="cpu",
    )
    X, y, d, seq_lens, train_sizes = dataset.get_batch()
    X_np = X[0].detach().cpu().numpy().astype(float)
    y_np = y[0].detach().cpu().numpy().astype(float)
    d_value = int(d[0].item())
    seq_len_value = int(seq_lens[0].item())
    train_size_value = int(train_sizes[0].item())

    if X_np.shape != (int(seq_len), EXPECTED_EPIT_FEATURE_COUNT):
        raise RuntimeError(f"Expected synthetic shape {(int(seq_len), EXPECTED_EPIT_FEATURE_COUNT)}, got {X_np.shape}.")
    if d_value != EXPECTED_EPIT_FEATURE_COUNT:
        raise RuntimeError(f"Expected all {EXPECTED_EPIT_FEATURE_COUNT} synthetic features to remain active, got d={d_value}.")
    if not np.isfinite(X_np).all() or not np.isfinite(y_np).all():
        raise RuntimeError("Synthetic EPIT sample contains non-finite values.")
    if float(np.std(y_np, ddof=0)) <= 0.0:
        raise RuntimeError("Synthetic EPIT target has zero variance.")

    process_unique = np.unique(X_np[:, -1])
    return SyntheticEpitSample(
        X=X_np,
        y=y_np,
        d=d_value,
        seq_len=seq_len_value,
        train_size=train_size_value,
        synthetic_seed=int(synthetic_seed),
        process_unique_count=int(process_unique.size),
        process_unique_values=[float(value) for value in process_unique.tolist()],
    )


def spearman_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Spearman inputs must have the same shape, got {y_true.shape} and {y_pred.shape}.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Spearman inputs must be finite.")
    rho = float(spearmanr(y_true, y_pred).correlation)
    return rho if np.isfinite(rho) else 0.0


def fit_and_score_theta(processed: ProcessedEpitData, synthetic: SyntheticEpitSample) -> ThetaSurrogateScore:
    if synthetic.X.shape[1] != processed.n_features:
        raise RuntimeError(
            f"Synthetic/real feature width mismatch: {synthetic.X.shape[1]} vs {processed.n_features}."
        )

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(synthetic.X, synthetic.y)
    predictions = np.asarray(model.predict(processed.X), dtype=float).reshape(-1)
    if predictions.shape != processed.y_z.shape:
        raise RuntimeError(f"Expected predictions shape {processed.y_z.shape}, got {predictions.shape}.")
    if not np.isfinite(predictions).all():
        raise RuntimeError("Ridge surrogate produced non-finite predictions.")

    residuals = predictions - processed.y_z
    return ThetaSurrogateScore(
        synthetic_seed=synthetic.synthetic_seed,
        spearman=spearman_score(processed.y_z, predictions),
        standardized_mae=float(np.mean(np.abs(residuals))),
        standardized_rmse=float(np.sqrt(np.mean(residuals**2))),
        predictions=predictions,
    )



def spearman_loss(spearman: float) -> float:
    rho = float(np.clip(spearman, -1.0, 1.0))
    return (1.0 - rho) / 2.0


def weights_from_spearman(spearmans: list[float] | np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    values = np.asarray(spearmans, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("At least one Spearman score is required for weighting.")
    if not np.isfinite(values).all():
        raise ValueError("Spearman scores must be finite for weighting.")

    losses = np.asarray([spearman_loss(value) for value in values], dtype=float)
    scaled = -(losses - losses.min()) / temperature
    raw_weights = np.exp(scaled)
    total = float(raw_weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Synthetic-task weights collapsed to an invalid total.")
    return raw_weights / total


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("At least one weight is required for ESS.")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("Weights must be finite and non-negative.")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Weights must have positive total mass.")
    normalized = weights / total
    return float(1.0 / np.sum(normalized**2))


def collapse_threshold(n_synth: int) -> float:
    return max(5.0, 0.10 * float(n_synth))


def summarize_eta(
    processed: ProcessedEpitData,
    scores: list[ThetaSurrogateScore],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    eta_id: str = DEFAULT_ETA_ID,
    phase: str = "smoke",
    anchored_regime: str = "plain",
    core_anchor: str = "balanced",
) -> EtaSummary:
    if not scores:
        raise ValueError("At least one theta score is required to summarize an eta.")

    spearmans = np.asarray([score.spearman for score in scores], dtype=float)
    weights = weights_from_spearman(spearmans, temperature=temperature)
    predictions = np.vstack([score.predictions for score in scores])
    ensemble_predictions = weights @ predictions
    residuals = ensemble_predictions - processed.y_z
    ess = effective_sample_size(weights)
    threshold = collapse_threshold(len(scores))

    return EtaSummary(
        eta_id=eta_id,
        phase=phase,
        anchored_regime=anchored_regime,
        core_anchor=core_anchor,
        n_synth=len(scores),
        temperature=float(temperature),
        ensemble_spearman=spearman_score(processed.y_z, ensemble_predictions),
        standardized_mae=float(np.mean(np.abs(residuals))),
        standardized_rmse=float(np.sqrt(np.mean(residuals**2))),
        median_theta_spearman=float(np.median(spearmans)),
        max_theta_spearman=float(np.max(spearmans)),
        ess=ess,
        collapse_threshold=threshold,
        collapsed=bool(ess < threshold),
        weights=weights,
        ensemble_predictions=ensemble_predictions,
    )


def sample_and_score_eta(
    processed: ProcessedEpitData,
    *,
    n_synth: int,
    synthetic_seed_start: int,
    seq_len: int,
    temperature: float = DEFAULT_TEMPERATURE,
    eta_params: dict[str, Any] | None = None,
    eta_id: str = DEFAULT_ETA_ID,
    phase: str = "smoke",
    anchored_regime: str = "plain",
    core_anchor: str = "balanced",
) -> tuple[list[SyntheticEpitSample], list[ThetaSurrogateScore], EtaSummary]:
    if int(n_synth) <= 0:
        raise ValueError("n_synth must be positive.")

    samples: list[SyntheticEpitSample] = []
    scores: list[ThetaSurrogateScore] = []
    for offset in range(int(n_synth)):
        synthetic = sample_synthetic_dataset(
            category_count=processed.category_count,
            synthetic_seed=int(synthetic_seed_start) + offset,
            seq_len=seq_len,
            eta_params=eta_params,
        )
        samples.append(synthetic)
        scores.append(fit_and_score_theta(processed, synthetic))

    summary = summarize_eta(
        processed,
        scores,
        temperature=temperature,
        eta_id=eta_id,
        phase=phase,
        anchored_regime=anchored_regime,
        core_anchor=core_anchor,
    )
    return samples, scores, summary


def theta_scores_frame(scores: list[ThetaSurrogateScore], summary: EtaSummary) -> pd.DataFrame:
    rows = []
    for score in scores:
        row = {
            "eta_id": summary.eta_id,
            "phase": summary.phase,
            "anchored_regime": summary.anchored_regime,
            "core_anchor": summary.core_anchor,
        }
        row.update(score.metrics_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def weights_frame(scores: list[ThetaSurrogateScore], summary: EtaSummary) -> pd.DataFrame:
    rows = []
    for score, weight in zip(scores, summary.weights):
        rows.append(
            {
                "eta_id": summary.eta_id,
                "phase": summary.phase,
                "anchored_regime": summary.anchored_regime,
                "core_anchor": summary.core_anchor,
                "synthetic_seed": score.synthetic_seed,
                "spearman": score.spearman,
                "loss": spearman_loss(score.spearman),
                "weight": float(weight),
            }
        )
    return pd.DataFrame(rows)


def eta_candidates_frame(candidates: list[EtaCandidate]) -> pd.DataFrame:
    return pd.DataFrame([candidate.row_dict() for candidate in candidates])


def summaries_frame(summaries: list[EtaSummary]) -> pd.DataFrame:
    frame = pd.DataFrame([summary.metrics_dict() for summary in summaries])
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["ensemble_spearman", "ESS", "standardized_rmse"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame.insert(0, "selected_rank", np.arange(1, len(frame) + 1, dtype=int))
    return frame


def evaluate_candidate_eta(
    processed: ProcessedEpitData,
    candidate: EtaCandidate,
    *,
    n_synth: int,
    synthetic_seed_start: int,
    seq_len: int,
    temperature: float,
    phase: str,
) -> tuple[list[SyntheticEpitSample], list[ThetaSurrogateScore], EtaSummary]:
    return sample_and_score_eta(
        processed,
        n_synth=n_synth,
        synthetic_seed_start=synthetic_seed_start,
        seq_len=seq_len,
        temperature=temperature,
        eta_params=candidate.eta_params,
        eta_id=candidate.eta_id,
        phase=phase,
        anchored_regime=candidate.anchored_regime,
        core_anchor=candidate.core_anchor,
    )


def run_phase1(
    processed: ProcessedEpitData,
    *,
    n_synth: int,
    synthetic_seed_start: int,
    seq_len: int,
    temperature: float,
    max_etas: int = 0,
) -> tuple[
    list[EtaCandidate],
    SyntheticEpitSample,
    dict[str, list[ThetaSurrogateScore]],
    list[EtaSummary],
]:
    candidates = build_phase1_candidates(max_etas=max_etas)
    if not candidates:
        raise ValueError("Phase 1 requires at least one eta candidate.")

    first_sample: SyntheticEpitSample | None = None
    scores_by_eta: dict[str, list[ThetaSurrogateScore]] = {}
    summaries: list[EtaSummary] = []
    for idx, candidate in enumerate(candidates, start=1):
        samples, scores, summary = evaluate_candidate_eta(
            processed,
            candidate,
            n_synth=n_synth,
            synthetic_seed_start=synthetic_seed_start,
            seq_len=seq_len,
            temperature=temperature,
            phase="phase1",
        )
        if first_sample is None:
            first_sample = samples[0]
        scores_by_eta[candidate.eta_id] = scores
        summaries.append(summary)
        print(
            "Phase 1 eta "
            f"{idx}/{len(candidates)} {candidate.eta_id}: "
            f"spearman={summary.ensemble_spearman:.6f}, ESS={summary.ess:.2f}"
        )

    if first_sample is None:
        raise RuntimeError("Phase 1 did not produce a synthetic sample.")
    return candidates, first_sample, scores_by_eta, summaries


def write_phase1_outputs(
    processed: ProcessedEpitData,
    candidates: list[EtaCandidate],
    first_sample: SyntheticEpitSample,
    scores_by_eta: dict[str, list[ThetaSurrogateScore]],
    summaries: list[EtaSummary],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "phase": args.phase,
        "random_state": args.random_state,
        "synthetic_seed_start": args.synthetic_seed,
        "n_synth": args.n_synth,
        "temperature": args.temperature,
        "max_etas": args.max_etas,
        "n_etas": len(candidates),
        "task_id": EPIT_TASK_ID,
        "scope": "phase1_anchor_regime_grid",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output_dir / "real_schema.json").write_text(
        json.dumps(processed.schema_dict(), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "synthetic_smoke_schema.json").write_text(
        json.dumps(first_sample.schema_dict(), indent=2, sort_keys=True) + "\n"
    )
    eta_candidates_frame(candidates).to_csv(output_dir / "etas.csv", index=False)
    theta_frames = [theta_scores_frame(scores_by_eta[summary.eta_id], summary) for summary in summaries]
    weight_frames = [weights_frame(scores_by_eta[summary.eta_id], summary) for summary in summaries]
    pd.concat(theta_frames, ignore_index=True).to_csv(output_dir / "theta_scores.csv", index=False)
    pd.concat(weight_frames, ignore_index=True).to_csv(output_dir / "weights.csv", index=False)
    summaries_frame(summaries).to_csv(output_dir / "summary.csv", index=False)



def run_phase2(
    processed: ProcessedEpitData,
    *,
    phase1_source_dir: Path,
    n_synth: int,
    synthetic_seed_start: int,
    seq_len: int,
    temperature: float,
    n_core_samples: int,
    top_regimes: int,
    random_state: int,
) -> tuple[
    list[EtaCandidate],
    SyntheticEpitSample,
    dict[str, list[ThetaSurrogateScore]],
    list[EtaSummary],
]:
    phase1_summary = read_phase1_summary(phase1_source_dir)
    candidates = build_phase2_candidates(
        phase1_summary,
        top_regimes=top_regimes,
        n_core_samples=n_core_samples,
        random_state=random_state,
    )
    if not candidates:
        raise ValueError("Phase 2 requires at least one eta candidate.")

    first_sample: SyntheticEpitSample | None = None
    scores_by_eta: dict[str, list[ThetaSurrogateScore]] = {}
    summaries: list[EtaSummary] = []
    for idx, candidate in enumerate(candidates, start=1):
        samples, scores, summary = evaluate_candidate_eta(
            processed,
            candidate,
            n_synth=n_synth,
            synthetic_seed_start=synthetic_seed_start,
            seq_len=seq_len,
            temperature=temperature,
            phase="phase2",
        )
        if first_sample is None:
            first_sample = samples[0]
        scores_by_eta[candidate.eta_id] = scores
        summaries.append(summary)
        print(
            "Phase 2 eta "
            f"{idx}/{len(candidates)} {candidate.eta_id}: "
            f"spearman={summary.ensemble_spearman:.6f}, ESS={summary.ess:.2f}"
        )

    if first_sample is None:
        raise RuntimeError("Phase 2 did not produce a synthetic sample.")
    return candidates, first_sample, scores_by_eta, summaries


def write_phase2_outputs(
    processed: ProcessedEpitData,
    candidates: list[EtaCandidate],
    first_sample: SyntheticEpitSample,
    scores_by_eta: dict[str, list[ThetaSurrogateScore]],
    summaries: list[EtaSummary],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "phase": args.phase,
        "random_state": args.random_state,
        "synthetic_seed_start": args.synthetic_seed,
        "n_synth": args.n_synth,
        "temperature": args.temperature,
        "phase2_source_dir": str(args.phase2_source_dir),
        "phase2_top_regimes": args.phase2_top_regimes,
        "n_core_samples": args.n_core_samples,
        "n_etas": len(candidates),
        "task_id": EPIT_TASK_ID,
        "scope": "phase2_space_filling_core_search",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output_dir / "real_schema.json").write_text(
        json.dumps(processed.schema_dict(), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "synthetic_smoke_schema.json").write_text(
        json.dumps(first_sample.schema_dict(), indent=2, sort_keys=True) + "\n"
    )
    eta_candidates_frame(candidates).to_csv(output_dir / "etas.csv", index=False)
    theta_frames = [theta_scores_frame(scores_by_eta[summary.eta_id], summary) for summary in summaries]
    weight_frames = [weights_frame(scores_by_eta[summary.eta_id], summary) for summary in summaries]
    pd.concat(theta_frames, ignore_index=True).to_csv(output_dir / "theta_scores.csv", index=False)
    pd.concat(weight_frames, ignore_index=True).to_csv(output_dir / "weights.csv", index=False)
    summaries_frame(summaries).to_csv(output_dir / "summary.csv", index=False)



def default_output_dir(phase: str = "smoke") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"epit_direct_prior_{phase}_{stamp}"


def write_smoke_outputs(
    processed: ProcessedEpitData,
    synthetic_samples: list[SyntheticEpitSample],
    scores: list[ThetaSurrogateScore],
    summary: EtaSummary,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    if not synthetic_samples or not scores:
        raise ValueError("Smoke outputs require at least one synthetic sample and score.")

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "phase": args.phase,
        "random_state": args.random_state,
        "synthetic_seed_start": args.synthetic_seed,
        "n_synth": args.n_synth,
        "temperature": args.temperature,
        "task_id": EPIT_TASK_ID,
        "scope": "single_eta_direct_prior_smoke",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output_dir / "real_schema.json").write_text(
        json.dumps(processed.schema_dict(), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "synthetic_smoke_schema.json").write_text(
        json.dumps(synthetic_samples[0].schema_dict(), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "theta_smoke_score.json").write_text(
        json.dumps(scores[0].metrics_dict(), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "eta_smoke_summary.json").write_text(
        json.dumps(summary.metrics_dict(), indent=2, sort_keys=True) + "\n"
    )
    theta_scores_frame(scores, summary).to_csv(output_dir / "theta_scores.csv", index=False)
    weights_frame(scores, summary).to_csv(output_dir / "weights.csv", index=False)
    pd.DataFrame([summary.metrics_dict()]).to_csv(output_dir / "summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("smoke", "phase1", "phase2"),
        default="smoke",
        help="Direct-prior phase. Phase 1 runs the anchored regime/core-anchor grid.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--synthetic-seed", type=int, default=0)
    parser.add_argument(
        "--n-synth",
        type=int,
        default=1,
        help="Synthetic tasks per eta. Use 32 for Phase 1 and 64+ for Phase 2 planned runs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Fixed synthetic-task weighting temperature.",
    )
    parser.add_argument(
        "--synthetic-seq-len",
        type=int,
        default=0,
        help="Synthetic rows for smoke sampling. Default 0 uses the real EPIT row count.",
    )
    parser.add_argument(
        "--max-etas",
        type=int,
        default=0,
        help="Limit Phase 1 to the first N etas for debugging. Default 0 runs all 24.",
    )
    parser.add_argument(
        "--phase2-source-dir",
        type=Path,
        default=None,
        help="Phase 1 output directory containing summary.csv for Phase 2 regime selection.",
    )
    parser.add_argument(
        "--phase2-top-regimes",
        type=int,
        default=PHASE2_DEFAULT_TOP_REGIMES,
        help="Number of top anchored regimes to keep for Phase 2.",
    )
    parser.add_argument(
        "--n-core-samples",
        type=int,
        default=PHASE2_DEFAULT_CORE_SAMPLES,
        help="Space-filling core candidates per retained anchored regime for Phase 2.",
    )
    parser.add_argument("--no-write", action="store_true", help="Load and preprocess without writing artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = load_epit_task(random_state=args.random_state)
    processed = preprocess_real_epit(task)
    synthetic_seq_len = processed.n_rows if args.synthetic_seq_len == 0 else args.synthetic_seq_len
    output_dir = args.output_dir or default_output_dir(args.phase)

    print(
        "Loaded real EPIT direct schema: "
        f"rows={processed.n_rows}, features={processed.n_features}, "
        f"category_count={processed.category_count}"
    )

    if args.phase == "phase1":
        candidates, first_sample, scores_by_eta, summaries = run_phase1(
            processed,
            n_synth=args.n_synth,
            synthetic_seed_start=args.synthetic_seed,
            seq_len=synthetic_seq_len,
            temperature=args.temperature,
            max_etas=args.max_etas,
        )
        ranked = summaries_frame(summaries)
        if not args.no_write:
            write_phase1_outputs(
                processed,
                candidates,
                first_sample,
                scores_by_eta,
                summaries,
                output_dir,
                args,
            )
        best = ranked.iloc[0]
        best_eta = str(best["eta_id"])
        best_spearman = float(best["ensemble_spearman"])
        best_ess = float(best["ESS"])
        print(
            "Phase 1 complete: "
            f"n_etas={len(candidates)}, n_synth_per_eta={args.n_synth}, "
            f"best_eta={best_eta}, best_spearman={best_spearman:.6f}, best_ESS={best_ess:.2f}"
        )
        if not args.no_write:
            print(f"Wrote {output_dir}")
        return

    if args.phase == "phase2":
        if args.phase2_source_dir is None:
            raise SystemExit("--phase phase2 requires --phase2-source-dir pointing to a Phase 1 output directory.")
        candidates, first_sample, scores_by_eta, summaries = run_phase2(
            processed,
            phase1_source_dir=args.phase2_source_dir,
            n_synth=args.n_synth,
            synthetic_seed_start=args.synthetic_seed,
            seq_len=synthetic_seq_len,
            temperature=args.temperature,
            n_core_samples=args.n_core_samples,
            top_regimes=args.phase2_top_regimes,
            random_state=args.random_state,
        )
        ranked = summaries_frame(summaries)
        if not args.no_write:
            write_phase2_outputs(
                processed,
                candidates,
                first_sample,
                scores_by_eta,
                summaries,
                output_dir,
                args,
            )
        best = ranked.iloc[0]
        best_eta = str(best["eta_id"])
        best_spearman = float(best["ensemble_spearman"])
        best_ess = float(best["ESS"])
        print(
            "Phase 2 complete: "
            f"n_etas={len(candidates)}, n_synth_per_eta={args.n_synth}, "
            f"best_eta={best_eta}, best_spearman={best_spearman:.6f}, best_ESS={best_ess:.2f}"
        )
        if not args.no_write:
            print(f"Wrote {output_dir}")
        return

    if args.phase != "smoke":
        raise SystemExit(f"Unknown phase: {args.phase}")

    synthetic_samples, scores, summary = sample_and_score_eta(
        processed,
        n_synth=args.n_synth,
        synthetic_seed_start=args.synthetic_seed,
        seq_len=synthetic_seq_len,
        temperature=args.temperature,
    )
    first_sample = synthetic_samples[0]
    first_score = scores[0]
    if not args.no_write:
        write_smoke_outputs(processed, synthetic_samples, scores, summary, output_dir, args)

    print(
        "Sampled synthetic EPIT smoke dataset: "
        f"n_synth={len(synthetic_samples)}, rows={first_sample.X.shape[0]}, "
        f"features={first_sample.X.shape[1]}, d={first_sample.d}, "
        f"process_unique_count={first_sample.process_unique_count}"
    )
    print(
        "First synthetic Ridge surrogate: "
        f"spearman={first_score.spearman:.6f}, standardized_mae={first_score.standardized_mae:.6f}, "
        f"standardized_rmse={first_score.standardized_rmse:.6f}"
    )
    print(
        "One-eta weighted Ridge ensemble: "
        f"spearman={summary.ensemble_spearman:.6f}, standardized_mae={summary.standardized_mae:.6f}, "
        f"standardized_rmse={summary.standardized_rmse:.6f}, ESS={summary.ess:.2f}"
    )
    if not args.no_write:
        print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
