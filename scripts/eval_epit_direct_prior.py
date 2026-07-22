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
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_corrosion_datasets import EvalTask, make_tasks  # noqa: E402
from tabicl.prior.dataset import PriorDataset, SCMPrior  # noqa: E402
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
ENSEMBLE_WEIGHTING_SCHEME = "uniform_theta_average"
DEFAULT_ETA_ID = "plain_balanced"
PHASE1_DEFAULT_N_SYNTH = 32
PHASE2_DEFAULT_N_SYNTH = 64
PHASE2_DEFAULT_CORE_SAMPLES = 64
PHASE2_DEFAULT_TOP_REGIMES = 2
DEFAULT_CHUNK_SIZE = 4
DEFAULT_PROGRESS_INTERVAL_SECONDS = 30.0
DEFAULT_SCHEMA_RETRY_ATTEMPTS = 50
SCHEMA_RETRY_SEED_STRIDE = 1_000_003
THETA_ARTIFACT_FORMAT = "theta_npz_v4_mlp_mix_pren_epit_rule"
TARGET_RULE_DIAGNOSTIC = "epit_target_rule_oracle"
DEFAULT_INFORMED_MLP_PROB = 0.70
PHASE1_INFORMED_MLP_PROB_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
PHASE2_INFORMED_MLP_PROB_WINDOW = 0.25
BASELINE_PRIOR_CONTROL_ID = "tabicl_default_prior_conditioned_d21"
BASELINE_PRIOR_CONTROL_PHASE = "baseline_prior_control"
BASELINE_PRIOR_CONTROL_ARTIFACT_FORMAT = "baseline_prior_npz_v1_default_conditioned_d21"
BASELINE_PRIOR_CONTROL_FEATURE_MODE = "default_variable_conditioned_d21"
BASELINE_PRIOR_CONTROL_PRIOR_TYPE = "mix_scm"
BASELINE_PRIOR_CONTROL_MIN_FEATURES = 2
BASELINE_PRIOR_CONTROL_MAX_FEATURES = 100
DEFAULT_BASELINE_CONTROL_MAX_ATTEMPTS = 2_000


_WORKER_X_REAL: np.ndarray | None = None
_WORKER_X_REAL_RAW: np.ndarray | None = None
_WORKER_Y_REAL_Z: np.ndarray | None = None
_WORKER_CATEGORY_COUNT: int | None = None
_WORKER_SEQ_LEN: int | None = None


@dataclass(frozen=True)
class ProcessedEpitData:
    task_id: str
    X: np.ndarray
    X_raw: np.ndarray
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
            "raw_feature_representation": "mean_imputed_original_numeric_units_with_ordinal_process_codes",
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
    sampling_seed: int
    schema_attempts: int
    process_unique_count: int
    process_unique_values: list[float]
    target_rule: dict[str, Any]

    def schema_dict(self) -> dict[str, Any]:
        return {
            "n_rows": int(self.X.shape[0]),
            "n_features": int(self.X.shape[1]),
            "d": self.d,
            "seq_len": self.seq_len,
            "train_size": self.train_size,
            "synthetic_seed": self.synthetic_seed,
            "sampling_seed": self.sampling_seed,
            "schema_attempts": self.schema_attempts,
            "process_column_index": EXPECTED_EPIT_FEATURE_COUNT - 1,
            "process_unique_count": self.process_unique_count,
            "process_unique_values": self.process_unique_values,
            "target_rule_type": str(self.target_rule.get("rule_type", "")),
            "target_rule_synthetic_environment_mode": str(self.target_rule.get("synthetic_environment_mode", "")),
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
    weighting_scheme: str
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
            "weighting_scheme": self.weighting_scheme,
            "ensemble_spearman": self.ensemble_spearman,
            "standardized_mae": self.standardized_mae,
            "standardized_rmse": self.standardized_rmse,
            "median_theta_spearman": self.median_theta_spearman,
            "max_theta_spearman": self.max_theta_spearman,
            "ESS": self.ess,
            "collapse_threshold": self.collapse_threshold,
            "collapsed": self.collapsed,
        }


@dataclass(frozen=True)
class BaselinePriorControlRecord:
    score: ThetaSurrogateScore
    sampling_seed: int
    schema_attempts: int
    d: int
    seq_len: int
    train_size: int
    raw_feature_count: int


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
        X_raw=X_raw,
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
            "informed_target_mix_weight": 0.50,
        },
        "environment_dominant": {
            "epit_material_coef": 0.50,
            "epit_environment_coef": 0.65,
            "epit_interaction_coef": 0.65,
            "informed_feature_block_strength": 0.35,
            "informed_target_mix_weight": 0.50,
        },
        "interaction_dominant": {
            "epit_material_coef": 0.55,
            "epit_environment_coef": 0.45,
            "epit_interaction_coef": 0.95,
            "informed_feature_block_strength": 0.35,
            "informed_target_mix_weight": 0.75,
        },
        "balanced": {
            "epit_material_coef": 0.575,
            "epit_environment_coef": 0.50,
            "epit_interaction_coef": 0.775,
            "informed_feature_block_strength": 0.30,
            "informed_target_mix_weight": 0.35,
        },
        "weak_prior": {
            "epit_material_coef": 0.50,
            "epit_environment_coef": 0.40,
            "epit_interaction_coef": 0.65,
            "informed_feature_block_strength": 0.15,
            "informed_target_mix_weight": 0.20,
        },
        "strong_prior": {
            "epit_material_coef": 0.65,
            "epit_environment_coef": 0.60,
            "epit_interaction_coef": 0.90,
            "informed_feature_block_strength": 0.60,
            "informed_target_mix_weight": 0.85,
        },
    }


def build_phase1_candidates(max_etas: int = 0) -> list[EtaCandidate]:
    candidates: list[EtaCandidate] = []
    for regime_name, regime_params in anchored_regimes().items():
        for anchor_name, anchor_params in core_anchors().items():
            for mlp_prob in PHASE1_INFORMED_MLP_PROB_GRID:
                eta_params = {
                    **regime_params,
                    **anchor_params,
                    "informed_mlp_prob": float(mlp_prob),
                }
                candidates.append(
                    EtaCandidate(
                        eta_id=f"{regime_name}__{anchor_name}__mlp{_mlp_prob_label(mlp_prob)}",
                        anchored_regime=regime_name,
                        core_anchor=anchor_name,
                        eta_params=eta_params,
                    )
                )
    if max_etas > 0:
        return candidates[: int(max_etas)]
    return candidates


def _mlp_prob_label(value: float) -> str:
    return f"{int(round(float(value) * 100.0)):03d}"


def core_search_ranges() -> dict[str, tuple[float, float]]:
    return {
        "epit_material_coef": (0.45, 0.70),
        "epit_environment_coef": (0.35, 0.65),
        "epit_interaction_coef": (0.60, 0.95),
        "informed_feature_block_strength": (0.00, 0.95),
        "informed_target_mix_weight": (0.00, 1.00),
    }


def read_phase1_summary(source_dir: Path) -> pd.DataFrame:
    summary_path = Path(source_dir) / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Phase 2 requires Phase 1 summary.csv at {summary_path}.")
    frame = pd.read_csv(summary_path)
    required = {"eta_id", "phase", "anchored_regime", "ensemble_spearman", "ESS", "weighting_scheme"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Phase 1 summary.csv is missing required columns: {missing}. "
            "Rerun Phase 1 with the current uniform-theta scorer before Phase 2."
        )
    if frame.empty:
        raise ValueError("Phase 1 summary.csv is empty.")
    schemes = set(frame["weighting_scheme"].astype(str))
    if schemes != {ENSEMBLE_WEIGHTING_SCHEME}:
        raise ValueError(
            "Phase 2 requires a Phase 1 summary scored with "
            f"{ENSEMBLE_WEIGHTING_SCHEME}; got {sorted(schemes)}."
        )
    etas_path = Path(source_dir) / "etas.csv"
    if "informed_mlp_prob" not in frame.columns:
        if not etas_path.exists():
            raise ValueError(
                f"Phase 2 requires Phase 1 eta metadata with informed_mlp_prob at {etas_path}. "
                "Rerun Phase 1 with the current MLP/tree split search."
            )
        etas = pd.read_csv(etas_path)
        eta_required = {"eta_id", "informed_mlp_prob"}
        eta_missing = sorted(eta_required - set(etas.columns))
        if eta_missing:
            raise ValueError(
                f"Phase 1 etas.csv is missing required columns: {eta_missing}. "
                "Rerun Phase 1 with the current MLP/tree split search."
            )
        frame = frame.merge(etas[["eta_id", "informed_mlp_prob"]], on="eta_id", how="left")
    if frame["informed_mlp_prob"].isna().any():
        raise ValueError(
            "Phase 1 summary is missing informed_mlp_prob for one or more selected etas. "
            "Rerun Phase 1 with the current MLP/tree split search."
        )
    if "selected_rank" in frame.columns:
        frame = frame.sort_values("selected_rank", kind="mergesort")
    else:
        frame = frame.sort_values(["ensemble_spearman", "ESS"], ascending=[False, False], kind="mergesort")
    return frame.reset_index(drop=True)


def _validate_informed_mlp_prob(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"informed_mlp_prob must be finite and in [0, 1], got {value!r}.")
    return value


def phase2_mlp_prob_range(anchor_prob: float, window: float = PHASE2_INFORMED_MLP_PROB_WINDOW) -> tuple[float, float]:
    anchor_prob = _validate_informed_mlp_prob(anchor_prob)
    window = float(window)
    if not np.isfinite(window) or window < 0.0:
        raise ValueError(f"window must be finite and non-negative, got {window!r}.")
    return max(0.0, anchor_prob - window), min(1.0, anchor_prob + window)


def select_phase2_regimes(phase1_summary: pd.DataFrame, top_regimes: int) -> list[str]:
    return [regime for regime, _ in select_phase2_regime_anchors(phase1_summary, top_regimes=top_regimes)]


def select_phase2_regime_anchors(phase1_summary: pd.DataFrame, top_regimes: int) -> list[tuple[str, float]]:
    if int(top_regimes) <= 0:
        raise ValueError("top_regimes must be positive.")
    if "informed_mlp_prob" not in phase1_summary.columns:
        raise ValueError("Phase 2 regime selection requires informed_mlp_prob in the Phase 1 summary.")
    regime_anchors: list[tuple[str, float]] = []
    seen: set[str] = set()
    for _, row in phase1_summary.iterrows():
        regime = str(row["anchored_regime"])
        if regime not in anchored_regimes():
            raise ValueError(f"Unknown anchored regime in Phase 1 summary: {regime}")
        if regime not in seen:
            regime_anchors.append((regime, _validate_informed_mlp_prob(float(row["informed_mlp_prob"]))))
            seen.add(regime)
        if len(regime_anchors) == int(top_regimes):
            break
    if not regime_anchors:
        raise ValueError("Could not select any Phase 2 regimes from Phase 1 summary.")
    return regime_anchors


def latin_hypercube_core_params(
    n_samples: int,
    random_state: int,
    extra_ranges: dict[str, tuple[float, float]] | None = None,
) -> list[dict[str, float]]:
    if int(n_samples) <= 0:
        raise ValueError("n_core_samples must be positive.")
    n_samples = int(n_samples)
    rng = np.random.default_rng(int(random_state))
    ranges = core_search_ranges()
    if extra_ranges:
        ranges = {**ranges, **extra_ranges}
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
    selected_regimes = select_phase2_regime_anchors(phase1_summary, top_regimes=top_regimes)
    regimes = anchored_regimes()
    candidates: list[EtaCandidate] = []
    for regime, mlp_anchor_prob in selected_regimes:
        mlp_low, mlp_high = phase2_mlp_prob_range(mlp_anchor_prob)
        core_samples = latin_hypercube_core_params(
            n_core_samples,
            random_state=random_state,
            extra_ranges={"informed_mlp_prob": (mlp_low, mlp_high)},
        )
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

    eta_params = dict(eta_params or {})
    informed_mlp_prob = _validate_informed_mlp_prob(
        eta_params.pop("informed_mlp_prob", DEFAULT_INFORMED_MLP_PROB)
    )
    informed_mix_probs = (informed_mlp_prob, 1.0 - informed_mlp_prob)
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp.update(
        {
            "mix_probs": informed_mix_probs,
            "informed_mix_probs": informed_mix_probs,
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
            "informed_target_mix_weight": 0.35,
            "pitting_material_dirichlet_prob": 0.0,
            "pitting_material_dirichlet_concentration": 1.0,
            "pitting_material_dirichlet_active_prob": 0.45,
            "pitting_process_role": "test_method_category",
            "pitting_process_category_count": int(category_count),
            "pitting_fixed_epit_schema": True,
            "cat_prob": 0.0,
            "permute_features": False,
        }
    )
    fixed_hp.update(eta_params)
    return fixed_hp


def synthetic_train_size_bounds(seq_len: int) -> tuple[int, int]:
    train_size = int(int(seq_len) * DIRECT_EPIT_TRAIN_SIZE_RATIO)
    train_size = max(1, min(train_size, int(seq_len) - 1))
    return train_size, train_size + 1


def _schema_retry_seed(seed: int, attempt: int) -> int:
    return int((int(seed) + int(attempt) * SCHEMA_RETRY_SEED_STRIDE) % (2**32 - 1))


def sample_synthetic_dataset(
    *,
    category_count: int,
    synthetic_seed: int,
    seq_len: int,
    eta_params: dict[str, Any] | None = None,
    max_schema_attempts: int = DEFAULT_SCHEMA_RETRY_ATTEMPTS,
) -> SyntheticEpitSample:
    if int(seq_len) < 16:
        raise ValueError("seq_len must be at least 16 for a usable direct EPIT synthetic sample.")
    if int(max_schema_attempts) <= 0:
        raise ValueError("max_schema_attempts must be positive.")

    min_train_size, max_train_size = synthetic_train_size_bounds(int(seq_len))
    last_error: str | None = None

    for attempt in range(int(max_schema_attempts)):
        sampling_seed = _schema_retry_seed(int(synthetic_seed), attempt)
        np.random.seed(sampling_seed)
        random.seed(sampling_seed)
        torch.manual_seed(sampling_seed)

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
            last_error = f"expected synthetic shape {(int(seq_len), EXPECTED_EPIT_FEATURE_COUNT)}, got {X_np.shape}"
            continue
        if d_value != EXPECTED_EPIT_FEATURE_COUNT:
            # PriorDataset removes constant features and left-packs the remaining columns.
            # Direct EPIT scoring relies on fixed feature positions, so these draws must be
            # resampled rather than accepted with a corrupted schema.
            last_error = f"expected all {EXPECTED_EPIT_FEATURE_COUNT} synthetic features active, got d={d_value}"
            continue
        if not np.isfinite(X_np).all() or not np.isfinite(y_np).all():
            last_error = "synthetic EPIT sample contains non-finite values"
            continue
        if float(np.std(y_np, ddof=0)) <= 0.0:
            last_error = "synthetic EPIT target has zero variance"
            continue
        target_rule = getattr(dataset.prior, "last_pitting_target_rule", None)
        if not isinstance(target_rule, dict) or target_rule.get("rule_type") != "fixed_epit_target_rule_v2_pren_anchor":
            last_error = "synthetic EPIT draw did not produce a fixed-schema target-rule trace"
            continue

        process_unique = np.unique(X_np[:, -1])
        return SyntheticEpitSample(
            X=X_np,
            y=y_np,
            d=d_value,
            seq_len=seq_len_value,
            train_size=train_size_value,
            synthetic_seed=int(synthetic_seed),
            sampling_seed=int(sampling_seed),
            schema_attempts=int(attempt) + 1,
            process_unique_count=int(process_unique.size),
            process_unique_values=[float(value) for value in process_unique.tolist()],
            target_rule=target_rule,
        )

    raise RuntimeError(
        "Could not sample a fixed-schema EPIT synthetic dataset after "
        f"{int(max_schema_attempts)} attempts for requested seed {int(synthetic_seed)}. "
        f"Last error: {last_error}."
    )


def build_baseline_prior_control_fixed_hp() -> dict[str, Any]:
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp["mix_probs"] = (0.7, 0.3)
    return fixed_hp


def sample_baseline_prior_control_dataset(
    *,
    synthetic_seed: int,
    seq_len: int,
    max_schema_attempts: int = DEFAULT_BASELINE_CONTROL_MAX_ATTEMPTS,
) -> SyntheticEpitSample:
    if int(seq_len) < 16:
        raise ValueError("seq_len must be at least 16 for a usable baseline prior control sample.")
    if int(max_schema_attempts) <= 0:
        raise ValueError("max_schema_attempts must be positive.")

    last_error: str | None = None
    for attempt in range(int(max_schema_attempts)):
        sampling_seed = _schema_retry_seed(int(synthetic_seed), attempt)
        np.random.seed(sampling_seed)
        random.seed(sampling_seed)
        torch.manual_seed(sampling_seed)

        dataset = PriorDataset(
            batch_size=1,
            batch_size_per_gp=1,
            min_features=BASELINE_PRIOR_CONTROL_MIN_FEATURES,
            max_features=BASELINE_PRIOR_CONTROL_MAX_FEATURES,
            max_classes=0,
            min_seq_len=None,
            max_seq_len=int(seq_len),
            min_train_size=0.1,
            max_train_size=0.9,
            prior_type=BASELINE_PRIOR_CONTROL_PRIOR_TYPE,
            scm_fixed_hp=build_baseline_prior_control_fixed_hp(),
            scm_sampled_hp=DEFAULT_SAMPLED_HP,
            n_jobs=1,
            device="cpu",
        )
        X, y, d, seq_lens, train_sizes = dataset.get_batch()
        X_np_full = X[0].detach().cpu().numpy().astype(float)
        y_np = y[0].detach().cpu().numpy().astype(float)
        d_value = int(d[0].item())
        seq_len_value = int(seq_lens[0].item())
        train_size_value = int(train_sizes[0].item())

        if X_np_full.shape[0] != int(seq_len):
            last_error = f"expected synthetic row count {int(seq_len)}, got {X_np_full.shape[0]}"
            continue
        if d_value != EXPECTED_EPIT_FEATURE_COUNT:
            last_error = f"expected accepted default-prior draw with d={EXPECTED_EPIT_FEATURE_COUNT}, got d={d_value}"
            continue
        X_np = X_np_full[:, :EXPECTED_EPIT_FEATURE_COUNT]
        if X_np.shape != (int(seq_len), EXPECTED_EPIT_FEATURE_COUNT):
            last_error = f"expected accepted synthetic shape {(int(seq_len), EXPECTED_EPIT_FEATURE_COUNT)}, got {X_np.shape}"
            continue
        if not np.isfinite(X_np).all() or not np.isfinite(y_np).all():
            last_error = "baseline prior control sample contains non-finite values"
            continue
        if float(np.std(y_np, ddof=0)) <= 0.0:
            last_error = "baseline prior control target has zero variance"
            continue

        final_column_unique = np.unique(X_np[:, -1])
        return SyntheticEpitSample(
            X=X_np,
            y=y_np,
            d=d_value,
            seq_len=seq_len_value,
            train_size=train_size_value,
            synthetic_seed=int(synthetic_seed),
            sampling_seed=int(sampling_seed),
            schema_attempts=int(attempt) + 1,
            process_unique_count=int(final_column_unique.size),
            process_unique_values=[float(value) for value in final_column_unique.tolist()],
            target_rule={},
        )

    raise RuntimeError(
        "Could not sample a default TabICL baseline prior control dataset conditioned on d=21 after "
        f"{int(max_schema_attempts)} attempts for requested seed {int(synthetic_seed)}. "
        f"Last error: {last_error}."
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


SURROGATE_MODEL_NAMES = ("ridge", "extra_trees", "hist_gradient_boosting")


def build_surrogate_model(model_name: str = "ridge", *, random_state: int = 42) -> Any:
    """Build a deterministic single-synthetic-task transfer surrogate.

    Scaling is kept in every pipeline so real inputs are transformed with the
    synthetic task's fitted statistics, matching the historical Ridge test.
    Tree estimators use one thread because trial-level multiprocessing provides
    the outer parallelism in the CPU scoring workflow.
    """
    model_name = str(model_name).strip().lower()
    if model_name == "ridge":
        estimator = Ridge(alpha=1.0)
    elif model_name == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=3,
            max_features=1.0,
            n_jobs=1,
            random_state=int(random_state),
        )
    elif model_name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=31,
            min_samples_leaf=10,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=int(random_state),
        )
    else:
        choices = ", ".join(SURROGATE_MODEL_NAMES)
        raise ValueError(f"Unknown surrogate model {model_name!r}; choose one of: {choices}.")
    return make_pipeline(StandardScaler(), estimator)


def fit_and_score_theta(
    processed: ProcessedEpitData,
    synthetic: SyntheticEpitSample,
    *,
    model_name: str = "ridge",
    random_state: int = 42,
) -> ThetaSurrogateScore:
    if synthetic.X.shape[1] != processed.n_features:
        raise RuntimeError(
            f"Synthetic/real feature width mismatch: {synthetic.X.shape[1]} vs {processed.n_features}."
        )

    model = build_surrogate_model(model_name, random_state=random_state)
    model.fit(synthetic.X, synthetic.y)
    predictions = np.asarray(model.predict(processed.X), dtype=float).reshape(-1)
    if predictions.shape != processed.y_z.shape:
        raise RuntimeError(f"Expected predictions shape {processed.y_z.shape}, got {predictions.shape}.")
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{model_name} surrogate produced non-finite predictions.")

    residuals = predictions - processed.y_z
    return ThetaSurrogateScore(
        synthetic_seed=synthetic.synthetic_seed,
        spearman=spearman_score(processed.y_z, predictions),
        standardized_mae=float(np.mean(np.abs(residuals))),
        standardized_rmse=float(np.sqrt(np.mean(residuals**2))),
        predictions=predictions,
    )



def standardize_prediction(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size <= 1:
        return np.zeros_like(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def score_target_rule_theta(processed: ProcessedEpitData, synthetic: SyntheticEpitSample) -> ThetaSurrogateScore:
    predictions = SCMPrior.evaluate_fixed_epit_target_rule_numpy(
        processed.X_raw,
        synthetic.target_rule,
        environment_mode="raw",
    )
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if predictions.shape != processed.y_z.shape:
        raise RuntimeError(f"Expected target-rule predictions shape {processed.y_z.shape}, got {predictions.shape}.")
    if not np.isfinite(predictions).all():
        raise RuntimeError("Target-rule oracle produced non-finite predictions.")

    scored_predictions = standardize_prediction(predictions)
    residuals = scored_predictions - processed.y_z
    return ThetaSurrogateScore(
        synthetic_seed=synthetic.synthetic_seed,
        spearman=spearman_score(processed.y_z, predictions),
        standardized_mae=float(np.mean(np.abs(residuals))),
        standardized_rmse=float(np.sqrt(np.mean(residuals**2))),
        predictions=predictions,
    )


def summarize_target_rule_eta(
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
        raise ValueError("At least one target-rule score is required to summarize an eta.")

    spearmans = np.asarray([score.spearman for score in scores], dtype=float)
    weights = uniform_theta_weights(len(scores))
    predictions = np.vstack([score.predictions for score in scores])
    ensemble_predictions = weights @ predictions
    scored_ensemble = standardize_prediction(ensemble_predictions)
    residuals = scored_ensemble - processed.y_z
    ess = effective_sample_size(weights)
    threshold = collapse_threshold(len(scores))

    return EtaSummary(
        eta_id=eta_id,
        phase=phase,
        anchored_regime=anchored_regime,
        core_anchor=core_anchor,
        n_synth=len(scores),
        temperature=float(temperature),
        weighting_scheme=ENSEMBLE_WEIGHTING_SCHEME,
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


def spearman_loss(spearman: float) -> float:
    rho = float(np.clip(spearman, -1.0, 1.0))
    return (1.0 - rho) / 2.0


def uniform_theta_weights(n_scores: int) -> np.ndarray:
    n_scores = int(n_scores)
    if n_scores <= 0:
        raise ValueError("At least one theta score is required for uniform weighting.")
    return np.full(n_scores, 1.0 / float(n_scores), dtype=float)


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
    weights = uniform_theta_weights(len(scores))
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
        weighting_scheme=ENSEMBLE_WEIGHTING_SCHEME,
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


def sample_and_score_baseline_prior_control(
    processed: ProcessedEpitData,
    *,
    n_synth: int,
    synthetic_seed_start: int,
    seq_len: int,
    temperature: float = DEFAULT_TEMPERATURE,
    phase: str = "baseline_control",
    max_schema_attempts: int = DEFAULT_BASELINE_CONTROL_MAX_ATTEMPTS,
) -> tuple[list[SyntheticEpitSample], list[ThetaSurrogateScore], EtaSummary]:
    if int(n_synth) <= 0:
        raise ValueError("n_synth must be positive.")

    samples: list[SyntheticEpitSample] = []
    scores: list[ThetaSurrogateScore] = []
    for offset in range(int(n_synth)):
        synthetic = sample_baseline_prior_control_dataset(
            synthetic_seed=int(synthetic_seed_start) + offset,
            seq_len=seq_len,
            max_schema_attempts=max_schema_attempts,
        )
        samples.append(synthetic)
        scores.append(fit_and_score_theta(processed, synthetic))

    summary = summarize_eta(
        processed,
        scores,
        temperature=temperature,
        eta_id=BASELINE_PRIOR_CONTROL_ID,
        phase=phase,
        anchored_regime="tabicl_default_prior",
        core_anchor=BASELINE_PRIOR_CONTROL_FEATURE_MODE,
    )
    return samples, scores, summary


def baseline_prior_control_records(
    samples: list[SyntheticEpitSample], scores: list[ThetaSurrogateScore]
) -> list[BaselinePriorControlRecord]:
    if len(samples) != len(scores):
        raise ValueError(f"Expected equal sample and score counts, got {len(samples)} and {len(scores)}.")
    records: list[BaselinePriorControlRecord] = []
    for sample, score in zip(samples, scores):
        records.append(
            BaselinePriorControlRecord(
                score=score,
                sampling_seed=sample.sampling_seed,
                schema_attempts=sample.schema_attempts,
                d=sample.d,
                seq_len=sample.seq_len,
                train_size=sample.train_size,
                raw_feature_count=BASELINE_PRIOR_CONTROL_MAX_FEATURES,
            )
        )
    return records


def summarize_baseline_prior_control_records(
    processed: ProcessedEpitData,
    records: list[BaselinePriorControlRecord],
    *,
    temperature: float,
    phase: str,
) -> EtaSummary:
    return summarize_eta(
        processed,
        [record.score for record in records],
        temperature=temperature,
        eta_id=BASELINE_PRIOR_CONTROL_ID,
        phase=phase,
        anchored_regime="tabicl_default_prior",
        core_anchor=BASELINE_PRIOR_CONTROL_FEATURE_MODE,
    )


def sample_target_rule_scores_for_eta(
    processed: ProcessedEpitData,
    candidate: EtaCandidate,
    *,
    n_synth: int,
    synthetic_seed_start: int,
    seq_len: int,
    temperature: float,
    phase: str,
) -> tuple[list[ThetaSurrogateScore], EtaSummary]:
    if int(n_synth) <= 0:
        raise ValueError("n_synth must be positive.")

    scores: list[ThetaSurrogateScore] = []
    for offset in range(int(n_synth)):
        synthetic = sample_synthetic_dataset(
            category_count=processed.category_count,
            synthetic_seed=int(synthetic_seed_start) + offset,
            seq_len=seq_len,
            eta_params=candidate.eta_params,
        )
        scores.append(score_target_rule_theta(processed, synthetic))

    summary = summarize_target_rule_eta(
        processed,
        scores,
        temperature=temperature,
        eta_id=candidate.eta_id,
        phase=phase,
        anchored_regime=candidate.anchored_regime,
        core_anchor=candidate.core_anchor,
    )
    return scores, summary


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
                "weighting_scheme": summary.weighting_scheme,
                "weight": float(weight),
            }
        )
    return pd.DataFrame(rows)


def baseline_prior_control_metadata(records: list[BaselinePriorControlRecord]) -> dict[str, Any]:
    attempts = np.asarray([record.schema_attempts for record in records], dtype=float)
    total_attempts = int(attempts.sum()) if attempts.size else 0
    return {
        "diagnostic": "ridge_direct_prior_control",
        "control_id": BASELINE_PRIOR_CONTROL_ID,
        "prior_type": BASELINE_PRIOR_CONTROL_PRIOR_TYPE,
        "feature_mode": BASELINE_PRIOR_CONTROL_FEATURE_MODE,
        "training_prior_min_features": BASELINE_PRIOR_CONTROL_MIN_FEATURES,
        "training_prior_max_features": BASELINE_PRIOR_CONTROL_MAX_FEATURES,
        "ridge_feature_count": EXPECTED_EPIT_FEATURE_COUNT,
        "target_rule_diagnostic": "not_applicable",
        "n_requested": len(records),
        "n_accepted_d21": len(records),
        "total_schema_attempts": total_attempts,
        "acceptance_rate": (float(len(records)) / float(total_attempts)) if total_attempts > 0 else np.nan,
        "median_schema_attempts": float(np.median(attempts)) if attempts.size else np.nan,
        "max_schema_attempts": int(np.max(attempts)) if attempts.size else 0,
    }


def baseline_prior_summary_frame(summary: EtaSummary, records: list[BaselinePriorControlRecord]) -> pd.DataFrame:
    frame = summaries_frame([summary])
    for key, value in baseline_prior_control_metadata(records).items():
        frame[key] = value
    return frame


def baseline_prior_theta_scores_frame(
    records: list[BaselinePriorControlRecord], summary: EtaSummary
) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "eta_id": summary.eta_id,
            "phase": summary.phase,
            "anchored_regime": summary.anchored_regime,
            "core_anchor": summary.core_anchor,
            "control_id": BASELINE_PRIOR_CONTROL_ID,
            "prior_type": BASELINE_PRIOR_CONTROL_PRIOR_TYPE,
            "feature_mode": BASELINE_PRIOR_CONTROL_FEATURE_MODE,
            "sampling_seed": record.sampling_seed,
            "schema_attempts": record.schema_attempts,
            "d": record.d,
            "seq_len": record.seq_len,
            "train_size": record.train_size,
            "raw_feature_count": record.raw_feature_count,
        }
        row.update(record.score.metrics_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def eta_candidates_frame(candidates: list[EtaCandidate]) -> pd.DataFrame:
    return pd.DataFrame([candidate.row_dict() for candidate in candidates])



def diagnostic_comparison_frame(ridge_summaries: list[EtaSummary], target_rule_summaries: list[EtaSummary]) -> pd.DataFrame:
    ridge = summaries_frame(ridge_summaries).rename(
        columns={
            "selected_rank": "ridge_rank",
            "ensemble_spearman": "ridge_ensemble_spearman",
            "standardized_mae": "ridge_standardized_mae",
            "standardized_rmse": "ridge_standardized_rmse",
            "median_theta_spearman": "ridge_median_theta_spearman",
            "max_theta_spearman": "ridge_max_theta_spearman",
        }
    )
    target = summaries_frame(target_rule_summaries).rename(
        columns={
            "selected_rank": "target_rule_rank",
            "ensemble_spearman": "target_rule_ensemble_spearman",
            "standardized_mae": "target_rule_standardized_mae",
            "standardized_rmse": "target_rule_standardized_rmse",
            "median_theta_spearman": "target_rule_median_theta_spearman",
            "max_theta_spearman": "target_rule_max_theta_spearman",
        }
    )
    ridge_cols = [
        "eta_id",
        "phase",
        "anchored_regime",
        "core_anchor",
        "ridge_rank",
        "ridge_ensemble_spearman",
        "ridge_standardized_mae",
        "ridge_standardized_rmse",
        "ridge_median_theta_spearman",
        "ridge_max_theta_spearman",
    ]
    target_cols = [
        "eta_id",
        "target_rule_rank",
        "target_rule_ensemble_spearman",
        "target_rule_standardized_mae",
        "target_rule_standardized_rmse",
        "target_rule_median_theta_spearman",
        "target_rule_max_theta_spearman",
    ]
    frame = ridge[ridge_cols].merge(target[target_cols], on="eta_id", how="left")
    frame["rank_delta_target_minus_ridge"] = frame["target_rule_rank"] - frame["ridge_rank"]
    return frame.sort_values("ridge_rank", kind="mergesort").reset_index(drop=True)


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




def resolve_n_workers(requested_workers: int = 0) -> int:
    if int(requested_workers) > 0:
        return int(requested_workers)
    slurm_value = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_value:
        try:
            return max(1, int(slurm_value))
        except ValueError:
            pass
    return 1


def _eta_artifact_dirname(eta_id: str) -> str:
    return str(eta_id).replace("/", "_").replace(os.sep, "_")


def theta_artifact_path(output_dir: Path, eta_id: str, synthetic_seed: int) -> Path:
    return Path(output_dir) / "theta_artifacts" / _eta_artifact_dirname(eta_id) / f"seed_{int(synthetic_seed):06d}.npz"


def baseline_prior_artifact_path(output_dir: Path, synthetic_seed: int) -> Path:
    return (
        Path(output_dir)
        / "baseline_prior_artifacts"
        / _eta_artifact_dirname(BASELINE_PRIOR_CONTROL_ID)
        / f"seed_{int(synthetic_seed):06d}.npz"
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_progress_log(output_dir: Path, message: str) -> None:
    log_path = Path(output_dir) / "progress.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(message + "\n")


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(float(seconds)) or float(seconds) < 0.0:
        return "unknown"
    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def write_progress_status(output_dir: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _atomic_write_json(Path(output_dir) / "progress.json", payload)


def write_theta_artifact(
    output_dir: Path,
    *,
    eta_id: str,
    phase: str,
    anchored_regime: str,
    core_anchor: str,
    score: ThetaSurrogateScore,
    target_rule_score: ThetaSurrogateScore,
    synthetic: SyntheticEpitSample,
) -> Path:
    path = theta_artifact_path(output_dir, eta_id, score.synthetic_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            artifact_format=np.asarray(THETA_ARTIFACT_FORMAT),
            eta_id=np.asarray(str(eta_id)),
            phase=np.asarray(str(phase)),
            anchored_regime=np.asarray(str(anchored_regime)),
            core_anchor=np.asarray(str(core_anchor)),
            synthetic_seed=np.asarray(int(score.synthetic_seed), dtype=np.int64),
            sampling_seed=np.asarray(int(synthetic.sampling_seed), dtype=np.int64),
            schema_attempts=np.asarray(int(synthetic.schema_attempts), dtype=np.int64),
            spearman=np.asarray(float(score.spearman), dtype=np.float64),
            standardized_mae=np.asarray(float(score.standardized_mae), dtype=np.float64),
            standardized_rmse=np.asarray(float(score.standardized_rmse), dtype=np.float64),
            predictions=np.asarray(score.predictions, dtype=np.float32),
            target_rule_spearman=np.asarray(float(target_rule_score.spearman), dtype=np.float64),
            target_rule_standardized_mae=np.asarray(float(target_rule_score.standardized_mae), dtype=np.float64),
            target_rule_standardized_rmse=np.asarray(float(target_rule_score.standardized_rmse), dtype=np.float64),
            target_rule_predictions=np.asarray(target_rule_score.predictions, dtype=np.float32),
            target_rule_type=np.asarray(str(synthetic.target_rule.get("rule_type", ""))),
            target_rule_material_anchor_type=np.asarray(str(synthetic.target_rule.get("material_anchor_type", ""))),
            target_rule_environment_rule_type=np.asarray(str(synthetic.target_rule.get("environment_rule_type", ""))),
            target_rule_synthetic_environment_mode=np.asarray(str(synthetic.target_rule.get("synthetic_environment_mode", ""))),
            target_rule_material_cols=np.asarray(synthetic.target_rule["material_cols"], dtype=np.int64),
            target_rule_material_anchor_weights=np.asarray(
                synthetic.target_rule.get("material_anchor_weights", np.zeros_like(synthetic.target_rule["material_weights"])),
                dtype=np.float32,
            ),
            target_rule_material_weights=np.asarray(synthetic.target_rule["material_weights"], dtype=np.float32),
            target_rule_environment_weights=np.asarray(synthetic.target_rule["environment_weights"], dtype=np.float32),
            target_rule_process_offsets=np.asarray(synthetic.target_rule["process_offsets"], dtype=np.float32),
            target_rule_temperature_col=np.asarray(int(synthetic.target_rule["temperature_col"]), dtype=np.int64),
            target_rule_chloride_col=np.asarray(int(synthetic.target_rule["chloride_col"]), dtype=np.int64),
            target_rule_ph_col=np.asarray(int(synthetic.target_rule["ph_col"]), dtype=np.int64),
            target_rule_process_col=np.asarray(int(synthetic.target_rule["process_col"]), dtype=np.int64),
            target_rule_ph_neutral=np.asarray(float(synthetic.target_rule["ph_neutral"]), dtype=np.float64),
            target_rule_material_coef=np.asarray(float(synthetic.target_rule["material_coef"]), dtype=np.float64),
            target_rule_environment_coef=np.asarray(float(synthetic.target_rule["environment_coef"]), dtype=np.float64),
            target_rule_interaction_coef=np.asarray(float(synthetic.target_rule["interaction_coef"]), dtype=np.float64),
            target_rule_process_coef=np.asarray(float(synthetic.target_rule["process_coef"]), dtype=np.float64),
            target_rule_target_mix_weight=np.asarray(float(synthetic.target_rule["target_mix_weight"]), dtype=np.float64),
            n_rows=np.asarray(int(synthetic.X.shape[0]), dtype=np.int64),
            n_features=np.asarray(int(synthetic.X.shape[1]), dtype=np.int64),
            d=np.asarray(int(synthetic.d), dtype=np.int64),
            seq_len=np.asarray(int(synthetic.seq_len), dtype=np.int64),
            train_size=np.asarray(int(synthetic.train_size), dtype=np.int64),
            process_unique_count=np.asarray(int(synthetic.process_unique_count), dtype=np.int64),
            target_mean=np.asarray(float(np.mean(synthetic.y)), dtype=np.float64),
            target_std=np.asarray(float(np.std(synthetic.y, ddof=0)), dtype=np.float64),
        )
    tmp_path.replace(path)
    return path


def write_baseline_prior_artifact(
    output_dir: Path,
    *,
    phase: str,
    score: ThetaSurrogateScore,
    synthetic: SyntheticEpitSample,
) -> Path:
    path = baseline_prior_artifact_path(output_dir, score.synthetic_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(
            handle,
            artifact_format=np.asarray(BASELINE_PRIOR_CONTROL_ARTIFACT_FORMAT),
            control_id=np.asarray(BASELINE_PRIOR_CONTROL_ID),
            phase=np.asarray(str(phase)),
            prior_type=np.asarray(BASELINE_PRIOR_CONTROL_PRIOR_TYPE),
            feature_mode=np.asarray(BASELINE_PRIOR_CONTROL_FEATURE_MODE),
            synthetic_seed=np.asarray(int(score.synthetic_seed), dtype=np.int64),
            sampling_seed=np.asarray(int(synthetic.sampling_seed), dtype=np.int64),
            schema_attempts=np.asarray(int(synthetic.schema_attempts), dtype=np.int64),
            spearman=np.asarray(float(score.spearman), dtype=np.float64),
            standardized_mae=np.asarray(float(score.standardized_mae), dtype=np.float64),
            standardized_rmse=np.asarray(float(score.standardized_rmse), dtype=np.float64),
            predictions=np.asarray(score.predictions, dtype=np.float32),
            n_rows=np.asarray(int(synthetic.X.shape[0]), dtype=np.int64),
            n_features=np.asarray(int(synthetic.X.shape[1]), dtype=np.int64),
            d=np.asarray(int(synthetic.d), dtype=np.int64),
            seq_len=np.asarray(int(synthetic.seq_len), dtype=np.int64),
            train_size=np.asarray(int(synthetic.train_size), dtype=np.int64),
            raw_feature_count=np.asarray(BASELINE_PRIOR_CONTROL_MAX_FEATURES, dtype=np.int64),
            target_mean=np.asarray(float(np.mean(synthetic.y)), dtype=np.float64),
            target_std=np.asarray(float(np.std(synthetic.y, ddof=0)), dtype=np.float64),
        )
    tmp_path.replace(path)
    return path


def read_baseline_prior_artifact(
    output_dir: Path,
    synthetic_seed: int,
    *,
    expected_n_rows: int,
    strict: bool = False,
) -> BaselinePriorControlRecord | None:
    path = baseline_prior_artifact_path(output_dir, synthetic_seed)
    try:
        with np.load(path, allow_pickle=False) as data:
            if "artifact_format" not in data or str(data["artifact_format"]) != BASELINE_PRIOR_CONTROL_ARTIFACT_FORMAT:
                got = "missing" if "artifact_format" not in data else str(data["artifact_format"])
                raise ValueError(
                    f"Baseline prior artifact {path} has format {got!r}; "
                    f"expected {BASELINE_PRIOR_CONTROL_ARTIFACT_FORMAT!r}."
                )
            seed = int(data["synthetic_seed"])
            if seed != int(synthetic_seed):
                raise ValueError(f"Baseline prior artifact seed mismatch for {path}: {seed} != {synthetic_seed}")
            predictions = np.asarray(data["predictions"], dtype=float).reshape(-1)
            if predictions.shape != (int(expected_n_rows),):
                raise ValueError(f"Baseline prior predictions shape mismatch for {path}: {predictions.shape}")
            if not np.isfinite(predictions).all():
                raise ValueError(f"Baseline prior predictions contain non-finite values: {path}")
            score = ThetaSurrogateScore(
                synthetic_seed=seed,
                spearman=float(data["spearman"]),
                standardized_mae=float(data["standardized_mae"]),
                standardized_rmse=float(data["standardized_rmse"]),
                predictions=predictions,
            )
            metrics = np.asarray([score.spearman, score.standardized_mae, score.standardized_rmse], dtype=float)
            if not np.isfinite(metrics).all():
                raise ValueError(f"Baseline prior artifact metrics contain non-finite values: {path}")
            return BaselinePriorControlRecord(
                score=score,
                sampling_seed=int(data["sampling_seed"]),
                schema_attempts=int(data["schema_attempts"]),
                d=int(data["d"]),
                seq_len=int(data["seq_len"]),
                train_size=int(data["train_size"]),
                raw_feature_count=int(data["raw_feature_count"]),
            )
    except Exception:
        if strict:
            raise
        return None


def _validate_theta_artifact_arrays(
    data: np.lib.npyio.NpzFile,
    path: Path,
    synthetic_seed: int,
    expected_n_rows: int,
) -> int:
    if "artifact_format" not in data or str(data["artifact_format"]) != THETA_ARTIFACT_FORMAT:
        got = "missing" if "artifact_format" not in data else str(data["artifact_format"])
        raise ValueError(f"Artifact {path} has format {got!r}; expected {THETA_ARTIFACT_FORMAT!r}.")
    seed = int(data["synthetic_seed"])
    if seed != int(synthetic_seed):
        raise ValueError(f"Artifact seed mismatch for {path}: {seed} != {synthetic_seed}")
    for key in ("predictions", "target_rule_predictions"):
        predictions = np.asarray(data[key], dtype=float).reshape(-1)
        if predictions.shape != (int(expected_n_rows),):
            raise ValueError(f"Artifact {key} shape mismatch for {path}: {predictions.shape}")
        if not np.isfinite(predictions).all():
            raise ValueError(f"Artifact {key} contains non-finite values: {path}")
    return seed


def read_theta_artifact(
    output_dir: Path,
    eta_id: str,
    synthetic_seed: int,
    *,
    expected_n_rows: int,
    strict: bool = False,
) -> ThetaSurrogateScore | None:
    path = theta_artifact_path(output_dir, eta_id, synthetic_seed)
    try:
        with np.load(path, allow_pickle=False) as data:
            seed = _validate_theta_artifact_arrays(data, path, synthetic_seed, expected_n_rows)
            score = ThetaSurrogateScore(
                synthetic_seed=seed,
                spearman=float(data["spearman"]),
                standardized_mae=float(data["standardized_mae"]),
                standardized_rmse=float(data["standardized_rmse"]),
                predictions=np.asarray(data["predictions"], dtype=float).reshape(-1),
            )
            metrics = np.asarray([score.spearman, score.standardized_mae, score.standardized_rmse], dtype=float)
            if not np.isfinite(metrics).all():
                raise ValueError(f"Artifact metrics contain non-finite values: {path}")
            return score
    except Exception:
        if strict:
            raise
        return None


def read_target_rule_artifact(
    output_dir: Path,
    eta_id: str,
    synthetic_seed: int,
    *,
    expected_n_rows: int,
    strict: bool = False,
) -> ThetaSurrogateScore | None:
    path = theta_artifact_path(output_dir, eta_id, synthetic_seed)
    try:
        with np.load(path, allow_pickle=False) as data:
            seed = _validate_theta_artifact_arrays(data, path, synthetic_seed, expected_n_rows)
            score = ThetaSurrogateScore(
                synthetic_seed=seed,
                spearman=float(data["target_rule_spearman"]),
                standardized_mae=float(data["target_rule_standardized_mae"]),
                standardized_rmse=float(data["target_rule_standardized_rmse"]),
                predictions=np.asarray(data["target_rule_predictions"], dtype=float).reshape(-1),
            )
            metrics = np.asarray([score.spearman, score.standardized_mae, score.standardized_rmse], dtype=float)
            if not np.isfinite(metrics).all():
                raise ValueError(f"Target-rule artifact metrics contain non-finite values: {path}")
            return score
    except Exception:
        if strict:
            raise
        return None


def load_scores_for_candidate(
    output_dir: Path,
    candidate: EtaCandidate,
    *,
    synthetic_seed_start: int,
    n_synth: int,
    expected_n_rows: int,
) -> list[ThetaSurrogateScore]:
    scores: list[ThetaSurrogateScore] = []
    for offset in range(int(n_synth)):
        seed = int(synthetic_seed_start) + offset
        score = read_theta_artifact(
            output_dir,
            candidate.eta_id,
            seed,
            expected_n_rows=expected_n_rows,
            strict=True,
        )
        if score is None:
            raise RuntimeError(f"Missing theta artifact for {candidate.eta_id} seed {seed}.")
        scores.append(score)
    return scores



def load_target_rule_scores_for_candidate(
    output_dir: Path,
    candidate: EtaCandidate,
    *,
    synthetic_seed_start: int,
    n_synth: int,
    expected_n_rows: int,
) -> list[ThetaSurrogateScore]:
    scores: list[ThetaSurrogateScore] = []
    for offset in range(int(n_synth)):
        seed = int(synthetic_seed_start) + offset
        score = read_target_rule_artifact(
            output_dir,
            candidate.eta_id,
            seed,
            expected_n_rows=expected_n_rows,
            strict=True,
        )
        if score is None:
            raise RuntimeError(f"Missing target-rule artifact for {candidate.eta_id} seed {seed}.")
        scores.append(score)
    return scores


def load_baseline_prior_control_records(
    output_dir: Path,
    *,
    synthetic_seed_start: int,
    n_synth: int,
    expected_n_rows: int,
) -> list[BaselinePriorControlRecord]:
    records: list[BaselinePriorControlRecord] = []
    for offset in range(int(n_synth)):
        seed = int(synthetic_seed_start) + offset
        record = read_baseline_prior_artifact(
            output_dir,
            seed,
            expected_n_rows=expected_n_rows,
            strict=True,
        )
        if record is None:
            raise RuntimeError(f"Missing baseline prior control artifact for seed {seed}.")
        records.append(record)
    return records


def write_baseline_prior_control_outputs(
    processed: ProcessedEpitData,
    records: list[BaselinePriorControlRecord],
    output_dir: Path,
    *,
    phase: str,
    temperature: float,
) -> EtaSummary:
    summary = summarize_baseline_prior_control_records(
        processed,
        records,
        temperature=temperature,
        phase=phase,
    )
    baseline_prior_theta_scores_frame(records, summary).to_csv(
        Path(output_dir) / "baseline_prior_theta_scores.csv", index=False
    )
    weights_frame([record.score for record in records], summary).to_csv(
        Path(output_dir) / "baseline_prior_weights.csv", index=False
    )
    baseline_prior_summary_frame(summary, records).to_csv(
        Path(output_dir) / "baseline_prior_summary.csv", index=False
    )
    return summary


def write_theta_manifest(
    output_dir: Path,
    candidates: list[EtaCandidate],
    *,
    phase: str,
    n_synth: int,
    synthetic_seed_start: int,
) -> None:
    rows = []
    for candidate in candidates:
        for offset in range(int(n_synth)):
            seed = int(synthetic_seed_start) + offset
            rows.append(
                {
                    "eta_id": candidate.eta_id,
                    "phase": phase,
                    "anchored_regime": candidate.anchored_regime,
                    "core_anchor": candidate.core_anchor,
                    "synthetic_seed": seed,
                    "artifact_path": str(theta_artifact_path(output_dir, candidate.eta_id, seed).relative_to(output_dir)),
                }
            )
    pd.DataFrame(rows).to_csv(Path(output_dir) / "manifest.csv", index=False)


def _init_theta_worker(
    X_real: np.ndarray,
    X_real_raw: np.ndarray,
    y_real_z: np.ndarray,
    category_count: int,
    seq_len: int,
) -> None:
    global _WORKER_X_REAL, _WORKER_X_REAL_RAW, _WORKER_Y_REAL_Z, _WORKER_CATEGORY_COUNT, _WORKER_SEQ_LEN
    _WORKER_X_REAL = np.asarray(X_real, dtype=float)
    _WORKER_X_REAL_RAW = np.asarray(X_real_raw, dtype=float)
    _WORKER_Y_REAL_Z = np.asarray(y_real_z, dtype=float)
    _WORKER_CATEGORY_COUNT = int(category_count)
    _WORKER_SEQ_LEN = int(seq_len)
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass


def _fit_and_score_theta_worker_arrays(synthetic: SyntheticEpitSample) -> ThetaSurrogateScore:
    if _WORKER_X_REAL is None or _WORKER_Y_REAL_Z is None:
        raise RuntimeError("Theta worker was not initialized with real EPIT arrays.")
    if synthetic.X.shape[1] != _WORKER_X_REAL.shape[1]:
        raise RuntimeError(
            f"Synthetic/real feature width mismatch: {synthetic.X.shape[1]} vs {_WORKER_X_REAL.shape[1]}."
        )
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(synthetic.X, synthetic.y)
    predictions = np.asarray(model.predict(_WORKER_X_REAL), dtype=float).reshape(-1)
    if predictions.shape != _WORKER_Y_REAL_Z.shape:
        raise RuntimeError(f"Expected predictions shape {_WORKER_Y_REAL_Z.shape}, got {predictions.shape}.")
    if not np.isfinite(predictions).all():
        raise RuntimeError("Ridge surrogate produced non-finite predictions.")
    residuals = predictions - _WORKER_Y_REAL_Z
    return ThetaSurrogateScore(
        synthetic_seed=synthetic.synthetic_seed,
        spearman=spearman_score(_WORKER_Y_REAL_Z, predictions),
        standardized_mae=float(np.mean(np.abs(residuals))),
        standardized_rmse=float(np.sqrt(np.mean(residuals**2))),
        predictions=predictions,
    )



def _score_target_rule_worker_arrays(synthetic: SyntheticEpitSample) -> ThetaSurrogateScore:
    if _WORKER_X_REAL_RAW is None or _WORKER_Y_REAL_Z is None:
        raise RuntimeError("Theta worker was not initialized with raw real EPIT arrays.")
    predictions = SCMPrior.evaluate_fixed_epit_target_rule_numpy(
        _WORKER_X_REAL_RAW,
        synthetic.target_rule,
        environment_mode="raw",
    )
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if predictions.shape != _WORKER_Y_REAL_Z.shape:
        raise RuntimeError(f"Expected target-rule predictions shape {_WORKER_Y_REAL_Z.shape}, got {predictions.shape}.")
    if not np.isfinite(predictions).all():
        raise RuntimeError("Target-rule oracle produced non-finite predictions.")
    scored_predictions = standardize_prediction(predictions)
    residuals = scored_predictions - _WORKER_Y_REAL_Z
    return ThetaSurrogateScore(
        synthetic_seed=synthetic.synthetic_seed,
        spearman=spearman_score(_WORKER_Y_REAL_Z, predictions),
        standardized_mae=float(np.mean(np.abs(residuals))),
        standardized_rmse=float(np.sqrt(np.mean(residuals**2))),
        predictions=predictions,
    )


def _score_theta_chunk(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if _WORKER_CATEGORY_COUNT is None or _WORKER_SEQ_LEN is None:
        raise RuntimeError("Theta worker was not initialized with synthetic sampling settings.")
    output_dir = Path(payload["output_dir"])
    eta_params = dict(payload["eta_params"])
    rows: list[dict[str, Any]] = []
    for seed in payload["synthetic_seeds"]:
        synthetic = sample_synthetic_dataset(
            category_count=_WORKER_CATEGORY_COUNT,
            synthetic_seed=int(seed),
            seq_len=_WORKER_SEQ_LEN,
            eta_params=eta_params,
        )
        score = _fit_and_score_theta_worker_arrays(synthetic)
        target_rule_score = _score_target_rule_worker_arrays(synthetic)
        write_theta_artifact(
            output_dir,
            eta_id=payload["eta_id"],
            phase=payload["phase"],
            anchored_regime=payload["anchored_regime"],
            core_anchor=payload["core_anchor"],
            score=score,
            target_rule_score=target_rule_score,
            synthetic=synthetic,
        )
        rows.append(
            {
                "eta_id": payload["eta_id"],
                "synthetic_seed": int(seed),
                "spearman": score.spearman,
                "standardized_mae": score.standardized_mae,
                "standardized_rmse": score.standardized_rmse,
                "target_rule_spearman": target_rule_score.spearman,
                "target_rule_standardized_mae": target_rule_score.standardized_mae,
                "target_rule_standardized_rmse": target_rule_score.standardized_rmse,
            }
        )
    return rows


def _score_baseline_prior_control_chunk(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if _WORKER_SEQ_LEN is None:
        raise RuntimeError("Theta worker was not initialized with synthetic sampling settings.")
    output_dir = Path(payload["output_dir"])
    max_schema_attempts = int(payload.get("max_schema_attempts", DEFAULT_BASELINE_CONTROL_MAX_ATTEMPTS))
    rows: list[dict[str, Any]] = []
    for seed in payload["synthetic_seeds"]:
        synthetic = sample_baseline_prior_control_dataset(
            synthetic_seed=int(seed),
            seq_len=_WORKER_SEQ_LEN,
            max_schema_attempts=max_schema_attempts,
        )
        score = _fit_and_score_theta_worker_arrays(synthetic)
        write_baseline_prior_artifact(
            output_dir,
            phase=payload["phase"],
            score=score,
            synthetic=synthetic,
        )
        rows.append(
            {
                "synthetic_seed": int(seed),
                "sampling_seed": synthetic.sampling_seed,
                "schema_attempts": synthetic.schema_attempts,
                "spearman": score.spearman,
                "standardized_mae": score.standardized_mae,
                "standardized_rmse": score.standardized_rmse,
            }
        )
    return rows


def _chunk_values(values: list[int], chunk_size: int) -> list[list[int]]:
    chunk_size = max(1, int(chunk_size))
    return [values[idx : idx + chunk_size] for idx in range(0, len(values), chunk_size)]


def _build_missing_theta_chunks(
    output_dir: Path,
    candidates: list[EtaCandidate],
    *,
    phase: str,
    n_synth: int,
    synthetic_seed_start: int,
    expected_n_rows: int,
    chunk_size: int,
    resume: bool,
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    chunks: list[dict[str, Any]] = []
    skipped = 0
    completed_by_eta = {candidate.eta_id: 0 for candidate in candidates}
    for candidate in candidates:
        missing_seeds: list[int] = []
        for offset in range(int(n_synth)):
            seed = int(synthetic_seed_start) + offset
            existing = None
            if resume:
                existing = read_theta_artifact(
                    output_dir,
                    candidate.eta_id,
                    seed,
                    expected_n_rows=expected_n_rows,
                    strict=False,
                )
            if existing is not None:
                skipped += 1
                completed_by_eta[candidate.eta_id] += 1
            else:
                missing_seeds.append(seed)
        for seed_chunk in _chunk_values(missing_seeds, chunk_size):
            chunks.append(
                {
                    "output_dir": str(output_dir),
                    "phase": phase,
                    "eta_id": candidate.eta_id,
                    "anchored_regime": candidate.anchored_regime,
                    "core_anchor": candidate.core_anchor,
                    "eta_params": dict(candidate.eta_params),
                    "synthetic_seeds": seed_chunk,
                }
            )
    return chunks, skipped, completed_by_eta


def _build_missing_baseline_prior_control_chunks(
    output_dir: Path,
    *,
    n_synth: int,
    synthetic_seed_start: int,
    expected_n_rows: int,
    phase: str,
    chunk_size: int,
    resume: bool,
    max_schema_attempts: int,
) -> tuple[list[dict[str, Any]], int]:
    chunks: list[dict[str, Any]] = []
    skipped = 0
    missing_seeds: list[int] = []
    for offset in range(int(n_synth)):
        seed = int(synthetic_seed_start) + offset
        existing = None
        if resume:
            existing = read_baseline_prior_artifact(
                output_dir,
                seed,
                expected_n_rows=expected_n_rows,
                strict=False,
            )
        if existing is not None:
            skipped += 1
        else:
            missing_seeds.append(seed)

    for seed_chunk in _chunk_values(missing_seeds, chunk_size):
        chunks.append(
            {
                "output_dir": str(output_dir),
                "phase": phase,
                "synthetic_seeds": seed_chunk,
                "max_schema_attempts": int(max_schema_attempts),
            }
        )
    return chunks, skipped


def baseline_prior_control_enabled(args: argparse.Namespace) -> bool:
    phase = str(getattr(args, "phase", ""))
    if phase == BASELINE_PRIOR_CONTROL_PHASE:
        return True
    return phase in {"phase1", "phase2"} and not bool(getattr(args, "no_baseline_prior_control", True))


def baseline_control_max_attempts(args: argparse.Namespace) -> int:
    return int(getattr(args, "baseline_control_max_attempts", DEFAULT_BASELINE_CONTROL_MAX_ATTEMPTS))


def baseline_prior_control_config(enabled: bool, max_attempts: int) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "control_id": BASELINE_PRIOR_CONTROL_ID,
        "prior_type": BASELINE_PRIOR_CONTROL_PRIOR_TYPE,
        "feature_mode": BASELINE_PRIOR_CONTROL_FEATURE_MODE,
        "training_prior_min_features": BASELINE_PRIOR_CONTROL_MIN_FEATURES,
        "training_prior_max_features": BASELINE_PRIOR_CONTROL_MAX_FEATURES,
        "ridge_feature_count": EXPECTED_EPIT_FEATURE_COUNT,
        "same_n_synth_as_eta": True,
        "target_rule_diagnostic": "not_applicable",
        "max_schema_attempts_per_theta": int(max_attempts),
    }


def write_baseline_prior_control_run_metadata(
    processed: ProcessedEpitData,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    n_workers: int,
    chunk_size: int,
    resume: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "phase": args.phase,
        "random_state": args.random_state,
        "synthetic_seed_start": args.synthetic_seed,
        "n_synth": args.n_synth,
        "temperature": args.temperature,
        "ensemble_weighting_scheme": ENSEMBLE_WEIGHTING_SCHEME,
        "n_workers": n_workers,
        "chunk_size": chunk_size,
        "resume": bool(resume),
        "progress_interval_seconds": args.progress_interval,
        "n_etas": 1,
        "task_id": EPIT_TASK_ID,
        "scope": "baseline_prior_control_default_tabicl_regression_prior",
        "artifact_format": BASELINE_PRIOR_CONTROL_ARTIFACT_FORMAT,
        "target_rule_diagnostic": "not_applicable",
        "baseline_prior_only": True,
        "baseline_prior_control": baseline_prior_control_config(True, baseline_control_max_attempts(args)),
    }
    _atomic_write_json(output_dir / "config.json", config)
    _atomic_write_json(output_dir / "real_schema.json", processed.schema_dict())


def _artifact_run_config(
    args: argparse.Namespace,
    *,
    candidates: list[EtaCandidate],
    scope: str,
    n_workers: int,
    chunk_size: int,
    resume: bool,
) -> dict[str, Any]:
    config = {
        "phase": args.phase,
        "random_state": args.random_state,
        "synthetic_seed_start": args.synthetic_seed,
        "n_synth": args.n_synth,
        "temperature": args.temperature,
        "ensemble_weighting_scheme": ENSEMBLE_WEIGHTING_SCHEME,
        "n_workers": n_workers,
        "chunk_size": chunk_size,
        "resume": bool(resume),
        "progress_interval_seconds": args.progress_interval,
        "n_etas": len(candidates),
        "task_id": EPIT_TASK_ID,
        "scope": scope,
        "artifact_format": THETA_ARTIFACT_FORMAT,
        "target_rule_diagnostic": TARGET_RULE_DIAGNOSTIC,
        "target_rule_real_feature_input": "raw_imputed_original_units",
        "informed_mlp_prob_search": {
            "phase1_grid": list(PHASE1_INFORMED_MLP_PROB_GRID),
            "phase2_window": PHASE2_INFORMED_MLP_PROB_WINDOW,
            "full_training_mapping": "--informed_mix_probs p 1-p",
        },
        "baseline_prior_control": baseline_prior_control_config(
            baseline_prior_control_enabled(args), baseline_control_max_attempts(args)
        ),
    }

    if args.phase == "phase1":
        config["max_etas"] = args.max_etas
    if args.phase == "phase2":
        config["phase2_source_dir"] = str(args.phase2_source_dir)
        config["phase2_top_regimes"] = args.phase2_top_regimes
        config["n_core_samples"] = args.n_core_samples
    return config


def write_artifact_run_metadata(
    processed: ProcessedEpitData,
    candidates: list[EtaCandidate],
    first_sample: SyntheticEpitSample,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    scope: str,
    n_workers: int,
    chunk_size: int,
    resume: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        output_dir / "config.json",
        _artifact_run_config(
            args,
            candidates=candidates,
            scope=scope,
            n_workers=n_workers,
            chunk_size=chunk_size,
            resume=resume,
        ),
    )
    _atomic_write_json(output_dir / "real_schema.json", processed.schema_dict())
    _atomic_write_json(output_dir / "synthetic_smoke_schema.json", first_sample.schema_dict())
    eta_candidates_frame(candidates).to_csv(output_dir / "etas.csv", index=False)
    write_theta_manifest(
        output_dir,
        candidates,
        phase=args.phase,
        n_synth=args.n_synth,
        synthetic_seed_start=args.synthetic_seed,
    )


def write_artifact_final_outputs(
    processed: ProcessedEpitData,
    candidates: list[EtaCandidate],
    output_dir: Path,
    *,
    phase: str,
    n_synth: int,
    synthetic_seed_start: int,
    temperature: float,
) -> tuple[dict[str, list[ThetaSurrogateScore]], list[EtaSummary]]:
    scores_by_eta: dict[str, list[ThetaSurrogateScore]] = {}
    target_rule_scores_by_eta: dict[str, list[ThetaSurrogateScore]] = {}
    summaries: list[EtaSummary] = []
    target_rule_summaries: list[EtaSummary] = []
    for candidate in candidates:
        scores = load_scores_for_candidate(
            output_dir,
            candidate,
            synthetic_seed_start=synthetic_seed_start,
            n_synth=n_synth,
            expected_n_rows=processed.n_rows,
        )
        target_scores = load_target_rule_scores_for_candidate(
            output_dir,
            candidate,
            synthetic_seed_start=synthetic_seed_start,
            n_synth=n_synth,
            expected_n_rows=processed.n_rows,
        )
        scores_by_eta[candidate.eta_id] = scores
        target_rule_scores_by_eta[candidate.eta_id] = target_scores
        summaries.append(
            summarize_eta(
                processed,
                scores,
                temperature=temperature,
                eta_id=candidate.eta_id,
                phase=phase,
                anchored_regime=candidate.anchored_regime,
                core_anchor=candidate.core_anchor,
            )
        )
        target_rule_summaries.append(
            summarize_target_rule_eta(
                processed,
                target_scores,
                temperature=temperature,
                eta_id=candidate.eta_id,
                phase=phase,
                anchored_regime=candidate.anchored_regime,
                core_anchor=candidate.core_anchor,
            )
        )

    theta_frames = [theta_scores_frame(scores_by_eta[summary.eta_id], summary) for summary in summaries]
    weight_frames = [weights_frame(scores_by_eta[summary.eta_id], summary) for summary in summaries]
    target_theta_frames = [
        theta_scores_frame(target_rule_scores_by_eta[summary.eta_id], summary) for summary in target_rule_summaries
    ]
    target_weight_frames = [
        weights_frame(target_rule_scores_by_eta[summary.eta_id], summary) for summary in target_rule_summaries
    ]
    pd.concat(theta_frames, ignore_index=True).to_csv(output_dir / "theta_scores.csv", index=False)
    pd.concat(weight_frames, ignore_index=True).to_csv(output_dir / "weights.csv", index=False)
    summaries_frame(summaries).to_csv(output_dir / "summary.csv", index=False)
    pd.concat(target_theta_frames, ignore_index=True).to_csv(output_dir / "target_rule_theta_scores.csv", index=False)
    pd.concat(target_weight_frames, ignore_index=True).to_csv(output_dir / "target_rule_weights.csv", index=False)
    summaries_frame(target_rule_summaries).to_csv(output_dir / "target_rule_summary.csv", index=False)
    diagnostic_comparison_frame(summaries, target_rule_summaries).to_csv(
        output_dir / "eta_diagnostic_comparison.csv", index=False
    )
    return scores_by_eta, summaries


def run_baseline_prior_control_with_artifacts(
    processed: ProcessedEpitData,
    output_dir: Path,
    *,
    phase: str,
    n_synth: int,
    synthetic_seed_start: int,
    seq_len: int,
    temperature: float,
    n_workers: int,
    chunk_size: int,
    resume: bool,
    progress_interval: float,
    max_schema_attempts: int,
) -> tuple[list[BaselinePriorControlRecord], EtaSummary]:
    if int(n_synth) <= 0:
        raise ValueError("n_synth must be positive.")

    chunks, skipped = _build_missing_baseline_prior_control_chunks(
        output_dir,
        n_synth=n_synth,
        synthetic_seed_start=synthetic_seed_start,
        expected_n_rows=processed.n_rows,
        phase=phase,
        chunk_size=chunk_size,
        resume=resume,
        max_schema_attempts=max_schema_attempts,
    )
    total_theta = int(n_synth)
    completed_theta = skipped
    start_time = time.monotonic()
    last_progress = 0.0

    print(
        f"Starting baseline prior control: id={BASELINE_PRIOR_CONTROL_ID}, n_synth={n_synth}, "
        f"skipped={skipped}, missing={sum(len(chunk['synthetic_seeds']) for chunk in chunks)}, "
        f"workers={n_workers}, chunk_size={chunk_size}, resume={resume}",
        flush=True,
    )

    def emit_progress(force: bool = False) -> None:
        nonlocal last_progress
        now = time.monotonic()
        if not force and (now - last_progress) < float(progress_interval):
            return
        elapsed = max(0.0, now - start_time)
        fresh_completed = max(0, completed_theta - skipped)
        rate = fresh_completed / elapsed if elapsed > 0.0 and fresh_completed > 0 else 0.0
        remaining = max(0, total_theta - completed_theta)
        eta_seconds = remaining / rate if rate > 0.0 else None
        payload = {
            "phase": phase,
            "control_id": BASELINE_PRIOR_CONTROL_ID,
            "total_theta": total_theta,
            "completed_theta": completed_theta,
            "skipped_theta": skipped,
            "missing_theta": remaining,
            "elapsed_seconds": elapsed,
            "theta_per_second": rate,
            "estimated_remaining_seconds": eta_seconds,
            "n_workers": n_workers,
            "chunk_size": chunk_size,
            "max_schema_attempts": int(max_schema_attempts),
        }
        payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_write_json(Path(output_dir) / "baseline_prior_progress.json", payload)
        message = (
            f"Progress baseline prior control: theta {completed_theta}/{total_theta} "
            f"({100.0 * completed_theta / max(1, total_theta):.1f}%), skipped {skipped}, "
            f"rate {rate:.2f} theta/s, elapsed {_format_seconds(elapsed)}, ETA {_format_seconds(eta_seconds)}"
        )
        print(message, flush=True)
        _append_progress_log(output_dir, message)
        last_progress = now

    emit_progress(force=True)
    _init_theta_worker(processed.X, processed.X_raw, processed.y_z, processed.category_count, seq_len)

    def mark_completed(rows: list[dict[str, Any]]) -> None:
        nonlocal completed_theta
        completed_theta += len(rows)
        emit_progress(force=False)

    if chunks:
        if n_workers <= 1:
            for chunk in chunks:
                rows = _score_baseline_prior_control_chunk(chunk)
                mark_completed(rows)
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_theta_worker,
                initargs=(processed.X, processed.X_raw, processed.y_z, processed.category_count, seq_len),
            ) as executor:
                future_to_size = {
                    executor.submit(_score_baseline_prior_control_chunk, chunk): len(chunk['synthetic_seeds'])
                    for chunk in chunks
                }
                for future in as_completed(future_to_size):
                    rows = future.result()
                    mark_completed(rows)

    emit_progress(force=True)
    if completed_theta != total_theta:
        raise RuntimeError(f"Incomplete baseline prior control artifacts: completed {completed_theta}/{total_theta}.")

    records = load_baseline_prior_control_records(
        output_dir,
        synthetic_seed_start=synthetic_seed_start,
        n_synth=n_synth,
        expected_n_rows=processed.n_rows,
    )
    summary = write_baseline_prior_control_outputs(
        processed,
        records,
        output_dir,
        phase=phase,
        temperature=temperature,
    )
    metadata = baseline_prior_control_metadata(records)
    final_message = (
        f"Baseline prior control complete: n_synth={n_synth}, "
        f"spearman={summary.ensemble_spearman:.6f}, MAE={summary.standardized_mae:.6f}, "
        f"RMSE={summary.standardized_rmse:.6f}, acceptance_rate={float(metadata['acceptance_rate']):.6f}"
    )
    print(final_message, flush=True)
    _append_progress_log(output_dir, final_message)
    return records, summary


def run_candidates_with_artifacts(
    processed: ProcessedEpitData,
    candidates: list[EtaCandidate],
    output_dir: Path,
    args: argparse.Namespace,
    *,
    phase: str,
    scope: str,
    n_synth: int,
    synthetic_seed_start: int,
    seq_len: int,
    temperature: float,
    n_workers: int,
    chunk_size: int,
    resume: bool,
    progress_interval: float,
) -> tuple[SyntheticEpitSample, dict[str, list[ThetaSurrogateScore]], list[EtaSummary]]:
    if not candidates:
        raise ValueError(f"{phase} requires at least one eta candidate.")
    if int(n_synth) <= 0:
        raise ValueError("n_synth must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    first_sample = sample_synthetic_dataset(
        category_count=processed.category_count,
        synthetic_seed=int(synthetic_seed_start),
        seq_len=seq_len,
        eta_params=candidates[0].eta_params,
    )
    write_artifact_run_metadata(
        processed,
        candidates,
        first_sample,
        output_dir,
        args,
        scope=scope,
        n_workers=n_workers,
        chunk_size=chunk_size,
        resume=resume,
    )

    chunks, skipped, completed_by_eta = _build_missing_theta_chunks(
        output_dir,
        candidates,
        phase=phase,
        n_synth=n_synth,
        synthetic_seed_start=synthetic_seed_start,
        expected_n_rows=processed.n_rows,
        chunk_size=chunk_size,
        resume=resume,
    )
    total_theta = len(candidates) * int(n_synth)
    completed_theta = skipped
    start_time = time.monotonic()
    last_progress = 0.0
    summaries_by_eta: dict[str, EtaSummary] = {}

    def summarize_newly_completed_etas() -> None:
        nonlocal summaries_by_eta
        for candidate in candidates:
            eta_id = candidate.eta_id
            if eta_id in summaries_by_eta or completed_by_eta.get(eta_id, 0) < int(n_synth):
                continue
            scores = load_scores_for_candidate(
                output_dir,
                candidate,
                synthetic_seed_start=synthetic_seed_start,
                n_synth=n_synth,
                expected_n_rows=processed.n_rows,
            )
            summary = summarize_eta(
                processed,
                scores,
                temperature=temperature,
                eta_id=eta_id,
                phase=phase,
                anchored_regime=candidate.anchored_regime,
                core_anchor=candidate.core_anchor,
            )
            summaries_by_eta[eta_id] = summary
            message = (
                f"Eta complete {len(summaries_by_eta)}/{len(candidates)} {eta_id}: "
                f"spearman={summary.ensemble_spearman:.6f}, "
                f"MAE={summary.standardized_mae:.6f}, RMSE={summary.standardized_rmse:.6f}, ESS={summary.ess:.2f}"
            )
            print(message, flush=True)
            _append_progress_log(output_dir, message)

    def emit_progress(force: bool = False) -> None:
        nonlocal last_progress
        now = time.monotonic()
        if not force and (now - last_progress) < float(progress_interval):
            return
        elapsed = max(0.0, now - start_time)
        fresh_completed = max(0, completed_theta - skipped)
        rate = fresh_completed / elapsed if elapsed > 0.0 and fresh_completed > 0 else 0.0
        remaining = max(0, total_theta - completed_theta)
        eta_seconds = remaining / rate if rate > 0.0 else None
        best_summary = None
        if summaries_by_eta:
            best_summary = max(summaries_by_eta.values(), key=lambda item: (item.ensemble_spearman, item.ess))
        eta_complete = sum(1 for count in completed_by_eta.values() if count >= int(n_synth))
        payload = {
            "phase": phase,
            "total_etas": len(candidates),
            "completed_etas": eta_complete,
            "total_theta": total_theta,
            "completed_theta": completed_theta,
            "skipped_theta": skipped,
            "missing_theta": remaining,
            "elapsed_seconds": elapsed,
            "theta_per_second": rate,
            "estimated_remaining_seconds": eta_seconds,
            "n_workers": n_workers,
            "chunk_size": chunk_size,
            "current_best_eta": None if best_summary is None else best_summary.eta_id,
            "current_best_spearman": None if best_summary is None else best_summary.ensemble_spearman,
        }
        write_progress_status(output_dir, payload)
        message = (
            f"Progress {phase}: theta {completed_theta}/{total_theta} "
            f"({100.0 * completed_theta / max(1, total_theta):.1f}%), "
            f"etas complete {eta_complete}/{len(candidates)}, "
            f"skipped {skipped}, rate {rate:.2f} theta/s, "
            f"elapsed {_format_seconds(elapsed)}, ETA {_format_seconds(eta_seconds)}"
        )
        if best_summary is not None:
            message += f", best {best_summary.eta_id} spearman={best_summary.ensemble_spearman:.6f}"
        print(message, flush=True)
        _append_progress_log(output_dir, message)
        last_progress = now

    summarize_newly_completed_etas()
    print(
        f"Starting {phase}: etas={len(candidates)}, n_synth={n_synth}, total_theta={total_theta}, "
        f"skipped={skipped}, missing={sum(len(chunk['synthetic_seeds']) for chunk in chunks)}, "
        f"workers={n_workers}, chunk_size={chunk_size}, resume={resume}",
        flush=True,
    )
    emit_progress(force=True)

    _init_theta_worker(processed.X, processed.X_raw, processed.y_z, processed.category_count, seq_len)

    def mark_completed(rows: list[dict[str, Any]]) -> None:
        nonlocal completed_theta
        completed_theta += len(rows)
        for row in rows:
            completed_by_eta[str(row["eta_id"])] += 1
        summarize_newly_completed_etas()
        emit_progress(force=False)

    if chunks:
        if n_workers <= 1:
            for chunk in chunks:
                rows = _score_theta_chunk(chunk)
                mark_completed(rows)
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_theta_worker,
                initargs=(processed.X, processed.X_raw, processed.y_z, processed.category_count, seq_len),
            ) as executor:
                future_to_size = {executor.submit(_score_theta_chunk, chunk): len(chunk['synthetic_seeds']) for chunk in chunks}
                for future in as_completed(future_to_size):
                    rows = future.result()
                    mark_completed(rows)

    summarize_newly_completed_etas()
    emit_progress(force=True)

    if completed_theta != total_theta:
        raise RuntimeError(f"Incomplete theta artifacts: completed {completed_theta}/{total_theta}.")

    scores_by_eta, summaries = write_artifact_final_outputs(
        processed,
        candidates,
        output_dir,
        phase=phase,
        n_synth=n_synth,
        synthetic_seed_start=synthetic_seed_start,
        temperature=temperature,
    )
    ranked = summaries_frame(summaries)
    best = ranked.iloc[0]
    final_message = (
        f"{phase} artifact run complete: n_etas={len(candidates)}, n_synth_per_eta={n_synth}, "
        f"best_eta={best['eta_id']}, best_spearman={float(best['ensemble_spearman']):.6f}, "
        f"best_ESS={float(best['ESS']):.2f}"
    )
    print(final_message, flush=True)
    _append_progress_log(output_dir, final_message)

    if baseline_prior_control_enabled(args):
        run_baseline_prior_control_with_artifacts(
            processed,
            output_dir,
            phase=phase,
            n_synth=n_synth,
            synthetic_seed_start=synthetic_seed_start,
            seq_len=seq_len,
            temperature=temperature,
            n_workers=n_workers,
            chunk_size=chunk_size,
            resume=resume,
            progress_interval=progress_interval,
            max_schema_attempts=baseline_control_max_attempts(args),
        )
    return first_sample, scores_by_eta, summaries

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
        "ensemble_weighting_scheme": ENSEMBLE_WEIGHTING_SCHEME,
        "target_rule_diagnostic": TARGET_RULE_DIAGNOSTIC,
        "target_rule_real_feature_input": "raw_imputed_original_units",
        "informed_mlp_prob_search": {
            "phase1_grid": list(PHASE1_INFORMED_MLP_PROB_GRID),
            "phase2_window": PHASE2_INFORMED_MLP_PROB_WINDOW,
            "full_training_mapping": "--informed_mix_probs p 1-p",
        },
        "baseline_prior_control": baseline_prior_control_config(
            baseline_prior_control_enabled(args), baseline_control_max_attempts(args)
        ),
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
    target_rule_scores_by_eta: dict[str, list[ThetaSurrogateScore]] = {}
    target_rule_summaries: list[EtaSummary] = []
    for candidate in candidates:
        target_scores, target_summary = sample_target_rule_scores_for_eta(
            processed,
            candidate,
            n_synth=args.n_synth,
            synthetic_seed_start=args.synthetic_seed,
            seq_len=first_sample.seq_len,
            temperature=args.temperature,
            phase="phase1",
        )
        target_rule_scores_by_eta[candidate.eta_id] = target_scores
        target_rule_summaries.append(target_summary)
    target_theta_frames = [
        theta_scores_frame(target_rule_scores_by_eta[summary.eta_id], summary) for summary in target_rule_summaries
    ]
    target_weight_frames = [
        weights_frame(target_rule_scores_by_eta[summary.eta_id], summary) for summary in target_rule_summaries
    ]
    pd.concat(theta_frames, ignore_index=True).to_csv(output_dir / "theta_scores.csv", index=False)
    pd.concat(weight_frames, ignore_index=True).to_csv(output_dir / "weights.csv", index=False)
    summaries_frame(summaries).to_csv(output_dir / "summary.csv", index=False)
    pd.concat(target_theta_frames, ignore_index=True).to_csv(output_dir / "target_rule_theta_scores.csv", index=False)
    pd.concat(target_weight_frames, ignore_index=True).to_csv(output_dir / "target_rule_weights.csv", index=False)
    summaries_frame(target_rule_summaries).to_csv(output_dir / "target_rule_summary.csv", index=False)
    diagnostic_comparison_frame(summaries, target_rule_summaries).to_csv(
        output_dir / "eta_diagnostic_comparison.csv", index=False
    )
    if baseline_prior_control_enabled(args):
        baseline_samples, baseline_scores, _ = sample_and_score_baseline_prior_control(
            processed,
            n_synth=args.n_synth,
            synthetic_seed_start=args.synthetic_seed,
            seq_len=first_sample.seq_len,
            temperature=args.temperature,
            phase="phase1",
            max_schema_attempts=baseline_control_max_attempts(args),
        )
        write_baseline_prior_control_outputs(
            processed,
            baseline_prior_control_records(baseline_samples, baseline_scores),
            output_dir,
            phase="phase1",
            temperature=args.temperature,
        )



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
        "ensemble_weighting_scheme": ENSEMBLE_WEIGHTING_SCHEME,
        "target_rule_diagnostic": TARGET_RULE_DIAGNOSTIC,
        "target_rule_real_feature_input": "raw_imputed_original_units",
        "informed_mlp_prob_search": {
            "phase1_grid": list(PHASE1_INFORMED_MLP_PROB_GRID),
            "phase2_window": PHASE2_INFORMED_MLP_PROB_WINDOW,
            "full_training_mapping": "--informed_mix_probs p 1-p",
        },
        "baseline_prior_control": baseline_prior_control_config(
            baseline_prior_control_enabled(args), baseline_control_max_attempts(args)
        ),
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
    target_rule_scores_by_eta: dict[str, list[ThetaSurrogateScore]] = {}
    target_rule_summaries: list[EtaSummary] = []
    for candidate in candidates:
        target_scores, target_summary = sample_target_rule_scores_for_eta(
            processed,
            candidate,
            n_synth=args.n_synth,
            synthetic_seed_start=args.synthetic_seed,
            seq_len=first_sample.seq_len,
            temperature=args.temperature,
            phase="phase2",
        )
        target_rule_scores_by_eta[candidate.eta_id] = target_scores
        target_rule_summaries.append(target_summary)
    target_theta_frames = [
        theta_scores_frame(target_rule_scores_by_eta[summary.eta_id], summary) for summary in target_rule_summaries
    ]
    target_weight_frames = [
        weights_frame(target_rule_scores_by_eta[summary.eta_id], summary) for summary in target_rule_summaries
    ]
    pd.concat(theta_frames, ignore_index=True).to_csv(output_dir / "theta_scores.csv", index=False)
    pd.concat(weight_frames, ignore_index=True).to_csv(output_dir / "weights.csv", index=False)
    summaries_frame(summaries).to_csv(output_dir / "summary.csv", index=False)
    pd.concat(target_theta_frames, ignore_index=True).to_csv(output_dir / "target_rule_theta_scores.csv", index=False)
    pd.concat(target_weight_frames, ignore_index=True).to_csv(output_dir / "target_rule_weights.csv", index=False)
    summaries_frame(target_rule_summaries).to_csv(output_dir / "target_rule_summary.csv", index=False)
    diagnostic_comparison_frame(summaries, target_rule_summaries).to_csv(
        output_dir / "eta_diagnostic_comparison.csv", index=False
    )
    if baseline_prior_control_enabled(args):
        baseline_samples, baseline_scores, _ = sample_and_score_baseline_prior_control(
            processed,
            n_synth=args.n_synth,
            synthetic_seed_start=args.synthetic_seed,
            seq_len=first_sample.seq_len,
            temperature=args.temperature,
            phase="phase2",
            max_schema_attempts=baseline_control_max_attempts(args),
        )
        write_baseline_prior_control_outputs(
            processed,
            baseline_prior_control_records(baseline_samples, baseline_scores),
            output_dir,
            phase="phase2",
            temperature=args.temperature,
        )



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

    target_rule_scores = [score_target_rule_theta(processed, sample) for sample in synthetic_samples]
    target_rule_summary = summarize_target_rule_eta(
        processed,
        target_rule_scores,
        temperature=args.temperature,
        eta_id=summary.eta_id,
        phase=summary.phase,
        anchored_regime=summary.anchored_regime,
        core_anchor=summary.core_anchor,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "phase": args.phase,
        "random_state": args.random_state,
        "synthetic_seed_start": args.synthetic_seed,
        "n_synth": args.n_synth,
        "temperature": args.temperature,
        "ensemble_weighting_scheme": ENSEMBLE_WEIGHTING_SCHEME,
        "target_rule_diagnostic": TARGET_RULE_DIAGNOSTIC,
        "target_rule_real_feature_input": "raw_imputed_original_units",
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
    theta_scores_frame(target_rule_scores, target_rule_summary).to_csv(
        output_dir / "target_rule_theta_scores.csv", index=False
    )
    weights_frame(target_rule_scores, target_rule_summary).to_csv(output_dir / "target_rule_weights.csv", index=False)
    pd.DataFrame([target_rule_summary.metrics_dict()]).to_csv(output_dir / "target_rule_summary.csv", index=False)
    diagnostic_comparison_frame([summary], [target_rule_summary]).to_csv(
        output_dir / "eta_diagnostic_comparison.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("smoke", "phase1", "phase2", BASELINE_PRIOR_CONTROL_PHASE),
        default="smoke",
        help="Direct-prior phase. Phase 1/2 search informed etas; baseline_prior_control runs only the default TabICL prior control.",
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
        help=(
            "Retained for compatibility with older weighted runs; "
            "current eta scoring uses uniform theta weights."
        ),
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
        help="Limit Phase 1 to the first N etas for debugging. Default 0 runs the full grid.",
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
    parser.add_argument(
        "--n-workers",
        type=int,
        default=0,
        help="Parallel theta workers. Default 0 uses SLURM_CPUS_PER_TASK when available, else 1.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Synthetic seeds per worker task. Larger chunks reduce multiprocessing overhead.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=DEFAULT_PROGRESS_INTERVAL_SECONDS,
        help="Seconds between progress status lines and progress.json updates.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute theta artifacts instead of skipping valid existing artifacts.",
    )
    parser.add_argument(
        "--no-baseline-prior-control",
        action="store_true",
        help=(
            "Skip the default TabICL prior Ridge control. By default phase1/phase2 also score one "
            "unsearched mix_scm baseline prior conditioned to accepted d=21 draws."
        ),
    )
    parser.add_argument(
        "--baseline-control-max-attempts",
        type=int,
        default=DEFAULT_BASELINE_CONTROL_MAX_ATTEMPTS,
        help="Maximum default-prior sampling attempts per accepted d=21 baseline-control theta.",
    )
    parser.add_argument("--no-write", action="store_true", help="Load and preprocess without writing artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = load_epit_task(random_state=args.random_state)
    processed = preprocess_real_epit(task)
    synthetic_seq_len = processed.n_rows if args.synthetic_seq_len == 0 else args.synthetic_seq_len
    output_dir = args.output_dir or default_output_dir(args.phase)
    n_workers = resolve_n_workers(args.n_workers)
    chunk_size = max(1, int(args.chunk_size))
    resume = not bool(args.no_resume)
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass

    print(
        "Loaded real EPIT direct schema: "
        f"rows={processed.n_rows}, features={processed.n_features}, "
        f"category_count={processed.category_count}"
    )
    print(
        "Execution settings: "
        f"workers={n_workers}, chunk_size={chunk_size}, resume={resume}, "
        f"progress_interval={args.progress_interval}s, "
        f"baseline_prior_control={baseline_prior_control_enabled(args)}"
    )


    if args.phase == BASELINE_PRIOR_CONTROL_PHASE:
        if args.no_baseline_prior_control:
            raise SystemExit("--no-baseline-prior-control cannot be used with --phase baseline_prior_control.")
        if args.no_write:
            samples, scores, summary = sample_and_score_baseline_prior_control(
                processed,
                n_synth=args.n_synth,
                synthetic_seed_start=args.synthetic_seed,
                seq_len=synthetic_seq_len,
                temperature=args.temperature,
                phase=BASELINE_PRIOR_CONTROL_PHASE,
                max_schema_attempts=baseline_control_max_attempts(args),
            )
            records = baseline_prior_control_records(samples, scores)
            metadata = baseline_prior_control_metadata(records)
            print(
                "Baseline prior control complete: "
                f"n_synth={args.n_synth}, spearman={summary.ensemble_spearman:.6f}, "
                f"MAE={summary.standardized_mae:.6f}, RMSE={summary.standardized_rmse:.6f}, "
                f"acceptance_rate={float(metadata['acceptance_rate']):.6f}"
            )
        else:
            write_baseline_prior_control_run_metadata(
                processed,
                output_dir,
                args,
                n_workers=n_workers,
                chunk_size=chunk_size,
                resume=resume,
            )
            run_baseline_prior_control_with_artifacts(
                processed,
                output_dir,
                phase=BASELINE_PRIOR_CONTROL_PHASE,
                n_synth=args.n_synth,
                synthetic_seed_start=args.synthetic_seed,
                seq_len=synthetic_seq_len,
                temperature=args.temperature,
                n_workers=n_workers,
                chunk_size=chunk_size,
                resume=resume,
                progress_interval=args.progress_interval,
                max_schema_attempts=baseline_control_max_attempts(args),
            )
            print(f"Wrote {output_dir}")
        return

    if args.phase == "phase1":
        candidates = build_phase1_candidates(max_etas=args.max_etas)
        if args.no_write:
            candidates, first_sample, scores_by_eta, summaries = run_phase1(
                processed,
                n_synth=args.n_synth,
                synthetic_seed_start=args.synthetic_seed,
                seq_len=synthetic_seq_len,
                temperature=args.temperature,
                max_etas=args.max_etas,
            )
        else:
            first_sample, scores_by_eta, summaries = run_candidates_with_artifacts(
                processed,
                candidates,
                output_dir,
                args,
                phase="phase1",
                scope="phase1_anchor_regime_grid",
                n_synth=args.n_synth,
                synthetic_seed_start=args.synthetic_seed,
                seq_len=synthetic_seq_len,
                temperature=args.temperature,
                n_workers=n_workers,
                chunk_size=chunk_size,
                resume=resume,
                progress_interval=args.progress_interval,
            )
        ranked = summaries_frame(summaries)
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
        phase1_summary = read_phase1_summary(args.phase2_source_dir)
        candidates = build_phase2_candidates(
            phase1_summary,
            top_regimes=args.phase2_top_regimes,
            n_core_samples=args.n_core_samples,
            random_state=args.random_state,
        )
        if args.no_write:
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
        else:
            first_sample, scores_by_eta, summaries = run_candidates_with_artifacts(
                processed,
                candidates,
                output_dir,
                args,
                phase="phase2",
                scope="phase2_space_filling_core_search",
                n_synth=args.n_synth,
                synthetic_seed_start=args.synthetic_seed,
                seq_len=synthetic_seq_len,
                temperature=args.temperature,
                n_workers=n_workers,
                chunk_size=chunk_size,
                resume=resume,
                progress_interval=args.progress_interval,
            )
        ranked = summaries_frame(summaries)
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
    first_target_rule_score = score_target_rule_theta(processed, first_sample)
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
        "First synthetic target-rule oracle: "
        f"spearman={first_target_rule_score.spearman:.6f}, "
        f"standardized_mae={first_target_rule_score.standardized_mae:.6f}, "
        f"standardized_rmse={first_target_rule_score.standardized_rmse:.6f}"
    )
    print(
        "One-eta uniform Ridge ensemble: "
        f"spearman={summary.ensemble_spearman:.6f}, standardized_mae={summary.standardized_mae:.6f}, "
        f"standardized_rmse={summary.standardized_rmse:.6f}, ESS={summary.ess:.2f}"
    )
    if not args.no_write:
        print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
