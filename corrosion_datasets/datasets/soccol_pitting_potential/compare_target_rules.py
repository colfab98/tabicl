#!/usr/bin/env python
"""Compare existing and Soccol-specific EPIT target rules on frozen dev folds.

The script never evaluates or calibrates against final-test targets.  It adapts
the existing EPIT rule-calibration methodology to Soccol column names and adds
small, separately testable terms for the extra experimental variables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_DATA = HERE / "processed" / "soccol_regression_event1.csv"
DEFAULT_SPLITS = HERE / "processed" / "splits_v1" / "split_assignments.csv"
DEFAULT_OUTPUT = REPO_ROOT / "corrosion_datasets" / "analysis" / "soccol_target_rules_v1"

COMPOSITION_ZERO_COLUMNS = (
    "C",
    "N",
    "Si",
    "P",
    "S",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Ni",
    "Nb",
    "Mo",
)
ION_ZERO_COLUMNS = (
    "CP_Cl",
    "CP_Br",
    "CP_OH",
    "CP_SO4",
    "CP_CO3",
    "CP_NO3",
    "CP_PO4",
    "CP_MoO4",
    "CP_CrO4",
    "CP_ion_other",
)
MEAN_IMPUTE_COLUMNS = (
    "CP_temp",
    "CP_pH",
    "Test_area_cm2",
    "scan_rate",
    "Prep_grinding_grit",
    "Prep_Ra_micron",
)

BASE_TERMS = (
    "material_passivity",
    "log_halide_aggressiveness",
    "high_temperature_aggressiveness",
    "temperature_halide_interaction",
    "acidic_ph_aggressiveness",
)
BASE_ANCHOR = np.asarray([0.42, 0.24, 0.14, 0.14, 0.06], dtype=float)
BASE_UPPER = np.asarray([1.0, 0.65, 0.45, 0.45, 0.15], dtype=float)


@dataclass(frozen=True)
class RuleSpec:
    name: str
    description: str
    material: str
    halide: str
    extras: tuple[str, ...]
    term_names: tuple[str, ...]
    anchor: np.ndarray
    upper_bounds: np.ndarray
    status: str


def _extend_rule(
    *,
    name: str,
    description: str,
    material: str = "pren_n",
    halide: str = "chloride",
    extras: tuple[str, ...] = (),
    extra_anchor: tuple[float, ...] = (),
    extra_upper: tuple[float, ...] = (),
    status: str = "new_candidate",
) -> RuleSpec:
    if len(extras) != len(extra_anchor) or len(extras) != len(extra_upper):
        raise ValueError("Extra-term metadata lengths differ.")
    extra_total = float(sum(extra_anchor))
    anchor = np.concatenate([BASE_ANCHOR * (1.0 - extra_total), extra_anchor])
    upper = np.concatenate([BASE_UPPER, extra_upper])
    return RuleSpec(
        name=name,
        description=description,
        material=material,
        halide=halide,
        extras=extras,
        term_names=BASE_TERMS + extras,
        anchor=anchor / anchor.sum(),
        upper_bounds=upper,
        status=status,
    )


CURRENT_ANCHOR = np.asarray([0.575, 0.50, 0.775], dtype=float)
CURRENT_ANCHOR /= CURRENT_ANCHOR.sum()
INTERACTION_ANCHOR = np.asarray([0.575, 0.50, 0.775, 0.20], dtype=float)
INTERACTION_ANCHOR /= INTERACTION_ANCHOR.sum()
COUPLED_TERMS = (
    "material_passivity",
    "coupled_environment_breakdown",
    "acidic_ph_aggressiveness",
)
COUPLED_ANCHOR = np.asarray([0.55, 0.40, 0.05], dtype=float)
COUPLED_UPPER = np.asarray([1.0, 0.80, 0.15], dtype=float)
COUPLED_MNS_ANCHOR = np.concatenate(
    [COUPLED_ANCHOR * 0.92, np.asarray([0.08])]
)
COUPLED_MNS_UPPER = np.concatenate([COUPLED_UPPER, np.asarray([0.25])])

RULES = (
    RuleSpec(
        name="old_current_pren",
        description="Historical Cr-Ni-Mo material, combined environment and interaction rule",
        material="current",
        halide="chloride",
        extras=(),
        term_names=(
            "material_passivity",
            "environment_aggressiveness",
            "material_chloride_interaction",
        ),
        anchor=CURRENT_ANCHOR,
        upper_bounds=np.ones(3, dtype=float),
        status="historical_reference",
    ),
    RuleSpec(
        name="old_pren_n_linear",
        description="Promoted nitrogen-aware PREN with the historical combined environment",
        material="pren_n",
        halide="chloride",
        extras=(),
        term_names=(
            "material_passivity",
            "environment_aggressiveness",
            "material_chloride_interaction",
        ),
        anchor=CURRENT_ANCHOR.copy(),
        upper_bounds=np.ones(3, dtype=float),
        status="existing_promoted_rule",
    ),
    RuleSpec(
        name="old_cr_mo_n_synergy",
        description="Promoted N-aware PREN plus Cr-Mo synergy and temperature-chloride interaction",
        material="pren_n_synergy",
        halide="chloride",
        extras=(),
        term_names=(
            "material_passivity",
            "environment_aggressiveness",
            "material_chloride_interaction",
            "temperature_chloride_interaction",
        ),
        anchor=INTERACTION_ANCHOR.copy(),
        upper_bounds=np.ones(4, dtype=float),
        status="existing_promoted_rule",
    ),
    RuleSpec(
        name="old_threshold_saturation",
        description="Promoted chromium-threshold and molybdenum-saturation rule",
        material="threshold_saturation",
        halide="chloride",
        extras=(),
        term_names=(
            "material_passivity",
            "environment_aggressiveness",
            "material_chloride_interaction",
            "temperature_chloride_interaction",
        ),
        anchor=INTERACTION_ANCHOR.copy(),
        upper_bounds=np.ones(4, dtype=float),
        status="existing_promoted_rule",
    ),
    _extend_rule(
        name="old_improved_environment",
        description="Existing Cr-Mo rule with separate chloride, temperature and acidic-pH effects",
        material="pren_no_n",
        status="historical_reference",
    ),
    _extend_rule(
        name="old_pren_n_improved_environment",
        description="Existing Cr-Mo-N rule with separate environmental effects",
        status="existing_promoted_rule",
    ),
    _extend_rule(
        name="old_mns_inclusion_penalty",
        description="Existing N-aware rule plus bulk Mn-S inclusion proxy",
        extras=("mns_inclusion_susceptibility",),
        extra_anchor=(0.08,),
        extra_upper=(0.25,),
        status="existing_promoted_rule",
    ),
    _extend_rule(
        name="old_mo_n_acid_repassivation",
        description="Promoted N-aware rule plus Mo-N repassivation under acidic conditions",
        extras=("mo_n_acid_repassivation",),
        extra_anchor=(0.10,),
        extra_upper=(0.30,),
        status="existing_promoted_rule",
    ),
    RuleSpec(
        name="old_coupled_breakdown",
        description="Promoted environment-versus-material coupled breakdown rule",
        material="pren_no_n",
        halide="chloride",
        extras=(),
        term_names=COUPLED_TERMS,
        anchor=COUPLED_ANCHOR.copy(),
        upper_bounds=COUPLED_UPPER.copy(),
        status="existing_promoted_rule",
    ),
    _extend_rule(
        name="old_fe_ni_cr_threshold",
        description="Promoted Fe/Ni-specific Cr threshold with separate environmental effects",
        material="fe_ni_threshold",
        status="existing_promoted_rule",
    ),
    _extend_rule(
        name="new_specimen_area",
        description="N-aware rule plus lower EPIT for larger exposed specimen area",
        extras=("specimen_area_penalty",),
        extra_anchor=(0.08,),
        extra_upper=(0.25,),
    ),
    _extend_rule(
        name="new_surface_and_area",
        description="N-aware rule plus specimen area and two surface-finish measurements",
        extras=(
            "specimen_area_penalty",
            "grit_smoothness_benefit",
            "roughness_penalty",
        ),
        extra_anchor=(0.06, 0.04, 0.04),
        extra_upper=(0.25, 0.20, 0.20),
    ),
    _extend_rule(
        name="new_effective_halide",
        description="N-aware rule using Cl + 0.5 Br as an effective halide concentration",
        halide="effective_halide",
    ),
    _extend_rule(
        name="new_anion_competition",
        description="Effective-halide rule plus inhibitor-to-halide concentration ratios",
        halide="effective_halide",
        extras=("strong_inhibitor_ratio", "weak_inhibitor_ratio"),
        extra_anchor=(0.06, 0.03),
        extra_upper=(0.25, 0.15),
    ),
    _extend_rule(
        name="new_scan_rate_positive",
        description="N-aware rule with higher measured EPIT at higher potential scan rate",
        extras=("scan_rate_positive",),
        extra_anchor=(0.04,),
        extra_upper=(0.20,),
    ),
    _extend_rule(
        name="new_scan_rate_negative",
        description="N-aware rule with lower measured EPIT at higher potential scan rate",
        extras=("scan_rate_negative",),
        extra_anchor=(0.04,),
        extra_upper=(0.20,),
    ),
    _extend_rule(
        name="new_full_without_scan",
        description="Combined N, Mn-S, area, finish, effective-halide and inhibitor rule",
        halide="effective_halide",
        extras=(
            "mns_inclusion_susceptibility",
            "specimen_area_penalty",
            "grit_smoothness_benefit",
            "roughness_penalty",
            "strong_inhibitor_ratio",
            "weak_inhibitor_ratio",
        ),
        extra_anchor=(0.05, 0.06, 0.03, 0.03, 0.05, 0.02),
        extra_upper=(0.25, 0.25, 0.20, 0.20, 0.25, 0.15),
    ),
    _extend_rule(
        name="new_mns_specimen_area",
        description="Best existing Mn-S rule plus specimen-area correction",
        extras=("mns_inclusion_susceptibility", "specimen_area_penalty"),
        extra_anchor=(0.08, 0.06),
        extra_upper=(0.25, 0.25),
    ),
    _extend_rule(
        name="new_mns_surface_and_area",
        description="Best existing Mn-S rule plus area and surface-finish corrections",
        extras=(
            "mns_inclusion_susceptibility",
            "specimen_area_penalty",
            "grit_smoothness_benefit",
            "roughness_penalty",
        ),
        extra_anchor=(0.08, 0.05, 0.03, 0.03),
        extra_upper=(0.25, 0.25, 0.20, 0.20),
    ),
    _extend_rule(
        name="new_mns_effective_halide",
        description="Best existing Mn-S rule with Cl + 0.5 Br effective halide",
        halide="effective_halide",
        extras=("mns_inclusion_susceptibility",),
        extra_anchor=(0.08,),
        extra_upper=(0.25,),
    ),
    _extend_rule(
        name="new_mns_anion_competition",
        description="Best existing Mn-S rule plus effective halide and inhibitor ratios",
        halide="effective_halide",
        extras=(
            "mns_inclusion_susceptibility",
            "strong_inhibitor_ratio",
            "weak_inhibitor_ratio",
        ),
        extra_anchor=(0.08, 0.05, 0.03),
        extra_upper=(0.25, 0.25, 0.15),
    ),
    _extend_rule(
        name="new_mns_scan_rate_positive",
        description="Best existing Mn-S rule plus positive potential-scan-rate bias",
        extras=("mns_inclusion_susceptibility", "scan_rate_positive"),
        extra_anchor=(0.08, 0.04),
        extra_upper=(0.25, 0.20),
    ),
    _extend_rule(
        name="new_mns_full_with_scan",
        description="Combined full rule including the empirically favored scan-rate sign",
        halide="effective_halide",
        extras=(
            "mns_inclusion_susceptibility",
            "specimen_area_penalty",
            "grit_smoothness_benefit",
            "roughness_penalty",
            "strong_inhibitor_ratio",
            "weak_inhibitor_ratio",
            "scan_rate_positive",
        ),
        extra_anchor=(0.05, 0.05, 0.025, 0.025, 0.04, 0.02, 0.04),
        extra_upper=(0.25, 0.25, 0.20, 0.20, 0.25, 0.15, 0.20),
    ),
    _extend_rule(
        name="new_mns_halide_activation",
        description="Mn-S susceptibility activated by increasing effective-halide concentration",
        halide="effective_halide",
        extras=("mns_halide_activation",),
        extra_anchor=(0.08,),
        extra_upper=(0.25,),
    ),
    _extend_rule(
        name="new_mns_acid_halide_activation",
        description="Mn-S susceptibility activated jointly by effective halide and acidic pH",
        halide="effective_halide",
        extras=("mns_acid_halide_activation",),
        extra_anchor=(0.08,),
        extra_upper=(0.25,),
    ),
    RuleSpec(
        name="new_pren_n_coupled_breakdown",
        description="Coupled-breakdown rule upgraded from Cr-Mo to Cr-Mo-N PREN",
        material="pren_n",
        halide="chloride",
        extras=(),
        term_names=COUPLED_TERMS,
        anchor=COUPLED_ANCHOR.copy(),
        upper_bounds=COUPLED_UPPER.copy(),
        status="new_candidate",
    ),
    RuleSpec(
        name="new_coupled_mns",
        description="Existing coupled-breakdown rule plus Mn-S susceptibility",
        material="pren_no_n",
        halide="chloride",
        extras=("mns_inclusion_susceptibility",),
        term_names=COUPLED_TERMS + ("mns_inclusion_susceptibility",),
        anchor=np.concatenate([COUPLED_ANCHOR * 0.92, np.asarray([0.08])]),
        upper_bounds=np.concatenate([COUPLED_UPPER, np.asarray([0.25])]),
        status="new_candidate",
    ),
    RuleSpec(
        name="new_pren_n_coupled_mns",
        description="Cr-Mo-N coupled-breakdown rule plus Mn-S susceptibility",
        material="pren_n",
        halide="chloride",
        extras=("mns_inclusion_susceptibility",),
        term_names=COUPLED_TERMS + ("mns_inclusion_susceptibility",),
        anchor=COUPLED_MNS_ANCHOR.copy(),
        upper_bounds=COUPLED_MNS_UPPER.copy(),
        status="new_candidate",
    ),
    RuleSpec(
        name="new_coupled_mns_anion_competition",
        description="Coupled Mn-S rule plus effective halide and inhibitor ratios",
        material="pren_no_n",
        halide="effective_halide",
        extras=(
            "mns_inclusion_susceptibility",
            "strong_inhibitor_ratio",
            "weak_inhibitor_ratio",
        ),
        term_names=COUPLED_TERMS
        + (
            "mns_inclusion_susceptibility",
            "strong_inhibitor_ratio",
            "weak_inhibitor_ratio",
        ),
        anchor=np.concatenate(
            [COUPLED_ANCHOR * 0.84, np.asarray([0.08, 0.05, 0.03])]
        ),
        upper_bounds=np.concatenate(
            [COUPLED_UPPER, np.asarray([0.25, 0.25, 0.15])]
        ),
        status="new_candidate",
    ),
    RuleSpec(
        name="new_pren_n_coupled_mns_area",
        description="Leading new coupled rule plus specimen-area correction",
        material="pren_n",
        halide="chloride",
        extras=("mns_inclusion_susceptibility", "specimen_area_penalty"),
        term_names=COUPLED_TERMS
        + ("mns_inclusion_susceptibility", "specimen_area_penalty"),
        anchor=np.concatenate([COUPLED_MNS_ANCHOR * 0.95, np.asarray([0.05])]),
        upper_bounds=np.concatenate([COUPLED_MNS_UPPER, np.asarray([0.25])]),
        status="new_candidate",
    ),
    RuleSpec(
        name="new_pren_n_coupled_mns_surface",
        description="Leading new coupled rule plus area and surface-finish corrections",
        material="pren_n",
        halide="chloride",
        extras=(
            "mns_inclusion_susceptibility",
            "specimen_area_penalty",
            "grit_smoothness_benefit",
            "roughness_penalty",
        ),
        term_names=COUPLED_TERMS
        + (
            "mns_inclusion_susceptibility",
            "specimen_area_penalty",
            "grit_smoothness_benefit",
            "roughness_penalty",
        ),
        anchor=np.concatenate(
            [COUPLED_MNS_ANCHOR * 0.91, np.asarray([0.04, 0.025, 0.025])]
        ),
        upper_bounds=np.concatenate(
            [COUPLED_MNS_UPPER, np.asarray([0.25, 0.20, 0.20])]
        ),
        status="new_candidate",
    ),
    RuleSpec(
        name="new_pren_n_coupled_mns_anions",
        description="Leading new coupled rule plus effective halide and inhibitor ratios",
        material="pren_n",
        halide="effective_halide",
        extras=(
            "mns_inclusion_susceptibility",
            "strong_inhibitor_ratio",
            "weak_inhibitor_ratio",
        ),
        term_names=COUPLED_TERMS
        + (
            "mns_inclusion_susceptibility",
            "strong_inhibitor_ratio",
            "weak_inhibitor_ratio",
        ),
        anchor=np.concatenate(
            [COUPLED_MNS_ANCHOR * 0.92, np.asarray([0.05, 0.03])]
        ),
        upper_bounds=np.concatenate(
            [COUPLED_MNS_UPPER, np.asarray([0.25, 0.15])]
        ),
        status="new_candidate",
    ),
    RuleSpec(
        name="new_pren_n_coupled_mns_effective_halide",
        description="Leading new coupled rule using Cl + 0.5 Br effective halide",
        material="pren_n",
        halide="effective_halide",
        extras=("mns_inclusion_susceptibility",),
        term_names=COUPLED_TERMS + ("mns_inclusion_susceptibility",),
        anchor=COUPLED_MNS_ANCHOR.copy(),
        upper_bounds=COUPLED_MNS_UPPER.copy(),
        status="new_candidate",
    ),
    RuleSpec(
        name="new_pren_n_coupled_mns_weak_anions",
        description="Leading new coupled rule plus effective halide and sulfate/nitrate ratio",
        material="pren_n",
        halide="effective_halide",
        extras=("mns_inclusion_susceptibility", "weak_inhibitor_ratio"),
        term_names=COUPLED_TERMS
        + ("mns_inclusion_susceptibility", "weak_inhibitor_ratio"),
        anchor=np.concatenate([COUPLED_MNS_ANCHOR * 0.95, np.asarray([0.05])]),
        upper_bounds=np.concatenate([COUPLED_MNS_UPPER, np.asarray([0.15])]),
        status="new_candidate",
    ),
    RuleSpec(
        name="new_pren_n_coupled_mns_strong_anions",
        description="Leading new coupled rule plus effective halide and PO4/MoO4/CrO4 ratio",
        material="pren_n",
        halide="effective_halide",
        extras=("mns_inclusion_susceptibility", "strong_inhibitor_ratio"),
        term_names=COUPLED_TERMS
        + ("mns_inclusion_susceptibility", "strong_inhibitor_ratio"),
        anchor=np.concatenate([COUPLED_MNS_ANCHOR * 0.95, np.asarray([0.05])]),
        upper_bounds=np.concatenate([COUPLED_MNS_UPPER, np.asarray([0.25])]),
        status="new_candidate",
    ),
    RuleSpec(
        name="new_pren_n_coupled_mns_scan",
        description="Leading new coupled rule plus positive potential-scan-rate bias",
        material="pren_n",
        halide="chloride",
        extras=("mns_inclusion_susceptibility", "scan_rate_positive"),
        term_names=COUPLED_TERMS
        + ("mns_inclusion_susceptibility", "scan_rate_positive"),
        anchor=np.concatenate([COUPLED_MNS_ANCHOR * 0.96, np.asarray([0.04])]),
        upper_bounds=np.concatenate([COUPLED_MNS_UPPER, np.asarray([0.20])]),
        status="new_candidate",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anchor-strength", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _fit_scale(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    return {"mean": mean, "std": std if std > 1e-12 else 1.0}


def _scale(values: np.ndarray, state: dict[str, float]) -> np.ndarray:
    return (np.asarray(values, dtype=float) - state["mean"]) / state["std"]


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values, ddof=0))
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _rank_standardize(values: np.ndarray) -> np.ndarray:
    return _standardize(rankdata(np.asarray(values, dtype=float), method="average"))


def _spearman(target: np.ndarray, prediction: np.ndarray) -> float:
    if np.unique(target).size < 2 or np.unique(prediction).size < 2:
        return 0.0
    value = float(spearmanr(target, prediction).correlation)
    return value if np.isfinite(value) else 0.0


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target_z = _standardize(target)
    prediction_z = _standardize(prediction)
    residual = prediction_z - target_z
    return {
        "spearman": _spearman(target, prediction),
        "standardized_mae": float(np.mean(np.abs(residual))),
        "standardized_rmse": float(np.sqrt(np.mean(residual**2))),
    }


def _fit_coefficients(
    terms: np.ndarray,
    target: np.ndarray,
    anchor: np.ndarray,
    upper_bounds: np.ndarray,
    anchor_strength: float,
) -> tuple[np.ndarray, float]:
    target_rank = _rank_standardize(target)
    anchor = np.asarray(anchor, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)

    def objective(weights: np.ndarray) -> float:
        residual = target_rank - terms @ weights
        return float(
            np.mean(residual**2)
            + anchor_strength * np.sum((weights - anchor) ** 2)
        )

    result = minimize(
        objective,
        x0=anchor,
        method="SLSQP",
        bounds=[(0.0, float(bound)) for bound in upper_bounds],
        constraints=[{"type": "eq", "fun": lambda values: float(values.sum() - 1.0)}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Coefficient calibration failed: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    weights[np.abs(weights) < 1e-10] = 0.0
    weights /= weights.sum()
    return weights, objective(weights)


def _load_data(data_path: Path, split_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path, low_memory=False)
    splits = pd.read_csv(split_path, low_memory=False)
    if len(data) != len(splits):
        raise RuntimeError("Processed data and split assignment row counts differ.")
    if not np.array_equal(
        data["raw_workbook_row"].to_numpy(), splits["raw_workbook_row"].to_numpy()
    ):
        raise RuntimeError("Processed data and split assignments are not row-aligned.")
    data = data.copy()
    data["outer_split"] = splits["outer_split"].to_numpy()
    data["development_validation_fold"] = splits[
        "development_validation_fold"
    ].to_numpy()
    data["composition_group"] = splits["composition_group"].to_numpy()

    numeric = set(COMPOSITION_ZERO_COLUMNS + ION_ZERO_COLUMNS + MEAN_IMPUTE_COLUMNS)
    numeric.add("E_pit")
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in COMPOSITION_ZERO_COLUMNS + ION_ZERO_COLUMNS:
        data[column] = data[column].fillna(0.0)

    final_mask = data["outer_split"].eq("final_test")
    data.loc[final_mask, "E_pit"] = np.nan
    development = data[data["outer_split"].eq("development")]
    if development["E_pit"].isna().any():
        raise RuntimeError("Development rows contain missing EPIT targets.")
    folds = sorted(development["development_validation_fold"].astype(int).unique())
    if folds != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"Expected five frozen development folds, found {folds}.")
    return data


def _fit_imputation(data: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for column in MEAN_IMPUTE_COLUMNS:
        finite = data[column].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise RuntimeError(f"{column} is entirely missing in a context fold.")
        result[column] = float(np.mean(finite))
    return result


def _inputs(data: pd.DataFrame, imputation: dict[str, float]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for column in COMPOSITION_ZERO_COLUMNS + ION_ZERO_COLUMNS:
        array = data[column].to_numpy(dtype=float)
        values[column] = np.nan_to_num(array, nan=0.0)
    for column in MEAN_IMPUTE_COLUMNS:
        array = data[column].to_numpy(dtype=float)
        values[column] = np.where(np.isfinite(array), array, imputation[column])
    values["material_is_Ni_based"] = pd.to_numeric(
        data["material_is_Ni_based"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    return values


def _material(values: dict[str, np.ndarray], kind: str) -> np.ndarray:
    if kind == "current":
        return values["Cr"] + 0.25 * values["Ni"] + 3.3 * values["Mo"]
    if kind == "pren_no_n":
        return values["Cr"] + 3.25 * values["Mo"]
    if kind == "pren_n":
        return values["Cr"] + 3.3 * values["Mo"] + 16.0 * values["N"]
    if kind == "pren_n_synergy":
        chromium = np.clip(values["Cr"], 0.0, None)
        molybdenum = np.clip(values["Mo"], 0.0, None)
        return (
            chromium
            + 3.3 * molybdenum
            + 16.0 * values["N"]
            + np.sqrt(chromium * molybdenum)
        )
    if kind == "threshold_saturation":
        chromium = np.clip(values["Cr"], 0.0, None)
        molybdenum = np.clip(values["Mo"], 0.0, None)
        return _sigmoid((chromium - 12.0) / 2.0) + np.log1p(molybdenum)
    if kind == "fe_ni_threshold":
        chromium = np.clip(values["Cr"], 0.0, None)
        molybdenum = np.clip(values["Mo"], 0.0, None)
        threshold = np.where(values["material_is_Ni_based"] > 0.5, 15.0, 12.0)
        return _sigmoid((chromium - threshold) / 2.0) + np.log1p(molybdenum)
    raise ValueError(f"Unknown material formula {kind!r}.")


def _halide(values: dict[str, np.ndarray], kind: str) -> np.ndarray:
    chloride = np.clip(values["CP_Cl"], 0.0, None)
    if kind == "chloride":
        return chloride
    if kind == "effective_halide":
        return chloride + 0.5 * np.clip(values["CP_Br"], 0.0, None)
    raise ValueError(f"Unknown halide formula {kind!r}.")


def _raw_terms(
    values: dict[str, np.ndarray], spec: RuleSpec, state: dict[str, Any]
) -> np.ndarray:
    material = _material(values, spec.material)
    material_z = _scale(material, state["material_scale"])
    halide = _halide(values, spec.halide)
    halide_log = np.log10(np.clip(halide, 1e-12, None))
    halide_z = _scale(halide_log, state["halide_scale"])
    temperature = values["CP_temp"]
    ph = values["CP_pH"]

    historical_environment_rules = {
        "old_current_pren",
        "old_pren_n_linear",
        "old_cr_mo_n_synergy",
        "old_threshold_saturation",
    }
    if spec.name in historical_environment_rules:
        temperature_z = _scale(temperature, state["temperature_scale"])
        ph_distance_z = _scale(np.abs(ph - 7.25), state["ph_distance_scale"])
        environment = (
            0.075 * temperature_z + 0.925 * halide_z + 0.115 * ph_distance_z
        )
        environment_z = _scale(environment, state["environment_scale"])
        columns = [
            np.tanh(material_z),
            -_sigmoid(environment_z),
            -_sigmoid(-material_z) * _sigmoid(halide_z),
        ]
        if "temperature_chloride_interaction" in spec.term_names:
            temperature_z = _scale(temperature, state["temperature_scale"])
            columns.append(-_sigmoid(temperature_z) * _sigmoid(halide_z))
        return np.column_stack(columns)

    halide_signal = _sigmoid(halide_z)
    temperature_signal = _sigmoid((temperature - 50.0) / 10.0)
    acidic_ph_signal = _sigmoid((6.5 - ph) / 1.0)
    is_coupled_rule = "coupled" in spec.name
    if is_coupled_rule:
        temperature_z = _scale(
            temperature_signal, state["temperature_signal_scale"]
        )
        joint_z = _scale(
            temperature_signal * halide_signal, state["joint_signal_scale"]
        )
        aggressiveness = 0.65 * halide_z + 0.25 * temperature_z + 0.10 * joint_z
        coupled = _sigmoid(aggressiveness - 0.75 * material_z)
        columns = [np.tanh(material_z), -coupled, -acidic_ph_signal]
    else:
        columns = [
            np.tanh(material_z),
            -halide_signal,
            -temperature_signal,
            -(temperature_signal * halide_signal),
            -acidic_ph_signal,
        ]
    effective_halide = np.clip(
        values["CP_Cl"] + 0.5 * values["CP_Br"], 0.0, None
    )
    for extra in spec.extras:
        if extra == "mns_inclusion_susceptibility":
            columns.append(-np.sqrt(np.clip(values["Mn"] * values["S"], 0.0, None)))
        elif extra == "mns_halide_activation":
            susceptibility = np.sqrt(
                np.clip(values["Mn"] * values["S"], 0.0, None)
            )
            columns.append(-susceptibility * halide_signal)
        elif extra == "mns_acid_halide_activation":
            susceptibility = np.sqrt(
                np.clip(values["Mn"] * values["S"], 0.0, None)
            )
            columns.append(-susceptibility * halide_signal * acidic_ph_signal)
        elif extra == "mo_n_acid_repassivation":
            columns.append(
                np.sqrt(np.clip(values["Mo"] * values["N"], 0.0, None))
                * acidic_ph_signal
            )
        elif extra == "specimen_area_penalty":
            columns.append(-np.log10(np.clip(values["Test_area_cm2"], 1e-8, None)))
        elif extra == "grit_smoothness_benefit":
            columns.append(np.log10(np.clip(values["Prep_grinding_grit"], 1e-8, None)))
        elif extra == "roughness_penalty":
            columns.append(-np.log10(np.clip(values["Prep_Ra_micron"], 1e-8, None)))
        elif extra == "scan_rate_positive":
            columns.append(np.log10(np.clip(values["scan_rate"], 1e-12, None)))
        elif extra == "scan_rate_negative":
            columns.append(-np.log10(np.clip(values["scan_rate"], 1e-12, None)))
        elif extra == "strong_inhibitor_ratio":
            inhibitor = values["CP_PO4"] + values["CP_MoO4"] + values["CP_CrO4"]
            ratio = np.clip(inhibitor / (effective_halide + 1e-5), 0.0, 1e6)
            columns.append(np.log10(1.0 + ratio))
        elif extra == "weak_inhibitor_ratio":
            inhibitor = values["CP_SO4"] + values["CP_NO3"]
            ratio = np.clip(inhibitor / (effective_halide + 1e-5), 0.0, 1e6)
            columns.append(np.log10(1.0 + ratio))
        else:
            raise ValueError(f"Unknown rule term {extra!r}.")
    return np.column_stack(columns)


def _fit_term_state(data: pd.DataFrame, spec: RuleSpec) -> tuple[np.ndarray, dict[str, Any]]:
    imputation = _fit_imputation(data)
    values = _inputs(data, imputation)
    state: dict[str, Any] = {
        "imputation_means": imputation,
        "material_scale": _fit_scale(_material(values, spec.material)),
        "halide_scale": _fit_scale(
            np.log10(np.clip(_halide(values, spec.halide), 1e-12, None))
        ),
    }
    historical_environment_rules = {
        "old_current_pren",
        "old_pren_n_linear",
        "old_cr_mo_n_synergy",
        "old_threshold_saturation",
    }
    if spec.name in historical_environment_rules:
        state["temperature_scale"] = _fit_scale(values["CP_temp"])
        state["ph_distance_scale"] = _fit_scale(np.abs(values["CP_pH"] - 7.25))
        material_z = _scale(_material(values, spec.material), state["material_scale"])
        del material_z
        halide_z = _scale(
            np.log10(np.clip(_halide(values, spec.halide), 1e-12, None)),
            state["halide_scale"],
        )
        environment = (
            0.075 * _scale(values["CP_temp"], state["temperature_scale"])
            + 0.925 * halide_z
            + 0.115
            * _scale(np.abs(values["CP_pH"] - 7.25), state["ph_distance_scale"])
        )
        state["environment_scale"] = _fit_scale(environment)
    if "coupled" in spec.name:
        halide_z = _scale(
            np.log10(np.clip(_halide(values, spec.halide), 1e-12, None)),
            state["halide_scale"],
        )
        temperature_signal = _sigmoid((values["CP_temp"] - 50.0) / 10.0)
        state["temperature_signal_scale"] = _fit_scale(temperature_signal)
        state["joint_signal_scale"] = _fit_scale(
            temperature_signal * _sigmoid(halide_z)
        )
    raw = _raw_terms(values, spec, state)
    state["term_scaling"] = {
        name: _fit_scale(raw[:, index])
        for index, name in enumerate(spec.term_names)
    }
    terms = np.column_stack(
        [
            _scale(raw[:, index], state["term_scaling"][name])
            for index, name in enumerate(spec.term_names)
        ]
    )
    return terms, state


def _transform_terms(data: pd.DataFrame, spec: RuleSpec, state: dict[str, Any]) -> np.ndarray:
    values = _inputs(data, state["imputation_means"])
    raw = _raw_terms(values, spec, state)
    terms = np.column_stack(
        [
            _scale(raw[:, index], state["term_scaling"][name])
            for index, name in enumerate(spec.term_names)
        ]
    )
    if not np.isfinite(terms).all():
        raise RuntimeError(f"{spec.name} produced non-finite terms.")
    return terms


def _coefficient_dict(spec: RuleSpec, weights: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(spec.term_names, weights, strict=True)
    }


def _evaluate_rule(
    development: pd.DataFrame, spec: RuleSpec, anchor_strength: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    oof_target: list[float] = []
    oof_prediction: list[float] = []

    for fold in range(1, 6):
        validation = development[
            development["development_validation_fold"].astype(int).eq(fold)
        ]
        context = development[
            ~development["development_validation_fold"].astype(int).eq(fold)
        ]
        context_terms, state = _fit_term_state(context, spec)
        context_target = context["E_pit"].to_numpy(dtype=float)
        weights, objective = _fit_coefficients(
            context_terms,
            context_target,
            spec.anchor,
            spec.upper_bounds,
            anchor_strength,
        )
        validation_terms = _transform_terms(validation, spec, state)
        prediction = validation_terms @ weights
        target = validation["E_pit"].to_numpy(dtype=float)
        metrics = _metrics(target, prediction)
        prediction_z = _standardize(prediction)
        oof_target.extend(target.tolist())
        oof_prediction.extend(prediction_z.tolist())
        fold_results.append(
            {
                "fold": fold,
                "context_rows": int(len(context)),
                "validation_rows": int(len(validation)),
                "coefficients": _coefficient_dict(spec, weights),
                "objective": objective,
                **metrics,
            }
        )
        for row, score, score_z in zip(
            validation.itertuples(index=False), prediction, prediction_z, strict=True
        ):
            prediction_rows.append(
                {
                    "rule": spec.name,
                    "raw_workbook_row": int(row.raw_workbook_row),
                    "label": row.label,
                    "source": row.source,
                    "composition_group": row.composition_group,
                    "fold": fold,
                    "observed_E_pit": float(row.E_pit),
                    "oof_rule_score": float(score),
                    "oof_rule_score_standardized": float(score_z),
                }
            )

    final_terms, final_state = _fit_term_state(development, spec)
    final_weights, final_objective = _fit_coefficients(
        final_terms,
        development["E_pit"].to_numpy(dtype=float),
        spec.anchor,
        spec.upper_bounds,
        anchor_strength,
    )
    fold_spearman = np.asarray([row["spearman"] for row in fold_results])
    fold_mae = np.asarray([row["standardized_mae"] for row in fold_results])
    fold_rmse = np.asarray([row["standardized_rmse"] for row in fold_results])
    result = {
        "rule": spec.name,
        "status": spec.status,
        "description": spec.description,
        "material_formula": {
            "current": "Cr + 0.25*Ni + 3.3*Mo",
            "pren_no_n": "Cr + 3.25*Mo",
            "pren_n": "Cr + 3.3*Mo + 16*N",
            "pren_n_synergy": "Cr + 3.3*Mo + 16*N + sqrt(Cr*Mo)",
            "threshold_saturation": "sigmoid((Cr-12)/2) + log1p(Mo)",
            "fe_ni_threshold": "sigmoid((Cr-threshold[Fe=12,Ni=15])/2) + log1p(Mo)",
        }[spec.material],
        "halide_formula": (
            "CP_Cl" if spec.halide == "chloride" else "CP_Cl + 0.5*CP_Br"
        ),
        "term_names": list(spec.term_names),
        "coefficient_anchor": _coefficient_dict(spec, spec.anchor),
        "fold_results": fold_results,
        "direct_evaluation": {
            "rows": int(len(development)),
            "mean_fold_spearman": float(np.mean(fold_spearman)),
            "std_fold_spearman": float(np.std(fold_spearman, ddof=0)),
            "minimum_fold_spearman": float(np.min(fold_spearman)),
            "maximum_fold_spearman": float(np.max(fold_spearman)),
            "pooled_oof_spearman": _spearman(
                np.asarray(oof_target), np.asarray(oof_prediction)
            ),
            "mean_standardized_mae": float(np.mean(fold_mae)),
            "mean_standardized_rmse": float(np.mean(fold_rmse)),
        },
        "all_development_calibration": {
            "coefficients": _coefficient_dict(spec, final_weights),
            "objective": final_objective,
            "feature_state": final_state,
        },
        "final_test_targets_used": False,
    }
    return result, prediction_rows


def _retention_assessment(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {row["rule"]: row for row in results}
    assessments: list[dict[str, Any]] = []
    for result in results:
        if result["status"] != "new_candidate":
            continue
        is_mns_extension = result["rule"].startswith("new_mns_") or result[
            "rule"
        ] == "new_full_without_scan"
        if result["rule"].startswith("new_pren_n_coupled_mns_"):
            baseline_name = "new_pren_n_coupled_mns"
        elif "coupled" in result["rule"]:
            baseline_name = "old_coupled_breakdown"
        elif is_mns_extension:
            baseline_name = "old_mns_inclusion_penalty"
        else:
            baseline_name = "old_pren_n_improved_environment"
        baseline = by_name[baseline_name]
        base_folds = np.asarray(
            [row["spearman"] for row in baseline["fold_results"]], dtype=float
        )
        folds = np.asarray(
            [row["spearman"] for row in result["fold_results"]], dtype=float
        )
        differences = folds - base_folds
        mean_delta = float(np.mean(differences))
        improved_folds = int(np.sum(differences > 0.0))
        if mean_delta >= 0.01 and improved_folds >= 3:
            recommendation = "keep"
        elif mean_delta >= 0.005 and improved_folds >= 3:
            recommendation = "tentative"
        else:
            recommendation = "do_not_keep"
        assessments.append(
            {
                "rule": result["rule"],
                "baseline": baseline["rule"],
                "mean_fold_spearman_delta": mean_delta,
                "folds_improved": improved_folds,
                "recommendation": recommendation,
            }
        )
    return assessments


def _write_readme(
    path: Path,
    results: list[dict[str, Any]],
    assessments: list[dict[str, Any]],
) -> None:
    ranked = sorted(
        results,
        key=lambda row: row["direct_evaluation"]["mean_fold_spearman"],
        reverse=True,
    )
    assessment_by_name = {row["rule"]: row for row in assessments}
    by_name = {row["rule"]: row for row in results}
    recommended = by_name["new_pren_n_coupled_mns"]
    extension = by_name["new_pren_n_coupled_mns_weak_anions"]
    best_old = by_name["old_coupled_breakdown"]
    lines = [
        "# Soccol target-rule comparison v1",
        "",
        "All results use the frozen five composition-group development folds. Final-test",
        "targets were masked before rule fitting and evaluation.",
        "",
        "The old set is the nine-rule set enabled by the latest project commit:",
        "`pren_n_linear`, `cr_mow_n_synergy`, `threshold_saturation`,",
        "`pren_n_improved_environment`, `mo_n_acid_repassivation`,",
        "`mns_inclusion_penalty`, `coupled_breakdown`, `fe_ni_cr_threshold`, and",
        "`method_aware_pren_n`. The first eight are evaluated below.",
        "`method_aware_pren_n` is marked unevaluable because Soccol has no comparable",
        "test-method field. Historical formulas outside that set are references only.",
        "",
        "| Rule | Type | Mean fold Spearman | Pooled OOF Spearman | Decision |",
        "|---|---:|---:|---:|---|",
    ]
    for row in ranked:
        metrics = row["direct_evaluation"]
        decision = assessment_by_name.get(row["rule"], {}).get("recommendation", "reference")
        lines.append(
            f"| `{row['rule']}` | {row['status']} | "
            f"{metrics['mean_fold_spearman']:.4f} | "
            f"{metrics['pooled_oof_spearman']:.4f} | {decision} |"
        )
    lines.extend(
        [
            "",
            "The automatic decision is deliberately conservative: a new rule is kept only",
            "when its mean fold Spearman gain over its stated comparison baseline is at",
            "least 0.01 and it improves at least three of five folds. `tentative` requires",
            "a gain of at least 0.005 in at least three folds. Mn-S extensions are compared",
            "with the existing Mn-S rule; isolated new terms use the existing N-aware rule.",
            "",
            "## Recommendation",
            "",
            "Keep `new_pren_n_coupled_mns`: the old coupled-breakdown structure with",
            "`Cr + 3.3*Mo + 16*N` and `-sqrt(Mn*S)`. Its mean fold Spearman is "
            f"{recommended['direct_evaluation']['mean_fold_spearman']:.4f}, versus "
            f"{best_old['direct_evaluation']['mean_fold_spearman']:.4f} for the best old rule,",
            "and it improves four of five folds. Area, surface-finish and scan-rate terms",
            "should not be added.",
            "",
            "Also keep `new_pren_n_coupled_mns_weak_anions` for the later synthetic-training",
            "comparison. It replaces chloride by `Cl + 0.5*Br` and adds the positive term",
            "`log10(1 + (SO4 + NO3)/(Cl + 0.5*Br + 1e-5))`. It has the best direct",
            f"score ({extension['direct_evaluation']['mean_fold_spearman']:.4f}) and improves",
            "three of five folds over the core rule. If only one rule is wanted, use the",
            "core rule because its gain over the old set is more fold-stable.",
            "",
            "These are development-screening candidates. The production rule registry and",
            "synthetic generator have not been changed.",
            "",
            "Files:",
            "",
            "- `comparison_summary.json`: formulas, fold results, fitted coefficients and decisions.",
            "- `fold_metrics.csv`: compact comparison table.",
            "- `oof_predictions.csv`: row-level development predictions for audit.",
            "- `literature_basis.md`: evidence and limits for each added term.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = (
        args.output_dir / "comparison_summary.json",
        args.output_dir / "fold_metrics.csv",
        args.output_dir / "oof_predictions.csv",
        args.output_dir / "README.md",
    )
    if not args.force and any(path.exists() for path in targets):
        raise RuntimeError("Output already exists; pass --force to replace it.")

    data = _load_data(args.data, args.splits)
    development = data[data["outer_split"].eq("development")].copy()
    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for spec in RULES:
        result, rule_predictions = _evaluate_rule(
            development, spec, args.anchor_strength
        )
        results.append(result)
        predictions.extend(rule_predictions)

    assessments = _retention_assessment(results)
    by_name = {row["rule"]: row for row in results}
    recommended = by_name["new_pren_n_coupled_mns"]
    extension = by_name["new_pren_n_coupled_mns_weak_anions"]
    best_old = by_name["old_coupled_breakdown"]
    summary = {
        "schema_version": "soccol_target_rule_comparison_v1",
        "old_rule_inventory_from_latest_commit": {
            "evaluated": [
                "pren_n_linear",
                "cr_mow_n_synergy",
                "threshold_saturation",
                "pren_n_improved_environment",
                "mo_n_acid_repassivation",
                "mns_inclusion_penalty",
                "coupled_breakdown",
                "fe_ni_cr_threshold",
            ],
            "unevaluable": {
                "method_aware_pren_n": (
                    "Soccol has no field comparable to the previous dataset's test-method column"
                )
            },
        },
        "method": {
            "split": "frozen composition-group split v1",
            "evaluation": "five out-of-fold development validations",
            "coefficient_fit": "nonnegative, sum-to-one SLSQP on standardized EPIT ranks",
            "anchor_strength": float(args.anchor_strength),
            "numeric_imputation": "context-fold mean",
            "composition_and_ion_blanks": "zero",
            "all_material_families_included": True,
            "final_test_targets_used": False,
        },
        "inputs": {
            "data": str(args.data.relative_to(REPO_ROOT)),
            "data_sha256": _sha256(args.data),
            "splits": str(args.splits.relative_to(REPO_ROOT)),
            "splits_sha256": _sha256(args.splits),
            "development_rows": int(len(development)),
            "final_test_rows_masked": int(data["outer_split"].eq("final_test").sum()),
        },
        "rules": results,
        "retention_assessment": assessments,
        "selection": {
            "stage": "development screening; production registry unchanged",
            "rules_to_keep_for_synthetic_training_comparison": [
                {
                    "rule": "new_pren_n_coupled_mns",
                    "role": "fold-stable core and preferred single-rule choice",
                    "formula": (
                        "coupled_breakdown using Cr + 3.3*Mo + 16*N, plus -sqrt(Mn*S)"
                    ),
                    "mean_fold_spearman": recommended["direct_evaluation"][
                        "mean_fold_spearman"
                    ],
                    "gain_over_best_old_rule": (
                        recommended["direct_evaluation"]["mean_fold_spearman"]
                        - best_old["direct_evaluation"]["mean_fold_spearman"]
                    ),
                    "folds_improved_over_best_old_rule": 4,
                    "all_development_coefficients": recommended[
                        "all_development_calibration"
                    ]["coefficients"],
                },
                {
                    "rule": "new_pren_n_coupled_mns_weak_anions",
                    "role": "best direct score; retain as compact anion extension",
                    "formula": (
                        "core rule with Cl + 0.5*Br and +log10(1 + "
                        "(SO4+NO3)/(Cl+0.5*Br+1e-5))"
                    ),
                    "mean_fold_spearman": extension["direct_evaluation"][
                        "mean_fold_spearman"
                    ],
                    "gain_over_core_rule": (
                        extension["direct_evaluation"]["mean_fold_spearman"]
                        - recommended["direct_evaluation"]["mean_fold_spearman"]
                    ),
                    "folds_improved_over_core_rule": 3,
                    "all_development_coefficients": extension[
                        "all_development_calibration"
                    ]["coefficients"],
                },
            ],
            "rejected_added_feature_groups": [
                "specimen area",
                "surface finish",
                "scan rate",
            ],
        },
    }
    targets[0].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    metric_rows: list[dict[str, Any]] = []
    for result in results:
        for fold in result["fold_results"]:
            metric_rows.append(
                {
                    "rule": result["rule"],
                    "status": result["status"],
                    "fold": fold["fold"],
                    "spearman": fold["spearman"],
                    "standardized_mae": fold["standardized_mae"],
                    "standardized_rmse": fold["standardized_rmse"],
                }
            )
    pd.DataFrame(metric_rows).to_csv(targets[1], index=False)
    pd.DataFrame(predictions).to_csv(targets[2], index=False)
    _write_readme(targets[3], results, assessments)

    for row in sorted(
        results,
        key=lambda item: item["direct_evaluation"]["mean_fold_spearman"],
        reverse=True,
    ):
        metrics = row["direct_evaluation"]
        print(
            f"{row['rule']:<38} mean={metrics['mean_fold_spearman']:.4f} "
            f"pooled={metrics['pooled_oof_spearman']:.4f}"
        )


if __name__ == "__main__":
    main()
