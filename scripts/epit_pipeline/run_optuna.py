#!/usr/bin/env python
"""Stage 3: run the original Magpie Optuna workflow on fixed EPIT folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import optuna_pitting_magpie_prior_search as original
from scripts.epit_pipeline.artifact_hashes import load_frozen_split, sha256_file
from tabicl.prior.dataset import (
    EPIT_FE_NI_FAMILY_NAMES,
    EPIT_FE_NI_FAMILY_PROBS,
    EPIT_TARGET_RULE_COEFFICIENTS,
)
from tabicl.prior.epit_composition_profile import EPIT_COMPOSITION_FAMILY_PROBS
from tabicl.prior.epit_feature_profile import (
    EPIT_FEATURE_PROFILE,
    load_epit_feature_profile,
)
from tabicl.prior.magpie_features import (
    EPIT_BASE_FEATURE_COUNT,
    EPIT_MAGPIE_DESCRIPTOR_NAMES,
    EPIT_MAGPIE_RANGE_MIN_ATOMIC_FRACTION,
    EPIT_MAGPIE_TOTAL_FEATURE_COUNT,
    EPIT_MAGPIE_VERSION,
)


PIPELINE_ROOT = (
    REPO_ROOT / "corrosion_datasets" / "analysis" / "epit_pipeline"
)
DEFAULT_SPLIT_MANIFEST = PIPELINE_ROOT / "splits_v2" / "split_manifest.json"
DEFAULT_TARGET_RULE_SUMMARY = (
    PIPELINE_ROOT / "target_rules_v2" / "calibration_summary.json"
)
DEFAULT_OPTUNA_ROOT = PIPELINE_ROOT / "optuna_v2"
FIXED_VALIDATION_FOLDS = (1, 2, 3, 4, 5)
PIPELINE_FINGERPRINT_SCHEMA = "epit_pipeline_stage3_fingerprint_v2"
STUDY_FINGERPRINT_ATTR = "epit_pipeline_fingerprint"
STUDY_FINGERPRINT_SHA256_ATTR = "epit_pipeline_fingerprint_sha256"
PITTING_COMPOSITION_MODES = (
    "legacy",
    "fe_ni_softmax",
    "empirical_features",
    "empirical_features_scm_target",
)
EMPIRICAL_FEATURE_MODES = {
    "empirical_features",
    "empirical_features_scm_target",
}
EPIT_COMPOSITION_PERTURB_STRENGTH_RANGE = (0.0, 0.15)
GENERIC_SCM_MIX_PROBS = (0.7, 0.3)


@dataclass(frozen=True)
class TrialParams:
    use_magpie: bool
    informed_prior_ratio: float
    mlp_prob: float
    informed_feature_block_strength: float
    informed_target_mix_weight: float
    pitting_material_dirichlet_prob: float
    pitting_material_dirichlet_concentration: float | None
    pitting_material_dirichlet_active_prob: float | None
    pitting_composition_perturb_strength: float | None = None


@dataclass(frozen=True)
class TargetRuleConfig:
    scores: dict[str, float]
    probabilities: dict[str, float]
    coefficients: dict[str, dict[str, float]]
    artifacts: dict[str, str]
    artifact_sha256s: dict[str, str]
    summary_path: str
    summary_sha256: str
    split_manifest_sha256: str
    split_lock_path: str
    split_lock_sha256: str
    source_sha256: str

    def score_cli_values(self) -> list[str]:
        return [
            f"{name}={format_float(score)}"
            for name, score in self.scores.items()
        ]

    def coefficient_cli_values(self) -> list[str]:
        return [
            f"{family}.{term}={format_float(value)}"
            for family, values in self.coefficients.items()
            for term, value in values.items()
        ]

    def exact_score_cli_values(self) -> list[str]:
        return [
            f"{name}={repr(float(score))}"
            for name, score in self.scores.items()
        ]

    def exact_coefficient_cli_values(self) -> list[str]:
        return [
            f"{family}.{term}={repr(float(value))}"
            for family, values in self.coefficients.items()
            for term, value in values.items()
        ]


class SuggestTrial(Protocol):
    def suggest_categorical(self, name: str, choices: list[Any]) -> Any: ...

    def suggest_float(self, name: str, low: float, high: float) -> float: ...


class RandomTrial:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.params: dict[str, Any] = {}

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        value = self.rng.choice(choices)
        self.params[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float) -> float:
        value = float(self.rng.uniform(low, high))
        self.params[name] = value
        return value


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pipeline_artifact_identity(rules: TargetRuleConfig) -> dict[str, Any]:
    """Return the frozen data/rule identity shared by Stages 3 and 4."""
    return {
        "split_manifest_sha256": rules.split_manifest_sha256,
        "split_lock_sha256": rules.split_lock_sha256,
        "source_sha256": rules.source_sha256,
        "target_rule_summary_sha256": rules.summary_sha256,
        "target_rule_artifact_sha256s": rules.artifact_sha256s,
        "target_rule_scores": rules.scores,
        "target_rule_probabilities": rules.probabilities,
        "target_rule_coefficients": rules.coefficients,
    }


def empirical_feature_profile_identity() -> dict[str, Any]:
    """Return the current immutable identity of the v5 physical-feature prior."""
    feature_profile = load_epit_feature_profile(EPIT_FEATURE_PROFILE)
    feature_family_counts = [
        int((feature_profile.composition_family_indices == index).sum())
        for index in range(len(feature_profile.families))
    ]
    feature_family_total = sum(feature_family_counts)
    feature_family_probabilities = [
        count / feature_family_total for count in feature_family_counts
    ]
    return {
        "feature_profile": EPIT_FEATURE_PROFILE,
        "feature_profile_metadata_sha256": canonical_json_sha256(
            feature_profile.metadata
        ),
        "feature_profile_csv_sha256": feature_profile.metadata["csv_sha256"],
        "composition_profile": feature_profile.composition_profile.name,
        "composition_family_probabilities": list(EPIT_COMPOSITION_FAMILY_PROBS),
        "synthetic_material_families": list(feature_profile.families),
        "synthetic_material_family_probabilities": feature_family_probabilities,
    }


def build_pipeline_fingerprint(
    args: argparse.Namespace,
    rules: TargetRuleConfig,
) -> dict[str, Any]:
    """Bind one Optuna study to one immutable EPIT Stage 3 definition."""
    composition_mode = str(args.pitting_composition_mode)
    search_space: dict[str, Any] = {
        "informed_prior_ratio": list(original.INFORMED_PRIOR_RATIO_GRID),
        "mlp_prob": list(original.MLP_PROB_GRID),
    }
    fixed_prior: dict[str, Any] = {
        "reference_workflow": "pitting_magpie_full_v1",
        "prior_type": "hybrid_scm",
        "block_allocation": list(original.FIXED_BLOCK_ALLOCATION),
        "material_style": "composition_like",
        "composition_mode": composition_mode,
        "physical_marginal_probability": 1.0,
        "feature_permutation": False,
    }
    if composition_mode in EMPIRICAL_FEATURE_MODES:
        feature_profile_identity = empirical_feature_profile_identity()
        search_space["pitting_composition_perturb_strength"] = list(
            EPIT_COMPOSITION_PERTURB_STRENGTH_RANGE
        )
        fixed_prior.update(
            {
                "reference_workflow": (
                    "epit_empirical_features_scm_target_v2"
                    if composition_mode == "empirical_features_scm_target"
                    else "epit_empirical_features_v1"
                ),
                "material_style": "empirical_feature_profile",
                **feature_profile_identity,
                "use_magpie": False,
                "informed_feature_block_strength": 0.0,
                "informed_target_policy": (
                    "standardized_scm_plus_calibrated_epit_v1"
                    if composition_mode == "empirical_features_scm_target"
                    else "epit_only"
                ),
                "pitting_material_dirichlet_prob": 0.0,
            }
        )
        if composition_mode == "empirical_features_scm_target":
            search_space["informed_target_mix_weight"] = [0.0, 1.0]
            fixed_prior.update(
                {
                    "scm_target_input_standardization": (
                        "per_dataset_column_population_zscore"
                    ),
                    "scm_target_constant_column_policy": "zero",
                    "scm_target_topology": "direct_noncausal",
                    "epit_target_input": "raw_physical_features",
                    "final_target_standardization": "population_zscore",
                    "transformer_input_standardization": (
                        "per_dataset_column_population_zscore"
                    ),
                    "informed_reg2cls_policy": "bypass",
                    "generic_scm_mix_probs": list(GENERIC_SCM_MIX_PROBS),
                }
            )
        else:
            fixed_prior["informed_target_mix_weight"] = 1.0
    else:
        search_space["use_magpie"] = [False, True]
        search_space["informed_feature_block_strength"] = [0.0, 0.95]
    if composition_mode == "legacy":
        search_space.update(
            {
                "informed_target_mix_weight": [0.0, 1.0],
                "pitting_material_dirichlet_prob": list(original.DIRICHLET_PROB_GRID),
                "pitting_material_dirichlet_concentration": list(original.DIRICHLET_CONCENTRATION_GRID),
                "pitting_material_dirichlet_active_prob": list(original.DIRICHLET_ACTIVE_PROB_GRID),
            }
        )
    elif composition_mode == "fe_ni_softmax":
        fixed_prior.update(
            {
                "informed_target_policy": "epit_only",
                "informed_target_mix_weight": 1.0,
                "pitting_material_dirichlet_prob": 0.0,
                "synthetic_material_families": list(EPIT_FE_NI_FAMILY_NAMES),
                "synthetic_material_family_probabilities": list(EPIT_FE_NI_FAMILY_PROBS),
            }
        )
    elif composition_mode not in EMPIRICAL_FEATURE_MODES:
        raise ValueError(
            f"Unsupported pitting composition mode: {composition_mode!r}"
        )
    return {
        "schema_version": PIPELINE_FINGERPRINT_SCHEMA,
        "artifacts": pipeline_artifact_identity(rules),
        "search_space": search_space,
        "fixed_prior": fixed_prior,
        "proxy_training": {
            "device": str(args.device),
            "max_steps": int(args.max_steps),
            "scheduler_total_steps": int(args.scheduler_total_steps),
            "np_seed": int(args.np_seed),
            "torch_seed": int(args.torch_seed),
            "nproc_per_node": int(args.nproc_per_node),
            "prior_n_jobs": int(args.prior_n_jobs),
            "dataloader_num_workers": int(args.dataloader_num_workers),
            "dataloader_prefetch_factor": int(
                args.dataloader_prefetch_factor
            ),
        },
        "evaluation": {
            "device": str(args.device),
            "validation_folds": list(FIXED_VALIDATION_FOLDS),
            "n_estimators": int(args.eval_n_estimators),
            "feature_shuffle_method": "none",
            "norm_methods": ["none"],
            "regression_output": "median",
        },
        "optuna_sampler": {
            "name": "TPESampler",
            "n_startup_trials": int(args.n_startup_trials),
            "worker_seed_policy": "per_worker_recorded_per_trial",
        },
    }


def validate_pipeline_fingerprint(
    payload: Any,
    expected_sha256: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("Optuna study is missing its EPIT pipeline fingerprint.")
    if payload.get("schema_version") != PIPELINE_FINGERPRINT_SCHEMA:
        raise RuntimeError("Optuna study has an unsupported pipeline fingerprint.")
    observed_sha256 = canonical_json_sha256(payload)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "Optuna study pipeline fingerprint is internally inconsistent."
        )
    return payload


def bind_study_pipeline_fingerprint(
    study: Any,
    fingerprint: dict[str, Any],
) -> str:
    """Set a new study fingerprint or reject reuse with another pipeline."""
    fingerprint_sha256 = canonical_json_sha256(fingerprint)
    existing_payload = study.user_attrs.get(STUDY_FINGERPRINT_ATTR)
    existing_sha256 = study.user_attrs.get(STUDY_FINGERPRINT_SHA256_ATTR)
    if existing_payload is None and existing_sha256 is None:
        if study.trials:
            raise RuntimeError(
                "Existing Optuna study has trials but no EPIT pipeline fingerprint. "
                "Use a new study name instead of mixing legacy and locked trials."
            )
        study.set_user_attr(STUDY_FINGERPRINT_ATTR, fingerprint)
        study.set_user_attr(STUDY_FINGERPRINT_SHA256_ATTR, fingerprint_sha256)
        return fingerprint_sha256
    if existing_payload is None or existing_sha256 is None:
        raise RuntimeError("Optuna study has an incomplete EPIT pipeline fingerprint.")
    validate_pipeline_fingerprint(existing_payload, str(existing_sha256))
    if str(existing_sha256) != fingerprint_sha256 or existing_payload != fingerprint:
        raise RuntimeError(
            "Optuna study belongs to a different split, target-rule set, or "
            "Stage 3 configuration. Use a new study name."
        )
    return fingerprint_sha256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("optuna", "random"), default="optuna")
    parser.add_argument("--study-name", default="epit_pipeline_optuna_v3")
    parser.add_argument(
        "--pitting-composition-mode",
        choices=PITTING_COMPOSITION_MODES,
        default="legacy",
        help="Fixed informed composition mode for this versioned study.",
    )
    parser.add_argument("--storage", default=None)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-startup-trials", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--base-run-name", default=None)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
    )
    parser.add_argument(
        "--target-rule-summary",
        type=Path,
        default=DEFAULT_TARGET_RULE_SUMMARY,
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "checkpoints",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_OPTUNA_ROOT / "trials",
    )
    parser.add_argument(
        "--eval-output-root",
        type=Path,
        default=DEFAULT_OPTUNA_ROOT / "evaluations",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--scheduler-total-steps", type=int, default=10000)
    parser.add_argument("--np-seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--prior-n-jobs", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=4)
    parser.add_argument("--eval-n-estimators", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def slugify(text: str) -> str:
    return original.slugify(text)


def format_float(value: float) -> str:
    return original.format_float(value)


def default_base_run_name(study_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tabicl_epit_pipeline_{slugify(study_name)}_{stamp}"


def load_target_rule_config(
    *,
    summary_path: Path,
    split_manifest_path: Path,
) -> TargetRuleConfig:
    summary_path = summary_path.expanduser().resolve()
    split_manifest_path = split_manifest_path.expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frozen_split = load_frozen_split(split_manifest_path)
    manifest = frozen_split.manifest
    if summary.get("schema_version") != "epit_target_rule_calibration_summary_v2":
        raise RuntimeError("Unsupported target-rule calibration summary.")
    if summary.get("split_manifest_sha256") != frozen_split.manifest_sha256:
        raise RuntimeError("Target-rule calibration used a different split manifest.")
    if summary.get("split_lock_sha256") != frozen_split.lock_sha256:
        raise RuntimeError("Target-rule calibration used a different split lock.")
    source_sha256 = str(manifest.get("dataset", {}).get("source_sha256", ""))
    if str(summary.get("source_sha256", "")) != source_sha256:
        raise RuntimeError("Target-rule calibration and split manifest use different data.")
    if bool(summary.get("final_test_targets_used", True)):
        raise RuntimeError("Target-rule calibration used final-test targets.")
    if int(summary.get("development_rows", -1)) != int(
        manifest.get("split_design", {}).get("development_rows", -2)
    ):
        raise RuntimeError("Target-rule calibration development count is inconsistent.")
    if int(summary.get("final_test_rows_excluded", -1)) != int(
        manifest.get("split_design", {}).get("final_test_rows", -2)
    ):
        raise RuntimeError("Target-rule calibration final-test count is inconsistent.")

    scores: dict[str, float] = {}
    probabilities: dict[str, float] = {}
    coefficients: dict[str, dict[str, float]] = {}
    artifacts: dict[str, str] = {}
    artifact_sha256s: dict[str, str] = {}
    for record in summary.get("rules", []):
        if record.get("evaluation_role") != "candidate":
            continue
        family = str(record.get("rule_family", ""))
        if family not in EPIT_TARGET_RULE_COEFFICIENTS:
            raise RuntimeError(
                f"Calibrated rule {family!r} is not implemented by the prior."
            )
        score = float(record.get("mean_fold_spearman", math.nan))
        if not math.isfinite(score) or score <= 0.0:
            raise RuntimeError(
                f"Calibrated rule {family!r} has a non-positive score."
            )
        artifact_path = summary_path.parent / str(record.get("artifact", ""))
        artifact_sha256 = sha256_file(artifact_path)
        if artifact_sha256 != record.get("artifact_sha256"):
            raise RuntimeError(f"Target-rule artifact changed for {family!r}.")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact.get("schema_version") != "epit_target_rule_v2":
            raise RuntimeError(f"Unsupported target-rule artifact for {family!r}.")
        if artifact.get("rule_family") != family:
            raise RuntimeError(f"Target-rule artifact mismatch for {family!r}.")
        if artifact.get("split_manifest_sha256") != frozen_split.manifest_sha256:
            raise RuntimeError(f"Target-rule artifact used a different split for {family!r}.")
        if artifact.get("split_lock_sha256") != frozen_split.lock_sha256:
            raise RuntimeError(f"Target-rule artifact used a different split lock for {family!r}.")
        if artifact.get("source_sha256") != source_sha256:
            raise RuntimeError(f"Target-rule artifact used different data for {family!r}.")
        values = {
            str(term): float(value)
            for term, value in artifact["final_development_calibration"][
                "coefficients"
            ].items()
        }
        expected_terms = set(EPIT_TARGET_RULE_COEFFICIENTS[family])
        if set(values) != expected_terms:
            raise RuntimeError(
                f"Calibrated coefficients for {family!r} do not match the prior."
            )
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise RuntimeError(
                f"Calibrated coefficients for {family!r} are invalid."
            )
        if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-8):
            raise RuntimeError(
                f"Calibrated coefficients for {family!r} do not sum to one."
            )
        scores[family] = score
        coefficients[family] = values
        artifacts[family] = str(artifact_path.resolve())
        artifact_sha256s[family] = artifact_sha256

    if not scores:
        raise RuntimeError("No candidate target rules were found.")
    score_sum = sum(scores.values())
    probabilities = {
        family: score / score_sum for family, score in scores.items()
    }
    return TargetRuleConfig(
        scores=scores,
        probabilities=probabilities,
        coefficients=coefficients,
        artifacts=artifacts,
        artifact_sha256s=artifact_sha256s,
        summary_path=str(summary_path),
        summary_sha256=sha256_file(summary_path),
        split_manifest_sha256=frozen_split.manifest_sha256,
        split_lock_path=str(frozen_split.lock_path),
        split_lock_sha256=frozen_split.lock_sha256,
        source_sha256=source_sha256,
    )


def sample_params(
    trial: SuggestTrial, *, composition_mode: str = "legacy"
) -> TrialParams:
    if composition_mode in EMPIRICAL_FEATURE_MODES:
        use_magpie = False
    else:
        use_magpie = bool(
            trial.suggest_categorical("use_magpie", [False, True])
        )
    informed_prior_ratio = float(
        trial.suggest_categorical(
            "informed_prior_ratio",
            list(original.INFORMED_PRIOR_RATIO_GRID),
        )
    )
    mlp_prob = float(
        trial.suggest_categorical("mlp_prob", list(original.MLP_PROB_GRID))
    )
    if composition_mode in EMPIRICAL_FEATURE_MODES:
        block_strength = 0.0
    else:
        block_strength = float(
            trial.suggest_float("informed_feature_block_strength", 0.0, 0.95)
        )
    perturb_strength = None
    if composition_mode in {"fe_ni_softmax", "empirical_features"}:
        target_mix = 1.0
        dirichlet_prob = 0.0
        concentration = None
        active_prob = None
        if composition_mode in EMPIRICAL_FEATURE_MODES:
            perturb_strength = float(
                trial.suggest_float(
                    "pitting_composition_perturb_strength",
                    *EPIT_COMPOSITION_PERTURB_STRENGTH_RANGE,
                )
            )
    elif composition_mode == "empirical_features_scm_target":
        target_mix = float(
            trial.suggest_float("informed_target_mix_weight", 0.0, 1.0)
        )
        dirichlet_prob = 0.0
        concentration = None
        active_prob = None
        perturb_strength = float(
            trial.suggest_float(
                "pitting_composition_perturb_strength",
                *EPIT_COMPOSITION_PERTURB_STRENGTH_RANGE,
            )
        )
    elif composition_mode == "legacy":
        target_mix = float(
            trial.suggest_float("informed_target_mix_weight", 0.0, 1.0)
        )
        dirichlet_prob = float(
            trial.suggest_categorical(
                "pitting_material_dirichlet_prob",
                list(original.DIRICHLET_PROB_GRID),
            )
        )
        if dirichlet_prob > 0.0:
            concentration = float(
                trial.suggest_categorical(
                    "pitting_material_dirichlet_concentration",
                    list(original.DIRICHLET_CONCENTRATION_GRID),
                )
            )
            active_prob = float(
                trial.suggest_categorical(
                    "pitting_material_dirichlet_active_prob",
                    list(original.DIRICHLET_ACTIVE_PROB_GRID),
                )
            )
        else:
            concentration = None
            active_prob = None
    else:
        raise ValueError(f"Unsupported pitting composition mode: {composition_mode!r}")
    return TrialParams(
        use_magpie=use_magpie,
        informed_prior_ratio=informed_prior_ratio,
        mlp_prob=mlp_prob,
        informed_feature_block_strength=block_strength,
        informed_target_mix_weight=target_mix,
        pitting_material_dirichlet_prob=dirichlet_prob,
        pitting_material_dirichlet_concentration=concentration,
        pitting_material_dirichlet_active_prob=active_prob,
        pitting_composition_perturb_strength=perturb_strength,
    )


def trial_params_from_mapping(
    values: dict[str, Any], *, composition_mode: str = "legacy"
) -> TrialParams:
    """Reconstruct the conditional search configuration from saved values."""
    required = {
        "informed_prior_ratio",
        "mlp_prob",
    }
    if composition_mode in EMPIRICAL_FEATURE_MODES:
        required.add("pitting_composition_perturb_strength")
    else:
        required.update({"use_magpie", "informed_feature_block_strength"})
    if composition_mode == "legacy":
        required.update(
            {
                "informed_target_mix_weight",
                "pitting_material_dirichlet_prob",
            }
        )
    elif composition_mode == "empirical_features_scm_target":
        required.add("informed_target_mix_weight")
    elif composition_mode not in {"fe_ni_softmax", *EMPIRICAL_FEATURE_MODES}:
        raise ValueError(
            f"Unsupported pitting composition mode: {composition_mode!r}"
        )
    missing = sorted(required.difference(values))
    if missing:
        raise RuntimeError(f"Optuna trial is missing parameters: {missing}")

    if composition_mode in EMPIRICAL_FEATURE_MODES:
        use_magpie = False
        block_strength = 0.0
        perturb_strength = float(
            values["pitting_composition_perturb_strength"]
        )
    else:
        use_magpie = bool(values["use_magpie"])
        block_strength = float(values["informed_feature_block_strength"])
        perturb_strength = None

    if composition_mode in {"fe_ni_softmax", "empirical_features"}:
        target_mix = 1.0
        dirichlet_prob = 0.0
        concentration = None
        active_prob = None
    elif composition_mode == "empirical_features_scm_target":
        target_mix = float(values["informed_target_mix_weight"])
        dirichlet_prob = 0.0
        concentration = None
        active_prob = None
    else:
        target_mix = float(values["informed_target_mix_weight"])
        dirichlet_prob = float(values["pitting_material_dirichlet_prob"])
        if dirichlet_prob > 0.0:
            for name in (
                "pitting_material_dirichlet_concentration",
                "pitting_material_dirichlet_active_prob",
            ):
                if name not in values:
                    raise RuntimeError(
                        f"Enabled Dirichlet Optuna trial is missing {name!r}."
                    )
            concentration = float(
                values["pitting_material_dirichlet_concentration"]
            )
            active_prob = float(values["pitting_material_dirichlet_active_prob"])
        else:
            concentration = None
            active_prob = None
    return TrialParams(
        use_magpie=use_magpie,
        informed_prior_ratio=float(values["informed_prior_ratio"]),
        mlp_prob=float(values["mlp_prob"]),
        informed_feature_block_strength=block_strength,
        informed_target_mix_weight=target_mix,
        pitting_material_dirichlet_prob=dirichlet_prob,
        pitting_material_dirichlet_concentration=concentration,
        pitting_material_dirichlet_active_prob=active_prob,
        pitting_composition_perturb_strength=perturb_strength,
    )


def trial_params_payload(params: TrialParams) -> dict[str, Any]:
    values = asdict(params)
    if values["pitting_composition_perturb_strength"] is None:
        values.pop("pitting_composition_perturb_strength")
    return values


def _original_params(params: TrialParams) -> original.TrialParams:
    values = trial_params_payload(params)
    values.pop("pitting_composition_perturb_strength", None)
    return original.TrialParams(
        **values,
        epit_material_coef=0.575,
        epit_environment_coef=0.50,
        epit_interaction_coef=0.775,
    )


def _remove_option(command: list[str], option: str, *, value_count: int = 1) -> None:
    index = command.index(option)
    del command[index : index + 1 + value_count]


def _replace_option(command: list[str], option: str, value: str) -> None:
    index = command.index(option)
    command[index + 1] = value


def training_command(
    args: argparse.Namespace,
    params: TrialParams,
    checkpoint_dir: Path,
    rules: TargetRuleConfig,
) -> list[str]:
    command = original.training_command(
        args,
        _original_params(params),
        checkpoint_dir,
    )
    composition_mode = str(
        getattr(args, "pitting_composition_mode", "legacy")
    )
    if composition_mode not in PITTING_COMPOSITION_MODES:
        raise ValueError(
            f"Unsupported pitting composition mode: {composition_mode!r}"
        )
    fixed_target_mode = composition_mode in {
        "fe_ni_softmax",
        "empirical_features",
    }
    if fixed_target_mode and (
        params.informed_target_mix_weight != 1.0
        or params.pitting_material_dirichlet_prob != 0.0
    ):
        raise ValueError(
            f"{composition_mode} requires a fixed target policy and no Dirichlet mixture."
        )
    if composition_mode == "empirical_features_scm_target" and (
        not math.isfinite(params.informed_target_mix_weight)
        or not 0.0 <= params.informed_target_mix_weight <= 1.0
        or params.pitting_material_dirichlet_prob != 0.0
    ):
        raise ValueError(
            "empirical_features_scm_target requires a target mix weight in "
            "[0, 1] and no Dirichlet mixture."
        )
    if composition_mode in EMPIRICAL_FEATURE_MODES:
        low, high = EPIT_COMPOSITION_PERTURB_STRENGTH_RANGE
        perturb_strength = params.pitting_composition_perturb_strength
        valid_perturb_strength = (
            perturb_strength is not None
            and math.isfinite(perturb_strength)
            and (
                perturb_strength >= 0.0
                if composition_mode == "empirical_features_scm_target"
                else low <= perturb_strength <= high
            )
        )
        valid_magpie_setting = (
            not params.use_magpie
            or composition_mode == "empirical_features_scm_target"
        )
        if (
            not valid_magpie_setting
            or params.informed_feature_block_strength != 0.0
            or not valid_perturb_strength
        ):
            raise ValueError(
                f"{composition_mode} requires a supported Magpie setting, "
                "zero block coupling, and a valid composition perturbation "
                "strength."
            )
    _replace_option(command, "--pitting_composition_mode", composition_mode)
    if composition_mode == "empirical_features_scm_target":
        mix_index = command.index("--mix_probs")
        command[mix_index + 1 : mix_index + 3] = [
            format_float(value) for value in GENERIC_SCM_MIX_PROBS
        ]
    for option in (
        "--epit_material_coef",
        "--epit_environment_coef",
        "--epit_interaction_coef",
    ):
        _remove_option(command, option)
    if composition_mode in EMPIRICAL_FEATURE_MODES:
        _remove_option(command, "--pitting_material_style_probs", value_count=5)
        if params.use_magpie:
            command.extend(
                ["--pitting_material_style_probs", "1", "0", "0", "0", "0"]
            )
        command.extend(
            [
                "--pitting_feature_profile",
                EPIT_FEATURE_PROFILE,
                "--pitting_composition_family_probs",
                *(format_float(value) for value in EPIT_COMPOSITION_FAMILY_PROBS),
                "--pitting_composition_perturb_strength",
                format_float(params.pitting_composition_perturb_strength),
            ]
        )
    if composition_mode == "empirical_features_scm_target":
        score_cli_values = rules.exact_score_cli_values()
        coefficient_cli_values = rules.exact_coefficient_cli_values()
    else:
        score_cli_values = rules.score_cli_values()
        coefficient_cli_values = rules.coefficient_cli_values()
    command.extend(
        [
            "--pitting_target_rule_scores",
            *score_cli_values,
            "--pitting_target_rule_coefficients",
            *coefficient_cli_values,
        ]
    )
    return command


def eval_command(
    args: argparse.Namespace,
    params: TrialParams,
    checkpoint_path: Path,
    model_label: str,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "epit_pipeline" / "evaluate_optuna_folds.py"),
        "--local-ckpt-path",
        str(checkpoint_path),
        "--model-label",
        model_label,
        "--split-manifest",
        str(args.split_manifest),
        "--validation-folds",
        *(str(fold) for fold in FIXED_VALIDATION_FOLDS),
        "--device",
        args.device,
        "--n-estimators",
        str(args.eval_n_estimators),
        "--tabicl-feat-shuffle-method",
        "none",
        "--tabicl-norm-methods",
        "none",
        "--output-dir",
        str(output_dir),
    ]
    if params.use_magpie:
        command.append("--pitting-magpie-features")
    return command


def study_root(root: Path, study_name: str) -> Path:
    return root.expanduser().resolve() / slugify(study_name)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify_no_power_fold_evaluation(eval_dir: Path) -> dict[str, Any]:
    summary_path = eval_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Fold-evaluation provenance not found: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    norm_methods = payload.get("settings", {}).get("tabicl_norm_methods")
    if norm_methods != ["none"]:
        raise RuntimeError(
            "Optuna fold evaluation did not explicitly disable the power transform."
        )
    return payload


def verify_trial_fold_evaluation(
    payload: dict[str, Any],
    *,
    rules: TargetRuleConfig,
    checkpoint_path: Path,
    params: TrialParams,
) -> None:
    """Reject stale evaluation output before it can enter Optuna storage."""
    if payload.get("split_manifest_sha256") != rules.split_manifest_sha256:
        raise RuntimeError("Trial evaluation used a different split manifest.")
    if payload.get("split_lock_sha256") != rules.split_lock_sha256:
        raise RuntimeError("Trial evaluation used a different split lock.")
    if payload.get("source_sha256") != rules.source_sha256:
        raise RuntimeError("Trial evaluation used a different source dataset.")
    source = payload.get("source", {})
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if Path(str(source.get("local_ckpt_path", ""))).resolve() != checkpoint_path:
        raise RuntimeError("Trial evaluation names a different checkpoint.")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if source.get("local_ckpt_sha256") != checkpoint_sha256:
        raise RuntimeError("Trial evaluation checkpoint hash does not match.")
    settings = payload.get("settings", {})
    if "pitting_magpie_features" not in settings or bool(
        settings["pitting_magpie_features"]
    ) != bool(params.use_magpie):
        raise RuntimeError("Trial evaluation changed the Magpie feature setting.")
    if settings.get("tabicl_feat_shuffle_method") != "none":
        raise RuntimeError("Trial evaluation changed the fixed feature order.")


def run_trial(
    args: argparse.Namespace,
    trial_number: int,
    params: TrialParams,
) -> dict[str, Any]:
    rules: TargetRuleConfig = args.target_rules
    base_run_name = args.base_run_name or default_base_run_name(args.study_name)
    trial_name = f"{base_run_name}_trial_{trial_number:04d}"
    checkpoint_dir = (
        study_root(args.checkpoint_root, args.study_name) / trial_name
    )
    checkpoint_path = checkpoint_dir / f"step-{args.max_steps}.ckpt"
    trial_dir = (
        study_root(args.work_dir, args.study_name)
        / slugify(base_run_name)
        / f"trial_{trial_number:04d}"
    )
    eval_dir = (
        study_root(args.eval_output_root, args.study_name)
        / f"development_folds_{trial_name}"
    )
    model_label = f"trial_{trial_number:04d}"
    pipeline_fingerprint = getattr(args, "pipeline_fingerprint", None)
    if pipeline_fingerprint is None:
        pipeline_fingerprint = build_pipeline_fingerprint(args, rules)
    pipeline_fingerprint_sha256 = canonical_json_sha256(pipeline_fingerprint)
    train_cmd = training_command(args, params, checkpoint_dir, rules)
    eval_cmd = eval_command(
        args,
        params,
        checkpoint_path,
        model_label,
        eval_dir,
    )
    fixed_prior = pipeline_fingerprint["fixed_prior"]
    metadata = {
        "schema_version": "epit_pipeline_optuna_trial_v2",
        "pipeline_fingerprint": pipeline_fingerprint,
        "pipeline_fingerprint_sha256": pipeline_fingerprint_sha256,
        "trial_number": trial_number,
        "trial_name": trial_name,
        "sampler_seed": int(args.random_seed),
        "params": trial_params_payload(params),
        "reference_workflow": fixed_prior["reference_workflow"],
        "fixed_base_feature_count": EPIT_BASE_FEATURE_COUNT,
        "final_feature_count": (
            EPIT_MAGPIE_TOTAL_FEATURE_COUNT
            if params.use_magpie
            else EPIT_BASE_FEATURE_COUNT
        ),
        "fixed_block_allocation": original.FIXED_BLOCK_ALLOCATION,
        "fixed_material_style": fixed_prior["material_style"],
        "fixed_composition_mode": args.pitting_composition_mode,
        "fixed_physical_marginal_prob": 1.0,
        "fixed_prior_type": "hybrid_scm",
        "magpie_version": EPIT_MAGPIE_VERSION if params.use_magpie else None,
        "magpie_descriptor_names": (
            list(EPIT_MAGPIE_DESCRIPTOR_NAMES)
            if params.use_magpie
            else []
        ),
        "magpie_range_min_atomic_fraction": (
            EPIT_MAGPIE_RANGE_MIN_ATOMIC_FRACTION
            if params.use_magpie
            else None
        ),
        "split_manifest": str(args.split_manifest.expanduser().resolve()),
        "split_manifest_sha256": rules.split_manifest_sha256,
        "split_lock": rules.split_lock_path,
        "split_lock_sha256": rules.split_lock_sha256,
        "development_validation_folds": list(FIXED_VALIDATION_FOLDS),
        "final_test_rows_used": False,
        "evaluation_norm_methods": ["none"],
        "evaluation_feature_shuffle_method": "none",
        "target_rule_summary": rules.summary_path,
        "target_rule_summary_sha256": rules.summary_sha256,
        "target_rule_scores": rules.scores,
        "target_rule_probabilities": rules.probabilities,
        "target_rule_coefficients": rules.coefficients,
        "target_rule_artifacts": rules.artifacts,
        "target_rule_artifact_sha256s": rules.artifact_sha256s,
        "checkpoint_path": str(checkpoint_path),
        "eval_dir": str(eval_dir),
        "train_command": train_cmd,
        "eval_command": eval_cmd,
    }
    if args.pitting_composition_mode in EMPIRICAL_FEATURE_MODES:
        metadata.update(
            {
                "fixed_feature_profile": fixed_prior["feature_profile"],
                "fixed_feature_profile_metadata_sha256": fixed_prior[
                    "feature_profile_metadata_sha256"
                ],
                "pitting_composition_perturb_strength_range": pipeline_fingerprint[
                    "search_space"
                ]["pitting_composition_perturb_strength"],
            }
        )
    trial_config_path = trial_dir / "trial_config.json"
    if trial_config_path.is_file():
        existing = json.loads(trial_config_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise RuntimeError(
                "Existing trial configuration belongs to another pipeline or "
                f"parameter set: {trial_config_path}"
            )
    else:
        write_json(trial_config_path, metadata)
    if args.dry_run:
        return {
            **metadata,
            "status": "dry_run",
            "mean_test_spearman": math.nan,
            "median_test_spearman": math.nan,
            "std_test_spearman": math.nan,
            "mean_test_mae": math.nan,
            "median_test_mae": math.nan,
            "mean_test_rmse": math.nan,
            "mean_test_r2": math.nan,
            "summary_csv": "",
            "rows_csv": "",
        }

    if not (args.skip_existing and checkpoint_path.is_file()):
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise FileExistsError(
                f"Checkpoint directory already exists and is non-empty: {checkpoint_dir}"
            )
        original.search_utils.run_command(
            train_cmd,
            cwd=REPO_ROOT,
            log_path=trial_dir / "train.log",
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Expected checkpoint was not created: {checkpoint_path}"
        )
    checkpoint_sha256 = sha256_file(checkpoint_path)

    if not (args.skip_existing and (eval_dir / "summary.csv").is_file()):
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        original.search_utils.run_command(
            eval_cmd,
            cwd=REPO_ROOT,
            log_path=trial_dir / "eval.log",
        )
    summary = original.search_utils.load_eval_summary(
        eval_dir / "summary.csv",
        model_label,
    )
    evaluation_payload = verify_no_power_fold_evaluation(eval_dir)
    verify_trial_fold_evaluation(
        evaluation_payload,
        rules=rules,
        checkpoint_path=checkpoint_path,
        params=params,
    )
    summary_csv_path = (eval_dir / "summary.csv").resolve()
    rows_csv_path = (eval_dir / "rows.csv").resolve()
    summary_json_path = (eval_dir / "summary.json").resolve()
    trial_result_path = (trial_dir / "trial_result.json").resolve()
    result = {
        **metadata,
        "status": "completed",
        "mean_test_spearman": float(summary["mean_test_spearman"]),
        "median_test_spearman": float(summary["median_test_spearman"]),
        "std_test_spearman": float(summary["std_test_spearman"]),
        "mean_test_mae": float(summary["mean_test_mae"]),
        "median_test_mae": float(summary["median_test_mae"]),
        "mean_test_rmse": float(summary["mean_test_rmse"]),
        "mean_test_r2": float(summary["mean_test_r2"]),
        "checkpoint_sha256": checkpoint_sha256,
        "summary_csv": str(summary_csv_path),
        "summary_csv_sha256": sha256_file(summary_csv_path),
        "summary_json": str(summary_json_path),
        "summary_json_sha256": sha256_file(summary_json_path),
        "rows_csv": str(rows_csv_path),
        "rows_csv_sha256": sha256_file(rows_csv_path),
        "trial_result_json": str(trial_result_path),
    }
    write_json(trial_result_path, result)
    result["trial_result_json_sha256"] = sha256_file(trial_result_path)
    return result


def run_optuna(args: argparse.Namespace) -> None:
    try:
        import optuna
    except ImportError as error:
        raise SystemExit("Optuna is not installed in this environment.") from error

    args.base_run_name = (
        args.base_run_name or default_base_run_name(args.study_name)
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=original.search_utils.build_optuna_storage(args.storage),
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=args.random_seed,
            n_startup_trials=args.n_startup_trials,
        ),
    )
    fingerprint_sha256 = bind_study_pipeline_fingerprint(
        study,
        args.pipeline_fingerprint,
    )

    def objective(trial: Any) -> float:
        trial.set_user_attr(STUDY_FINGERPRINT_SHA256_ATTR, fingerprint_sha256)
        params = sample_params(
            trial, composition_mode=args.pitting_composition_mode
        )
        result = run_trial(args, int(trial.number), params)
        for key in (
            "mean_test_mae",
            "median_test_mae",
            "mean_test_rmse",
            "mean_test_r2",
            "median_test_spearman",
            "std_test_spearman",
            "summary_csv",
            "summary_csv_sha256",
            "summary_json",
            "summary_json_sha256",
            "rows_csv",
            "rows_csv_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "trial_result_json",
            "trial_result_json_sha256",
            "sampler_seed",
        ):
            if key in result:
                trial.set_user_attr(key, result[key])
        trial.set_user_attr(
            "pipeline_artifact_identity",
            pipeline_artifact_identity(args.target_rules),
        )
        trial.set_user_attr("pitting_magpie_features", bool(params.use_magpie))
        trial.set_user_attr("tabicl_norm_methods", ["none"])
        return float(result["mean_test_spearman"])

    study.optimize(objective, n_trials=args.n_trials)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Mean development-fold Spearman: {study.best_value:.6g}")
    print(f"Parameters: {study.best_trial.params}")


def append_random_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_random(args: argparse.Namespace) -> None:
    args.base_run_name = (
        args.base_run_name or default_base_run_name(args.study_name)
    )
    rng = random.Random(args.random_seed)
    output = (
        study_root(args.work_dir, args.study_name)
        / slugify(args.base_run_name)
        / "random_results.csv"
    )
    for trial_number in range(args.n_trials):
        params = sample_params(
            RandomTrial(rng), composition_mode=args.pitting_composition_mode
        )
        result = run_trial(args, trial_number, params)
        append_random_result(
            output,
            {
                "trial_number": trial_number,
                "status": result["status"],
                "mean_test_spearman": result["mean_test_spearman"],
                "mean_test_mae": result["mean_test_mae"],
                **trial_params_payload(params),
            },
        )
    print(f"Wrote random-search results to {output}")


def run_search(args: argparse.Namespace) -> None:
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive.")
    if args.n_startup_trials < 0:
        raise ValueError("--n-startup-trials must be non-negative.")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if args.scheduler_total_steps < args.max_steps:
        raise ValueError("--scheduler-total-steps must be >= --max-steps.")
    if args.eval_n_estimators <= 0:
        raise ValueError("--eval-n-estimators must be positive.")
    args.split_manifest = args.split_manifest.expanduser().resolve()
    if (
        args.pitting_composition_mode in EMPIRICAL_FEATURE_MODES
        and args.prior_n_jobs != 1
    ):
        raise ValueError(
            f"{args.pitting_composition_mode} requires --prior-n-jobs 1."
        )
    args.target_rule_summary = args.target_rule_summary.expanduser().resolve()
    args.target_rules = load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )
    args.pipeline_fingerprint = build_pipeline_fingerprint(
        args,
        args.target_rules,
    )
    if args.backend == "optuna":
        run_optuna(args)
    else:
        run_random(args)


def main() -> None:
    run_search(parse_args())


if __name__ == "__main__":
    main()
