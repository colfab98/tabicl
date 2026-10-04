#!/usr/bin/env python
"""Run the previous EPIT Optuna methodology on the frozen Soccol task."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline import run_optuna as base  # noqa: E402
from scripts.epit_pipeline.artifact_hashes import load_frozen_split  # noqa: E402
from tabicl.prior.dataset import SOCCOL_TARGET_RULE_COEFFICIENTS  # noqa: E402
from tabicl.prior.soccol_feature_profile import load_soccol_feature_profile  # noqa: E402
from tabicl.prior.soccol_schema import (  # noqa: E402
    SOCCOL_BASE_FEATURE_COUNT,
    SOCCOL_FEATURE_PROFILE,
    SOCCOL_FIXED_BLOCK_ALLOCATION,
)


PIPELINE_ROOT = REPO_ROOT / "corrosion_datasets" / "analysis" / "soccol_pipeline"
PROMOTED_RULES = (
    "pren_n_linear",
    "cr_mow_n_synergy",
    "threshold_saturation",
    "pren_n_improved_environment",
    "mo_n_acid_repassivation",
    "mns_inclusion_penalty",
    "coupled_breakdown",
    "pren_n_coupled_mns",
    "pren_n_coupled_mns_weak_anions",
)
_base_training_command = base.training_command


def empirical_feature_profile_identity(
    composition_profile_name: str | None = None,
) -> dict[str, object]:
    del composition_profile_name
    profile = load_soccol_feature_profile(SOCCOL_FEATURE_PROFILE)
    return {
        "feature_profile": SOCCOL_FEATURE_PROFILE,
        "feature_profile_metadata_sha256": base.canonical_json_sha256(profile.metadata),
        "feature_profile_csv_sha256": profile.metadata["csv_sha256"],
        "composition_profile": "all_soccol_rows_v1",
        "composition_family_probabilities": [1.0],
        "synthetic_material_families": ["all_soccol_rows"],
        "synthetic_material_family_probabilities": [1.0],
    }


def eval_command(
    args: object,
    params: base.TrialParams,
    checkpoint_path: Path,
    model_label: str,
    output_dir: Path,
) -> list[str]:
    if params.use_magpie:
        raise ValueError("The 37-feature Soccol pipeline does not use EPIT Magpie features.")
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "soccol_pipeline" / "evaluate_optuna_folds.py"),
        "--local-ckpt-path",
        str(checkpoint_path),
        "--model-label",
        model_label,
        "--split-manifest",
        str(args.split_manifest),
        "--validation-folds",
        *(str(fold) for fold in base.FIXED_VALIDATION_FOLDS),
        "--device",
        str(args.device),
        "--n-estimators",
        str(args.eval_n_estimators),
        "--tabicl-feat-shuffle-method",
        "none",
        "--tabicl-norm-methods",
        "none",
        "--output-dir",
        str(output_dir),
    ]


def training_command(
    args: object,
    params: base.TrialParams,
    checkpoint_dir: Path,
    rules: base.TargetRuleConfig,
) -> list[str]:
    command = _base_training_command(args, params, checkpoint_dir, rules)
    for option in ("--pitting_process_role", "--pitting_process_category_count"):
        base._remove_option(command, option)
    return command


def default_base_run_name(study_name: str) -> str:
    stamp = base.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tabicl_soccol_pipeline_{base.slugify(study_name)}_{stamp}"


def configure() -> None:
    base.PIPELINE_ROOT = PIPELINE_ROOT
    base.DEFAULT_SPLIT_MANIFEST = (
        REPO_ROOT
        / "corrosion_datasets"
        / "datasets"
        / "soccol_pitting_potential"
        / "processed"
        / "splits_v1"
        / "split_manifest.json"
    )
    base.DEFAULT_TARGET_RULE_SUMMARY = PIPELINE_ROOT / "target_rules_v1" / "calibration_summary.json"
    base.DEFAULT_OPTUNA_ROOT = PIPELINE_ROOT / "optuna_v1"
    base.DEFAULT_STUDY_NAME = "soccol_pipeline_optuna_empirical_features_scm_target_v1"
    base.PIPELINE_FINGERPRINT_SCHEMA = "soccol_pipeline_stage3_fingerprint_v1"
    base.STUDY_FINGERPRINT_ATTR = "soccol_pipeline_fingerprint"
    base.STUDY_FINGERPRINT_SHA256_ATTR = "soccol_pipeline_fingerprint_sha256"
    base.FIXED_BLOCK_ALLOCATION = SOCCOL_FIXED_BLOCK_ALLOCATION
    base.EPIT_BASE_FEATURE_COUNT = SOCCOL_BASE_FEATURE_COUNT
    base.EPIT_FEATURE_PROFILE = SOCCOL_FEATURE_PROFILE
    base.TARGET_RULE_COEFFICIENTS = SOCCOL_TARGET_RULE_COEFFICIENTS
    # The shared training CLI retains five legacy family slots. The Soccol
    # generator uses one all-row bank and ignores this compatibility vector.
    base.EPIT_COMPOSITION_FAMILY_PROBS = (1.0, 0.0, 0.0, 0.0, 0.0)
    base.PROMOTED_TARGET_RULE_FAMILIES = PROMOTED_RULES
    base.load_frozen_split = partial(
        load_frozen_split,
        expected_manifest_schema="soccol_composition_split_manifest_v1",
    )
    base.empirical_feature_profile_identity = empirical_feature_profile_identity
    base.training_command = training_command
    base.eval_command = eval_command
    base.default_base_run_name = default_base_run_name


def main() -> None:
    configure()
    base.main()


if __name__ == "__main__":
    main()
