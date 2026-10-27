#!/usr/bin/env python
"""Run the reviewed EPIT Optuna methodology on accepted Soccol development rows."""

from __future__ import annotations

import sys
import json
import argparse
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
from tabicl.prior.soccol_reviewed_schema import (
    SOCCOL_REVIEWED_FEATURE_PROFILE, SOCCOL_REVIEWED_FEATURE_COLUMNS,
    SOCCOL_REVIEWED_FEATURE_COUNT, SOCCOL_REVIEWED_BLOCK_ALLOCATION,
)


PIPELINE_ROOT = REPO_ROOT / "corrosion_datasets" / "analysis" / "soccol_pipeline"
from scripts.soccol_pipeline.target_rules import PROMOTED_RULES
_base_training_command = base.training_command
_base_load_target_rule_config = base.load_target_rule_config


def empirical_feature_profile_identity(
    composition_profile_name: str | None = None,
) -> dict[str, object]:
    del composition_profile_name
    name = SOCCOL_REVIEWED_FEATURE_PROFILE if base.EPIT_FEATURE_PROFILE == SOCCOL_REVIEWED_FEATURE_PROFILE else SOCCOL_FEATURE_PROFILE
    profile = load_soccol_feature_profile(name)
    if profile.name == SOCCOL_REVIEWED_FEATURE_PROFILE:
        return {
            "feature_profile": profile.name,
            "feature_profile_metadata_sha256": base.canonical_json_sha256(profile.metadata),
            "feature_profile_csv_sha256": profile.metadata["csv_sha256"],
            "composition_profile": "accepted_soccol_development_v2",
            "composition_family_probabilities": [1.0],
            "synthetic_material_families": ["all_accepted_soccol_materials"],
            "synthetic_material_family_probabilities": [1.0],
            "feature_columns": list(SOCCOL_REVIEWED_FEATURE_COLUMNS),
            "feature_count": SOCCOL_REVIEWED_FEATURE_COUNT,
            "development_selection": profile.metadata["development_selection"],
            "missing_value_policy": "paired_environment_row_nan",
            "col_missing_value_embedding": True,
            "formula_input_policy": "existing_soccol_formulas_with_named_column_lookup",
            "composition_sampling_policy": profile.metadata["sampling"]["composition_sampling_policy"],
            "distinct_composition_templates": len(profile.composition_template_indices),
        }
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
        raise ValueError("The Soccol pipeline does not use EPIT Magpie features.")
    command = [
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
    if base.EPIT_FEATURE_PROFILE == SOCCOL_REVIEWED_FEATURE_PROFILE:
        command.append("--soccol-reviewed-data")
    return command


def training_command(
    args: object,
    params: base.TrialParams,
    checkpoint_dir: Path,
    rules: base.TargetRuleConfig,
) -> list[str]:
    command = _base_training_command(args, params, checkpoint_dir, rules)
    for option in ("--pitting_process_role", "--pitting_process_category_count"):
        base._remove_option(command, option)
    if base.EPIT_FEATURE_PROFILE == SOCCOL_REVIEWED_FEATURE_PROFILE:
        if args.pitting_composition_mode != "empirical_features_scm_target" or params.use_magpie:
            raise ValueError("Reviewed Soccol requires empirical_features_scm_target without Magpie.")
        command.extend(["--col_missing_value_embedding", "true"])
    return command


def default_base_run_name(study_name: str) -> str:
    stamp = base.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tabicl_soccol_pipeline_{base.slugify(study_name)}_{stamp}"


def load_reviewed_target_rule_config(*, summary_path: Path, split_manifest_path: Path):
    """Do not allow historical coefficients to masquerade as the reviewed fit."""
    if not summary_path.is_file():
        raise RuntimeError("Reviewed Soccol schema is connected, but its formula calibration is not ready. Refit formulas on the 2,950 accepted development rows before Optuna.")
    metadata = json.loads(summary_path.read_text())
    expected = load_soccol_feature_profile(SOCCOL_REVIEWED_FEATURE_PROFILE).metadata["development_selection"]
    if metadata.get("development_selection") != expected or metadata.get("development_rows") != expected["source_rows"]:
        raise RuntimeError("Reviewed Soccol training requires formula calibration from the same 2,950 accepted development rows; historical calibration cannot be reused.")
    return _base_load_target_rule_config(summary_path=summary_path, split_manifest_path=split_manifest_path)


def configure(*, reviewed_data: bool = True) -> None:
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
    base.load_target_rule_config = _base_load_target_rule_config
    if reviewed_data:
        base.EPIT_FEATURE_PROFILE = SOCCOL_REVIEWED_FEATURE_PROFILE
        base.EPIT_BASE_FEATURE_COUNT = SOCCOL_REVIEWED_FEATURE_COUNT
        base.FIXED_BLOCK_ALLOCATION = SOCCOL_REVIEWED_BLOCK_ALLOCATION
        base.DEFAULT_TARGET_RULE_SUMMARY = PIPELINE_ROOT / "target_rules_v2" / "calibration_summary.json"
        base.DEFAULT_OPTUNA_ROOT = PIPELINE_ROOT / "optuna_v4"
        base.DEFAULT_STUDY_NAME = "soccol_pipeline_optuna_empirical_features_scm_target_v4"
        base.PIPELINE_FINGERPRINT_SCHEMA = "soccol_pipeline_stage3_fingerprint_v4"
        base.load_target_rule_config = load_reviewed_target_rule_config


def data_mode_parser() -> argparse.ArgumentParser:
    mode_parser = argparse.ArgumentParser(add_help=False)
    group = mode_parser.add_mutually_exclusive_group()
    group.add_argument(
        "--soccol-reviewed-data", action="store_true",
        help="Use reviewed 58-column Soccol data (default).",
    )
    group.add_argument(
        "--legacy-data", action="store_true",
        help="Reproduce the historical 37-column v1 workflow.",
    )
    return mode_parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    mode_parser = data_mode_parser()
    mode, _ = mode_parser.parse_known_args(argv)
    configure(reviewed_data=not mode.legacy_data)
    return base.parse_args(argv, parents=(mode_parser,))


def main() -> None:
    base.run_search(parse_args())


if __name__ == "__main__":
    main()
