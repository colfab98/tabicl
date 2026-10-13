#!/usr/bin/env python3
"""Freeze the selected EPIT and Soccol prior regimes for combined training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


DATASET_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATASET_DIR.parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tabicl.prior.combined_corrosion import COMBINED_FEATURE_COLUMNS  # noqa: E402
from tabicl.train.train_config import build_parser  # noqa: E402


DEFAULT_EPIT_LOG = (
    REPO_ROOT
    / "corrosion_datasets/analysis/epit_pipeline/final_v2"
    / "epit_pipeline_optuna_empirical_features_scm_target_v8/train.log"
)
DEFAULT_SOCCOL_LOG = (
    REPO_ROOT
    / "corrosion_datasets/analysis/soccol_pipeline/final_v1"
    / "soccol_pipeline_optuna_empirical_features_scm_target_v1/train.log"
)
DEFAULT_OUTPUT = DATASET_DIR / "processed/combined_prior_config.json"

TUPLE_OPTIONS = (
    "mix_probs",
    "informed_mix_probs",
    "informed_task_family_probs",
    "informed_normal_block_allocation",
    "informed_normal_block_allocation_min_counts",
    "informed_inhibitor_block_allocation",
    "informed_inhibitor_block_allocation_min_counts",
    "pitting_material_style_probs",
    "pitting_composition_family_probs",
    "pitting_target_rule_scores",
    "pitting_target_rule_coefficients",
)
SCALAR_OPTIONS = (
    "informed_feature_block_strength",
    "informed_interaction_strength",
    "informed_target_mix_weight",
    "informed_history_strength",
    "informed_intervention_strength",
    "informed_target_family",
    "informed_physical_marginal_prob",
    "informed_physical_marginal_profile",
    "pitting_material_dirichlet_prob",
    "pitting_material_dirichlet_concentration",
    "pitting_material_dirichlet_active_prob",
    "pitting_composition_mode",
    "pitting_composition_profile",
    "pitting_feature_profile",
    "pitting_composition_perturb_strength",
    "pitting_material_latent_count",
    "pitting_process_role",
    "pitting_process_category_count",
    "pitting_fixed_epit_schema",
    "pitting_magpie_features",
    "cat_prob",
    "permute_features",
    "epit_material_coef",
    "epit_environment_coef",
    "epit_interaction_coef",
    "pitting_coefficient_variation",
    "pitting_coefficient_variation_seed",
    "pitting_coefficient_upper_bounds",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_training_log(path: Path) -> tuple[list[str], argparse.Namespace]:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    # The training logger stores an argv-style command joined by spaces, not a
    # shell-escaped command. Splitting literally preserves JSON quotes in
    # values such as pitting_coefficient_upper_bounds.
    command = first_line.split()
    run_index = next(
        index for index, value in enumerate(command) if value.endswith("src/tabicl/train/run.py")
    )
    args = build_parser().parse_args(command[run_index + 1 :])
    return command, args


def fixed_hp_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for name in TUPLE_OPTIONS:
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = list(value)
    for name in SCALAR_OPTIONS:
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    return overrides


def source_regime(args: argparse.Namespace, probability: float) -> dict[str, Any]:
    if args.prior_type != "hybrid_scm":
        raise ValueError("Source model must use hybrid_scm.")
    if args.max_classes != 0 or args.min_features != args.max_features:
        raise ValueError("Source model must be fixed-width regression.")
    overrides = fixed_hp_overrides(args)
    overrides["cat_prob"] = 0.0
    overrides["permute_features"] = False
    return {
        "probability": probability,
        "prior_type": "informed_scm",
        "num_features": int(args.max_features),
        "fixed_hp": overrides,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epit-train-log", type=Path, default=DEFAULT_EPIT_LOG)
    parser.add_argument("--soccol-train-log", type=Path, default=DEFAULT_SOCCOL_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Use --overwrite to replace {output}.")
    epit_log = args.epit_train_log.expanduser().resolve()
    soccol_log = args.soccol_train_log.expanduser().resolve()
    epit_command, epit_args = parse_training_log(epit_log)
    soccol_command, soccol_args = parse_training_log(soccol_log)

    source_model_probability = 0.5
    epit_informed_probability = source_model_probability * float(epit_args.informed_prior_ratio)
    soccol_informed_probability = source_model_probability * float(soccol_args.informed_prior_ratio)
    generic_probability = 1.0 - epit_informed_probability - soccol_informed_probability
    payload = {
        "schema_version": "combined_corrosion_prior_v1",
        "feature_columns_in_order": list(COMBINED_FEATURE_COLUMNS),
        "source_commands": {
            "EPIT": {
                "path": str(epit_log.relative_to(REPO_ROOT)),
                "sha256": sha256_file(epit_log),
                "command": epit_command,
            },
            "Soccol": {
                "path": str(soccol_log.relative_to(REPO_ROOT)),
                "sha256": sha256_file(soccol_log),
                "command": soccol_command,
            },
        },
        "regimes": {
            "generic": {
                "probability": generic_probability,
                "prior_type": "mix_scm",
                "num_features": len(COMBINED_FEATURE_COLUMNS),
                "fixed_hp": {
                    "mix_probs": [0.7, 0.3],
                    "cat_prob": 0.2,
                    "permute_features": True,
                    "pitting_coefficient_variation": 0.0,
                },
            },
            "EPIT": source_regime(epit_args, epit_informed_probability),
            "Soccol": source_regime(soccol_args, soccol_informed_probability),
        },
        "policy": {
            "source_model_probabilities": {"EPIT": 0.5, "Soccol": 0.5},
            "formula_coefficients": "source-specific; never averaged",
            "generic_cat_prob": 0.2,
            "informed_cat_prob": 0.0,
            "mpea_used": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: regime["probability"] for name, regime in payload["regimes"].items()}, indent=2))


if __name__ == "__main__":
    main()

