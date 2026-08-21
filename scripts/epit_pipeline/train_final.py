#!/usr/bin/env python
"""Stage 4: train the selected EPIT configuration once and freeze it."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline import run_optuna as search
from scripts.epit_pipeline.artifact_hashes import (
    FINAL_MODEL_LOCK_NAME,
    build_final_model_lock,
    load_frozen_split,
    sha256_file,
)


DEFAULT_FINAL_ROOT = search.PIPELINE_ROOT / "final_v1"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "epit_pipeline_final_v1"
FINAL_MODEL_MANIFEST_NAME = "final_model_manifest.json"
CHECKPOINT_RE = re.compile(r"^step-(\d+)\.ckpt$")
PITTING_TASK_ID = "electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg"


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected true or false, received {value!r}."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-name", default="epit_pipeline_optuna_v3")
    parser.add_argument(
        "--selected-trial-number",
        type=int,
        default=None,
        help=(
            "Pin a completed trial after verifying that it is still the best "
            "completed trial. By default Stage 4 selects study.best_trial."
        ),
    )
    parser.add_argument(
        "--allow-stale-running-trials",
        action="store_true",
        help=(
            "Deprecated compatibility flag. RUNNING/WAITING trials are always "
            "non-blocking; Stage 4 selects the current best completed trial."
        ),
    )
    parser.add_argument(
        "--allow-missing-selected-trial-artifacts",
        action="store_true",
        help=(
            "Use the completed Optuna journal record when an explicitly pinned "
            "trial's immutable proxy artifacts are no longer locally available."
        ),
    )
    parser.add_argument(
        "--storage",
        required=True,
        help="The Stage 3 Optuna storage URL (including journal://).",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=search.DEFAULT_SPLIT_MANIFEST,
    )
    parser.add_argument(
        "--target-rule-summary",
        type=Path,
        default=search.DEFAULT_TARGET_RULE_SUMMARY,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--final-run-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--scheduler-total-steps", type=int, default=10000)
    parser.add_argument("--save-temp-every", type=int, default=100)
    parser.add_argument("--save-perm-every", type=int, default=500)
    parser.add_argument("--checkpoint-selection-min-step", type=int, default=500)
    parser.add_argument("--checkpoint-selection-interval", type=int, default=500)
    parser.add_argument("--np-seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--prior-n-jobs", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=4)
    parser.add_argument("--eval-n-estimators", type=int, default=8)
    parser.add_argument("--override-informed-prior-ratio", type=float, default=None)
    parser.add_argument("--override-mlp-prob", type=float, default=None)
    parser.add_argument(
        "--override-pitting-composition-perturb-strength",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--override-use-magpie",
        type=parse_bool,
        default=None,
        metavar="{true,false}",
    )
    parser.add_argument(
        "--override-pitting-composition-family-probs",
        type=float,
        nargs=5,
        default=None,
        metavar=("FE", "AL", "HEA", "NICRMO", "OTHER"),
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--nproc-per-node": args.nproc_per_node,
        "--max-steps": args.max_steps,
        "--scheduler-total-steps": args.scheduler_total_steps,
        "--save-temp-every": args.save_temp_every,
        "--save-perm-every": args.save_perm_every,
        "--checkpoint-selection-min-step": args.checkpoint_selection_min_step,
        "--checkpoint-selection-interval": args.checkpoint_selection_interval,
        "--eval-n-estimators": args.eval_n_estimators,
    }
    for option, value in positive.items():
        if value <= 0:
            raise ValueError(f"{option} must be positive.")
    if args.scheduler_total_steps < args.max_steps:
        raise ValueError("--scheduler-total-steps must be >= --max-steps.")
    if args.max_steps % args.save_perm_every:
        raise ValueError("--max-steps must be divisible by --save-perm-every.")
    if args.checkpoint_selection_interval % args.save_perm_every:
        raise ValueError(
            "--checkpoint-selection-interval must be divisible by --save-perm-every."
        )
    if args.selected_trial_number is not None and args.selected_trial_number < 0:
        raise ValueError("--selected-trial-number must be non-negative.")
    if (
        args.allow_missing_selected_trial_artifacts
        and args.selected_trial_number is None
    ):
        raise ValueError(
            "--allow-missing-selected-trial-artifacts requires "
            "--selected-trial-number."
        )


def load_selected_trial(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError as error:
        raise SystemExit("Optuna is not installed in this environment.") from error

    study = optuna.load_study(
        study_name=args.study_name,
        storage=search.original.search_utils.build_optuna_storage(args.storage),
    )
    completed = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
    ]
    if not completed:
        raise RuntimeError("The Optuna study has no completed finite trials.")
    best_completed = max(completed, key=lambda trial: float(trial.value))
    if args.selected_trial_number is None:
        return study, best_completed

    selected = next(
        (
            trial
            for trial in completed
            if int(trial.number) == int(args.selected_trial_number)
        ),
        None,
    )
    if selected is None:
        raise RuntimeError(
            f"Pinned trial {args.selected_trial_number} is not a completed "
            "finite trial in the study."
        )
    if int(selected.number) != int(best_completed.number):
        raise RuntimeError(
            f"Pinned trial {selected.number} is not the best completed trial; "
            f"current best is trial {best_completed.number}."
        )
    return study, selected


def verify_study_pipeline_identity(
    study: Any,
    rules: search.TargetRuleConfig,
) -> tuple[dict[str, Any], str]:
    """Require Stage 4 to consume the exact artifacts locked by Stage 3."""
    fingerprint = study.user_attrs.get(search.STUDY_FINGERPRINT_ATTR)
    fingerprint_sha256 = study.user_attrs.get(
        search.STUDY_FINGERPRINT_SHA256_ATTR
    )
    if fingerprint_sha256 is None:
        raise RuntimeError("Optuna study has no EPIT pipeline fingerprint hash.")
    fingerprint_sha256 = str(fingerprint_sha256)
    fingerprint = search.validate_pipeline_fingerprint(
        fingerprint,
        fingerprint_sha256,
    )
    current_artifacts = search.pipeline_artifact_identity(rules)
    if fingerprint.get("artifacts") != current_artifacts:
        raise RuntimeError(
            "Optuna study used a different split, dataset, or target-rule set."
        )
    return fingerprint, fingerprint_sha256


def verify_empirical_feature_profile_identity(
    fixed_prior: dict[str, Any],
) -> None:
    """Require final training to use the profile locked by the v5 study."""
    expected = search.empirical_feature_profile_identity()
    observed = {
        key: fixed_prior.get(key)
        for key in expected
    }
    if observed != expected:
        raise RuntimeError(
            "Current empirical feature profile differs from the selected Optuna study."
        )


def verify_trial_file(trial: Any, path_attr: str, sha_attr: str) -> Path:
    path_value = trial.user_attrs.get(path_attr)
    expected_sha256 = trial.user_attrs.get(sha_attr)
    if not path_value or not expected_sha256:
        raise RuntimeError(
            f"Selected trial is missing immutable artifact fields: "
            f"{path_attr}, {sha_attr}."
        )
    path = Path(str(path_value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Selected-trial artifact does not exist: {path}")
    if sha256_file(path) != str(expected_sha256):
        raise RuntimeError(f"Selected-trial artifact hash changed: {path}")
    return path


def require_same_finite_metric(
    observed: Any,
    expected: Any,
    *,
    label: str,
) -> None:
    try:
        observed_value = float(observed)
        expected_value = float(expected)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric.") from error
    if not math.isfinite(observed_value) or not math.isfinite(expected_value):
        raise RuntimeError(f"{label} is not finite.")
    if not math.isclose(
        observed_value,
        expected_value,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"{label} does not match the Optuna objective.")


def verify_selected_trial_evaluation(
    trial: Any,
    *,
    split_manifest: Path,
    rules: search.TargetRuleConfig,
    params: search.TrialParams,
    study_fingerprint_sha256: str,
) -> dict[str, Any]:
    if (
        str(trial.user_attrs.get(search.STUDY_FINGERPRINT_SHA256_ATTR, ""))
        != study_fingerprint_sha256
    ):
        raise RuntimeError("Selected trial belongs to another pipeline fingerprint.")
    if trial.user_attrs.get("pipeline_artifact_identity") != (
        search.pipeline_artifact_identity(rules)
    ):
        raise RuntimeError("Selected trial belongs to another artifact set.")
    if "pitting_magpie_features" not in trial.user_attrs or bool(
        trial.user_attrs["pitting_magpie_features"]
    ) != bool(params.use_magpie):
        raise RuntimeError("Selected trial has inconsistent Magpie provenance.")

    summary_csv = verify_trial_file(
        trial,
        "summary_csv",
        "summary_csv_sha256",
    )
    summary_json = verify_trial_file(
        trial,
        "summary_json",
        "summary_json_sha256",
    )
    rows_csv = verify_trial_file(trial, "rows_csv", "rows_csv_sha256")
    checkpoint = verify_trial_file(
        trial,
        "checkpoint_path",
        "checkpoint_sha256",
    )
    trial_result_path = verify_trial_file(
        trial,
        "trial_result_json",
        "trial_result_json_sha256",
    )
    if summary_json != summary_csv.parent / "summary.json":
        raise RuntimeError("Selected trial summary files come from different runs.")
    eval_dir = summary_csv.parent
    payload = search.verify_no_power_fold_evaluation(eval_dir)
    if not bool(payload.get("development_rows_only", False)):
        raise RuntimeError("Selected trial was not evaluated development-only.")
    if payload.get("validation_folds") != list(search.FIXED_VALIDATION_FOLDS):
        raise RuntimeError("Selected trial did not use all five frozen folds.")
    if Path(str(payload.get("split_manifest", ""))).resolve() != split_manifest:
        raise RuntimeError("Selected trial used a different split manifest.")
    if payload.get("split_manifest_sha256") != rules.split_manifest_sha256:
        raise RuntimeError("Selected trial used a changed split manifest.")
    if payload.get("split_lock_sha256") != rules.split_lock_sha256:
        raise RuntimeError("Selected trial used a changed split lock.")
    if str(payload.get("source_sha256", "")) != rules.source_sha256:
        raise RuntimeError("Selected trial evaluation used a different dataset.")
    source = payload.get("source", {})
    if Path(str(source.get("local_ckpt_path", ""))).resolve() != checkpoint:
        raise RuntimeError("Selected trial evaluation names another checkpoint.")
    if source.get("local_ckpt_sha256") != sha256_file(checkpoint):
        raise RuntimeError("Selected trial evaluation checkpoint hash changed.")
    settings = payload.get("settings", {})
    if settings.get("tabicl_feat_shuffle_method") != "none":
        raise RuntimeError("Selected trial changed the fixed feature order.")
    if "pitting_magpie_features" not in settings or bool(
        settings["pitting_magpie_features"]
    ) != bool(params.use_magpie):
        raise RuntimeError("Selected trial evaluation changed Magpie features.")
    if Path(str(payload.get("summary_csv", ""))).resolve() != summary_csv:
        raise RuntimeError("Selected trial summary JSON names another summary CSV.")
    if Path(str(payload.get("rows_csv", ""))).resolve() != rows_csv:
        raise RuntimeError("Selected trial summary JSON names another rows CSV.")

    summary_records = payload.get("summary")
    if not isinstance(summary_records, list) or len(summary_records) != 1:
        raise RuntimeError("Selected trial has an invalid fold summary record.")
    require_same_finite_metric(
        summary_records[0].get("mean_test_spearman"),
        trial.value,
        label="Selected trial fold Spearman",
    )

    trial_result = json.loads(trial_result_path.read_text(encoding="utf-8"))
    if not isinstance(trial_result, dict):
        raise RuntimeError("Selected trial result is not a JSON object.")
    expected_result_fields = {
        "status": "completed",
        "trial_number": int(trial.number),
        "sampler_seed": trial.user_attrs.get("sampler_seed"),
        "params": search.trial_params_payload(params),
        "pipeline_fingerprint_sha256": study_fingerprint_sha256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "summary_csv": str(summary_csv),
        "summary_csv_sha256": sha256_file(summary_csv),
        "summary_json": str(summary_json),
        "summary_json_sha256": sha256_file(summary_json),
        "rows_csv": str(rows_csv),
        "rows_csv_sha256": sha256_file(rows_csv),
    }
    if not isinstance(trial.user_attrs.get("sampler_seed"), int):
        raise RuntimeError("Selected trial has no recorded sampler seed.")
    for key, expected in expected_result_fields.items():
        if trial_result.get(key) != expected:
            raise RuntimeError(
                f"Selected trial result has inconsistent field: {key}."
            )
    require_same_finite_metric(
        trial_result.get("mean_test_spearman"),
        trial.value,
        label="Selected trial result Spearman",
    )
    return payload


def verify_selected_trial_journal_record(
    trial: Any,
    *,
    rules: search.TargetRuleConfig,
    params: search.TrialParams,
    study_fingerprint: dict[str, Any],
    study_fingerprint_sha256: str,
) -> dict[str, Any]:
    """Validate provenance retained in Optuna when worker-local files are gone."""
    if (
        str(trial.user_attrs.get(search.STUDY_FINGERPRINT_SHA256_ATTR, ""))
        != study_fingerprint_sha256
    ):
        raise RuntimeError("Selected trial belongs to another pipeline fingerprint.")
    if trial.user_attrs.get("pipeline_artifact_identity") != (
        search.pipeline_artifact_identity(rules)
    ):
        raise RuntimeError("Selected trial belongs to another artifact set.")
    if "pitting_magpie_features" not in trial.user_attrs or bool(
        trial.user_attrs["pitting_magpie_features"]
    ) != bool(params.use_magpie):
        raise RuntimeError("Selected trial has inconsistent Magpie provenance.")
    if trial.user_attrs.get("tabicl_norm_methods") != ["none"]:
        raise RuntimeError("Selected trial did not record disabled power normalization.")
    if not isinstance(trial.user_attrs.get("sampler_seed"), int):
        raise RuntimeError("Selected trial has no recorded sampler seed.")

    evaluation = study_fingerprint.get("evaluation", {})
    if evaluation.get("validation_folds") != list(search.FIXED_VALIDATION_FOLDS):
        raise RuntimeError("Study fingerprint does not contain all development folds.")
    if evaluation.get("feature_shuffle_method") != "none":
        raise RuntimeError("Study fingerprint changed the fixed feature order.")
    if evaluation.get("norm_methods") != ["none"]:
        raise RuntimeError("Study fingerprint did not disable power normalization.")
    if evaluation.get("regression_output") != "median":
        raise RuntimeError("Study fingerprint changed regression output.")

    artifacts: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for path_attr, sha_attr in (
        ("summary_csv", "summary_csv_sha256"),
        ("summary_json", "summary_json_sha256"),
        ("rows_csv", "rows_csv_sha256"),
        ("checkpoint_path", "checkpoint_sha256"),
        ("trial_result_json", "trial_result_json_sha256"),
    ):
        path_value = trial.user_attrs.get(path_attr)
        sha_value = str(trial.user_attrs.get(sha_attr, ""))
        if not path_value or re.fullmatch(r"[0-9a-f]{64}", sha_value) is None:
            raise RuntimeError(
                "Selected trial has an incomplete journal artifact record: "
                f"{path_attr}, {sha_attr}."
            )
        path = Path(str(path_value)).expanduser().resolve()
        artifacts[path_attr] = {"path": str(path), "sha256": sha_value}
        if not path.is_file():
            missing.append(str(path))

    metrics = {"mean_test_spearman": float(trial.value)}
    if not math.isfinite(metrics["mean_test_spearman"]):
        raise RuntimeError("Selected trial objective is not finite.")
    for key in (
        "mean_test_mae",
        "median_test_mae",
        "mean_test_rmse",
        "mean_test_r2",
        "median_test_spearman",
        "std_test_spearman",
    ):
        try:
            value = float(trial.user_attrs[key])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"Selected trial has no finite journal metric: {key}."
            ) from error
        if not math.isfinite(value):
            raise RuntimeError(f"Selected trial journal metric is not finite: {key}.")
        metrics[key] = value

    return {
        "schema_version": "epit_selected_trial_journal_provenance_v1",
        "verification_mode": "optuna_journal_record",
        "immutable_files_verified": False,
        "missing_local_artifacts": missing,
        "development_rows_only": True,
        "validation_folds": list(search.FIXED_VALIDATION_FOLDS),
        "settings": {
            "n_estimators": int(evaluation["n_estimators"]),
            "pitting_magpie_features": bool(params.use_magpie),
            "tabicl_feat_shuffle_method": "none",
            "tabicl_norm_methods": ["none"],
            "regression_output": "median",
        },
        "pipeline_fingerprint_sha256": study_fingerprint_sha256,
        "artifact_identity": search.pipeline_artifact_identity(rules),
        "recorded_artifacts": artifacts,
        "recorded_metrics": metrics,
    }


def replace_option(command: list[str], option: str, value: Any) -> None:
    index = command.index(option)
    command[index + 1] = str(value)


def replace_option_values(
    command: list[str],
    option: str,
    values: list[Any],
    *,
    expected_count: int,
) -> None:
    if len(values) != expected_count:
        raise ValueError(
            f"{option} requires exactly {expected_count} values."
        )
    index = command.index(option)
    command[index + 1 : index + 1 + expected_count] = [
        search.format_float(float(value)) for value in values
    ]


def apply_parameter_overrides(
    args: argparse.Namespace,
    selected_params: search.TrialParams,
    *,
    composition_mode: str,
) -> tuple[search.TrialParams, dict[str, Any]]:
    requested = {
        "informed_prior_ratio": args.override_informed_prior_ratio,
        "mlp_prob": args.override_mlp_prob,
        "pitting_composition_perturb_strength": (
            args.override_pitting_composition_perturb_strength
        ),
        "use_magpie": args.override_use_magpie,
    }
    family_probs = args.override_pitting_composition_family_probs
    if not any(value is not None for value in requested.values()) and (
        family_probs is None
    ):
        return selected_params, {}
    if composition_mode != "empirical_features_scm_target":
        raise ValueError(
            "Final parameter overrides are supported only for "
            "empirical_features_scm_target studies."
        )

    for name in ("informed_prior_ratio", "mlp_prob"):
        value = requested[name]
        if value is not None and (
            not math.isfinite(value) or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"--override-{name.replace('_', '-')} must be in [0, 1].")
    perturb_strength = requested["pitting_composition_perturb_strength"]
    if perturb_strength is not None and (
        not math.isfinite(perturb_strength) or perturb_strength < 0.0
    ):
        raise ValueError(
            "--override-pitting-composition-perturb-strength must be finite "
            "and non-negative."
        )

    trial_param_overrides = {
        name: value
        for name, value in requested.items()
        if value is not None
    }
    effective_params = replace(selected_params, **trial_param_overrides)
    provenance: dict[str, Any] = {
        "trial_parameter_overrides": trial_param_overrides,
    }
    if family_probs is not None:
        probabilities = [float(value) for value in family_probs]
        if (
            any(not math.isfinite(value) or value < 0.0 for value in probabilities)
            or sum(probabilities) <= 0.0
        ):
            raise ValueError(
                "--override-pitting-composition-family-probs values must be "
                "finite and non-negative, with at least one positive value."
            )
        provenance["fixed_prior_overrides"] = {
            "pitting_composition_family_probs": probabilities,
        }
    return effective_params, provenance


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.match(path.name)
    if match is None:
        raise ValueError(f"Not a step checkpoint: {path}")
    return int(match.group(1))


def checkpoint_candidates(args: argparse.Namespace, checkpoint_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in checkpoint_dir.glob("step-*.ckpt"):
        try:
            step = checkpoint_step(path)
        except ValueError:
            continue
        if (
            args.checkpoint_selection_min_step <= step <= args.max_steps
            and step % args.checkpoint_selection_interval == 0
        ):
            candidates.append(path.resolve())
    candidates.sort(key=checkpoint_step)
    final_checkpoint = checkpoint_dir / f"step-{args.max_steps}.ckpt"
    if not final_checkpoint.is_file():
        raise FileNotFoundError(
            f"Final training did not create its terminal checkpoint: {final_checkpoint}"
        )
    if final_checkpoint.resolve() not in candidates:
        candidates.append(final_checkpoint.resolve())
        candidates.sort(key=checkpoint_step)
    return candidates


def validate_fold_evaluation_payload(
    payload: dict[str, Any],
    *,
    checkpoint: Path,
    split_manifest: Path,
    rules: search.TargetRuleConfig,
    use_magpie: bool,
    n_estimators: int,
) -> None:
    if payload.get("settings", {}).get("tabicl_norm_methods") != ["none"]:
        raise RuntimeError("Checkpoint evaluation did not disable power.")
    if not bool(payload.get("development_rows_only", False)):
        raise RuntimeError("Checkpoint selection touched final-test rows.")
    if payload.get("validation_folds") != list(search.FIXED_VALIDATION_FOLDS):
        raise RuntimeError("Checkpoint selection did not use all five folds.")
    if Path(str(payload.get("split_manifest", ""))).resolve() != split_manifest:
        raise RuntimeError("Checkpoint selection used a different split manifest.")
    if payload.get("split_manifest_sha256") != rules.split_manifest_sha256:
        raise RuntimeError("Checkpoint selection used a changed split manifest.")
    if payload.get("split_lock_sha256") != rules.split_lock_sha256:
        raise RuntimeError("Checkpoint selection used a changed split lock.")
    if payload.get("source_sha256") != rules.source_sha256:
        raise RuntimeError("Checkpoint selection used a different source dataset.")
    source = payload.get("source", {})
    if Path(str(source.get("local_ckpt_path", ""))).resolve() != checkpoint:
        raise RuntimeError("Checkpoint evaluation provenance names another model.")
    if source.get("local_ckpt_sha256") != sha256_file(checkpoint):
        raise RuntimeError("Checkpoint evaluation model hash changed.")
    settings = payload.get("settings", {})
    if int(settings.get("n_estimators", -1)) != n_estimators:
        raise RuntimeError("Checkpoint evaluation changed n_estimators.")
    if settings.get("tabicl_feat_shuffle_method") != "none":
        raise RuntimeError("Checkpoint evaluation changed the feature order.")
    if "pitting_magpie_features" not in settings or bool(
        settings["pitting_magpie_features"]
    ) != bool(use_magpie):
        raise RuntimeError("Checkpoint evaluation changed Magpie features.")


def evaluation_record(
    args: argparse.Namespace,
    params: search.TrialParams,
    *,
    checkpoint: Path,
    output_dir: Path,
    split_manifest: Path,
    rules: search.TargetRuleConfig,
) -> dict[str, Any]:
    step = checkpoint_step(checkpoint)
    model_label = f"final_step_{step:05d}"
    command = search.eval_command(
        args,
        params,
        checkpoint,
        model_label,
        output_dir,
    )
    summary_csv = output_dir / "summary.csv"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Checkpoint evaluation already exists: {output_dir}")
    search.original.search_utils.run_command(
        command,
        cwd=REPO_ROOT,
        log_path=output_dir.parent / f"step_{step:05d}.log",
    )

    payload = search.verify_no_power_fold_evaluation(output_dir)
    validate_fold_evaluation_payload(
        payload,
        checkpoint=checkpoint,
        split_manifest=split_manifest,
        rules=rules,
        use_magpie=params.use_magpie,
        n_estimators=args.eval_n_estimators,
    )
    summary = search.original.search_utils.load_eval_summary(
        summary_csv,
        model_label,
    )
    spearman = float(summary["mean_test_spearman"])
    mae = float(summary["mean_test_mae"])
    if not math.isfinite(spearman):
        raise RuntimeError(f"Checkpoint step {step} has non-finite Spearman.")
    return {
        "step": step,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "mean_test_spearman": spearman,
        "mean_test_mae": mae,
        "mean_test_rmse": float(summary["mean_test_rmse"]),
        "mean_test_r2": float(summary["mean_test_r2"]),
        "summary_csv": str(summary_csv.resolve()),
        "summary_csv_sha256": sha256_file(summary_csv),
        "summary_json": str((output_dir / "summary.json").resolve()),
        "summary_json_sha256": sha256_file(output_dir / "summary.json"),
        "evaluation_command": command,
    }


def output_dir_for(
    args: argparse.Namespace,
    *,
    run_name: str,
    isolate_run: bool,
) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    study_root = DEFAULT_FINAL_ROOT / search.slugify(args.study_name)
    if isolate_run:
        return (study_root / search.slugify(run_name)).resolve()
    return study_root.resolve()


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    args.split_manifest = args.split_manifest.expanduser().resolve()
    args.target_rule_summary = args.target_rule_summary.expanduser().resolve()
    frozen_split = load_frozen_split(args.split_manifest)
    rules = search.load_target_rule_config(
        summary_path=args.target_rule_summary,
        split_manifest_path=args.split_manifest,
    )
    study, trial = load_selected_trial(args)
    unfinished_trials_at_selection = [
        {"number": int(candidate.number), "state": candidate.state.name}
        for candidate in study.trials
        if candidate.state.name in {"RUNNING", "WAITING"}
    ]
    if unfinished_trials_at_selection:
        print(
            "Selecting the current best completed trial while Optuna still has "
            f"{len(unfinished_trials_at_selection)} RUNNING/WAITING trial(s).",
            flush=True,
        )
    study_fingerprint, study_fingerprint_sha256 = (
        verify_study_pipeline_identity(study, rules)
    )
    fixed_prior = study_fingerprint.get("fixed_prior", {})
    composition_mode = str(fixed_prior.get("composition_mode", "legacy"))
    if composition_mode not in search.PITTING_COMPOSITION_MODES:
        raise RuntimeError(
            "Selected study has an unsupported pitting composition mode: "
            f"{composition_mode!r}."
        )
    args.pitting_composition_mode = composition_mode
    if composition_mode in search.EMPIRICAL_FEATURE_MODES:
        verify_empirical_feature_profile_identity(fixed_prior)
    proxy_training = study_fingerprint.get("proxy_training", {})
    proxy_training_overrides: dict[str, dict[str, int]] = {}
    for argument_name, fingerprint_name in (
        ("np_seed", "np_seed"),
        ("torch_seed", "torch_seed"),
        ("prior_n_jobs", "prior_n_jobs"),
    ):
        expected = int(proxy_training.get(fingerprint_name, -1))
        observed = int(getattr(args, argument_name))
        if observed != expected:
            proxy_training_overrides[argument_name] = {
                "selected_study_value": expected,
                "effective_final_value": observed,
            }
    if args.prior_n_jobs != 1 and composition_mode in search.EMPIRICAL_FEATURE_MODES:
        print(
            "Warning: --prior-n-jobs greater than 1 is recorded as an "
            "experimental override and may not reproduce the single-worker run.",
            flush=True,
        )
    selected_params = search.trial_params_from_mapping(
        dict(trial.params), composition_mode=composition_mode
    )
    try:
        trial_eval = verify_selected_trial_evaluation(
            trial,
            split_manifest=args.split_manifest,
            rules=rules,
            params=selected_params,
            study_fingerprint_sha256=study_fingerprint_sha256,
        )
    except FileNotFoundError:
        if not args.allow_missing_selected_trial_artifacts:
            raise
        trial_eval = verify_selected_trial_journal_record(
            trial,
            rules=rules,
            params=selected_params,
            study_fingerprint=study_fingerprint,
            study_fingerprint_sha256=study_fingerprint_sha256,
        )
        print(
            "Selected-trial proxy files are unavailable locally; "
            "validated the completed Optuna journal record instead.",
            flush=True,
        )
    params, overrides = apply_parameter_overrides(
        args,
        selected_params,
        composition_mode=composition_mode,
    )
    if proxy_training_overrides:
        overrides["proxy_training_overrides"] = proxy_training_overrides
    default_run_name = (
        f"final_{search.slugify(args.study_name)}_trial_{trial.number:04d}"
    )
    is_experimental = bool(overrides)
    if is_experimental:
        override_sha256 = search.canonical_json_sha256(overrides)
        default_run_name += f"_override_{override_sha256[:10]}"
    run_name = args.final_run_name or default_run_name
    isolate_run = is_experimental or args.final_run_name is not None
    output_dir = output_dir_for(
        args,
        run_name=run_name,
        isolate_run=isolate_run,
    )
    checkpoint_dir = (
        args.checkpoint_root.expanduser().resolve() / search.slugify(run_name)
    )
    manifest_path = output_dir / FINAL_MODEL_MANIFEST_NAME
    lock_path = output_dir / FINAL_MODEL_LOCK_NAME
    if manifest_path.exists() or lock_path.exists():
        raise FileExistsError(
            f"Final model is already frozen under {output_dir}; refusing to replace it."
        )

    train_command = search.training_command(args, params, checkpoint_dir, rules)
    fixed_prior_overrides = overrides.get("fixed_prior_overrides", {})
    family_probs = fixed_prior_overrides.get(
        "pitting_composition_family_probs"
    )
    if family_probs is not None:
        replace_option_values(
            train_command,
            "--pitting_composition_family_probs",
            family_probs,
            expected_count=5,
        )
    replace_option(train_command, "--save_temp_every", args.save_temp_every)
    replace_option(train_command, "--save_perm_every", args.save_perm_every)
    training_configuration = {
        "study_name": args.study_name,
        "storage": args.storage,
        "trial_number": int(trial.number),
        "trial_value": float(trial.value),
        "trial_params": search.trial_params_payload(params),
        "selected_trial_params": search.trial_params_payload(selected_params),
        "effective_trial_params": search.trial_params_payload(params),
        "overrides": overrides,
        "experimental_override_run": is_experimental,
        "split_manifest_sha256": frozen_split.manifest_sha256,
        "split_lock_sha256": frozen_split.lock_sha256,
        "target_rule_summary_sha256": rules.summary_sha256,
        "checkpoint_dir": str(checkpoint_dir),
        "max_steps": args.max_steps,
        "scheduler_total_steps": args.scheduler_total_steps,
        "save_temp_every": args.save_temp_every,
        "save_perm_every": args.save_perm_every,
        "checkpoint_selection_min_step": args.checkpoint_selection_min_step,
        "checkpoint_selection_interval": args.checkpoint_selection_interval,
        "train_command": train_command,
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Final output directory is non-empty: {output_dir}")
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise FileExistsError(
            f"Final checkpoint directory is non-empty: {checkpoint_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    search.original.search_utils.run_command(
        train_command,
        cwd=REPO_ROOT,
        log_path=output_dir / "train.log",
    )

    evaluation_root = output_dir / "development_checkpoint_evaluations"
    records = [
        evaluation_record(
            args,
            params,
            checkpoint=checkpoint,
            output_dir=evaluation_root / f"step_{checkpoint_step(checkpoint):05d}",
            split_manifest=args.split_manifest,
            rules=rules,
        )
        for checkpoint in checkpoint_candidates(args, checkpoint_dir)
    ]
    selected = max(records, key=lambda record: float(record["mean_test_spearman"]))
    selected_checkpoint = Path(selected["checkpoint_path"])
    evaluation_configuration = {
        "task_id": PITTING_TASK_ID,
        "device": args.device,
        "random_state": 42,
        "n_estimators": args.eval_n_estimators,
        "tabicl_feat_shuffle_method": "none",
        "tabicl_norm_methods": ["none"],
        "regression_output": "median",
        "regression_uncertainty": False,
        "pitting_magpie_features": bool(params.use_magpie),
        "expected_n_features": 31 if params.use_magpie else 21,
        "max_samples_per_task": 0,
    }
    manifest = {
        "schema_version": "epit_final_model_manifest_v1",
        "frozen_utc": utc_now(),
        "study": {
            "name": args.study_name,
            "storage": args.storage,
            "total_trials": len(study.trials),
            "selection_mode": (
                "explicit_current_best_completed_trial"
                if args.selected_trial_number is not None
                else "automatic_current_best_completed_trial"
            ),
            "unfinished_trials_at_selection": unfinished_trials_at_selection,
            "selected_trial_number": int(trial.number),
            "selected_trial_value": float(trial.value),
            "selected_trial_params": search.trial_params_payload(
                selected_params
            ),
            "selected_trial_user_attrs": dict(trial.user_attrs),
            "selected_trial_fold_evaluation": trial_eval,
            "pipeline_fingerprint": study_fingerprint,
            "pipeline_fingerprint_sha256": study_fingerprint_sha256,
        },
        "split": {
            "manifest": str(frozen_split.manifest_path),
            "manifest_sha256": frozen_split.manifest_sha256,
            "lock": str(frozen_split.lock_path),
            "lock_sha256": frozen_split.lock_sha256,
            "source_sha256": rules.source_sha256,
            "development_rows": int(
                frozen_split.manifest["split_design"]["development_rows"]
            ),
            "final_test_rows": int(
                frozen_split.manifest["split_design"]["final_test_rows"]
            ),
        },
        "target_rules": {
            "summary": rules.summary_path,
            "summary_sha256": rules.summary_sha256,
            "scores": rules.scores,
            "probabilities": rules.probabilities,
            "coefficients": rules.coefficients,
            "artifacts": rules.artifacts,
            "artifact_sha256s": rules.artifact_sha256s,
        },
        "training": training_configuration,
        "checkpoint_selection": {
            "data": "five frozen development folds only",
            "primary_metric": "mean_test_spearman",
            "candidates": records,
        },
        "selected_checkpoint": {
            "path": str(selected_checkpoint),
            "sha256": selected["checkpoint_sha256"],
            "step": selected["step"],
            "mean_development_fold_spearman": selected[
                "mean_test_spearman"
            ],
            "mean_development_fold_mae": selected["mean_test_mae"],
        },
        "evaluation_configuration": evaluation_configuration,
        "final_test_rows_used": False,
    }
    write_json(manifest_path, manifest, exclusive=True)
    lock = build_final_model_lock(
        manifest_path=manifest_path,
        immutable_artifacts={"selected_checkpoint": selected_checkpoint},
    )
    write_json(lock_path, lock, exclusive=True)
    print(f"Frozen final checkpoint: {selected_checkpoint}")
    print(f"Development-fold mean Spearman: {selected['mean_test_spearman']:.6g}")
    print(f"Final-model manifest: {manifest_path}")
    print(f"Final-model lock: {lock_path}")
    return manifest_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
