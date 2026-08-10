#!/usr/bin/env python
"""Stage 4: train the selected EPIT configuration once and freeze it."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-name", default="epit_pipeline_optuna_v1")
    parser.add_argument(
        "--storage",
        required=True,
        help="The completed Stage 3 Optuna storage URL (including journal://).",
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
    active = [
        trial.number
        for trial in study.trials
        if trial.state in (TrialState.RUNNING, TrialState.WAITING)
    ]
    if active:
        raise RuntimeError(
            "Stage 3 is still active; final training cannot freeze a winner. "
            f"Active trials: {active}"
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
    return study, study.best_trial


def verify_selected_trial_evaluation(
    trial: Any,
    *,
    split_manifest: Path,
    source_sha256: str,
) -> dict[str, Any]:
    summary_csv = trial.user_attrs.get("summary_csv")
    if not summary_csv:
        raise RuntimeError("Selected trial has no saved fold-evaluation summary.")
    eval_dir = Path(str(summary_csv)).expanduser().resolve().parent
    payload = search.verify_no_power_fold_evaluation(eval_dir)
    if not bool(payload.get("development_rows_only", False)):
        raise RuntimeError("Selected trial was not evaluated development-only.")
    if payload.get("validation_folds") != list(search.FIXED_VALIDATION_FOLDS):
        raise RuntimeError("Selected trial did not use all five frozen folds.")
    if Path(str(payload.get("split_manifest", ""))).resolve() != split_manifest:
        raise RuntimeError("Selected trial used a different split manifest.")
    if str(payload.get("source_sha256", "")) != source_sha256:
        raise RuntimeError("Selected trial evaluation used a different dataset.")
    if payload.get("settings", {}).get("tabicl_feat_shuffle_method") != "none":
        raise RuntimeError("Selected trial changed the fixed feature order.")
    return payload


def replace_option(command: list[str], option: str, value: Any) -> None:
    index = command.index(option)
    command[index + 1] = str(value)


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
    source = payload.get("source", {})
    if Path(str(source.get("local_ckpt_path", ""))).resolve() != checkpoint:
        raise RuntimeError("Checkpoint evaluation provenance names another model.")
    settings = payload.get("settings", {})
    if int(settings.get("n_estimators", -1)) != n_estimators:
        raise RuntimeError("Checkpoint evaluation changed n_estimators.")
    if settings.get("tabicl_feat_shuffle_method") != "none":
        raise RuntimeError("Checkpoint evaluation changed the feature order.")


def evaluation_record(
    args: argparse.Namespace,
    params: search.TrialParams,
    *,
    checkpoint: Path,
    output_dir: Path,
    split_manifest: Path,
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


def output_dir_for(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    return (DEFAULT_FINAL_ROOT / search.slugify(args.study_name)).resolve()


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
    trial_eval = verify_selected_trial_evaluation(
        trial,
        split_manifest=args.split_manifest,
        source_sha256=rules.source_sha256,
    )
    params = search.trial_params_from_mapping(dict(trial.params))
    run_name = args.final_run_name or (
        f"final_{search.slugify(args.study_name)}_trial_{trial.number:04d}"
    )
    output_dir = output_dir_for(args)
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
    replace_option(train_command, "--save_temp_every", args.save_temp_every)
    replace_option(train_command, "--save_perm_every", args.save_perm_every)
    training_configuration = {
        "study_name": args.study_name,
        "storage": args.storage,
        "trial_number": int(trial.number),
        "trial_value": float(trial.value),
        "trial_params": asdict(params),
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
            "selected_trial_number": int(trial.number),
            "selected_trial_value": float(trial.value),
            "selected_trial_params": asdict(params),
            "selected_trial_user_attrs": dict(trial.user_attrs),
            "selected_trial_fold_evaluation": trial_eval,
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
