#!/usr/bin/env python
"""Tune corrosion-informed prior parameters for the alloy pitting task.

Each trial trains one TabICL regression checkpoint to step 1000, evaluates that
checkpoint with repeated pitting-potential splits, and scores mean Spearman.
The script is designed to be launched from a Slurm job that owns the GPUs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_SEEDS = (1001, 1002, 1003, 1004, 1005)
DEFAULT_INHIBITOR_ALLOCATION = (0.05, 0.08, 0.02, 0.02, 0.0, 0.05, 0.78, 0.0, 0.0)


@dataclass(frozen=True)
class TrialParams:
    informed_prior_ratio: float
    informed_feature_block_strength: float
    informed_interaction_strength: float
    informed_physical_marginal_prob: float
    alloc_material: float
    alloc_environment: float
    alloc_process_history: float

    @property
    def normal_block_allocation(self) -> tuple[float, ...]:
        return (
            self.alloc_material,
            self.alloc_environment,
            self.alloc_process_history,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )


class SuggestTrial(Protocol):
    def suggest_categorical(self, name: str, choices: list[float]) -> float: ...

    def suggest_float(self, name: str, low: float, high: float) -> float: ...


class RandomTrial:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.params: dict[str, float] = {}

    def suggest_categorical(self, name: str, choices: list[float]) -> float:
        value = float(self.rng.choice(choices))
        self.params[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float) -> float:
        value = float(self.rng.uniform(low, high))
        self.params[name] = value
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("optuna", "random"), default="optuna")
    parser.add_argument("--study-name", type=str, default="pitting_prior_search")
    parser.add_argument("--storage", type=str, default=None, help="Optional Optuna storage URL, e.g. sqlite:///study.db.")
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--base-run-name", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "corrosion_datasets" / "analysis" / "pitting_prior_search",
    )
    parser.add_argument(
        "--eval-output-root",
        type=Path,
        default=REPO_ROOT / "corrosion_datasets" / "analysis" / "eval_results",
    )
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(DEFAULT_SPLIT_SEEDS))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--scheduler-total-steps",
        type=int,
        default=None,
        help="LR scheduler horizon passed as --scheduler_total_steps to training. Defaults to --max-steps.",
    )
    parser.add_argument("--np-seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--prior-n-jobs", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=4)
    parser.add_argument("--eval-n-estimators", type=int, default=8)
    parser.add_argument(
        "--physical-profile",
        type=str,
        default="pitting_potential_v1",
        help="Informed physical marginal profile passed to training.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing checkpoints/evals.")
    parser.add_argument("--dry-run", action="store_true", help="Write commands/metadata without running training or eval.")
    return parser.parse_args()


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    return "_".join(part for part in cleaned.split("_") if part) or "run"


def default_base_run_name(study_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tabicl_pitting_prior_{slugify(study_name)}_{stamp}"


def study_output_root(root: Path, study_name: str) -> Path:
    return root.expanduser().resolve() / slugify(study_name)


def build_optuna_storage(storage: str | None) -> Any:
    if storage is None or not storage.startswith("journal://"):
        return storage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock, JournalStorage

    journal_path = Path(storage.removeprefix("journal://")).expanduser().resolve()
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    lock = JournalFileOpenLock(str(journal_path), grace_period=120)
    return JournalStorage(JournalFileBackend(str(journal_path), lock_obj=lock))


def sample_params(trial: SuggestTrial) -> TrialParams:
    informed_prior_ratio = trial.suggest_categorical("informed_prior_ratio", [0.75, 1.00])
    informed_feature_block_strength = trial.suggest_float("informed_feature_block_strength", 0.15, 0.45)
    informed_interaction_strength = trial.suggest_float("informed_interaction_strength", 0.25, 0.55)
    informed_physical_marginal_prob = trial.suggest_categorical(
        "informed_physical_marginal_prob",
        [0.25, 0.50, 0.75, 1.00],
    )

    alloc_material = trial.suggest_float("alloc_material", 0.55, 0.85)
    process_low = max(0.0, 0.65 - alloc_material)
    process_high = min(0.15, 0.90 - alloc_material)
    alloc_process_history = trial.suggest_float("alloc_process_history", process_low, process_high)
    alloc_environment = 1.0 - alloc_material - alloc_process_history

    return TrialParams(
        informed_prior_ratio=float(informed_prior_ratio),
        informed_feature_block_strength=float(informed_feature_block_strength),
        informed_interaction_strength=float(informed_interaction_strength),
        informed_physical_marginal_prob=float(informed_physical_marginal_prob),
        alloc_material=float(alloc_material),
        alloc_environment=float(alloc_environment),
        alloc_process_history=float(alloc_process_history),
    )


def format_float(value: float) -> str:
    return f"{value:.10g}"


def training_command(args: argparse.Namespace, params: TrialParams, checkpoint_dir: Path) -> list[str]:
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(REPO_ROOT / "src" / "tabicl" / "train" / "run.py"),
        "--wandb_log",
        "False",
        "--wandb_project",
        "TabICL",
        "--wandb_name",
        "PittingPriorSearch",
        "--wandb_dir",
        "/home/fcolanto/wandb",
        "--wandb_mode",
        "disabled",
        "--device",
        args.device,
        "--dtype",
        "float32",
        "--np_seed",
        str(args.np_seed),
        "--torch_seed",
        str(args.torch_seed),
        "--max_steps",
        str(args.max_steps),
        "--batch_size",
        "512",
        "--micro_batch_size",
        "4",
        "--lr",
        "1e-4",
        "--scheduler",
        "cosine_warmup",
        "--warmup_proportion",
        "0.02",
        "--gradient_clipping",
        "1.0",
        "--prior_type",
        "hybrid_scm",
        "--informed_prior_ratio",
        format_float(params.informed_prior_ratio),
        "--mix_probs",
        "0.7",
        "0.3",
        "--informed_mix_probs",
        "0.7",
        "0.3",
        "--informed_task_family_probs",
        "1.0",
        "0.0",
        "--informed_normal_block_allocation",
        *(format_float(value) for value in params.normal_block_allocation),
        "--informed_inhibitor_block_allocation",
        *(format_float(value) for value in DEFAULT_INHIBITOR_ALLOCATION),
        "--informed_feature_block_strength",
        format_float(params.informed_feature_block_strength),
        "--informed_interaction_strength",
        format_float(params.informed_interaction_strength),
        "--informed_history_strength",
        "0.0",
        "--informed_intervention_strength",
        "0.0",
        "--informed_target_family",
        "pitting_potential",
        "--informed_physical_marginal_prob",
        format_float(params.informed_physical_marginal_prob),
        "--informed_physical_marginal_profile",
        args.physical_profile,
        "--prior_device",
        "cpu",
        "--prior_n_jobs",
        str(args.prior_n_jobs),
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--dataloader_prefetch_factor",
        str(args.dataloader_prefetch_factor),
        "--batch_size_per_gp",
        "4",
        "--min_features",
        "2",
        "--max_features",
        "100",
        "--max_classes",
        "0",
        "--num_quantiles",
        "999",
        "--max_seq_len",
        "1024",
        "--min_train_size",
        "0.1",
        "--max_train_size",
        "0.9",
        "--embed_dim",
        "128",
        "--col_num_blocks",
        "3",
        "--col_nhead",
        "4",
        "--col_num_inds",
        "128",
        "--row_num_blocks",
        "3",
        "--row_nhead",
        "8",
        "--row_num_cls",
        "4",
        "--row_rope_base",
        "100000",
        "--icl_num_blocks",
        "12",
        "--icl_nhead",
        "4",
        "--ff_factor",
        "2",
        "--norm_first",
        "True",
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--save_temp_every",
        str(args.max_steps),
        "--save_perm_every",
        str(args.max_steps),
    ]
    if args.scheduler_total_steps is not None:
        command.extend(["--scheduler_total_steps", str(args.scheduler_total_steps)])
    return command


def eval_command(args: argparse.Namespace, checkpoint_path: Path, model_label: str, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_pitting_repeated_splits.py"),
        "--local-ckpt-path",
        str(checkpoint_path),
        "--model-label",
        model_label,
        "--split-seeds",
        *(str(seed) for seed in args.split_seeds),
        "--device",
        args.device,
        "--n-estimators",
        str(args.eval_n_estimators),
        "--output-dir",
        str(output_dir),
    ]


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write(" ".join(command) + "\n\n")
        log_file.flush()
        subprocess.run(command, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT, check=True)


def load_eval_summary(summary_csv: Path, model_label: str) -> dict[str, Any]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Evaluation summary not found: {summary_csv}")
    df = pd.read_csv(summary_csv)
    rows = df[df["model"].astype(str) == model_label]
    if rows.empty:
        raise RuntimeError(f"Model {model_label!r} not found in {summary_csv}")
    return rows.iloc[0].to_dict()


def write_trial_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_random_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_trial(args: argparse.Namespace, trial_number: int, params: TrialParams) -> dict[str, Any]:
    base_run_name = args.base_run_name or default_base_run_name(args.study_name)
    trial_name = f"{base_run_name}_trial_{trial_number:04d}"
    checkpoint_dir = study_output_root(args.checkpoint_root, args.study_name) / trial_name
    checkpoint_path = checkpoint_dir / f"step-{args.max_steps}.ckpt"
    trial_dir = study_output_root(args.work_dir, args.study_name) / slugify(base_run_name) / f"trial_{trial_number:04d}"
    eval_dir = study_output_root(args.eval_output_root, args.study_name) / f"pitting_repeated_splits_{trial_name}"
    model_label = f"trial_{trial_number:04d}"

    train_cmd = training_command(args, params, checkpoint_dir)
    eval_cmd = eval_command(args, checkpoint_path, model_label, eval_dir)
    metadata = {
        "trial_number": trial_number,
        "trial_name": trial_name,
        "params": asdict(params),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": str(checkpoint_path),
        "eval_dir": str(eval_dir),
        "train_command": train_cmd,
        "eval_command": eval_cmd,
    }
    write_trial_metadata(trial_dir / "trial_config.json", metadata)

    if args.dry_run:
        return {
            **metadata,
            "mean_test_spearman": math.nan,
            "mean_test_mae": math.nan,
            "status": "dry_run",
        }

    if not (args.skip_existing and checkpoint_path.is_file()):
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise FileExistsError(
                f"Checkpoint directory already exists and is non-empty: {checkpoint_dir}. "
                "Use --skip-existing to reuse existing checkpoints."
            )
        run_command(train_cmd, cwd=REPO_ROOT, log_path=trial_dir / "train.log")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Expected checkpoint was not created: {checkpoint_path}")

    if not (args.skip_existing and (eval_dir / "summary.csv").is_file()):
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        run_command(eval_cmd, cwd=REPO_ROOT, log_path=trial_dir / "eval.log")

    summary = load_eval_summary(eval_dir / "summary.csv", model_label)
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
        "summary_csv": str(eval_dir / "summary.csv"),
        "rows_csv": str(eval_dir / "rows.csv"),
    }
    write_trial_metadata(trial_dir / "trial_result.json", result)
    return result


def run_optuna(args: argparse.Namespace) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed in this virtualenv. Install optuna or run with --backend random."
        ) from exc

    base_run_name = args.base_run_name or default_base_run_name(args.study_name)
    args.base_run_name = base_run_name
    study = optuna.create_study(
        study_name=args.study_name,
        storage=build_optuna_storage(args.storage),
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial: Any) -> float:
        params = sample_params(trial)
        result = run_trial(args, int(trial.number), params)
        for key in [
            "mean_test_mae",
            "median_test_mae",
            "mean_test_rmse",
            "mean_test_r2",
            "median_test_spearman",
            "std_test_spearman",
            "summary_csv",
            "rows_csv",
        ]:
            trial.set_user_attr(key, result[key])
        return float(result["mean_test_spearman"])

    study.optimize(objective, n_trials=args.n_trials)
    print("Best trial:")
    print(f"  number: {study.best_trial.number}")
    print(f"  value mean_test_spearman: {study.best_value:.6g}")
    print(f"  params: {study.best_trial.params}")
    print(f"  mean_test_mae: {study.best_trial.user_attrs.get('mean_test_mae')}")


def run_random(args: argparse.Namespace) -> None:
    base_run_name = args.base_run_name or default_base_run_name(args.study_name)
    args.base_run_name = base_run_name
    rng = random.Random(args.random_seed)
    results_csv = study_output_root(args.work_dir, args.study_name) / slugify(base_run_name) / "random_results.csv"
    best: dict[str, Any] | None = None

    for trial_number in range(args.n_trials):
        random_trial = RandomTrial(rng)
        params = sample_params(random_trial)
        result = run_trial(args, trial_number, params)
        flat = {
            "trial_number": trial_number,
            "status": result["status"],
            "mean_test_spearman": result["mean_test_spearman"],
            "mean_test_mae": result["mean_test_mae"],
            **asdict(params),
            "summary_csv": result.get("summary_csv", ""),
        }
        append_random_result(results_csv, flat)
        if result["status"] == "completed" and (best is None or result["mean_test_spearman"] > best["mean_test_spearman"]):
            best = result

    if best is not None:
        print("Best random trial:")
        print(f"  mean_test_spearman: {best['mean_test_spearman']:.6g}")
        print(f"  mean_test_mae: {best['mean_test_mae']:.6g}")
        print(f"  trial_name: {best['trial_name']}")
    print(f"Wrote random-search results to {results_csv}")


def main() -> None:
    args = parse_args()
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive.")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if args.scheduler_total_steps is not None and args.scheduler_total_steps < args.max_steps:
        raise ValueError("--scheduler-total-steps should be >= --max-steps for proxy runs.")
    if not args.split_seeds:
        raise ValueError("--split-seeds must contain at least one seed.")

    if args.backend == "optuna":
        run_optuna(args)
    else:
        run_random(args)


if __name__ == "__main__":
    main()
