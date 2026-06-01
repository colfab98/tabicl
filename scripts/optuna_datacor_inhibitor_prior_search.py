#!/usr/bin/env python
"""Tune inhibitor-efficiency informed-prior parameters for DATACOR aluminum inhibitors.

Each trial trains one TabICL regression checkpoint, evaluates it on the DATACOR
aluminum-inhibitor efficiency task with grouped molecular-descriptor splits, and
optimizes mean Spearman. The inhibitor-agent block allocation is fixed from the
retained DATACOR model-input table shape: 1 material, 1 environment, and 14
molecular-descriptor features out of 16 total features.
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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATACOR_DATASET = "datacor_aluminum_inhibitors"
DATACOR_TASK_ID = "datacor_aluminum_inhibitors__datacor_long_efficiency__efficiency_score"
PRETRAINED_LABEL = "pretrained_tabicl_v2"
DEFAULT_SPLIT_SEEDS = (11, 22, 33, 44, 55)

# DATACOR retained features: alloy=1, pH=1, molecular descriptors=14.
DATACOR_INHIBITOR_BLOCK_ALLOCATION = (
    0.0625,  # material
    0.0625,  # environment
    0.0,     # process_history
    0.0,     # exposure_duration
    0.0,     # temporal_history
    0.0,     # direct_intervention; DATACOR has no dose/control column
    0.875,   # molecular_descriptor
    0.0,     # electrochem_control
    0.0,     # electrochem_downstream
)


@dataclass(frozen=True)
class TrialParams:
    informed_prior_ratio: float
    informed_feature_block_strength: float
    informed_interaction_strength: float
    informed_intervention_strength: float
    informed_physical_marginal_prob: float


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
    parser.add_argument("--study-name", type=str, default="datacor_inhibitor_prior_search")
    parser.add_argument("--storage", type=str, default=None, help="Optional Optuna storage URL, e.g. sqlite:///study.db.")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--base-run-name", type=str, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "corrosion_datasets" / "analysis" / "datacor_inhibitor_prior_search",
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
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--batch-size-per-gp", type=int, default=4)
    parser.add_argument("--min-features", type=int, default=16)
    parser.add_argument("--max-features", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing checkpoints/evals.")
    parser.add_argument("--dry-run", action="store_true", help="Write commands/metadata without running training or eval.")
    return parser.parse_args()


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    return "_".join(part for part in cleaned.split("_") if part) or "run"


def default_base_run_name(study_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tabicl_datacor_inhibitor_prior_{slugify(study_name)}_{stamp}"


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
    informed_prior_ratio = trial.suggest_categorical("informed_prior_ratio", [0.25, 0.50, 0.75, 1.00])
    informed_feature_block_strength = trial.suggest_float("informed_feature_block_strength", 0.00, 0.65)
    informed_interaction_strength = trial.suggest_float("informed_interaction_strength", 0.00, 0.60)
    informed_intervention_strength = trial.suggest_float("informed_intervention_strength", 0.20, 1.00)
    informed_physical_marginal_prob = trial.suggest_categorical(
        "informed_physical_marginal_prob",
        [0.25, 0.50, 0.75, 1.00],
    )

    return TrialParams(
        informed_prior_ratio=float(informed_prior_ratio),
        informed_feature_block_strength=float(informed_feature_block_strength),
        informed_interaction_strength=float(informed_interaction_strength),
        informed_intervention_strength=float(informed_intervention_strength),
        informed_physical_marginal_prob=float(informed_physical_marginal_prob),
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
        "DatacorInhibitorPriorSearch",
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
        str(args.batch_size),
        "--micro_batch_size",
        str(args.micro_batch_size),
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
        "0.0",
        "1.0",
        "--informed_inhibitor_block_allocation",
        *(format_float(value) for value in DATACOR_INHIBITOR_BLOCK_ALLOCATION),
        "--informed_feature_block_strength",
        format_float(params.informed_feature_block_strength),
        "--informed_interaction_strength",
        format_float(params.informed_interaction_strength),
        "--informed_history_strength",
        "0.0",
        "--informed_intervention_strength",
        format_float(params.informed_intervention_strength),
        "--informed_target_family",
        "inhibitor_efficiency",
        "--informed_physical_marginal_prob",
        format_float(params.informed_physical_marginal_prob),
        "--informed_physical_marginal_profile",
        "inhibitor_efficiency_v1",
        "--prior_device",
        "cpu",
        "--prior_n_jobs",
        str(args.prior_n_jobs),
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--dataloader_prefetch_factor",
        str(args.dataloader_prefetch_factor),
        "--batch_size_per_gp",
        str(args.batch_size_per_gp),
        "--min_features",
        str(args.min_features),
        "--max_features",
        str(args.max_features),
        "--max_classes",
        "0",
        "--num_quantiles",
        "999",
        "--max_seq_len",
        str(args.max_seq_len),
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
        str(REPO_ROOT / "scripts" / "eval_corrosion_datasets.py"),
        "--local-ckpt-path",
        str(checkpoint_path),
        "--local-model-label",
        model_label,
        "--dataset",
        DATACOR_DATASET,
        "--task",
        DATACOR_TASK_ID,
        "--target-mode",
        "primary",
        "--target-binning",
        "continuous",
        "--split-seeds",
        *(str(seed) for seed in args.split_seeds),
        "--device",
        args.device,
        "--n-estimators",
        str(args.eval_n_estimators),
        "--no-regression-uncertainty",
        "--compare-pretrained-tabicl",
        "--output-json",
        str(output_dir / "results.json"),
        "--output-csv",
        str(output_dir / "rows.csv"),
        "--output-wide-csv",
        str(output_dir / "wide.csv"),
        "--output-summary-csv",
        str(output_dir / "summary.csv"),
        "--no-checkpoint-plots",
    ]


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write(" ".join(command) + "\n\n")
        log_file.flush()
        subprocess.run(command, cwd=cwd, stdout=log_file, stderr=subprocess.STDOUT, check=True)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def load_eval_metrics(summary_csv: Path, rows_csv: Path, model_label: str) -> dict[str, Any]:
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Evaluation summary not found: {summary_csv}")
    summary_df = pd.read_csv(summary_csv)
    summary_rows = summary_df[summary_df["model"].astype(str) == model_label]
    if summary_rows.empty:
        raise RuntimeError(f"Model {model_label!r} not found in {summary_csv}")
    summary = summary_rows.iloc[0].to_dict()

    metrics: dict[str, Any] = {
        "mean_test_spearman": safe_float(summary.get("mean_test_spearman")),
        "median_test_spearman": safe_float(summary.get("median_test_spearman")),
        "mean_test_mae": safe_float(summary.get("mean_test_mae")),
        "median_test_mae": safe_float(summary.get("median_test_mae")),
        "mean_test_nmae_iqr": safe_float(summary.get("mean_test_nmae_iqr")),
        "median_test_nmae_iqr": safe_float(summary.get("median_test_nmae_iqr")),
        "mean_test_rmse": safe_float(summary.get("mean_test_rmse")),
        "mean_test_r2": safe_float(summary.get("mean_test_r2")),
        "mean_delta_test_spearman_vs_pretrained_tabicl_v2": safe_float(
            summary.get("mean_delta_test_spearman_vs_pretrained_tabicl_v2")
        ),
        "mean_delta_test_mae_vs_pretrained_tabicl_v2": safe_float(
            summary.get("mean_delta_test_mae_vs_pretrained_tabicl_v2")
        ),
        "summary_wins_test_spearman": safe_float(summary.get("wins_test_spearman")),
    }

    if rows_csv.is_file():
        rows_df = pd.read_csv(rows_csv)
        local_rows = rows_df[rows_df["model"].astype(str) == model_label].reset_index(drop=True)
        pretrained_rows = rows_df[rows_df["model"].astype(str) == PRETRAINED_LABEL].reset_index(drop=True)
        if "test_spearman" in local_rows and len(local_rows) > 1:
            metrics["std_test_spearman"] = float(local_rows["test_spearman"].astype(float).std(ddof=0))
        else:
            metrics["std_test_spearman"] = math.nan
        if "test_spearman" in local_rows and "test_spearman" in pretrained_rows:
            n_pairs = min(len(local_rows), len(pretrained_rows))
            if n_pairs > 0:
                local_values = local_rows.loc[: n_pairs - 1, "test_spearman"].astype(float).to_numpy()
                pretrained_values = pretrained_rows.loc[: n_pairs - 1, "test_spearman"].astype(float).to_numpy()
                metrics["split_wins_test_spearman_vs_pretrained_tabicl_v2"] = int(
                    (local_values > pretrained_values).sum()
                )
                metrics["n_split_pairs_vs_pretrained_tabicl_v2"] = int(n_pairs)
            else:
                metrics["split_wins_test_spearman_vs_pretrained_tabicl_v2"] = math.nan
                metrics["n_split_pairs_vs_pretrained_tabicl_v2"] = 0
    else:
        metrics["std_test_spearman"] = math.nan
        metrics["split_wins_test_spearman_vs_pretrained_tabicl_v2"] = math.nan
        metrics["n_split_pairs_vs_pretrained_tabicl_v2"] = 0

    return metrics


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
    eval_dir = study_output_root(args.eval_output_root, args.study_name) / f"datacor_grouped_splits_{trial_name}"
    model_label = f"trial_{trial_number:04d}"

    train_cmd = training_command(args, params, checkpoint_dir)
    eval_cmd = eval_command(args, checkpoint_path, model_label, eval_dir)
    metadata = {
        "trial_number": trial_number,
        "trial_name": trial_name,
        "params": asdict(params),
        "fixed_inhibitor_block_allocation": DATACOR_INHIBITOR_BLOCK_ALLOCATION,
        "dataset": DATACOR_DATASET,
        "task_id": DATACOR_TASK_ID,
        "split_strategy": "grouped_molecular_descriptor",
        "split_seeds": list(args.split_seeds),
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
            "status": "dry_run",
            "mean_test_spearman": math.nan,
            "median_test_spearman": math.nan,
            "std_test_spearman": math.nan,
            "mean_test_mae": math.nan,
            "mean_test_nmae_iqr": math.nan,
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

    metrics = load_eval_metrics(eval_dir / "summary.csv", eval_dir / "rows.csv", model_label)
    result = {
        **metadata,
        "status": "completed",
        **metrics,
        "summary_csv": str(eval_dir / "summary.csv"),
        "rows_csv": str(eval_dir / "rows.csv"),
        "wide_csv": str(eval_dir / "wide.csv"),
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
            "median_test_spearman",
            "std_test_spearman",
            "mean_test_mae",
            "median_test_mae",
            "mean_test_nmae_iqr",
            "median_test_nmae_iqr",
            "mean_test_rmse",
            "mean_test_r2",
            "mean_delta_test_spearman_vs_pretrained_tabicl_v2",
            "mean_delta_test_mae_vs_pretrained_tabicl_v2",
            "split_wins_test_spearman_vs_pretrained_tabicl_v2",
            "n_split_pairs_vs_pretrained_tabicl_v2",
            "summary_csv",
            "rows_csv",
            "wide_csv",
        ]:
            if key in result:
                trial.set_user_attr(key, result[key])
        return float(result["mean_test_spearman"])

    study.optimize(objective, n_trials=args.n_trials)
    print("Best trial:")
    print(f"  number: {study.best_trial.number}")
    print(f"  value mean_test_spearman: {study.best_value:.6g}")
    print(f"  params: {study.best_trial.params}")
    print(f"  mean_test_mae: {study.best_trial.user_attrs.get('mean_test_mae')}")
    print(f"  mean_test_nmae_iqr: {study.best_trial.user_attrs.get('mean_test_nmae_iqr')}")
    print(
        "  split wins vs pretrained: "
        f"{study.best_trial.user_attrs.get('split_wins_test_spearman_vs_pretrained_tabicl_v2')}"
    )


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
            "median_test_spearman": result.get("median_test_spearman", math.nan),
            "std_test_spearman": result.get("std_test_spearman", math.nan),
            "mean_test_mae": result.get("mean_test_mae", math.nan),
            "mean_test_nmae_iqr": result.get("mean_test_nmae_iqr", math.nan),
            "split_wins_test_spearman_vs_pretrained_tabicl_v2": result.get(
                "split_wins_test_spearman_vs_pretrained_tabicl_v2", math.nan
            ),
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
        print(f"  mean_test_nmae_iqr: {best.get('mean_test_nmae_iqr')}")
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
    if args.min_features <= 0 or args.max_features < args.min_features:
        raise ValueError("Require 0 < --min-features <= --max-features.")

    if args.backend == "optuna":
        run_optuna(args)
    else:
        run_random(args)


if __name__ == "__main__":
    main()
