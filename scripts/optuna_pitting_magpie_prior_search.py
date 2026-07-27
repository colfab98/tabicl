#!/usr/bin/env python
"""Conditional Optuna study for composition-only EPIT priors with optional Magpie features.

The study keeps the successful fixed 17/3/1 legacy schema and changes only the
explicit search dimensions below.  Dirichlet concentration and active
probability are suggested only when the Dirichlet branch is enabled; inactive
parameters therefore cannot influence Optuna's search model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

try:
    from scripts import optuna_pitting_fixed_pren_prior_search as search_utils
except ImportError:
    import optuna_pitting_fixed_pren_prior_search as search_utils
from tabicl.prior.magpie_features import (
    EPIT_BASE_FEATURE_COUNT,
    EPIT_MAGPIE_DESCRIPTOR_NAMES,
    EPIT_MAGPIE_RANGE_MIN_ATOMIC_FRACTION,
    EPIT_MAGPIE_TOTAL_FEATURE_COUNT,
    EPIT_MAGPIE_VERSION,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV_SPLIT_SEEDS = (2001, 2002, 2003)
RESERVED_FINAL_SPLIT_SEEDS = (3001, 3002, 3003, 3004, 3005)
FIXED_BLOCK_ALLOCATION = (17, 3, 1, 0, 0, 0, 0, 0, 0)
DEFAULT_INHIBITOR_ALLOCATION = (0.05, 0.08, 0.02, 0.02, 0.0, 0.05, 0.78, 0.0, 0.0)
MLP_PROB_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
DIRICHLET_PROB_GRID = (0.25, 0.50, 0.75, 1.00)
DIRICHLET_CONCENTRATION_GRID = (0.10, 0.25, 0.50, 1.00, 2.00, 5.00)
DIRICHLET_ACTIVE_PROB_GRID = (0.20, 0.25, 0.35, 0.50, 0.70)
CONTROL_BRANCHES = (
    {"use_magpie": False, "use_dirichlet": False},
    {"use_magpie": False, "use_dirichlet": True},
    {"use_magpie": True, "use_dirichlet": False},
    {"use_magpie": True, "use_dirichlet": True},
)


@dataclass(frozen=True)
class TrialParams:
    use_magpie: bool
    use_dirichlet: bool
    mlp_prob: float
    informed_feature_block_strength: float
    informed_target_mix_weight: float
    pitting_material_dirichlet_prob: float
    pitting_material_dirichlet_concentration: float | None
    pitting_material_dirichlet_active_prob: float | None
    epit_material_coef: float
    epit_environment_coef: float
    epit_interaction_coef: float


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("optuna", "random"), default="optuna")
    parser.add_argument("--study-name", default="pitting_magpie_conditional_v1")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--base-run-name", default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "corrosion_datasets" / "analysis" / "pitting_magpie_prior_search",
    )
    parser.add_argument(
        "--eval-output-root",
        type=Path,
        default=REPO_ROOT / "corrosion_datasets" / "analysis" / "eval_results",
    )
    parser.add_argument(
        "--split-seeds",
        "--dev-split-seeds",
        dest="split_seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_DEV_SPLIT_SEEDS),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--scheduler-total-steps", type=int, default=8000)
    parser.add_argument("--np-seed", type=int, default=42)
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--prior-n-jobs", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=4)
    parser.add_argument("--eval-n-estimators", type=int, default=8)
    parser.add_argument(
        "--enqueue-controls",
        action="store_true",
        help="Ensure one initial queued trial for every Magpie/Dirichlet on/off branch.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def slugify(text: str) -> str:
    return search_utils.slugify(text)


def default_base_run_name(study_name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tabicl_pitting_magpie_{slugify(study_name)}_{stamp}"


def sample_params(trial: SuggestTrial) -> TrialParams:
    use_magpie = bool(trial.suggest_categorical("use_magpie", [False, True]))
    use_dirichlet = bool(trial.suggest_categorical("use_dirichlet", [False, True]))
    mlp_prob = float(trial.suggest_categorical("mlp_prob", list(MLP_PROB_GRID)))
    block_strength = float(trial.suggest_float("informed_feature_block_strength", 0.0, 0.95))
    target_mix = float(trial.suggest_float("informed_target_mix_weight", 0.0, 1.0))

    if use_dirichlet:
        dirichlet_prob = float(
            trial.suggest_categorical("pitting_material_dirichlet_prob", list(DIRICHLET_PROB_GRID))
        )
        concentration = float(
            trial.suggest_categorical(
                "pitting_material_dirichlet_concentration",
                list(DIRICHLET_CONCENTRATION_GRID),
            )
        )
        active_prob = float(
            trial.suggest_categorical(
                "pitting_material_dirichlet_active_prob",
                list(DIRICHLET_ACTIVE_PROB_GRID),
            )
        )
    else:
        dirichlet_prob = 0.0
        concentration = None
        active_prob = None

    return TrialParams(
        use_magpie=use_magpie,
        use_dirichlet=use_dirichlet,
        mlp_prob=mlp_prob,
        informed_feature_block_strength=block_strength,
        informed_target_mix_weight=target_mix,
        pitting_material_dirichlet_prob=dirichlet_prob,
        pitting_material_dirichlet_concentration=concentration,
        pitting_material_dirichlet_active_prob=active_prob,
        epit_material_coef=float(trial.suggest_float("epit_material_coef", 0.45, 0.70)),
        epit_environment_coef=float(trial.suggest_float("epit_environment_coef", 0.35, 0.65)),
        epit_interaction_coef=float(trial.suggest_float("epit_interaction_coef", 0.60, 0.95)),
    )


def format_float(value: float) -> str:
    return search_utils.format_float(value)


def training_command(args: argparse.Namespace, params: TrialParams, checkpoint_dir: Path) -> list[str]:
    feature_count = EPIT_MAGPIE_TOTAL_FEATURE_COUNT if params.use_magpie else EPIT_BASE_FEATURE_COUNT
    mlp_prob = params.mlp_prob
    if not 0.0 <= mlp_prob <= 1.0:
        raise ValueError("mlp_prob must be in [0, 1].")
    dirichlet_args = [
        "--pitting_material_dirichlet_prob",
        format_float(params.pitting_material_dirichlet_prob),
    ]
    if params.use_dirichlet:
        if (
            params.pitting_material_dirichlet_concentration is None
            or params.pitting_material_dirichlet_active_prob is None
        ):
            raise ValueError("Enabled Dirichlet trials require concentration and active probability.")
        dirichlet_args.extend(
            [
                "--pitting_material_dirichlet_concentration",
                format_float(params.pitting_material_dirichlet_concentration),
                "--pitting_material_dirichlet_active_prob",
                format_float(params.pitting_material_dirichlet_active_prob),
            ]
        )

    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(REPO_ROOT / "src" / "tabicl" / "train" / "run.py"),
        "--wandb_log", "False",
        "--wandb_project", "TabICL",
        "--wandb_name", "PittingMagpieConditionalSearch",
        "--wandb_dir", "/home/fcolanto/wandb",
        "--wandb_mode", "disabled",
        "--device", args.device,
        "--dtype", "float32",
        "--np_seed", str(args.np_seed),
        "--torch_seed", str(args.torch_seed),
        "--max_steps", str(args.max_steps),
        "--batch_size", "512",
        "--micro_batch_size", "4",
        "--lr", "1e-4",
        "--scheduler", "cosine_warmup",
        "--scheduler_total_steps", str(args.scheduler_total_steps),
        "--warmup_proportion", "0.02",
        "--gradient_clipping", "1.0",
        "--prior_type", "informed_scm",
        "--informed_prior_ratio", "1.0",
        "--mix_probs", format_float(mlp_prob), format_float(1.0 - mlp_prob),
        "--informed_mix_probs", format_float(mlp_prob), format_float(1.0 - mlp_prob),
        "--informed_task_family_probs", "1.0", "0.0",
        "--informed_normal_block_allocation", *(str(value) for value in FIXED_BLOCK_ALLOCATION),
        "--informed_normal_block_allocation_min_counts", *(str(value) for value in FIXED_BLOCK_ALLOCATION),
        "--informed_inhibitor_block_allocation", *(format_float(value) for value in DEFAULT_INHIBITOR_ALLOCATION),
        "--informed_feature_block_strength", format_float(params.informed_feature_block_strength),
        "--informed_target_mix_weight", format_float(params.informed_target_mix_weight),
        "--informed_history_strength", "0.0",
        "--informed_intervention_strength", "0.0",
        "--informed_target_family", "pitting_potential",
        "--informed_physical_marginal_prob", "1.0",
        "--informed_physical_marginal_profile", "pitting_potential_v1",
        "--pitting_composition_mode", "legacy",
        "--pitting_material_style_probs", "1.0", "0.0", "0.0", "0.0", "0.0",
        *dirichlet_args,
        "--pitting_process_role", "test_method_category",
        "--pitting_process_category_count", "52",
        "--pitting_fixed_epit_schema", "True",
        "--pitting_magpie_features", str(params.use_magpie),
        "--cat_prob", "0.0",
        "--permute_features", "False",
        "--epit_material_coef", format_float(params.epit_material_coef),
        "--epit_environment_coef", format_float(params.epit_environment_coef),
        "--epit_interaction_coef", format_float(params.epit_interaction_coef),
        "--prior_device", "cpu",
        "--prior_n_jobs", str(args.prior_n_jobs),
        "--dataloader_num_workers", str(args.dataloader_num_workers),
        "--dataloader_prefetch_factor", str(args.dataloader_prefetch_factor),
        "--batch_size_per_gp", "4",
        "--min_features", str(feature_count),
        "--max_features", str(feature_count),
        "--max_classes", "0",
        "--num_quantiles", "999",
        "--max_seq_len", "1024",
        "--min_train_size", "0.1",
        "--max_train_size", "0.9",
        "--embed_dim", "128",
        "--col_num_blocks", "3",
        "--col_nhead", "4",
        "--col_num_inds", "128",
        "--row_num_blocks", "3",
        "--row_nhead", "8",
        "--row_num_cls", "4",
        "--row_rope_base", "100000",
        "--icl_num_blocks", "12",
        "--icl_nhead", "4",
        "--ff_factor", "2",
        "--norm_first", "True",
        "--checkpoint_dir", str(checkpoint_dir),
        "--save_temp_every", str(args.max_steps),
        "--save_perm_every", str(args.max_steps),
    ]
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
        str(REPO_ROOT / "scripts" / "eval_pitting_repeated_splits.py"),
        "--local-ckpt-path", str(checkpoint_path),
        "--model-label", model_label,
        "--split-seeds", *(str(seed) for seed in args.split_seeds),
        "--device", args.device,
        "--n-estimators", str(args.eval_n_estimators),
        "--tabicl-feat-shuffle-method", "none",
        "--output-dir", str(output_dir),
    ]
    if params.use_magpie:
        command.append("--pitting-magpie-features")
    return command


def study_root(root: Path, study_name: str) -> Path:
    return root.expanduser().resolve() / slugify(study_name)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_trial(args: argparse.Namespace, trial_number: int, params: TrialParams) -> dict[str, Any]:
    base_run_name = args.base_run_name or default_base_run_name(args.study_name)
    trial_name = f"{base_run_name}_trial_{trial_number:04d}"
    checkpoint_dir = study_root(args.checkpoint_root, args.study_name) / trial_name
    checkpoint_path = checkpoint_dir / f"step-{args.max_steps}.ckpt"
    trial_dir = study_root(args.work_dir, args.study_name) / slugify(base_run_name) / f"trial_{trial_number:04d}"
    eval_dir = study_root(args.eval_output_root, args.study_name) / f"pitting_magpie_repeated_splits_{trial_name}"
    model_label = f"trial_{trial_number:04d}"
    train_cmd = training_command(args, params, checkpoint_dir)
    eval_cmd = eval_command(args, params, checkpoint_path, model_label, eval_dir)
    metadata = {
        "trial_number": trial_number,
        "trial_name": trial_name,
        "params": asdict(params),
        "fixed_base_feature_count": EPIT_BASE_FEATURE_COUNT,
        "final_feature_count": EPIT_MAGPIE_TOTAL_FEATURE_COUNT if params.use_magpie else EPIT_BASE_FEATURE_COUNT,
        "fixed_block_allocation": FIXED_BLOCK_ALLOCATION,
        "fixed_material_style": "composition_like",
        "fixed_composition_mode": "legacy",
        "fixed_physical_marginal_prob": 1.0,
        "fixed_prior_type": "informed_scm",
        "magpie_version": EPIT_MAGPIE_VERSION if params.use_magpie else None,
        "magpie_descriptor_names": list(EPIT_MAGPIE_DESCRIPTOR_NAMES) if params.use_magpie else [],
        "magpie_range_min_atomic_fraction": (
            EPIT_MAGPIE_RANGE_MIN_ATOMIC_FRACTION if params.use_magpie else None
        ),
        "development_split_seeds": list(args.split_seeds),
        "reserved_final_split_seeds": list(RESERVED_FINAL_SPLIT_SEEDS),
        "checkpoint_path": str(checkpoint_path),
        "eval_dir": str(eval_dir),
        "train_command": train_cmd,
        "eval_command": eval_cmd,
    }
    write_json(trial_dir / "trial_config.json", metadata)
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
            raise FileExistsError(f"Checkpoint directory already exists and is non-empty: {checkpoint_dir}")
        search_utils.run_command(train_cmd, cwd=REPO_ROOT, log_path=trial_dir / "train.log")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Expected checkpoint was not created: {checkpoint_path}")

    if not (args.skip_existing and (eval_dir / "summary.csv").is_file()):
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        search_utils.run_command(eval_cmd, cwd=REPO_ROOT, log_path=trial_dir / "eval.log")
    summary = search_utils.load_eval_summary(eval_dir / "summary.csv", model_label)
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
    write_json(trial_dir / "trial_result.json", result)
    return result


def enqueue_missing_controls(study: Any) -> None:
    existing = {
        (trial.params.get("use_magpie"), trial.params.get("use_dirichlet"))
        for trial in study.trials
        if "use_magpie" in trial.params and "use_dirichlet" in trial.params
    }
    for branch in CONTROL_BRANCHES:
        key = (branch["use_magpie"], branch["use_dirichlet"])
        if key not in existing:
            study.enqueue_trial(branch)


def run_optuna(args: argparse.Namespace) -> None:
    import optuna

    args.base_run_name = args.base_run_name or default_base_run_name(args.study_name)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=search_utils.build_optuna_storage(args.storage),
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.random_seed),
    )
    if args.enqueue_controls:
        enqueue_missing_controls(study)

    def objective(trial: Any) -> float:
        params = sample_params(trial)
        result = run_trial(args, int(trial.number), params)
        for key in (
            "mean_test_mae",
            "median_test_mae",
            "mean_test_rmse",
            "mean_test_r2",
            "median_test_spearman",
            "std_test_spearman",
            "summary_csv",
            "rows_csv",
        ):
            trial.set_user_attr(key, result[key])
        return float(result["mean_test_spearman"])

    study.optimize(objective, n_trials=args.n_trials)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Mean development Spearman: {study.best_value:.6g}")
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
    args.base_run_name = args.base_run_name or default_base_run_name(args.study_name)
    rng = random.Random(args.random_seed)
    output = study_root(args.work_dir, args.study_name) / slugify(args.base_run_name) / "random_results.csv"
    for trial_number in range(args.n_trials):
        params = sample_params(RandomTrial(rng))
        result = run_trial(args, trial_number, params)
        append_random_result(
            output,
            {
                "trial_number": trial_number,
                "status": result["status"],
                "mean_test_spearman": result["mean_test_spearman"],
                "mean_test_mae": result["mean_test_mae"],
                **asdict(params),
            },
        )
    print(f"Wrote random-search results to {output}")


def run_search(args: argparse.Namespace) -> None:
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive.")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if args.scheduler_total_steps < args.max_steps:
        raise ValueError("--scheduler-total-steps must be >= --max-steps.")
    if not args.split_seeds:
        raise ValueError("--split-seeds must contain at least one development seed.")
    if set(args.split_seeds).intersection(RESERVED_FINAL_SPLIT_SEEDS):
        raise ValueError("Development split seeds must not use the reserved final split seeds.")
    if args.backend == "optuna":
        run_optuna(args)
    else:
        run_random(args)


def main() -> None:
    run_search(parse_args())


if __name__ == "__main__":
    main()
