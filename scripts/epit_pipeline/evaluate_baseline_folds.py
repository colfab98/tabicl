#!/usr/bin/env python
"""Evaluate pretrained TabICL and CatBoost on frozen EPIT development folds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline.artifact_hashes import load_frozen_split, sha256_file
from scripts.epit_pipeline.evaluate_optuna_folds import (
    DEFAULT_SPLIT_MANIFEST,
    PITTING_TASK_ID,
    SUMMARY_METRICS,
    summarize_rows,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "corrosion_datasets"
    / "analysis"
    / "epit_pipeline"
    / "baseline_folds_v2"
)
DEFAULT_GENERIC_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints"
    / "tabicl_s1_regression_baseline"
    / "step-1000.ckpt"
)
VALIDATION_FOLDS = (1, 2, 3, 4, 5)
EXPECTED_MODELS = (
    "generic_baseline",
    "pretrained_tabicl_v2",
    "catboost",
)
PRETRAINED_CHECKPOINT_VERSION = "tabicl-regressor-v2-20260212.ckpt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--generic-checkpoint",
        type=Path,
        default=DEFAULT_GENERIC_CHECKPOINT,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--catboost-iterations", type=int, default=1000)
    parser.add_argument("--catboost-depth", type=int, default=6)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.03)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--catboost-thread-count", type=int, default=8)
    parser.add_argument("--print-subcommands", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.split_manifest.expanduser().is_file():
        raise FileNotFoundError(f"Split manifest not found: {args.split_manifest}")
    if not args.generic_checkpoint.expanduser().is_file():
        raise FileNotFoundError(
            f"Generic baseline checkpoint not found: {args.generic_checkpoint}"
        )
    positive = {
        "--n-estimators": args.n_estimators,
        "--catboost-iterations": args.catboost_iterations,
        "--catboost-depth": args.catboost_depth,
        "--catboost-learning-rate": args.catboost_learning_rate,
        "--catboost-thread-count": args.catboost_thread_count,
    }
    for option, value in positive.items():
        if value <= 0:
            raise ValueError(f"{option} must be positive.")
    if args.catboost_l2_leaf_reg < 0:
        raise ValueError("--catboost-l2-leaf-reg must be nonnegative.")


def fold_command(
    args: argparse.Namespace,
    *,
    fold: int,
    fold_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_corrosion_datasets.py"),
        "--local-ckpt-path",
        str(args.generic_checkpoint.expanduser().resolve()),
        "--local-model-label",
        "generic_baseline",
        "--task",
        PITTING_TASK_ID,
        "--target-mode",
        "primary",
        "--target-binning",
        "continuous",
        "--epit-split-manifest",
        str(args.split_manifest.expanduser().resolve()),
        "--epit-validation-fold",
        str(fold),
        "--device",
        args.device,
        "--n-estimators",
        str(args.n_estimators),
        "--random-state",
        str(args.random_state),
        "--tabicl-feat-shuffle-method",
        "none",
        "--tabicl-norm-methods",
        "none",
        "--regression-output",
        "median",
        "--no-regression-uncertainty",
        "--max-samples-per-task",
        "0",
        "--compare-pretrained-tabicl",
        "--pretrained-checkpoint-version",
        PRETRAINED_CHECKPOINT_VERSION,
        "--compare-catboost",
        "--catboost-iterations",
        str(args.catboost_iterations),
        "--catboost-depth",
        str(args.catboost_depth),
        "--catboost-learning-rate",
        str(args.catboost_learning_rate),
        "--catboost-l2-leaf-reg",
        str(args.catboost_l2_leaf_reg),
        "--catboost-thread-count",
        str(args.catboost_thread_count),
        "--output-json",
        str(fold_dir / "results.json"),
        "--output-csv",
        str(fold_dir / "rows.csv"),
        "--output-wide-csv",
        str(fold_dir / "wide.csv"),
        "--output-summary-csv",
        str(fold_dir / "summary.csv"),
    ]


def validate_fold_rows(rows: pd.DataFrame, *, fold: int) -> None:
    models = list(rows.get("model", pd.Series(dtype=str)).astype(str))
    if len(models) != len(EXPECTED_MODELS) or set(models) != set(EXPECTED_MODELS):
        raise RuntimeError(
            f"Fold {fold} did not produce exactly the two reference models: {models}"
        )
    expected_strategy = f"epit_pipeline_development_fold_{fold}"
    if set(rows["split_strategy"].astype(str)) != {expected_strategy}:
        raise RuntimeError(f"Fold {fold} did not use the frozen development split.")
    if set(pd.to_numeric(rows["n_train"], errors="raise")) not in (
        {486},
        {487},
    ):
        raise RuntimeError(f"Fold {fold} has an unexpected context size.")
    if set(pd.to_numeric(rows["n_test"], errors="raise")) not in (
        {121},
        {122},
    ):
        raise RuntimeError(f"Fold {fold} has an unexpected validation size.")
    if not all(
        int(train) + int(test) == 608
        for train, test in zip(rows["n_train"], rows["n_test"])
    ):
        raise RuntimeError(f"Fold {fold} did not use all 608 development rows.")
    if "pitting_magpie_features" in rows:
        magpie_values = {
            str(value).strip().lower()
            for value in rows["pitting_magpie_features"]
        }
        if not magpie_values.issubset({"false", "0"}):
            raise RuntimeError(
                "Reference models unexpectedly used Magpie descriptors."
            )
    for metric in SUMMARY_METRICS:
        values = pd.to_numeric(rows[metric], errors="coerce").to_numpy(dtype=float)
        if len(values) != len(EXPECTED_MODELS) or not np.isfinite(values).all():
            raise RuntimeError(f"Fold {fold} produced invalid {metric} values.")


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    args.split_manifest = args.split_manifest.expanduser().resolve()
    args.generic_checkpoint = args.generic_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Baseline output directory is non-empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_split = load_frozen_split(args.split_manifest)

    commands: list[list[str]] = []
    fold_rows: list[pd.DataFrame] = []
    for fold in VALIDATION_FOLDS:
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=False)
        command = fold_command(args, fold=fold, fold_dir=fold_dir)
        commands.append(command)
        if args.print_subcommands:
            print(" ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        result = json.loads((fold_dir / "results.json").read_text(encoding="utf-8"))
        if result.get("errors"):
            raise RuntimeError(
                f"Fold {fold} reported model errors: {result['errors']}"
            )
        rows = pd.read_csv(fold_dir / "rows.csv")
        validate_fold_rows(rows, fold=fold)
        rows.insert(0, "optuna_validation_fold", fold)
        rows.insert(1, "fold_output_dir", str(fold_dir))
        fold_rows.append(rows)

    combined = pd.concat(fold_rows, ignore_index=True)
    summary = summarize_rows(combined)
    if set(summary["model"].astype(str)) != set(EXPECTED_MODELS):
        raise RuntimeError("Combined baseline summary is missing a reference model.")
    if set(pd.to_numeric(summary["n_folds"], errors="raise")) != {5}:
        raise RuntimeError("Combined baseline summary did not use all five folds.")

    rows_path = output_dir / "rows.csv"
    summary_path = output_dir / "summary.csv"
    manifest_path = output_dir / "summary.json"
    combined.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    payload: dict[str, Any] = {
        "schema_version": "epit_development_baselines_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": PITTING_TASK_ID,
        "development_rows_only": True,
        "final_test_rows_used": False,
        "models": list(EXPECTED_MODELS),
        "generic_checkpoint": {
            "path": str(args.generic_checkpoint),
            "sha256": sha256_file(args.generic_checkpoint),
            "training_prior": "unmodified generic mix_scm",
            "training_step": 1000,
        },
        "validation_folds": list(VALIDATION_FOLDS),
        "split_manifest": str(frozen_split.manifest_path),
        "split_manifest_sha256": frozen_split.manifest_sha256,
        "split_lock": str(frozen_split.lock_path),
        "split_lock_sha256": frozen_split.lock_sha256,
        "source_sha256": frozen_split.manifest["dataset"]["source_sha256"],
        "settings": {
            "device": args.device,
            "n_estimators": args.n_estimators,
            "random_state": args.random_state,
            "tabicl_feat_shuffle_method": "none",
            "tabicl_norm_methods": ["none"],
            "regression_output": "median",
            "pitting_magpie_features": False,
            "pretrained_checkpoint_version": PRETRAINED_CHECKPOINT_VERSION,
            "catboost_iterations": args.catboost_iterations,
            "catboost_depth": args.catboost_depth,
            "catboost_learning_rate": args.catboost_learning_rate,
            "catboost_l2_leaf_reg": args.catboost_l2_leaf_reg,
            "catboost_thread_count": args.catboost_thread_count,
        },
        "commands": commands,
        "rows_csv": str(rows_path),
        "rows_csv_sha256": sha256_file(rows_path),
        "summary_csv": str(summary_path),
        "summary_csv_sha256": sha256_file(summary_path),
        "summary": summary.to_dict(orient="records"),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nWrote baseline rows to {rows_path}")
    print(f"Wrote baseline summary to {summary_path}")
    print(f"Wrote baseline provenance to {manifest_path}")
    return manifest_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
