#!/usr/bin/env python
"""Evaluate one Optuna trial on the five saved EPIT development folds."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_MANIFEST = (
    REPO_ROOT
    / "corrosion_datasets"
    / "analysis"
    / "epit_pipeline"
    / "splits_v2"
    / "split_manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "corrosion_datasets"
    / "analysis"
    / "epit_pipeline"
    / "optuna_v1"
    / "evaluations"
)
PITTING_TASK_ID = "electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg"
SUMMARY_METRICS = (
    "test_spearman",
    "test_mae",
    "test_rmse",
    "test_r2",
    "test_pearson",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-ckpt-path", type=Path, required=True)
    parser.add_argument("--model-label", default="pitting_candidate")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST,
    )
    parser.add_argument(
        "--validation-folds",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument(
        "--tabicl-feat-shuffle-method",
        choices=("none", "random", "latin", "shift"),
        default="none",
    )
    parser.add_argument(
        "--regression-output",
        choices=("mean", "median"),
        default="median",
    )
    parser.add_argument("--pitting-magpie-features", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--keep-fold-outputs", action="store_true")
    parser.add_argument("--print-subcommands", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.local_ckpt_path.expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.local_ckpt_path}")
    if not args.split_manifest.expanduser().is_file():
        raise FileNotFoundError(f"Split manifest not found: {args.split_manifest}")
    folds = list(args.validation_folds)
    if not folds or len(folds) != len(set(folds)):
        raise ValueError("--validation-folds must be non-empty and unique.")
    if any(fold not in range(1, 6) for fold in folds):
        raise ValueError("--validation-folds values must be between 1 and 5.")
    if args.n_estimators <= 0:
        raise ValueError("--n-estimators must be positive.")


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    return "_".join(part for part in cleaned.split("_") if part) or "model"


def default_output_dir(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"{slugify(args.model_label)}_{stamp}"


def fold_command(
    args: argparse.Namespace,
    *,
    fold: int,
    fold_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_corrosion_datasets.py"),
        "--local-ckpt-path",
        str(args.local_ckpt_path.expanduser()),
        "--local-model-label",
        args.model_label,
        "--task",
        PITTING_TASK_ID,
        "--target-mode",
        "primary",
        "--target-binning",
        "continuous",
        "--epit-split-manifest",
        str(args.split_manifest.expanduser()),
        "--epit-validation-fold",
        str(fold),
        "--device",
        args.device,
        "--n-estimators",
        str(args.n_estimators),
        "--tabicl-feat-shuffle-method",
        args.tabicl_feat_shuffle_method,
        "--regression-output",
        args.regression_output,
        "--no-regression-uncertainty",
        "--no-compare-pretrained-tabicl",
        "--output-json",
        str(fold_dir / "results.json"),
        "--output-csv",
        str(fold_dir / "rows.csv"),
        "--output-wide-csv",
        str(fold_dir / "wide.csv"),
        "--output-summary-csv",
        str(fold_dir / "summary.csv"),
    ]
    if args.pitting_magpie_features:
        command.append("--pitting-magpie-features")
    return command


def finite_metric_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for (model, task_id), group in rows.groupby(
        ["model", "task_id"],
        sort=False,
        dropna=False,
    ):
        record: dict[str, Any] = {
            "model": model,
            "task_id": task_id,
            "n_folds": int(group["optuna_validation_fold"].nunique()),
            "validation_folds": " ".join(
                str(int(fold))
                for fold in group["optuna_validation_fold"].tolist()
            ),
        }
        for meta_col in (
            "model_kind",
            "model_source",
            "dataset",
            "table",
            "target",
            "target_binning",
            "n_samples",
            "n_features",
            "feature_groups",
            "feature_group_counts",
            "task_quality_flags",
        ):
            if meta_col in group.columns:
                record[meta_col] = group.iloc[0][meta_col]
        if "n_train" in group.columns:
            record["context_rows_min"] = int(group["n_train"].min())
            record["context_rows_max"] = int(group["n_train"].max())
        if "n_test" in group.columns:
            record["validation_rows_min"] = int(group["n_test"].min())
            record["validation_rows_max"] = int(group["n_test"].max())

        for metric in SUMMARY_METRICS:
            if metric not in group.columns:
                continue
            values = finite_metric_values(group[metric])
            record[f"mean_{metric}"] = (
                float(values.mean()) if len(values) else math.nan
            )
            record[f"median_{metric}"] = (
                float(np.median(values)) if len(values) else math.nan
            )
            record[f"std_{metric}"] = (
                float(values.std(ddof=1)) if len(values) > 1 else math.nan
            )
            record[f"min_{metric}"] = (
                float(values.min()) if len(values) else math.nan
            )
            record[f"max_{metric}"] = (
                float(values.max()) if len(values) else math.nan
            )
        summaries.append(record)
    return pd.DataFrame(summaries)


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = (
        args.output_dir or default_output_dir(args)
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[pd.DataFrame] = []
    for fold in args.validation_folds:
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        command = fold_command(args, fold=fold, fold_dir=fold_dir)
        if args.print_subcommands:
            print(" ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        rows = pd.read_csv(fold_dir / "rows.csv")
        rows.insert(0, "optuna_validation_fold", fold)
        rows.insert(1, "fold_output_dir", str(fold_dir))
        fold_rows.append(rows)

    combined = pd.concat(fold_rows, ignore_index=True)
    summary = summarize_rows(combined)
    rows_path = output_dir / "rows.csv"
    summary_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"
    combined.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = json.loads(
        args.split_manifest.expanduser().read_text(encoding="utf-8")
    )
    summary_json_path.write_text(
        json.dumps(
            {
                "schema_version": "epit_optuna_fold_evaluation_v1",
                "task_id": PITTING_TASK_ID,
                "split_manifest": str(args.split_manifest.expanduser().resolve()),
                "split_manifest_schema": manifest.get("schema_version"),
                "source_sha256": manifest.get("dataset", {}).get("source_sha256"),
                "validation_folds": list(args.validation_folds),
                "development_rows_only": True,
                "final_test_rows_excluded": int(
                    manifest.get("split_design", {}).get("final_test_rows", -1)
                ),
                "source": {
                    "local_ckpt_path": str(
                        args.local_ckpt_path.expanduser().resolve()
                    ),
                    "model_label": args.model_label,
                },
                "settings": {
                    "device": args.device,
                    "n_estimators": args.n_estimators,
                    "tabicl_feat_shuffle_method": args.tabicl_feat_shuffle_method,
                    "regression_output": args.regression_output,
                    "pitting_magpie_features": bool(
                        args.pitting_magpie_features
                    ),
                },
                "rows_csv": str(rows_path),
                "summary_csv": str(summary_path),
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if not args.keep_fold_outputs:
        for fold in args.validation_folds:
            shutil.rmtree(output_dir / f"fold_{fold}")

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nWrote Optuna-fold rows to {rows_path}")
    print(f"Wrote Optuna-fold summary to {summary_path}")


if __name__ == "__main__":
    main()
