#!/usr/bin/env python
"""Repeated-split evaluation for the alloy pitting-potential regression task.

This is a thin wrapper around ``scripts/eval_corrosion_datasets.py``. It keeps
the existing task construction, model loading, and metric logic in one place,
while varying only the corrosion train/test split seed.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "corrosion_datasets" / "analysis" / "eval_results"
PITTING_TASK_ID = "electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg"
SUMMARY_METRICS = (
    "test_spearman",
    "test_mae",
    "test_rmse",
    "test_r2",
    "test_pearson",
)
DEFAULT_SPLIT_SEEDS = (1001, 1002, 1003, 1004, 1005)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--local-ckpt-path", type=Path, help="Local step-1000 .ckpt path to evaluate.")
    source.add_argument("--run", type=str, help="Checkpoint run directory or suffix to evaluate.")
    source.add_argument(
        "--pretrained-tabicl-only",
        action="store_true",
        help="Evaluate only the pretrained TabICL v2 model without a local checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="1000",
        help="Checkpoint step/name when --run is used. Defaults to 1000.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default=None,
        help="Optional run prefix passed through to eval_corrosion_datasets.py.",
    )
    parser.add_argument(
        "--model-label",
        "--local-model-label",
        dest="model_label",
        type=str,
        default="pitting_candidate",
        help="Model label used in output rows.",
    )
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(DEFAULT_SPLIT_SEEDS))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--regression-output", choices=("mean", "median"), default="median")
    parser.add_argument(
        "--regression-uncertainty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request quantile/uncertainty metrics from the underlying evaluator.",
    )
    parser.add_argument(
        "--compare-pretrained-tabicl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also evaluate the pretrained TabICL regressor for each split.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--keep-seed-outputs", action="store_true", help="Keep per-seed evaluator outputs.")
    parser.add_argument(
        "--print-subcommands",
        action="store_true",
        help="Print each eval_corrosion_datasets.py command before running it.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.local_ckpt_path is not None and not args.local_ckpt_path.expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.local_ckpt_path}")
    if args.run is not None and args.checkpoint == "all":
        raise ValueError("This repeated-split wrapper scores one forced checkpoint; do not pass --checkpoint all.")
    if args.pretrained_tabicl_only:
        args.compare_pretrained_tabicl = True
    if len(args.split_seeds) == 0:
        raise ValueError("--split-seeds must contain at least one seed.")


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    return "_".join(part for part in cleaned.split("_") if part) or "model"


def default_output_dir(args: argparse.Namespace) -> Path:
    label = "pretrained_tabicl_v2" if args.pretrained_tabicl_only else slugify(args.model_label)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"pitting_repeated_splits_{label}_{stamp}"


def source_args(args: argparse.Namespace) -> list[str]:
    if args.pretrained_tabicl_only:
        return []
    if args.local_ckpt_path is not None:
        return [
            "--local-ckpt-path",
            str(args.local_ckpt_path.expanduser()),
            "--local-model-label",
            args.model_label,
        ]

    values = ["--run", args.run, "--checkpoint", args.checkpoint]
    run_prefix = args.run_prefix
    if run_prefix is None and str(args.run).startswith("tabicl_"):
        # eval_corrosion_datasets.py drops empty string args, so use a relative prefix
        # that resolves to the same checkpoint directory.
        run_prefix = "./"
    if run_prefix is not None:
        values.extend(["--run-prefix", run_prefix])
    if args.checkpoint != "all":
        values.extend(["--local-model-label", args.model_label])
    return values


def run_seed_eval(args: argparse.Namespace, seed: int, seed_dir: Path) -> pd.DataFrame:
    seed_dir.mkdir(parents=True, exist_ok=True)
    output_json = seed_dir / "results.json"
    output_csv = seed_dir / "rows.csv"
    output_wide_csv = seed_dir / "wide.csv"
    output_summary_csv = seed_dir / "summary.csv"

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_corrosion_datasets.py"),
        *source_args(args),
        "--task",
        PITTING_TASK_ID,
        "--target-mode",
        "primary",
        "--target-binning",
        "continuous",
        "--random-state",
        str(seed),
        "--test-size",
        str(args.test_size),
        "--device",
        args.device,
        "--n-estimators",
        str(args.n_estimators),
        "--regression-output",
        args.regression_output,
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
        "--output-wide-csv",
        str(output_wide_csv),
        "--output-summary-csv",
        str(output_summary_csv),
    ]
    if not args.regression_uncertainty:
        command.append("--no-regression-uncertainty")

    if args.compare_pretrained_tabicl:
        command.append("--compare-pretrained-tabicl")
    else:
        command.append("--no-compare-pretrained-tabicl")

    if args.print_subcommands:
        print(" ".join(command), flush=True)

    subprocess.run(command, cwd=REPO_ROOT, check=True)

    rows = pd.read_csv(output_csv)
    rows.insert(0, "split_seed", seed)
    rows.insert(1, "seed_output_dir", str(seed_dir))
    return rows


def finite_metric_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for (model, task_id), group in rows.groupby(["model", "task_id"], sort=False, dropna=False):
        record: dict[str, Any] = {
            "model": model,
            "task_id": task_id,
            "n_splits": int(group["split_seed"].nunique()),
            "split_seeds": " ".join(str(seed) for seed in group["split_seed"].tolist()),
        }
        for meta_col in [
            "model_kind",
            "model_source",
            "dataset",
            "table",
            "target",
            "target_binning",
            "n_samples",
            "n_train",
            "n_test",
            "n_features",
            "feature_groups",
            "feature_group_counts",
            "task_quality_flags",
        ]:
            if meta_col in group.columns:
                record[meta_col] = group.iloc[0][meta_col]

        for metric in SUMMARY_METRICS:
            if metric not in group.columns:
                continue
            values = finite_metric_values(group[metric])
            record[f"mean_{metric}"] = float(values.mean()) if len(values) else math.nan
            record[f"median_{metric}"] = float(np.median(values)) if len(values) else math.nan
            record[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
            record[f"min_{metric}"] = float(values.min()) if len(values) else math.nan
            record[f"max_{metric}"] = float(values.max()) if len(values) else math.nan
        summaries.append(record)
    return pd.DataFrame(summaries)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = (args.output_dir or default_output_dir(args)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows: list[pd.DataFrame] = []
    for seed in args.split_seeds:
        seed_dir = output_dir / f"seed_{seed}"
        rows = run_seed_eval(args, seed, seed_dir)
        split_rows.append(rows)

    combined = pd.concat(split_rows, ignore_index=True)
    summary = summarize_rows(combined)

    rows_path = output_dir / "rows.csv"
    summary_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"
    combined.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_json(
        json_path,
        {
            "task_id": PITTING_TASK_ID,
            "split_seeds": list(args.split_seeds),
            "source": {
                "local_ckpt_path": str(args.local_ckpt_path) if args.local_ckpt_path is not None else None,
                "run": args.run,
                "checkpoint": args.checkpoint,
                "run_prefix": args.run_prefix,
                "model_label": args.model_label,
                "pretrained_tabicl_only": bool(args.pretrained_tabicl_only),
            },
            "settings": {
                "target_binning": "continuous",
                "test_size": args.test_size,
                "device": args.device,
                "n_estimators": args.n_estimators,
                "regression_output": args.regression_output,
                "regression_uncertainty": bool(args.regression_uncertainty),
                "compare_pretrained_tabicl": bool(args.compare_pretrained_tabicl),
            },
            "rows_csv": str(rows_path),
            "summary_csv": str(summary_path),
            "summary": summary.to_dict(orient="records"),
        },
    )

    if not args.keep_seed_outputs:
        for seed in args.split_seeds:
            seed_dir = output_dir / f"seed_{seed}"
            for path in seed_dir.glob("*"):
                path.unlink()
            seed_dir.rmdir()

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nWrote repeated-split rows to {rows_path}")
    print(f"Wrote repeated-split summary to {summary_path}")


if __name__ == "__main__":
    main()
