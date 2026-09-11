#!/usr/bin/env python
"""Combine fixed-fold checkpoint comparisons and render checkpoint trends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.eval_corrosion_datasets import (
    DEFAULT_REGRESSION_PLOT_METRICS,
    write_checkpoint_trend_plots,
)
from scripts.epit_pipeline.evaluate_optuna_folds import summarize_rows


VALIDATION_FOLDS = (1, 2, 3, 4, 5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def load_fold_rows(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    expected_models: set[str] | None = None
    expected_checkpoints: set[tuple[str, int]] | None = None
    for fold in VALIDATION_FOLDS:
        fold_dir = root / f"fold_{fold}"
        result = json.loads((fold_dir / "results.json").read_text(encoding="utf-8"))
        if result.get("errors"):
            raise RuntimeError(f"Fold {fold} reported errors: {result['errors']}")
        rows = pd.read_csv(fold_dir / "rows.csv")
        required = {"model", "checkpoint_name", "checkpoint_step"}
        missing = required.difference(rows.columns)
        if missing:
            raise RuntimeError(
                f"Fold {fold} is not an all-checkpoint evaluation; missing {sorted(missing)}."
            )
        observed_models = set(rows["model"].astype(str))
        observed_checkpoints = {
            (str(name), int(step))
            for name, step in zip(rows["checkpoint_name"], rows["checkpoint_step"])
        }
        if expected_models is None:
            expected_models = observed_models
            expected_checkpoints = observed_checkpoints
        elif observed_models != expected_models:
            raise RuntimeError(
                f"Fold {fold} model set differs: {sorted(observed_models)} != "
                f"{sorted(expected_models)}"
            )
        elif observed_checkpoints != expected_checkpoints:
            raise RuntimeError(f"Fold {fold} checkpoint set differs from earlier folds.")
        rows.insert(0, "optuna_validation_fold", fold)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def build_summary(combined: pd.DataFrame) -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []
    for (name, step), rows in combined.groupby(
        ["checkpoint_name", "checkpoint_step"], sort=False
    ):
        summary = summarize_rows(rows)
        summary.insert(0, "checkpoint_step", int(step))
        summary.insert(0, "checkpoint_name", str(name))
        summaries.append(summary)
    result = pd.concat(summaries, ignore_index=True)
    if set(pd.to_numeric(result["n_folds"], errors="raise")) != {5}:
        raise RuntimeError("Every model/checkpoint comparison must contain all five folds.")
    return result


def run(args: argparse.Namespace) -> Path:
    root = args.output_root.expanduser().resolve()
    combined = load_fold_rows(root)
    summary = build_summary(combined)
    combined.to_csv(root / "rows.csv", index=False)
    summary.to_csv(root / "summary.csv", index=False)
    plot_dir = root / "plots"
    plot_paths = write_checkpoint_trend_plots(
        summary,
        plot_dir,
        list(DEFAULT_REGRESSION_PLOT_METRICS),
    )
    if not plot_paths:
        raise RuntimeError("No checkpoint trend plots were generated.")
    print(summary.to_string(index=False))
    print(f"Combined results: {root}")
    print(f"Five-fold average checkpoint trend plots: {plot_dir}")
    for path in plot_paths:
        print(f"  {path}")
    return root


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
