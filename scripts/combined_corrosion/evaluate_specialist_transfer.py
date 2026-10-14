#!/usr/bin/env python
"""Evaluate frozen EPIT and Soccol specialists on the frozen MPEA query set."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts import eval_corrosion_datasets as corrosion_eval
from tabicl.prior.epit_schema import (
    EPIT_COMPOSITION_COLUMNS,
    EPIT_COMPOSITION_ELEMENTS,
    EPIT_ENVIRONMENT_COLUMNS,
    EPIT_METHOD_COLUMN,
)
from tabicl.prior.soccol_schema import (
    SOCCOL_CATEGORICAL_COLUMNS,
    SOCCOL_COMPOSITION_COLUMNS,
    SOCCOL_CONTINUOUS_COLUMNS,
    SOCCOL_FEATURE_COLUMNS,
    SOCCOL_ION_COLUMNS,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "corrosion_datasets"
    / "datasets"
    / "combined_epit_soccol_mpea"
    / "processed"
)
DEFAULT_EPIT_RUN = (
    REPO_ROOT
    / "checkpoints"
    / "epit_pipeline_final_v2"
    / "final_epit_pipeline_optuna_empirical_features_scm_target_v8_trial_0017"
)
DEFAULT_SOCCOL_RUN = (
    REPO_ROOT
    / "checkpoints"
    / "soccol_pipeline_final_v1"
    / "final_soccol_pipeline_optuna_empirical_features_scm_target_v1_trial_0019"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "corrosion_datasets"
    / "analysis"
    / "combined_epit_soccol_mpea"
    / "specialist_transfer_checkpoint_comparison"
)
EPIT_MODEL_LABEL = "epit_corrpfn"
SOCCOL_MODEL_LABEL = "soccol_corrpfn"
EPIT_SOURCE_COLUMNS = (
    *EPIT_COMPOSITION_ELEMENTS,
    "CP_temp",
    "CP_Cl",
    "CP_pH",
    "test_method",
)
EPIT_NATIVE_COLUMNS = (
    *EPIT_COMPOSITION_COLUMNS,
    *EPIT_ENVIRONMENT_COLUMNS,
    EPIT_METHOD_COLUMN,
)
REGRESSION_SORT_SPECS = [
    (corrosion_eval.REGRESSION_PRIMARY_METRIC, False),
    ("test_nmae_iqr", True),
    ("test_mae", True),
    ("test_rmse", True),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epit-run", type=Path, default=DEFAULT_EPIT_RUN)
    parser.add_argument("--soccol-run", type=Path, default=DEFAULT_SOCCOL_RUN)
    parser.add_argument("--combined-data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", default="all")
    parser.add_argument("--min-checkpoint-step", type=int, default=0)
    parser.add_argument("--checkpoint-step-interval", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-checkpoint-plots",
        dest="checkpoint_plots",
        action="store_false",
    )
    parser.set_defaults(checkpoint_plots=True)
    args = parser.parse_args(argv)
    if args.min_checkpoint_step < 0:
        parser.error("--min-checkpoint-step must be non-negative.")
    if args.checkpoint_step_interval < 0:
        parser.error("--checkpoint-step-interval must be non-negative.")
    if args.n_estimators <= 0:
        parser.error("--n-estimators must be positive.")
    return args


def available_checkpoints(run_dir: Path) -> dict[int, Path]:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {run_dir}")
    checkpoints: dict[int, Path] = {}
    for path in run_dir.glob("step-*.ckpt"):
        step = corrosion_eval.checkpoint_step_from_path(path)
        if step is not None:
            checkpoints[step] = path.resolve()
    if not checkpoints:
        raise FileNotFoundError(f"No step-*.ckpt files found in {run_dir}")
    return checkpoints


def select_checkpoint_pairs(
    epit_run: Path,
    soccol_run: Path,
    *,
    checkpoint: str,
    min_step: int,
    step_interval: int,
) -> list[tuple[int, Path, Path]]:
    epit = available_checkpoints(epit_run)
    soccol = available_checkpoints(soccol_run)
    common = sorted(set(epit).intersection(soccol))
    if not common:
        raise FileNotFoundError("EPIT and Soccol runs have no common checkpoint steps.")

    normalized = checkpoint.removeprefix("step-").removesuffix(".ckpt")
    if checkpoint == "all":
        eligible = [step for step in common if step >= min_step]
        selected = [
            step
            for step in eligible
            if step_interval == 0 or step % step_interval == 0
        ]
        if eligible and eligible[-1] not in selected:
            selected.append(eligible[-1])
    elif checkpoint == "latest":
        selected = [common[-1]]
    elif normalized.isdigit():
        step = int(normalized)
        if step not in common:
            raise FileNotFoundError(
                f"Checkpoint step {step} is not present in both specialist runs."
            )
        selected = [step]
    else:
        raise ValueError("--checkpoint must be all, latest, or a numeric step.")

    if not selected:
        raise FileNotFoundError(
            "No common checkpoints satisfy the minimum-step and interval filters."
        )
    return [(step, epit[step], soccol[step]) for step in selected]


def _project_epit_features(X: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(EPIT_SOURCE_COLUMNS).difference(X.columns))
    if missing:
        raise RuntimeError(f"Combined data lacks EPIT projection columns: {missing}")
    projected = X.loc[:, EPIT_SOURCE_COLUMNS].copy()
    projected.columns = EPIT_NATIVE_COLUMNS
    return projected


def _preprocess_native_projection(
    X: pd.DataFrame,
    split: corrosion_eval.EvalSplit,
    *,
    specialist: str,
) -> pd.DataFrame:
    train = X.iloc[split.train_index].reset_index(drop=True)
    test = X.iloc[split.test_index].reset_index(drop=True)
    if specialist == "epit":
        train, test = corrosion_eval.preprocess_epit_split(
            train,
            test,
            X_preprocessing=train,
        )
    elif specialist == "soccol":
        train, test = corrosion_eval.preprocess_soccol_split(
            train,
            test,
            X_preprocessing=train,
        )
    else:
        raise ValueError(f"Unknown specialist: {specialist}")
    projected = pd.concat((train, test), ignore_index=True)
    if not np.isfinite(projected.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{specialist} projection contains non-finite model inputs.")
    return projected


def build_specialist_tasks(
    data_dir: Path,
) -> dict[str, corrosion_eval.EvalTask]:
    base = corrosion_eval._make_combined_pipeline_task(data_dir)
    split = base.fixed_split
    if split is None:
        raise RuntimeError("Combined task does not provide its frozen context/query split.")

    epit_X = _preprocess_native_projection(
        _project_epit_features(base.X),
        split,
        specialist="epit",
    )
    soccol_X = _preprocess_native_projection(
        base.X.loc[:, SOCCOL_FEATURE_COLUMNS].copy(),
        split,
        specialist="soccol",
    )
    return {
        EPIT_MODEL_LABEL: replace(
            base,
            X=epit_X,
            feature_group_counts={"material": 24, "environment": 3, "process_history": 1},
        ),
        SOCCOL_MODEL_LABEL: replace(
            base,
            X=soccol_X,
            feature_group_counts={
                "material": len(SOCCOL_COMPOSITION_COLUMNS),
                "environment": len(SOCCOL_CONTINUOUS_COLUMNS) + len(SOCCOL_ION_COLUMNS),
                "process_history": len(SOCCOL_CATEGORICAL_COLUMNS),
            },
        ),
    }


def _error_record(
    *,
    model: str,
    task: corrosion_eval.EvalTask,
    checkpoint_path: Path,
    step: int,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "model": model,
        "model_kind": "local_tabicl_regressor",
        "model_source": str(checkpoint_path),
        "checkpoint_name": f"step-{step}",
        "checkpoint_step": step,
        "task_id": task.task_id,
        "dataset": task.dataset,
        "table": task.table,
        "target": task.target,
        "split_strategy": task.split_strategy,
        "include_in_summary": task.include_in_summary,
        "summary_exclusion_reason": task.summary_exclusion_reason,
        "feature_projection": model.removesuffix("_corrpfn"),
        "n_features": int(task.X.shape[1]),
        "error": repr(exc),
    }


def main() -> None:
    args = parse_args()
    tasks = build_specialist_tasks(args.combined_data_dir)
    checkpoint_pairs = select_checkpoint_pairs(
        args.epit_run,
        args.soccol_run,
        checkpoint=args.checkpoint,
        min_step=args.min_checkpoint_step,
        step_interval=args.checkpoint_step_interval,
    )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for step, epit_checkpoint, soccol_checkpoint in checkpoint_pairs:
        print(f"\nCheckpoint step-{step}")
        for model, checkpoint_path in (
            (EPIT_MODEL_LABEL, epit_checkpoint),
            (SOCCOL_MODEL_LABEL, soccol_checkpoint),
        ):
            task = tasks[model]
            try:
                row = corrosion_eval.evaluate_estimator(
                    model_label=model,
                    model_kind="local_tabicl_regressor",
                    estimator_factory=lambda checkpoint_path=checkpoint_path: corrosion_eval.make_tabicl_regressor(
                        model_path=str(checkpoint_path),
                        checkpoint_version="",
                        device=args.device,
                        n_estimators=args.n_estimators,
                        random_state=args.random_state,
                        allow_auto_download=False,
                        feat_shuffle_method="none",
                        norm_methods=["none"],
                    ),
                    task=task,
                    task_split=task.fixed_split,
                    test_size=0.2,
                    random_state=args.random_state,
                    regression_output="median",
                    regression_uncertainty=False,
                    preserve_feature_schema=True,
                )
                row.update(
                    {
                        "checkpoint_name": f"step-{step}",
                        "checkpoint_step": step,
                        "feature_projection": model.removesuffix("_corrpfn"),
                    }
                )
                rows.append(row)
                corrosion_eval.print_result_row(row)
            except Exception as exc:
                errors.append(
                    _error_record(
                        model=model,
                        task=task,
                        checkpoint_path=checkpoint_path,
                        step=step,
                        exc=exc,
                    )
                )
                print(f"  {model:<24} ERROR {exc!r}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "results.json"
    output_csv = output_dir / "rows.csv"
    output_wide_csv = output_dir / "wide.csv"
    output_summary_csv = output_dir / "summary.csv"
    output_plot_dir = output_dir / "plots"

    summary = corrosion_eval.print_summary(
        rows,
        errors,
        primary_metric=corrosion_eval.REGRESSION_PRIMARY_METRIC,
        metric_sort_specs=REGRESSION_SORT_SPECS,
    )
    wide = corrosion_eval.make_wide_results_dataframe(rows)
    plot_paths: list[Path] = []
    if args.checkpoint == "all" and args.checkpoint_plots:
        plot_paths = corrosion_eval.write_checkpoint_trend_plots(
            summary,
            output_plot_dir,
            list(corrosion_eval.DEFAULT_REGRESSION_PLOT_METRICS),
        )

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": "combined_specialist_transfer_evaluation_v1",
        "diagnostic_only": True,
        "task_id": corrosion_eval.COMBINED_PIPELINE_TASK_ID,
        "split_strategy": "combined_full_context_frozen_mpea_queries",
        "context_rows": 4787,
        "query_rows": 267,
        "combined_data_dir": str(args.combined_data_dir.expanduser().resolve()),
        "checkpoint": args.checkpoint,
        "checkpoint_steps": [step for step, _, _ in checkpoint_pairs],
        "epit_run": str(args.epit_run.expanduser().resolve()),
        "soccol_run": str(args.soccol_run.expanduser().resolve()),
        "device": args.device,
        "n_estimators": args.n_estimators,
        "random_state": args.random_state,
        "feat_shuffle_method": "none",
        "norm_methods": ["none"],
        "feature_projections": {
            EPIT_MODEL_LABEL: list(EPIT_NATIVE_COLUMNS),
            SOCCOL_MODEL_LABEL: list(SOCCOL_FEATURE_COLUMNS),
        },
        "interpretation_warning": (
            "Both specialists use the mixed-reference combined context; neither legacy "
            "checkpoint was trained for that missingness/reference pattern."
        ),
        "rows": rows,
        "errors": errors,
        "output_files": {
            "json": str(output_json),
            "csv": str(output_csv),
            "wide_csv": str(output_wide_csv),
            "summary_csv": str(output_summary_csv),
            "plot_dir": str(output_plot_dir) if plot_paths else None,
            "plots": [str(path) for path in plot_paths],
        },
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    wide.to_csv(output_wide_csv, index=False)
    summary.to_csv(output_summary_csv, index=False)
    print(f"\nSaved specialist-transfer evaluation to {output_dir}")


if __name__ == "__main__":
    main()
