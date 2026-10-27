#!/usr/bin/env python
"""Evaluate generic, pretrained and CatBoost baselines on reviewed Soccol development folds."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline import evaluate_baseline_folds as baseline
from scripts.epit_pipeline.artifact_hashes import load_frozen_split, sha256_file
from scripts.soccol_pipeline.data import SPLIT_MANIFEST, load_soccol_dataset, select_accepted_development
from scripts.soccol_pipeline.evaluate_checkpoint_comparison import SOCCOL_TASK_ID, CATBOOST_CATEGORICAL_COLUMNS

DEFAULT_OUTPUT_DIR = REPO_ROOT / "corrosion_datasets/analysis/soccol_pipeline/baseline_folds_reviewed_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    preview = argparse.ArgumentParser(add_help=False)
    preview.add_argument("--dry-run", action="store_true", help="Print all five commands without evaluating models.")
    return baseline.parse_args(
        ["--split-manifest", str(SPLIT_MANIFEST), "--output-dir", str(DEFAULT_OUTPUT_DIR),
         *(sys.argv[1:] if argv is None else argv)],
        description=__doc__, parents=(preview,),
    )


def fold_command(args: argparse.Namespace, *, fold: int, fold_dir: Path) -> list[str]:
    command = baseline.fold_command(args, fold=fold, fold_dir=fold_dir)
    command[command.index("--task") + 1] = SOCCOL_TASK_ID
    command.append("--soccol-reviewed-data")
    for column in CATBOOST_CATEGORICAL_COLUMNS:
        command.extend(["--catboost-categorical-column", column])
    return command


def fold_membership(selection: dict, fold: int) -> dict:
    pairs = list(zip(selection["source_task_row_indices"], selection["validation_folds"], strict=True))
    return {
        "soccol_development_selection": selection,
        "context_task_row_indices": [index for index, value in pairs if value != fold],
        "query_task_row_indices": [index for index, value in pairs if value == fold],
    }


def validate_fold_results(payload: dict, rows: pd.DataFrame, *, selection: dict, fold: int) -> None:
    expected = fold_membership(selection, fold)
    if payload.get("errors"):
        raise RuntimeError(f"Baseline fold {fold} reported model errors: {payload['errors']}")
    if payload.get("soccol_development_split") != expected:
        raise RuntimeError(f"Baseline fold {fold} changed the accepted development membership.")
    if len(rows) != 3 or set(rows.model) != set(baseline.EXPECTED_MODELS):
        raise RuntimeError(f"Baseline fold {fold} did not produce all three models.")
    for key, value in {
        "task_id": SOCCOL_TASK_ID, "n_features": 58,
        "split_strategy": f"epit_pipeline_development_fold_{fold}",
        "n_train": len(expected["context_task_row_indices"]), "n_test": len(expected["query_task_row_indices"]),
    }.items():
        if set(rows[key]) != {value}:
            raise RuntimeError(f"Baseline fold {fold} has incorrect {key}.")
    for metric in baseline.SUMMARY_METRICS:
        if not np.isfinite(pd.to_numeric(rows[metric], errors="coerce")).all():
            raise RuntimeError(f"Baseline fold {fold} produced invalid {metric}.")


def run(args: argparse.Namespace) -> Path:
    args.split_manifest = args.split_manifest.expanduser().resolve()
    args.generic_checkpoint = args.generic_checkpoint.expanduser().resolve()
    dataset = load_soccol_dataset(split_manifest=args.split_manifest)
    selection = select_accepted_development(dataset).identity
    output = args.output_dir.expanduser().resolve()
    if args.dry_run:
        for fold in baseline.VALIDATION_FOLDS:
            print(shlex.join(fold_command(args, fold=fold, fold_dir=output / f"fold_{fold}")))
        return output / "summary.json"
    baseline.validate_args(args)
    output = baseline.prepare_output_dir(output, auto_output_dir=args.auto_output_dir)
    frozen = load_frozen_split(args.split_manifest, expected_manifest_schema="soccol_composition_split_manifest_v1")
    commands, all_rows = [], []
    for fold in baseline.VALIDATION_FOLDS:
        directory = output / f"fold_{fold}"
        directory.mkdir()
        command = fold_command(args, fold=fold, fold_dir=directory)
        commands.append(command)
        if args.print_subcommands:
            print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        payload = json.loads((directory / "results.json").read_text())
        rows = pd.read_csv(directory / "rows.csv")
        validate_fold_results(payload, rows, selection=selection, fold=fold)
        rows.insert(0, "optuna_validation_fold", fold)
        rows.insert(1, "fold_output_dir", str(directory))
        all_rows.append(rows)
    combined = pd.concat(all_rows, ignore_index=True)
    summary = baseline.summarize_rows(combined)
    if set(summary.model) != set(baseline.EXPECTED_MODELS) or set(summary.n_folds) != {5}:
        raise RuntimeError("Baseline summary is missing models or development folds.")
    rows_path, summary_path = output / "rows.csv", output / "summary.csv"
    combined.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "schema_version": "soccol_reviewed_development_baselines_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "task_id": SOCCOL_TASK_ID,
        "development_rows_only": True, "final_test_rows_used": False, "final_test_rows_excluded": 805,
        "development_rows": 2950, "soccol_development_selection": selection,
        "validation_folds": list(baseline.VALIDATION_FOLDS), "models": list(baseline.EXPECTED_MODELS),
        "split_manifest": str(frozen.manifest_path), "split_manifest_sha256": frozen.manifest_sha256,
        "split_lock_sha256": frozen.lock_sha256,
        "source_sha256": dataset.identity["files_sha256"]["soccol_processed.csv"],
        "generic_checkpoint": {"path": str(args.generic_checkpoint.resolve()), "sha256": sha256_file(args.generic_checkpoint)},
        "settings": {"soccol_reviewed_data": True, "n_features": 58, "random_state": args.random_state,
                     "n_estimators": args.n_estimators, "tabicl_feat_shuffle_method": "none", "tabicl_norm_methods": ["none"],
                     "catboost_categorical_columns": list(CATBOOST_CATEGORICAL_COLUMNS),
                     "catboost_hyperparameter_policy": "fixed_predeclared_no_dataset_specific_tuning"},
        "commands": commands, "rows_csv": str(rows_path), "rows_csv_sha256": sha256_file(rows_path),
        "summary_csv": str(summary_path), "summary_csv_sha256": sha256_file(summary_path),
        "summary": summary.to_dict(orient="records"),
    }
    path = output / "summary.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))
    print(f"Wrote development baseline results to {output}")
    return path


if __name__ == "__main__":
    run(parse_args())
