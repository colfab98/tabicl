#!/usr/bin/env python
"""Run a CPU-only symbolic-regression baseline on the frozen EPIT folds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import spearmanr

from scripts.epit_pipeline.artifact_hashes import (
    load_frozen_split,
    sha256_file,
)

if TYPE_CHECKING:
    from scripts.epit_pipeline.split_data import EpitDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILE = (
    REPO_ROOT
    / "corrosion_datasets"
    / "datasets"
    / "electrochemical_metrics_alloys"
    / "raw"
    / "CRA_database_Scientific_Data_Publication_12102020.xlsx"
)
DEFAULT_SPLIT_DIR = (
    REPO_ROOT / "corrosion_datasets" / "analysis" / "epit_pipeline" / "splits_v2"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "corrosion_datasets" / "analysis" / "epit_pipeline" / "pysr_v1"
)
INNER_FOLD_COUNT = 5
DEFAULT_FULL_ITERATIONS = 1_000
DEFAULT_DRY_RUN_ITERATIONS = 100
DEFAULT_DRY_RUN_TIMEOUT_SECONDS = 600.0
BINARY_OPERATORS = ("+", "-", "*", "/")
UNARY_OPERATORS = ("square", "sqrt", "log", "exp")
NESTED_CONSTRAINTS = {
    outer: {inner: 0 for inner in UNARY_OPERATORS}
    for outer in UNARY_OPERATORS
}

# Human-readable PySR name, source-workbook column.
FEATURES = (
    ("Fe_wt_pct", "Composition, wt.% Fe"),
    ("Cr_wt_pct", "Composition, wt.% Cr"),
    ("Ni_wt_pct", "Composition, wt.% Ni"),
    ("Mo_wt_pct", "Composition, wt.% Mo"),
    ("W_wt_pct", "Composition, wt.% W"),
    ("N_wt_pct", "Composition, wt.% N"),
    ("temperature_C", "Test Temp. oC"),
    ("chloride_M", "[Cl-] M"),
    ("pH", "[Cl-] pH"),
)
MODEL_FEATURE_NAMES = tuple(name for name, _ in FEATURES)


@dataclass(frozen=True)
class SplitRows:
    development: np.ndarray
    final_test: np.ndarray
    validation_by_fold: dict[int, np.ndarray]
    manifest_sha256: str
    lock_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_DIR / "split_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Exact output directory; default creates a timestamped directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Fit one development fold for 100 iterations by default, report timing, "
            "and estimate a five-fold 1000-iteration run."
        ),
    )
    parser.add_argument(
        "--fold",
        action="append",
        type=int,
        choices=range(1, INNER_FOLD_COUNT + 1),
        help="Development validation fold to run; repeat as needed.",
    )
    parser.add_argument(
        "--niterations",
        type=int,
        help="PySR iterations per fold (dry-run: 100; normal: 1000).",
    )
    parser.add_argument("--populations", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--maxsize", type=int, default=15)
    parser.add_argument("--maxdepth", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--parallelism",
        choices=("serial", "multithreading", "multiprocessing"),
        default="multithreading",
    )
    parser.add_argument(
        "--procs",
        type=int,
        help="Worker count for multiprocessing; ignored for multithreading.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        help="Per-fold timeout (dry-run default: 600 seconds).",
    )
    parser.add_argument(
        "--projection-iterations",
        type=int,
        default=DEFAULT_FULL_ITERATIONS,
        help="Iteration count used for the rough full-run timing projection.",
    )
    return parser.parse_args()


def load_split_rows(manifest_path: Path, expected_rows: int) -> SplitRows:
    frozen = load_frozen_split(manifest_path)
    manifest = frozen.manifest
    if int(manifest.get("dataset", {}).get("usable_rows", -1)) != expected_rows:
        raise RuntimeError("Frozen split row count does not match the EPIT dataset.")
    if sha256_file(SOURCE_FILE) != manifest["dataset"]["source_sha256"]:
        raise RuntimeError("EPIT source workbook changed after the split was frozen.")

    records = manifest.get("rows", [])
    by_index = {int(record["task_row_index"]): record for record in records}
    if sorted(by_index) != list(range(expected_rows)):
        raise RuntimeError("Frozen split does not cover every EPIT row exactly once.")

    development = np.asarray(
        [i for i, record in by_index.items() if record["outer_split"] == "development"],
        dtype=int,
    )
    final_test = np.asarray(
        [i for i, record in by_index.items() if record["outer_split"] == "final_test"],
        dtype=int,
    )
    validation_by_fold = {
        fold: np.asarray(
            [
                i
                for i in development
                if int(by_index[int(i)]["optuna_validation_fold"]) == fold
            ],
            dtype=int,
        )
        for fold in range(1, INNER_FOLD_COUNT + 1)
    }
    if len(development) != 608 or len(final_test) != 152:
        raise RuntimeError("Expected the frozen 608-development/152-final EPIT split.")
    covered = np.concatenate(list(validation_by_fold.values()))
    if sorted(covered.tolist()) != sorted(development.tolist()):
        raise RuntimeError("Frozen validation folds do not cover development rows once.")

    return SplitRows(
        development=development,
        final_test=final_test,
        validation_by_fold=validation_by_fold,
        manifest_sha256=frozen.manifest_sha256,
        lock_sha256=frozen.lock_sha256,
    )


def extract_features(dataset: EpitDataset, row_indices: np.ndarray) -> np.ndarray:
    from scripts.epit_pipeline.split_data import _numeric

    return np.asarray(
        [
            [
                _numeric(dataset.table, dataset.rows[int(row_index)], column)
                for _, column in FEATURES
            ]
            for row_index in row_indices
        ],
        dtype=float,
    )


def fit_imputation_means(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    means = np.asarray(
        [
            np.mean(column[np.isfinite(column)])
            if np.isfinite(column).any()
            else np.nan
            for column in values.T
        ],
        dtype=float,
    )
    if not np.isfinite(means).all():
        missing = [FEATURES[i][0] for i in np.flatnonzero(~np.isfinite(means))]
        raise RuntimeError(f"Training fold has entirely missing features: {missing}.")
    return means


def apply_imputation(values: np.ndarray, means: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.where(np.isfinite(values), values, np.asarray(means, dtype=float))
    if not np.isfinite(result).all():
        raise RuntimeError("Feature imputation produced non-finite values.")
    return result


def rough_full_run_seconds(
    measured_seconds: float,
    measured_iterations: int,
    projected_iterations: int,
    projected_folds: int = INNER_FOLD_COUNT,
) -> float:
    if measured_seconds < 0 or measured_iterations <= 0 or projected_iterations <= 0:
        raise ValueError("Runtime projection inputs must be positive.")
    return measured_seconds * projected_iterations / measured_iterations * projected_folds


def _write_predictions(
    path: Path,
    fold: int,
    row_indices: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("fold", "task_row_index", "observed_epit_mV_SCE", "predicted_epit_mV_SCE"),
        )
        writer.writeheader()
        for row_index, y_true, y_pred in zip(
            row_indices, observed, predicted, strict=True
        ):
            writer.writerow(
                {
                    "fold": fold,
                    "task_row_index": int(row_index),
                    "observed_epit_mV_SCE": float(y_true),
                    "predicted_epit_mV_SCE": float(y_pred),
                }
            )


def _output_directory(path: Path | None, dry_run: bool) -> Path:
    if path is not None:
        output = path.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Output directory already exists: {output}")
        return output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = "dry_run" if dry_run else "fold_search"
    return DEFAULT_OUTPUT_ROOT / f"{prefix}_{timestamp}"


def run(args: argparse.Namespace) -> Path:
    if args.niterations is not None and args.niterations <= 0:
        raise ValueError("--niterations must be positive.")
    if args.procs is not None and args.procs <= 0:
        raise ValueError("--procs must be positive.")
    if args.procs is not None and args.parallelism != "multiprocessing":
        raise ValueError("--procs is only valid with --parallelism multiprocessing.")

    niterations = args.niterations or (
        DEFAULT_DRY_RUN_ITERATIONS if args.dry_run else DEFAULT_FULL_ITERATIONS
    )
    timeout_seconds = args.timeout_seconds
    if timeout_seconds is None and args.dry_run:
        timeout_seconds = DEFAULT_DRY_RUN_TIMEOUT_SECONDS
    folds = args.fold or ([1] if args.dry_run else list(range(1, 6)))
    if args.dry_run and len(folds) != 1:
        raise ValueError("--dry-run accepts exactly one --fold.")

    try:
        from pysr import PySRRegressor
    except ImportError as error:
        raise RuntimeError(
            "PySR is required. Install the corrosion extra with "
            "`pip install -e '.[corrosion-eval]'`."
        ) from error

    # Import the project loader after PySR/Juliacall. The loader reaches torch,
    # and Juliacall warns that importing torch first can cause a native crash.
    from scripts.epit_pipeline.split_data import load_epit_dataset

    dataset = load_epit_dataset()
    split = load_split_rows(args.split_manifest, dataset.n_rows)
    masked_target = dataset.target.copy()
    masked_target[split.final_test] = np.nan
    if not np.isnan(masked_target[split.final_test]).all():
        raise RuntimeError("Final-test target masking failed.")

    output_dir = _output_directory(args.output_dir, args.dry_run)
    output_dir.mkdir(parents=True, exist_ok=False)

    fold_results: list[dict[str, Any]] = []
    raw_feature_names = [name for name, _ in FEATURES]
    variable_names = list(MODEL_FEATURE_NAMES)
    for fold in folds:
        validation = split.validation_by_fold[int(fold)]
        context = np.setdiff1d(split.development, validation, assume_unique=True)
        train_raw = extract_features(dataset, context)
        validation_raw = extract_features(dataset, validation)
        means = fit_imputation_means(train_raw)
        X_train = apply_imputation(train_raw, means)
        X_validation = apply_imputation(validation_raw, means)
        y_train = masked_target[context]
        y_validation = masked_target[validation]

        run_id = f"fold_{fold}_seed_{args.random_state}"
        model = PySRRegressor(
            niterations=niterations,
            populations=args.populations,
            population_size=args.population_size,
            binary_operators=list(BINARY_OPERATORS),
            unary_operators=list(UNARY_OPERATORS),
            nested_constraints=NESTED_CONSTRAINTS,
            maxsize=args.maxsize,
            maxdepth=args.maxdepth,
            model_selection="best",
            parallelism=args.parallelism,
            procs=args.procs,
            random_state=args.random_state,
            timeout_in_seconds=timeout_seconds,
            progress=False,
            verbosity=1,
            output_directory=str(output_dir / "pysr_runs"),
            run_id=run_id,
        )
        started = time.perf_counter()
        model.fit(X_train, y_train, variable_names=variable_names)
        elapsed = time.perf_counter() - started
        predicted = np.asarray(model.predict(X_validation), dtype=float)
        if not np.isfinite(predicted).all():
            raise RuntimeError(f"Fold {fold} produced non-finite predictions.")

        correlation = float(spearmanr(y_validation, predicted).correlation)
        mae = float(np.mean(np.abs(y_validation - predicted)))
        equations = model.equations_.copy()
        export_columns = [
            column
            for column in ("pick", "score", "equation", "loss", "complexity")
            if column in equations.columns
        ]
        equations[export_columns].to_csv(
            output_dir / f"fold_{fold}_equations.csv", index=False
        )
        _write_predictions(
            output_dir / f"fold_{fold}_predictions.csv",
            fold,
            validation,
            y_validation,
            predicted,
        )
        fold_results.append(
            {
                "fold": int(fold),
                "context_rows": int(len(context)),
                "validation_rows": int(len(validation)),
                "fit_seconds": elapsed,
                "validation_spearman": correlation,
                "validation_mae_mV": mae,
                "selected_equation": str(model.sympy()),
                "imputation_means": {
                    name: float(value)
                    for name, value in zip(raw_feature_names, means, strict=True)
                },
            }
        )
        print(
            f"fold={fold} seconds={elapsed:.1f} "
            f"spearman={correlation:.4f} mae_mV={mae:.2f}"
        )
        print(f"equation={model.sympy()}")

    total_seconds = float(sum(result["fit_seconds"] for result in fold_results))
    projection = rough_full_run_seconds(
        measured_seconds=total_seconds / len(fold_results),
        measured_iterations=niterations,
        projected_iterations=args.projection_iterations,
    )
    summary = {
        "schema_version": "epit_pysr_exploratory_grammar_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run" if args.dry_run else "fold_search",
        "features": variable_names,
        "source_columns": [column for _, column in FEATURES],
        "operators": [*BINARY_OPERATORS, *UNARY_OPERATORS],
        "binary_operators": list(BINARY_OPERATORS),
        "unary_operators": list(UNARY_OPERATORS),
        "nested_constraints": NESTED_CONSTRAINTS,
        "niterations_per_fold": niterations,
        "populations": args.populations,
        "population_size": args.population_size,
        "maxsize": args.maxsize,
        "maxdepth": args.maxdepth,
        "random_state": args.random_state,
        "parallelism": args.parallelism,
        "procs": args.procs,
        "timeout_seconds_per_fold": timeout_seconds,
        "folds": folds,
        "final_test_rows": int(len(split.final_test)),
        "final_test_targets_used": False,
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": split.manifest_sha256,
        "split_lock_sha256": split.lock_sha256,
        "source_sha256": sha256_file(SOURCE_FILE),
        "pysr_version": version("pysr"),
        "runtime": {
            "measured_fit_seconds": total_seconds,
            "projection_iterations_per_fold": args.projection_iterations,
            "rough_projected_five_fold_seconds": projection,
            "projection_note": (
                "Linear indicator only; it includes first-run Julia startup/compilation "
                "and depends on CPU count and search behavior."
            ),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "fold_results": fold_results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"rough_projected_five_fold_seconds={projection:.1f}")
    print(f"output_dir={output_dir}")
    return output_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
