#!/usr/bin/env python
"""Discover EPIT formulas on the matched Fe/Ni scope with categorical offsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "auto")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BINARY_OPERATORS = ("+", "-", "*", "/")
UNARY_OPERATORS = ("square", "sqrt", "log", "exp")
NESTED_CONSTRAINTS = {
    outer: {inner: 0 for inner in UNARY_OPERATORS}
    for outer in UNARY_OPERATORS
}
EXPECTED_DEVELOPMENT_ROWS = 608
EXPECTED_ELIGIBLE_ROWS = 452


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing bundle_manifest.json and the exported CSV.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fold", action="append", type=int, choices=range(1, 6))
    parser.add_argument("--niterations", type=int)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--populations", type=int, default=16)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--select-k-features", type=int, default=12)
    parser.add_argument("--maxsize", type=int, default=15)
    parser.add_argument("--maxdepth", type=int, default=5)
    return parser.parse_args()


def load_bundle(bundle_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "epit_pysr_discovery_bundle_v2":
        raise RuntimeError("Expected the v2 EPIT discovery bundle.")
    data_path = bundle_dir / str(manifest["data_file"])
    if sha256(data_path) != manifest["data_sha256"]:
        raise RuntimeError("EPIT development CSV hash mismatch.")
    if manifest.get("final_test_rows_exported") != 0:
        raise RuntimeError("Bundle unexpectedly contains final-test rows.")
    data = pd.read_csv(data_path)
    if len(data) != EXPECTED_DEVELOPMENT_ROWS:
        raise RuntimeError("Expected 608 development rows.")
    if sorted(data["validation_fold"].unique()) != [1, 2, 3, 4, 5]:
        raise RuntimeError("Expected five frozen validation folds.")
    allowed = set(manifest["eligible_material_classes"])
    eligible = data[data["material_class"].isin(allowed)].copy()
    if len(eligible) != EXPECTED_ELIGIBLE_ROWS:
        raise RuntimeError("Expected 452 Fe/Ni development rows.")
    return eligible, manifest


def fit_imputation_means(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    means = frame[features].mean()
    if not np.isfinite(means.to_numpy(float)).all():
        missing = means.index[~np.isfinite(means.to_numpy(float))].tolist()
        raise RuntimeError(f"Training fold has entirely missing features: {missing}")
    return means


def varying_feature_mask(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.std(values, axis=0) > 1e-12


def method_indicators(
    labels: pd.Series,
    offset_categories: list[str],
) -> np.ndarray:
    strings = labels.fillna("other").astype(str).to_numpy()
    return np.column_stack([strings == category for category in offset_categories]).astype(
        float
    )


def method_feature_names(offset_categories: list[str]) -> list[str]:
    return [f"method_{category}" for category in offset_categories]


def make_template(
    physical_features: list[str],
    offset_categories: list[str],
):
    from pysr import TemplateExpressionSpec

    method_features = method_feature_names(offset_categories)
    formula = f"f({', '.join(physical_features)})"
    for index, name in enumerate(method_features, start=1):
        formula += f" + method_offset[{index}] * {name}"
    return TemplateExpressionSpec(
        combine=formula,
        expressions=["f"],
        variable_names=[*physical_features, *method_features],
        parameters={"method_offset": len(method_features)},
    )


def label_template_equation(equation: str, physical_features: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        position = int(match.group(1)) - 1
        if position < 0 or position >= len(physical_features):
            return match.group(0)
        return physical_features[position]

    return re.sub(r"#(\d+)", replace, equation)


def selected_equation(model: Any, physical_features: list[str]) -> str:
    best = model.get_best()
    if "equation" in best:
        return label_template_equation(str(best["equation"]), physical_features)
    return label_template_equation(str(best), physical_features)


def run(args: argparse.Namespace) -> Path:
    from pysr import PySRRegressor, __version__ as pysr_version
    from pysr.feature_selection import run_feature_selection

    bundle_dir = args.bundle_dir.expanduser().resolve()
    data, manifest = load_bundle(bundle_dir)
    physical_features = list(manifest["physical_features"])
    offset_categories = list(manifest["method_offset_categories"])
    niterations = args.niterations or (100 if args.dry_run else 5_000)
    folds = args.fold or ([1] if args.dry_run else [1, 2, 3, 4, 5])
    seeds = args.seed or [42]
    if niterations <= 0 or args.select_k_features <= 0:
        raise ValueError("Iterations and selected-feature count must be positive.")
    if args.dry_run and (len(folds) != 1 or len(seeds) != 1):
        raise ValueError("Dry run requires exactly one fold and one seed.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir or bundle_dir / "results" / f"discovery_{timestamp}"
    ).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    fold_results: list[dict[str, Any]] = []
    predictions_by_seed: dict[int, list[pd.DataFrame]] = {seed: [] for seed in seeds}
    total_seconds = 0.0
    for seed in seeds:
        for fold in folds:
            context = data[data["validation_fold"] != fold]
            validation = data[data["validation_fold"] == fold]
            means = fit_imputation_means(context, physical_features)
            train_all = context[physical_features].fillna(means).to_numpy(float)
            validation_all = validation[physical_features].fillna(means).to_numpy(float)
            y_train = context["epit_mV_SCE"].to_numpy(float)
            y_validation = validation["epit_mV_SCE"].to_numpy(float)

            varying = varying_feature_mask(train_all)
            candidate_features = [
                name
                for name, keep in zip(physical_features, varying, strict=True)
                if keep
            ]
            train_candidates = train_all[:, varying]
            validation_candidates = validation_all[:, varying]
            k = min(args.select_k_features, len(candidate_features))
            selected_mask = run_feature_selection(
                train_candidates,
                y_train,
                k,
                random_state=np.random.RandomState(seed),
            )
            selected_features = [
                name
                for name, keep in zip(candidate_features, selected_mask, strict=True)
                if keep
            ]
            X_train = np.column_stack(
                [
                    train_candidates[:, selected_mask],
                    method_indicators(context["test_method_family"], offset_categories),
                ]
            )
            X_validation = np.column_stack(
                [
                    validation_candidates[:, selected_mask],
                    method_indicators(
                        validation["test_method_family"], offset_categories
                    ),
                ]
            )
            variable_names = [
                *selected_features,
                *method_feature_names(offset_categories),
            ]
            model = PySRRegressor(
                niterations=niterations,
                populations=args.populations,
                population_size=args.population_size,
                binary_operators=list(BINARY_OPERATORS),
                unary_operators=list(UNARY_OPERATORS),
                nested_constraints=NESTED_CONSTRAINTS,
                expression_spec=make_template(selected_features, offset_categories),
                maxsize=args.maxsize,
                maxdepth=args.maxdepth,
                model_selection="best",
                parallelism="multithreading",
                random_state=seed,
                timeout_in_seconds=600.0 if args.dry_run else None,
                progress=False,
                verbosity=1,
                output_directory=str(output_dir / "pysr_runs"),
                run_id=f"seed_{seed}_fold_{fold}",
            )
            started = time.perf_counter()
            model.fit(X_train, y_train, variable_names=variable_names)
            elapsed = time.perf_counter() - started
            total_seconds += elapsed
            prediction = np.asarray(model.predict(X_validation), dtype=float)
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"Seed {seed}, fold {fold} produced non-finite predictions.")
            spearman = float(spearmanr(y_validation, prediction).correlation)
            mae = float(np.mean(np.abs(y_validation - prediction)))
            equation = selected_equation(model, selected_features)
            equations = model.equations_.copy()
            if "equation" in equations:
                equations["labeled_equation"] = equations["equation"].map(
                    lambda value: label_template_equation(
                        str(value), selected_features
                    )
                )
            equations.to_csv(
                output_dir / f"seed_{seed}_fold_{fold}_equations.csv", index=False
            )
            prediction_frame = pd.DataFrame(
                {
                    "seed": seed,
                    "fold": fold,
                    "task_row_index": validation["task_row_index"].astype(int),
                    "observed_epit_mV_SCE": y_validation,
                    "predicted_epit_mV_SCE": prediction,
                }
            )
            prediction_frame.to_csv(
                output_dir / f"seed_{seed}_fold_{fold}_predictions.csv", index=False
            )
            predictions_by_seed[seed].append(prediction_frame)
            fold_results.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "context_rows": len(context),
                    "validation_rows": len(validation),
                    "fit_seconds": elapsed,
                    "validation_spearman": spearman,
                    "validation_mae_mV": mae,
                    "selected_features": selected_features,
                    "selected_equation": equation,
                }
            )
            print(
                f"seed={seed} fold={fold} seconds={elapsed:.1f} "
                f"spearman={spearman:.4f} mae_mV={mae:.2f}"
            )
            print(f"features={','.join(selected_features)}")
            print(f"equation={equation}")

    seed_results = []
    for seed, frames in predictions_by_seed.items():
        combined = pd.concat(frames, ignore_index=True)
        observed = combined["observed_epit_mV_SCE"].to_numpy(float)
        predicted = combined["predicted_epit_mV_SCE"].to_numpy(float)
        combined.to_csv(output_dir / f"seed_{seed}_oof_predictions.csv", index=False)
        seed_results.append(
            {
                "seed": seed,
                "rows": len(combined),
                "pooled_oof_spearman": float(spearmanr(observed, predicted).correlation),
                "pooled_oof_mae_mV": float(np.mean(np.abs(observed - predicted))),
            }
        )

    summary = {
        "schema_version": "epit_pysr_discovery_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "material_classes": manifest["eligible_material_classes"],
            "development_rows": len(data),
            "final_test_targets_used": False,
        },
        "physical_feature_candidates": physical_features,
        "feature_selection": {
            "method": "PySR random-forest selector fitted within each context fold",
            "select_k_features": args.select_k_features,
        },
        "categorical_effects": {
            "column": "test_method_family",
            "form": "additive category offsets outside the symbolic physical formula",
            "baseline": manifest["method_baseline_category"],
            "offset_categories": offset_categories,
        },
        "operators": [*BINARY_OPERATORS, *UNARY_OPERATORS],
        "nested_constraints": NESTED_CONSTRAINTS,
        "niterations_per_fit": niterations,
        "populations": args.populations,
        "population_size": args.population_size,
        "maxsize": args.maxsize,
        "maxdepth": args.maxdepth,
        "seeds": seeds,
        "folds": folds,
        "pysr_version": pysr_version,
        "runtime": {
            "measured_fit_seconds": total_seconds,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "seed_results": seed_results,
        "fold_results": fold_results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"output_dir={output_dir}")
    return output_dir


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
