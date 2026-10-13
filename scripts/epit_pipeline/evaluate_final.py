#!/usr/bin/env python
"""Stage 5: evaluate the frozen model once on the untouched final-test set."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline import run_optuna as search
from scripts.epit_pipeline.artifact_hashes import (
    FrozenFinalModel,
    load_frozen_final_model,
    load_frozen_split,
)
from scripts.epit_pipeline.train_final import (
    DEFAULT_FINAL_ROOT,
    FINAL_MODEL_MANIFEST_NAME,
)


PITTING_TASK_ID = "electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg"
FINAL_MODEL_LABEL = "final_epit_model"
FINAL_EVALUATION_MANIFEST_NAME = "final_evaluation_manifest.json"
FINAL_CONTEXT_FOLD_EVALUATION_MANIFEST_NAME = (
    "final_context_fold_evaluation_manifest.json"
)
CONTEXT_MODES = ("full", "development-folds")
DEVELOPMENT_FOLDS = (1, 2, 3, 4, 5)
DEVELOPMENT_ROWS = 608
FINAL_TEST_ROWS = 152
REQUIRED_FINAL_METRICS = (
    "test_spearman",
    "test_mae",
    "test_rmse",
    "test_r2",
    "test_pearson",
)


def default_final_model_manifest() -> Path:
    return (
        DEFAULT_FINAL_ROOT
        / search.slugify(search.DEFAULT_STUDY_NAME)
        / FINAL_MODEL_MANIFEST_NAME
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-model-manifest",
        type=Path,
        default=default_final_model_manifest(),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--context-mode",
        choices=CONTEXT_MODES,
        default="full",
        help=(
            "full uses every development row as labeled context. "
            "development-folds evaluates each of the five saved development "
            "folds separately as context and averages their metrics. Both "
            "modes always use the frozen outer final-test rows as queries."
        ),
    )
    parser.add_argument(
        "--context-folds",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Equivalent to --context-mode development-folds, with N required "
            "to match the number of saved development folds."
        ),
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_frozen_model(frozen: FrozenFinalModel) -> dict[str, Any]:
    manifest = frozen.manifest
    split_meta = manifest.get("split", {})
    split_manifest = Path(str(split_meta.get("manifest", ""))).resolve()
    frozen_split = load_frozen_split(split_manifest)
    if frozen_split.manifest_sha256 != split_meta.get("manifest_sha256"):
        raise RuntimeError("Final model names a different frozen split manifest.")
    if frozen_split.lock_sha256 != split_meta.get("lock_sha256"):
        raise RuntimeError("Final model names a different frozen split lock.")
    split_design = frozen_split.manifest.get("split_design", {})
    if int(split_design.get("development_rows", -1)) != DEVELOPMENT_ROWS:
        raise RuntimeError(
            f"Final EPIT evaluation requires exactly {DEVELOPMENT_ROWS} context rows."
        )
    if int(split_design.get("final_test_rows", -1)) != FINAL_TEST_ROWS:
        raise RuntimeError(
            f"Final EPIT evaluation requires exactly {FINAL_TEST_ROWS} final-test rows."
        )

    config = manifest.get("evaluation_configuration", {})
    if config.get("task_id") != PITTING_TASK_ID:
        raise RuntimeError("Frozen evaluation task is not the EPIT task.")
    if config.get("tabicl_norm_methods") != ["none"]:
        raise RuntimeError("Final evaluation must explicitly disable power.")
    if config.get("tabicl_feat_shuffle_method") != "none":
        raise RuntimeError("Final evaluation must preserve the fixed feature order.")
    if config.get("regression_output") != "median":
        raise RuntimeError("Unsupported frozen regression output.")
    if bool(config.get("regression_uncertainty", True)):
        raise RuntimeError("Unexpected uncertainty setting in final configuration.")
    if int(config.get("n_estimators", 0)) <= 0:
        raise RuntimeError("Frozen n_estimators must be positive.")
    expected_n_features = (
        search.EPIT_MAGPIE_TOTAL_FEATURE_COUNT
        if config.get("pitting_magpie_features")
        else search.EPIT_BASE_FEATURE_COUNT
    )
    if int(config.get("expected_n_features", -1)) != expected_n_features:
        raise RuntimeError("Frozen EPIT feature count is inconsistent.")
    if int(config.get("max_samples_per_task", -1)) != 0:
        raise RuntimeError("Final evaluation must use every EPIT row.")
    return config


def final_evaluation_command(
    frozen: FrozenFinalModel,
    *,
    output_dir: Path,
    context_fold: int | None = None,
) -> list[str]:
    config = validate_frozen_model(frozen)
    if context_fold is not None and context_fold not in DEVELOPMENT_FOLDS:
        raise ValueError(f"Unsupported development context fold: {context_fold}.")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_corrosion_datasets.py"),
        "--local-ckpt-path",
        str(frozen.checkpoint_path),
        "--local-model-label",
        FINAL_MODEL_LABEL,
        "--fixed-schema-model",
        FINAL_MODEL_LABEL,
        "--task",
        PITTING_TASK_ID,
        "--target-mode",
        "primary",
        "--target-binning",
        "continuous",
        "--max-samples-per-task",
        "0",
        "--epit-split-manifest",
        str(frozen.manifest["split"]["manifest"]),
        "--epit-final-test",
        "--device",
        str(config["device"]),
        "--random-state",
        str(config["random_state"]),
        "--n-estimators",
        str(config["n_estimators"]),
        "--tabicl-feat-shuffle-method",
        "none",
        "--tabicl-norm-methods",
        "none",
        "--regression-output",
        "median",
        "--no-regression-uncertainty",
        "--no-compare-pretrained-tabicl",
        "--output-json",
        str(output_dir / "results.json"),
        "--output-csv",
        str(output_dir / "rows.csv"),
        "--output-wide-csv",
        str(output_dir / "wide.csv"),
        "--output-summary-csv",
        str(output_dir / "summary.csv"),
    ]
    if context_fold is not None:
        command.extend(["--epit-final-context-fold", str(context_fold)])
    if bool(config["pitting_magpie_features"]):
        command.extend(["--pitting-magpie-model", FINAL_MODEL_LABEL])
    return command


def validate_final_results(
    payload: dict[str, Any],
    *,
    frozen: FrozenFinalModel,
    config: dict[str, Any],
    expected_context_rows: int | None = None,
    expected_split_strategy: str = "epit_pipeline_final_test",
) -> dict[str, Any]:
    if expected_context_rows is None:
        expected_context_rows = DEVELOPMENT_ROWS
    if payload.get("tabicl_feat_shuffle_method") != "none":
        raise RuntimeError("Final evaluation changed the feature order.")
    if int(payload.get("n_estimators", -1)) != int(config["n_estimators"]):
        raise RuntimeError("Final evaluation changed n_estimators.")
    if int(payload.get("random_state", -1)) != int(config["random_state"]):
        raise RuntimeError("Final evaluation changed random_state.")
    if int(payload.get("max_samples_per_task", -1)) != 0:
        raise RuntimeError("Final evaluation did not use every EPIT row.")
    if bool(payload.get("regression_uncertainty", True)):
        raise RuntimeError("Final evaluation changed uncertainty inference.")
    if payload.get("regression_output") != config["regression_output"]:
        raise RuntimeError("Final evaluation changed the regression output.")
    if Path(str(payload.get("local_checkpoint", ""))).resolve() != frozen.checkpoint_path:
        raise RuntimeError("Final evaluation used a different checkpoint.")
    if payload.get("errors"):
        raise RuntimeError(f"Final evaluation reported errors: {payload['errors']}")

    rows = list(payload.get("rows", []))
    if len(rows) != 1 or rows[0].get("model") != FINAL_MODEL_LABEL:
        raise RuntimeError("Final evaluation must contain only the frozen model.")
    row = rows[0]
    if row.get("task_id") != PITTING_TASK_ID:
        raise RuntimeError("Final evaluation produced an unexpected task.")
    if row.get("split_strategy") != expected_split_strategy:
        raise RuntimeError("Final evaluation did not use the frozen outer split.")
    if (
        int(row.get("n_train", -1)) != expected_context_rows
        or int(row.get("n_test", -1)) != FINAL_TEST_ROWS
    ):
        raise RuntimeError(
            "Final evaluation row counts are not "
            f"{expected_context_rows}/{FINAL_TEST_ROWS}."
        )
    if int(row.get("n_features", -1)) != int(config["expected_n_features"]):
        raise RuntimeError("Final model received the wrong feature schema.")
    if bool(row.get("pitting_magpie_features")) != bool(
        config["pitting_magpie_features"]
    ):
        raise RuntimeError("Final model Magpie routing changed.")
    for metric in REQUIRED_FINAL_METRICS:
        if metric not in row:
            raise RuntimeError(f"Final evaluation omitted {metric}.")
        float(row[metric])
    return row


def development_fold_counts(frozen: FrozenFinalModel) -> dict[int, int]:
    split_manifest = Path(str(frozen.manifest["split"]["manifest"])).resolve()
    manifest = load_frozen_split(split_manifest).manifest
    fold_field = (
        "development_validation_fold"
        if manifest.get("schema_version") == "soccol_composition_split_manifest_v1"
        else "optuna_validation_fold"
    )
    counts = {fold: 0 for fold in DEVELOPMENT_FOLDS}
    for record in manifest.get("rows", []):
        if record.get("outer_split") != "development":
            continue
        fold = int(record[fold_field])
        if fold not in counts:
            raise RuntimeError(f"Unexpected saved development fold: {fold}.")
        counts[fold] += 1
    if any(count <= 0 for count in counts.values()):
        raise RuntimeError("Every saved development context fold must be non-empty.")
    if sum(counts.values()) != DEVELOPMENT_ROWS:
        raise RuntimeError("Saved development context folds have the wrong row count.")
    return counts


def execute_evaluation(
    frozen: FrozenFinalModel,
    *,
    config: dict[str, Any],
    output_dir: Path,
    context_fold: int | None = None,
    expected_context_rows: int | None = None,
) -> dict[str, Any]:
    if expected_context_rows is None:
        expected_context_rows = DEVELOPMENT_ROWS
    output_dir.mkdir(parents=True, exist_ok=True)
    command = final_evaluation_command(
        frozen,
        output_dir=output_dir,
        context_fold=context_fold,
    )
    with (output_dir / "evaluate.log").open("w", encoding="utf-8") as log_file:
        log_file.write(" ".join(command) + "\n\n")
        log_file.flush()
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

    evaluator_result = json.loads(
        (output_dir / "results.json").read_text(encoding="utf-8")
    )
    split_strategy = (
        "epit_pipeline_final_test"
        if context_fold is None
        else f"epit_pipeline_final_test_context_fold_{context_fold}"
    )
    return validate_final_results(
        evaluator_result,
        frozen=frozen,
        config=config,
        expected_context_rows=expected_context_rows,
        expected_split_strategy=split_strategy,
    )


def run_development_context_folds(
    frozen: FrozenFinalModel,
    *,
    config: dict[str, Any],
    output_dir: Path,
) -> Path:
    result_manifest_path = (
        output_dir / FINAL_CONTEXT_FOLD_EVALUATION_MANIFEST_NAME
    )
    if result_manifest_path.exists():
        raise FileExistsError("Final context-fold evaluation is already complete.")

    fold_counts = development_fold_counts(frozen)
    rows: list[dict[str, Any]] = []
    for fold in DEVELOPMENT_FOLDS:
        row = execute_evaluation(
            frozen,
            config=config,
            output_dir=output_dir / f"context_fold_{fold}",
            context_fold=fold,
            expected_context_rows=fold_counts[fold],
        )
        rows.append({"context_fold": fold, **row})

    aggregate: dict[str, float] = {}
    for metric in REQUIRED_FINAL_METRICS:
        values = [float(row[metric]) for row in rows]
        aggregate[f"mean_{metric}"] = statistics.fmean(values)
        aggregate[f"sample_sd_{metric}"] = statistics.stdev(values)

    result_manifest = {
        "schema_version": "epit_final_context_fold_evaluation_manifest_v1",
        "completed_utc": utc_now(),
        "final_model_manifest": str(frozen.manifest_path),
        "final_model_manifest_sha256": frozen.manifest_sha256,
        "selected_checkpoint": str(frozen.checkpoint_path),
        "selected_checkpoint_sha256": frozen.checkpoint_sha256,
        "split_manifest": frozen.manifest["split"]["manifest"],
        "query_partition": "frozen_outer_final_test",
        "final_test_rows_per_evaluation": FINAL_TEST_ROWS,
        "context_mode": "development-folds",
        "development_context_rows_by_fold": {
            str(fold): count for fold, count in fold_counts.items()
        },
        "metrics_averaged_across_context_folds": True,
        "predictions_averaged_across_context_folds": False,
        "selection_or_tuning_performed": False,
        "evaluation_configuration": config,
        "fold_results": rows,
        "aggregate": aggregate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_manifest_path, result_manifest)
    print(f"Final context-fold evaluation completed: {result_manifest_path}")
    print(
        f"Mean Spearman={aggregate['mean_test_spearman']:.6g}, "
        f"Mean MAE={aggregate['mean_test_mae']:.6g}, "
        f"Mean RMSE={aggregate['mean_test_rmse']:.6g}"
    )
    return result_manifest_path


def run(args: argparse.Namespace) -> Path:
    frozen = load_frozen_final_model(args.final_model_manifest)
    config = validate_frozen_model(frozen)
    use_context_folds = (
        args.context_mode == "development-folds" or args.context_folds is not None
    )
    if args.context_folds is not None and args.context_folds != len(DEVELOPMENT_FOLDS):
        raise ValueError(
            f"--context-folds must be {len(DEVELOPMENT_FOLDS)} for the frozen split."
        )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else frozen.manifest_path.parent
        / (
            "final_test_evaluation"
            if not use_context_folds
            else "final_test_context_fold_evaluation"
        )
    )
    if use_context_folds:
        return run_development_context_folds(
            frozen,
            config=config,
            output_dir=output_dir,
        )

    result_manifest_path = output_dir / FINAL_EVALUATION_MANIFEST_NAME
    if result_manifest_path.exists():
        raise FileExistsError("Final evaluation is already complete.")

    row = execute_evaluation(
        frozen=frozen,
        config=config,
        output_dir=output_dir,
    )
    result_manifest = {
        "schema_version": "epit_final_evaluation_manifest_v1",
        "completed_utc": utc_now(),
        "final_model_manifest": str(frozen.manifest_path),
        "final_model_manifest_sha256": frozen.manifest_sha256,
        "selected_checkpoint": str(frozen.checkpoint_path),
        "selected_checkpoint_sha256": frozen.checkpoint_sha256,
        "split_manifest": frozen.manifest["split"]["manifest"],
        "development_context_rows": DEVELOPMENT_ROWS,
        "final_test_rows": FINAL_TEST_ROWS,
        "selection_or_tuning_performed": False,
        "evaluation_configuration": config,
        "result": row,
    }
    write_json(result_manifest_path, result_manifest)
    print(f"Final evaluation completed: {result_manifest_path}")
    print(
        f"Spearman={float(row['test_spearman']):.6g}, "
        f"MAE={float(row['test_mae']):.6g}, "
        f"RMSE={float(row['test_rmse']):.6g}"
    )
    return result_manifest_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
