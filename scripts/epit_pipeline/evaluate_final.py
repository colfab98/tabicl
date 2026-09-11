#!/usr/bin/env python
"""Stage 5: evaluate the frozen model once on the untouched final-test set."""

from __future__ import annotations

import argparse
import json
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
    if int(split_design.get("development_rows", -1)) != 608:
        raise RuntimeError("Final EPIT evaluation requires exactly 608 context rows.")
    if int(split_design.get("final_test_rows", -1)) != 152:
        raise RuntimeError("Final EPIT evaluation requires exactly 152 final-test rows.")

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
    expected_n_features = 31 if config.get("pitting_magpie_features") else 21
    if int(config.get("expected_n_features", -1)) != expected_n_features:
        raise RuntimeError("Frozen EPIT feature count is inconsistent.")
    if int(config.get("max_samples_per_task", -1)) != 0:
        raise RuntimeError("Final evaluation must use every EPIT row.")
    return config


def final_evaluation_command(
    frozen: FrozenFinalModel,
    *,
    output_dir: Path,
) -> list[str]:
    config = validate_frozen_model(frozen)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "eval_corrosion_datasets.py"),
        "--local-ckpt-path",
        str(frozen.checkpoint_path),
        "--local-model-label",
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
    if bool(config["pitting_magpie_features"]):
        command.extend(["--pitting-magpie-model", FINAL_MODEL_LABEL])
    return command


def validate_final_results(
    payload: dict[str, Any],
    *,
    frozen: FrozenFinalModel,
    config: dict[str, Any],
) -> dict[str, Any]:
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
    if row.get("split_strategy") != "epit_pipeline_final_test":
        raise RuntimeError("Final evaluation did not use the frozen outer split.")
    if int(row.get("n_train", -1)) != 608 or int(row.get("n_test", -1)) != 152:
        raise RuntimeError("Final evaluation row counts are not 608/152.")
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


def run(args: argparse.Namespace) -> Path:
    frozen = load_frozen_final_model(args.final_model_manifest)
    config = validate_frozen_model(frozen)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else frozen.manifest_path.parent / "final_test_evaluation"
    )
    result_manifest_path = output_dir / FINAL_EVALUATION_MANIFEST_NAME
    if result_manifest_path.exists():
        raise FileExistsError("Final evaluation is already complete.")

    output_dir.mkdir(parents=True, exist_ok=True)
    command = final_evaluation_command(frozen, output_dir=output_dir)
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
    row = validate_final_results(
        evaluator_result,
        frozen=frozen,
        config=config,
    )
    result_manifest = {
        "schema_version": "epit_final_evaluation_manifest_v1",
        "completed_utc": utc_now(),
        "final_model_manifest": str(frozen.manifest_path),
        "final_model_manifest_sha256": frozen.manifest_sha256,
        "selected_checkpoint": str(frozen.checkpoint_path),
        "selected_checkpoint_sha256": frozen.checkpoint_sha256,
        "split_manifest": frozen.manifest["split"]["manifest"],
        "development_context_rows": 608,
        "final_test_rows": 152,
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
