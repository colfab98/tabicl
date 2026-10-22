#!/usr/bin/env python
"""Run the MPEA external-evaluation protocol on 56 frozen Zhang 2023 queries."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import gc
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pandas as pd

from scripts import eval_corrosion_datasets as ev
from scripts.combined_corrosion.evaluate_final import add_evaluation_arguments, comparison_command, CATEGORICAL_COLUMNS
from scripts.combined_corrosion.evaluate_shared_specialists import DEFAULT_COMBINED_RUN, DEFAULT_EPIT_RUN, DEFAULT_SOCCOL_RUN
from scripts.combined_corrosion.evaluate_mpea_folds import DEFAULT_G50_RUN, MISSING_AWARE_MODELS
from corrosion_datasets.datasets.zhang2023_duplex_pitting.prepare_evaluation import (
    HERE, ROOT, CONTEXT_DIR, QUERY_NAME, SCORING_NAME, TASK_ID, FEATURE_COLUMNS, TARGET_COLUMN, file_sha256,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "processed")
    parser.add_argument("--combined-run", type=Path, default=DEFAULT_COMBINED_RUN)
    parser.add_argument("--epit-run", type=Path, default=DEFAULT_EPIT_RUN)
    parser.add_argument("--soccol-run", type=Path, default=DEFAULT_SOCCOL_RUN)
    parser.add_argument("--g50-run", type=Path, default=DEFAULT_G50_RUN)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    add_evaluation_arguments(parser, default_output_dir=ROOT / "corrosion_datasets/analysis/zhang2023_duplex_pitting/comparison_cpu")
    parser.set_defaults(device="cpu")
    return parser.parse_args(argv)


def shared_options(options: argparse.Namespace) -> argparse.Namespace:
    runs = [("combined_corrpfn_v2", options.combined_run), ("epit_shared_v2", options.epit_run),
            ("soccol_shared_v2", options.soccol_run), ("combined_g50_epit25_soccol25", options.g50_run),
            ("generic_tabicl", options.generic_run)]
    args = ev.parse_args(comparison_command(options, runs, missing_aware_models=MISSING_AWARE_MODELS)[2:])
    args.combined_data_dir = None  # This driver supplies the separately frozen external task.
    args.task = [TASK_ID]
    ev.validate_args(args)
    labels = {name for name, _ in runs}
    ev.validate_fixed_schema_model_labels(args, labels)
    ev.validate_missing_aware_model_labels(args, labels)
    return args


def load_task(data_dir: Path) -> tuple[ev.EvalTask, list[str], dict]:
    manifest = json.loads((data_dir / "preparation_manifest.json").read_text())
    if manifest.get("preparation_version") != "zhang2023_external_v1" or manifest.get("task_id") != TASK_ID:
        raise ValueError("Unsupported external preparation manifest.")
    for name in ("epit_soccol_context.csv", "shared_feature_registry.json", QUERY_NAME, SCORING_NAME, "row_audit.csv"):
        if file_sha256(data_dir / name) != manifest["outputs"][name]["sha256"]:
            raise ValueError(f"Frozen external artifact changed: {name}")
    for name in ("epit_soccol_context.csv", "shared_feature_registry.json"):
        if file_sha256(data_dir / name) != file_sha256(CONTEXT_DIR / name):
            raise ValueError(f"External evaluation must use the unchanged original {name}.")
    registry = json.loads((data_dir / "shared_feature_registry.json").read_text())
    if registry["feature_columns_in_order"] != list(FEATURE_COLUMNS):
        raise ValueError("Expected the existing 50-column schema in its original order.")
    context = pd.read_csv(data_dir / "epit_soccol_context.csv", low_memory=False)
    queries, scoring = pd.read_csv(data_dir / QUERY_NAME), pd.read_csv(data_dir / SCORING_NAME)
    if len(context) != 4787 or len(queries) != 56 or len(scoring) != 56:
        raise ValueError("Expected 4,787 context rows and all 56 external queries.")
    if context.source_dataset.value_counts().to_dict() != {"Soccol": 4027, "EPIT": 760}:
        raise ValueError("Unexpected context sources.")
    if not queries.source_dataset.eq("Zhang2023").all() or TARGET_COLUMN in queries:
        raise ValueError("External queries must be target-free Zhang 2023 features.")
    ids = queries.row_id.tolist()
    if len(set(ids)) != 56 or ids != scoring.row_id.tolist() or set(ids) & set(context.row_id):
        raise ValueError("Query IDs must be unique, aligned to targets, and disjoint from context.")
    if not queries.reference_type.eq("SCE").all() or not scoring.reference_type.eq("SCE").all():
        raise ValueError("External targets must retain SCE reference.")
    if list(queries.columns) != list(context.columns[:-1]):
        raise ValueError("Context and external query columns must match exactly.")
    X = pd.concat([context.loc[:, FEATURE_COLUMNS], queries.loc[:, FEATURE_COLUMNS]], ignore_index=True)
    y = pd.to_numeric(pd.concat([context[TARGET_COLUMN], scoring[TARGET_COLUMN]], ignore_index=True), errors="raise")
    if not np.isfinite(y).all():
        raise ValueError("Context and scoring targets must be finite.")
    split = ev.EvalSplit(np.arange(4787), np.arange(4787, 4843), "full_epit_soccol_context_frozen_zhang2023_queries")
    task = ev.EvalTask(
        task_id=TASK_ID, dataset="zhang2023_duplex_pitting", table="frozen_external_56",
        target=TARGET_COLUMN, threshold=np.nan, target_binning="continuous", target_bins=0,
        bin_edges=[], class_labels=[], class_counts={}, X=X, y=y,
        y_ordinal=pd.Series(np.zeros(len(y), dtype=int)), task_family="normal_corrosion",
        feature_groups_used=["material", "environment", "process_history"],
        feature_group_counts={"material": 24, "environment": 20, "process_history": 6},
        dropped_feature_columns=[], quality_flags=["seven_distinct_input_groups"], split_groups=None,
        split_strategy=split.split_strategy, include_in_summary=True, summary_exclusion_reason="", fixed_split=split,
    )
    return task, ids, manifest


def evaluate_job(job: dict, args: argparse.Namespace, data_dir: Path, threads: int) -> dict:
    import torch
    torch.set_num_threads(threads)
    task, row_ids, _ = load_task(data_dir)
    label, path = job["model"], job["path"]
    if path is not None and file_sha256(Path(path)) != job["sha256"]:
        raise ValueError(f"Checkpoint changed after freezing: {path}")
    if label == "catboost":
        kind = "catboost_regressor"
        factory = partial(ev.make_catboost_regressor, iterations=args.catboost_iterations,
                          depth=args.catboost_depth, learning_rate=args.catboost_learning_rate,
                          l2_leaf_reg=args.catboost_l2_leaf_reg, random_state=args.random_state,
                          thread_count=args.catboost_thread_count, categorical_columns=CATEGORICAL_COLUMNS)
    else:
        pretrained = label == "pretrained_tabicl_v2"
        kind = "pretrained_tabicl_regressor" if pretrained else "local_tabicl_regressor"
        shuffle, norms = ev.tabicl_inference_settings(args, label, pretrained=pretrained)
        factory = partial(ev.make_tabicl_regressor, model_path=path,
                          checkpoint_version=args.pretrained_checkpoint_version, device=args.device,
                          n_estimators=args.n_estimators, random_state=args.random_state,
                          allow_auto_download=pretrained, feat_shuffle_method=shuffle, norm_methods=norms)
    captured = {}

    def capturing_factory():
        estimator = factory()
        original_predict = estimator.predict

        def predict(*a, **kw):
            prediction = original_predict(*a, **kw)
            captured["prediction"] = np.asarray(prediction, dtype=float).reshape(-1).copy()
            return prediction

        estimator.predict = predict
        return estimator

    start = time.monotonic()
    row = ev.evaluate_estimator(
        model_label=label, model_kind=kind, estimator_factory=capturing_factory, task=task,
        task_split=task.fixed_split, test_size=args.test_size, random_state=args.random_state,
        regression_output="median", regression_uncertainty=False,
        preserve_feature_schema=label in MISSING_AWARE_MODELS,
        expected_preserve_missing_values=None if label == "catboost" else label in MISSING_AWARE_MODELS,
    )
    pred = captured["prediction"]
    if pred.shape != (56,) or not np.isfinite(pred).all():
        raise ValueError("Every external row must have one finite prediction.")
    row["elapsed_seconds"] = time.monotonic() - start
    result = {"job": job, "row": row, "predictions": [
        {"row_id": row_id, "y_true_mV_SCE": float(y), "y_pred_mV_SCE": float(p)}
        for row_id, y, p in zip(row_ids, task.y.iloc[task.fixed_split.test_index], pred, strict=True)
    ]}
    gc.collect()
    return result


def write_results(output: Path, protocol: dict, completed: dict, errors: list[dict]) -> None:
    rows, predictions = [], []
    for checkpoint in protocol["checkpoints"]:
        for key in checkpoint["jobs"]:
            if key not in completed:
                continue
            extra = {"checkpoint_name": checkpoint["name"], "checkpoint_step": checkpoint["step"]}
            result = completed[key]
            rows.append({**result["row"], **extra})
            predictions.extend({**p, **extra, "model": result["row"]["model"]} for p in result["predictions"])
    if not rows:
        return
    summary = ev.make_summary_dataframe(rows, errors, metric_sort_specs=[("test_spearman", False)])
    pd.DataFrame(rows).to_csv(output / "rows.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    ev.make_wide_results_dataframe(rows).to_csv(output / "wide.csv", index=False)
    plots = ev.write_checkpoint_trend_plots(summary, output / "plots", ["test_spearman", "test_pearson", "test_mae", "test_rmse", "test_r2"])
    payload = {"protocol": protocol, "status": "complete" if len(completed) == len(protocol["jobs"]) else "in_progress",
               "completed_jobs": len(completed), "total_jobs": len(protocol["jobs"]), "errors": errors,
               "rows": rows, "summary": summary.to_dict("records"), "plots": [str(p) for p in plots]}
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    if options.workers < 1 or options.torch_threads < 1 or (options.device != "cpu" and options.workers != 1):
        raise ValueError("Use positive CPU worker/thread counts; non-CPU inference requires one worker.")
    args = shared_options(options)
    _, _, manifest = load_task(options.data_dir)
    specs = ev.resolve_checkpoint_eval_specs(args)
    jobs = {label: {"key": label, "model": label, "path": None, "sha256": None}
            for label in ("catboost", "pretrained_tabicl_v2")}
    checkpoints = []
    for checkpoint in specs:
        keys = []
        for local in checkpoint.local_specs:
            key = f"{checkpoint.checkpoint_name}__{local.label}"
            jobs[key] = {"key": key, "model": local.label, "path": str(local.checkpoint_path),
                         "sha256": file_sha256(local.checkpoint_path)}
            keys.append(key)
        checkpoints.append({"name": checkpoint.checkpoint_name, "step": checkpoint.checkpoint_step,
                            "jobs": keys + ["catboost", "pretrained_tabicl_v2"]})
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(options).items()
                if key not in {"workers", "prepare_only"}}
    protocol = {"task_id": TASK_ID, "settings": settings, "preparation": manifest,
                "manifest_sha256": file_sha256(options.data_dir / "preparation_manifest.json"),
                "evaluation_script_sha256": file_sha256(Path(__file__)),
                "shared_evaluator_sha256": file_sha256(Path(ev.__file__)),
                "query_partition": "all 56 external rows; identical for every model and checkpoint",
                "context_partition": "all 4,787 original EPIT/Soccol rows; no Zhang or MPEA labels",
                "preprocessing_partition": "fit on context only, using each model's original native/fixed-schema settings",
                "interpretation": "Checkpoint trajectories are diagnostics, not external-label model selection; 56 repeats represent seven feature vectors.",
                "inference": {label: ev.tabicl_inference_settings(args, label, pretrained=label == "pretrained_tabicl_v2")
                              for label in (*MISSING_AWARE_MODELS, "generic_tabicl", "pretrained_tabicl_v2")},
                "checkpoints": checkpoints, "jobs": jobs}
    # Normalize tuples exactly as JSON will reload them, for safe resume comparison.
    protocol = json.loads(json.dumps(protocol))
    output = options.output_dir
    (output / "jobs").mkdir(parents=True, exist_ok=True)
    frozen = output / "protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise ValueError("Evaluation protocol changed; use a new output directory instead of mixing runs.")
    frozen.write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Frozen {len(jobs)} model/checkpoint jobs, 4,787 context rows, 56 queries, 50 features.", flush=True)
    if options.prepare_only:
        return
    completed, errors = {}, []
    for key, job in jobs.items():
        path = output / "jobs" / f"{key}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["job"] != job:
                raise ValueError(f"Saved job differs from frozen protocol: {key}")
            completed[key] = result
    with ProcessPoolExecutor(max_workers=options.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(evaluate_job, job, args, options.data_dir, options.torch_threads): key
                   for key, job in jobs.items() if key not in completed}
        for future in as_completed(pending):
            key = pending[future]
            try:
                result = future.result()
                path = output / "jobs" / f"{key}.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(result, indent=2) + "\n")
                temporary.replace(path)
                completed[key] = result
                row = result["row"]
                print(f"{len(completed)}/{len(jobs)} {key}: rho={row['test_spearman']:.4f}, RMSE={row['test_rmse']:.2f}, {row['elapsed_seconds']:.1f}s", flush=True)
            except Exception as exc:
                errors.append({"job": key, "model": jobs[key]["model"], "error": repr(exc)})
                print(f"ERROR {key}: {exc!r}", flush=True)
            write_results(output, protocol, completed, errors)
    write_results(output, protocol, completed, errors)
    if len(completed) != len(jobs):
        (output / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
        raise RuntimeError("Incomplete evaluation; inspect errors.json and rerun unchanged command to resume.")


if __name__ == "__main__":
    main()
