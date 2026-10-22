#!/usr/bin/env python
"""Compare seven models on a separate, frozen MPEA-only five-fold benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from functools import partial
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import eval_corrosion_datasets as ev
from scripts.combined_corrosion.evaluate_final import (
    CATEGORICAL_COLUMNS, REPO_ROOT,
    add_evaluation_arguments, comparison_command,
)
from scripts.combined_corrosion.evaluate_shared_specialists import (
    DEFAULT_COMBINED_RUN, DEFAULT_EPIT_RUN, DEFAULT_SOCCOL_RUN, RUN_ROOT,
)
from scripts.epit_pipeline.split_data import cluster_composition_vectors
from tabicl.prior.combined_corrosion import COMBINED_COMPOSITION_COLUMNS, COMBINED_FEATURE_COLUMNS


TASK_ID = "mpea_only__pitting_potential_mv_sce"
N_FOLDS = 5
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "corrosion_datasets/analysis/combined_epit_soccol_mpea/mpea_only_fivefold_comparison"
)
DEFAULT_G50_RUN = RUN_ROOT / "final_v2_g50_epit25_soccol25_seed42"
MISSING_AWARE_MODELS = (
    "combined_corrpfn_v2", "epit_shared_v2", "soccol_shared_v2", "combined_g50_epit25_soccol25",
)
MODEL_LABELS = (*MISSING_AWARE_MODELS, "generic_tabicl", "pretrained_tabicl_v2", "catboost")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-run", type=Path, default=DEFAULT_COMBINED_RUN)
    parser.add_argument("--epit-run", type=Path, default=DEFAULT_EPIT_RUN)
    parser.add_argument("--soccol-run", type=Path, default=DEFAULT_SOCCOL_RUN)
    parser.add_argument("--g50-run", type=Path, default=DEFAULT_G50_RUN)
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="Frozen fold assignments; defaults to DATA_DIR/mpea_only_folds_v1.json.")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for target-blind group assignment; independent of model random state.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Freeze/verify the five outer folds and exit without loading models.")
    add_evaluation_arguments(parser, default_output_dir=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def shared_options(options: argparse.Namespace) -> argparse.Namespace:
    runs = [
        ("combined_corrpfn_v2", options.combined_run),
        ("epit_shared_v2", options.epit_run),
        ("soccol_shared_v2", options.soccol_run),
        ("combined_g50_epit25_soccol25", options.g50_run),
        ("generic_tabicl", options.generic_run),
    ]
    command = comparison_command(options, runs, missing_aware_models=MISSING_AWARE_MODELS)
    args = ev.parse_args(command[2:])
    # This driver constructs its own outer folds, never the combined final task.
    args.combined_data_dir = None
    args.task = [TASK_ID]
    ev.validate_args(args)
    ev.validate_fixed_schema_model_labels(args, {name for name, _ in runs})
    ev.validate_missing_aware_model_labels(args, {name for name, _ in runs})
    return args


def load_mpea(data_dir: Path) -> tuple[ev.EvalTask, list[str], dict[str, str]]:
    """Read only the frozen MPEA features/labels and shared registry, not source context."""
    preparation = json.loads((data_dir / ev.COMBINED_MANIFEST_NAME).read_text())
    if preparation.get("preparation_version") != "combined_context_query_v2":
        raise ValueError("Unsupported MPEA preparation version.")
    hashes = {}
    for name in (ev.COMBINED_QUERY_NAME, ev.COMBINED_SCORING_NAME, ev.COMBINED_REGISTRY_NAME):
        hashes[name] = ev._sha256_file(data_dir / name)
        if hashes[name] != preparation.get("outputs", {}).get(name, {}).get("sha256"):
            raise ValueError(f"Frozen MPEA artifact changed: {name}")
    queries = pd.read_csv(data_dir / ev.COMBINED_QUERY_NAME)
    scoring = pd.read_csv(data_dir / ev.COMBINED_SCORING_NAME)
    registry = json.loads((data_dir / ev.COMBINED_REGISTRY_NAME).read_text())
    columns = list(COMBINED_FEATURE_COLUMNS)
    target = "reported_pitting_potential_mV"
    if registry.get("feature_columns_in_order") != columns or registry.get("target_column") != target:
        raise ValueError("Expected the frozen 50-column feature registry.")
    ids = queries["row_id"].astype(str).tolist()
    if len(ids) != 267 or len(set(ids)) != 267 or ids != scoring["row_id"].astype(str).tolist():
        raise ValueError("Expected the same 267 unique, aligned MPEA rows and targets.")
    if not queries["source_dataset"].eq("MPEA").all():
        raise ValueError("The new benchmark must contain MPEA rows only.")
    if not queries["reference_type"].eq("SCE").all() or not scoring["reference_type"].eq("SCE").all():
        raise ValueError("Every MPEA row must use the SCE scale.")
    if target in queries:
        raise ValueError("Query features must not contain the target.")
    y = pd.to_numeric(scoring[target], errors="raise")
    if not np.isfinite(y).all():
        raise ValueError("MPEA targets must be finite.")
    task = ev.EvalTask(
        task_id=TASK_ID, dataset="mpea_only", table="frozen_mpea_sce_267", target=target,
        threshold=np.nan, target_binning="continuous", target_bins=0, bin_edges=[],
        class_labels=[], class_counts={}, X=queries[columns].copy(), y=y,
        y_ordinal=pd.Series(np.zeros(len(y), dtype=int)), task_family="normal_corrosion",
        feature_groups_used=["material", "environment", "process_history"],
        feature_group_counts={"material": 24, "environment": 20, "process_history": 6},
        dropped_feature_columns=[], quality_flags=[], split_groups=None,
        split_strategy="mpea_only_frozen_outer_fivefold", include_in_summary=True,
        summary_exclusion_reason="",
    )
    return task, ids, hashes


def composition_folds(X: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Target-blind, row-balanced assignment of complete composition components."""
    # Match the project's grouping convention; this fill never touches model inputs.
    vectors = X.loc[:, COMBINED_COMPOSITION_COLUMNS].fillna(0).to_numpy(dtype=float).round(2)
    if not np.isfinite(vectors).all():
        raise ValueError("Composition grouping requires finite reported values.")
    groups = cluster_composition_vectors(vectors, distance_threshold=1.0)
    counts = np.bincount(groups)
    if len(counts) < N_FOLDS:
        raise ValueError("At least five composition groups are required.")
    rng = np.random.default_rng(seed)
    order = sorted(rng.permutation(len(counts)), key=lambda group: -counts[group])
    sizes = np.zeros(N_FOLDS, dtype=int)
    assignments = np.zeros(len(counts), dtype=int)
    for group in order:
        fold = int(rng.choice(np.flatnonzero(sizes == sizes.min())))
        assignments[group] = fold + 1
        sizes[fold] += counts[group]
    if sizes.min() < 2:
        raise ValueError("Every outer query fold needs at least two rows.")
    return groups, assignments[groups]


def freeze_folds(task, row_ids, hashes, path: Path, seed: int) -> dict:
    groups, folds = composition_folds(task.X, seed)
    manifest = {
        "schema_version": "mpea_only_outer_fivefold_v1",
        "scope": "Exploratory MPEA adaptation benchmark; separate from combined final evaluation.",
        "n_folds": N_FOLDS, "split_seed": seed, "input_sha256": hashes,
        "feature_columns": list(task.X.columns),
        "grouping": {"columns": list(COMBINED_COMPOSITION_COLUMNS), "round_decimals": 2,
                     "missing_fill_for_grouping_only": 0, "l1_threshold_wt_percent": 1.0,
                     "method": "threshold_connected_components"},
        "assignment": "descending_group_size_greedy_row_balance_seeded_ties_v1",
        "targets_used_for_assignment": False,
        "rows": [{"row_id": row_id, "composition_group": int(group), "outer_fold": int(fold)}
                 for row_id, group, fold in zip(row_ids, groups, folds, strict=True)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Frozen MPEA folds differ from the data/settings; use a new split manifest.")
    else:
        with path.open("x") as handle:
            handle.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def fold_tasks(task: ev.EvalTask, manifest: dict) -> list[ev.EvalTask]:
    rows = pd.DataFrame(manifest["rows"])
    result = []
    for fold in range(1, N_FOLDS + 1):
        query = np.flatnonzero(rows.outer_fold.to_numpy() == fold)
        context = np.flatnonzero(rows.outer_fold.to_numpy() != fold)
        if set(rows.composition_group.iloc[context]) & set(rows.composition_group.iloc[query]):
            raise ValueError("A composition group crosses an outer fold boundary.")
        split = ev.EvalSplit(context, query, task.split_strategy)
        result.append(replace(
            task, task_id=f"{TASK_ID}__outer_fold_{fold}",
            split_groups=rows.composition_group.copy(), fixed_split=split,
        ))
    return result


def summarize(
    rows: list[dict], expected_models_by_checkpoint: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Equal fold means and sample SDs; never silently omit a missing/nonfinite metric."""
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No complete checkpoint results.")
    for checkpoint, block in df.groupby("checkpoint_name", sort=False):
        models = (
            expected_models_by_checkpoint[checkpoint]
            if expected_models_by_checkpoint is not None else MODEL_LABELS
        )
        expected = {(model, fold) for model in models for fold in range(1, N_FOLDS + 1)}
        actual = Counter(zip(block.model, block.outer_fold))
        if set(actual) != expected or any(count != 1 for count in actual.values()):
            raise ValueError(f"{checkpoint} requires every scheduled model on all five folds exactly once.")
    summary = ev.make_summary_dataframe(rows, [], metric_sort_specs=[("test_spearman", False)])
    grouped = df.groupby(["checkpoint_name", "model"], sort=False)
    metrics = [metric for metric in ev.METRIC_COLUMNS if metric in df]
    stats = grouped[metrics].std(ddof=1).add_prefix("std_").reset_index()
    summary = summary.merge(stats, on=["checkpoint_name", "model"], validate="one_to_one")
    summary["n_folds"] = N_FOLDS
    for metric in metrics:
        valid_counts = grouped[metric].count().rename(f"n_valid_folds_{metric}").reset_index()
        summary = summary.merge(valid_counts, on=["checkpoint_name", "model"], validate="one_to_one")
        incomplete = summary[f"n_valid_folds_{metric}"] != N_FOLDS
        # Correlations can be undefined for constant predictions. Expose that,
        # rather than reporting a four-fold score as a five-fold average.
        summary.loc[incomplete, [f"mean_{metric}", f"median_{metric}",
                                 f"weighted_mean_{metric}", f"std_{metric}"]] = np.nan
    return summary


def evaluate_model(args, tasks, label, path=None) -> list[dict]:
    if label == "catboost":
        kind = "catboost_regressor"
        factory = partial(
            ev.make_catboost_regressor, iterations=args.catboost_iterations,
            depth=args.catboost_depth, learning_rate=args.catboost_learning_rate,
            l2_leaf_reg=args.catboost_l2_leaf_reg, random_state=args.random_state,
            thread_count=args.catboost_thread_count, categorical_columns=CATEGORICAL_COLUMNS,
        )
    else:
        pretrained = label == "pretrained_tabicl_v2"
        kind = "pretrained_tabicl_regressor" if pretrained else "local_tabicl_regressor"
        shuffle, norms = ev.tabicl_inference_settings(args, label, pretrained=pretrained)
        factory = partial(
            ev.make_tabicl_regressor, model_path=str(path) if path is not None else None,
            checkpoint_version=args.pretrained_checkpoint_version, device=args.device,
            n_estimators=args.n_estimators, random_state=args.random_state,
            allow_auto_download=pretrained, feat_shuffle_method=shuffle, norm_methods=norms,
        )
        # Reuse weights across folds, but fit a fresh estimator each time.
        factory = ev.make_cached_tabicl_factory(factory, cache_key=(label, str(path)), model_cache={})
    records = []
    for fold, task in enumerate(tasks, 1):
        row = ev.evaluate_estimator(
            model_label=label, model_kind=kind, estimator_factory=factory,
            task=task, task_split=task.fixed_split, test_size=args.test_size,
            random_state=args.random_state, regression_output="median", regression_uncertainty=False,
            preserve_feature_schema=label in MISSING_AWARE_MODELS,
            expected_preserve_missing_values=(
                None if label == "catboost" else label in MISSING_AWARE_MODELS
            ),
        )
        row["outer_fold"] = fold
        records.append(row)
        print(f"{label} | {path.name if path else 'baseline'} | fold {fold}: "
              f"Spearman={row['test_spearman']:.4f}, RMSE={row['test_rmse']:.1f}", flush=True)
    return records


def write_results(output: Path, rows: list[dict], metadata: dict) -> None:
    expected_models = (
        {checkpoint["name"]: set(checkpoint["models"]) | {"pretrained_tabicl_v2", "catboost"}
         for checkpoint in metadata["checkpoint_evaluations"]}
        if "checkpoint_evaluations" in metadata else None
    )
    summary = summarize(rows, expected_models)
    # One wide row per checkpoint; the metrics here are fold averages and SDs.
    wide_rows = []
    for checkpoint, block in summary.groupby("checkpoint_name", sort=False):
        record = {"checkpoint_name": checkpoint, "checkpoint_step": block.checkpoint_step.iloc[0],
                  "task_id": TASK_ID, "n_folds": N_FOLDS}
        for _, row in block.iterrows():
            for metric in ev.METRIC_COLUMNS:
                for prefix in ("mean_", "std_"):
                    column = prefix + metric
                    if column in row:
                        record[f"{column}__{row['model']}"] = row[column]
        wide_rows.append(record)
    wide = pd.DataFrame(wide_rows)
    plots = ev.write_checkpoint_trend_plots(summary, output / "plots", [
        "test_spearman", "test_pearson", "test_mae", "test_rmse", "test_r2",
        "test_nmae_iqr", "test_nrmse_iqr",
    ])
    pd.DataFrame(rows).to_csv(output / "rows.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    wide.to_csv(output / "wide.csv", index=False)
    for fold in range(1, N_FOLDS + 1):
        directory = output / f"outer_fold_{fold}"
        directory.mkdir(exist_ok=True)
        selected = [row for row in rows if row["outer_fold"] == fold]
        pd.DataFrame(selected).to_csv(directory / "rows.csv", index=False)
        ev.make_wide_results_dataframe(selected).to_csv(directory / "wide.csv", index=False)
        ev.make_summary_dataframe(selected, [], metric_sort_specs=[("test_spearman", False)]).to_csv(
            directory / "summary.csv", index=False)
    payload = {**metadata, "rows": rows, "errors": [], "summary": summary.to_dict(orient="records"),
               "plots": [str(path) for path in plots]}
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    args = shared_options(options)
    data_dir = options.combined_data_dir.expanduser().resolve()
    task, row_ids, hashes = load_mpea(data_dir)
    manifest_path = (options.split_manifest or data_dir / "mpea_only_folds_v1.json").expanduser().resolve()
    manifest = freeze_folds(task, row_ids, hashes, manifest_path, options.split_seed)
    tasks = fold_tasks(task, manifest)
    print(f"Frozen MPEA outer folds: {[len(t.fixed_split.test_index) for t in tasks]} query rows; "
          f"{len(set(row['composition_group'] for row in manifest['rows']))} composition groups.")
    if options.prepare_only:
        print(f"Split manifest: {manifest_path}")
        return
    checkpoints = ev.resolve_checkpoint_eval_specs(args)
    expected_models = {
        checkpoint.checkpoint_name:
            {local.label for local in checkpoint.local_specs} | {"pretrained_tabicl_v2", "catboost"}
        for checkpoint in checkpoints
    }
    for checkpoint in checkpoints:
        steps = {ev.checkpoint_step_from_path(local.checkpoint_path) for local in checkpoint.local_specs}
        if len(steps) != 1 or None in steps:
            raise ValueError("All local models must use the same saved step; use --checkpoint all or an explicit step.")
    output = options.output_dir.expanduser().resolve()
    # Reuse the output directory; result writers replace the previous run's files.
    output.mkdir(parents=True, exist_ok=True)
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pd.DataFrame(manifest["rows"]).to_csv(output / "split_assignments.csv", index=False)
    metadata = {
        "benchmark": TASK_ID, "scope": manifest["scope"], "n_folds": N_FOLDS,
        "query_partition": "new_frozen_mpea_outer_folds", "context_partition": "other_four_mpea_folds",
        "preprocessing_partition": "current_fold_context_only",
        "aggregation": "unweighted mean and sample standard deviation of five fold metrics",
        "predictions_averaged_across_folds": False, "split_manifest": str(manifest_path),
        "split_manifest_sha256": ev._sha256_file(manifest_path), "input_sha256": hashes,
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(options).items()},
        "checkpoint_evaluations": [{"name": c.checkpoint_name, "step": c.checkpoint_step,
                                    "models": {s.label: str(s.checkpoint_path) for s in c.local_specs}}
                                   for c in checkpoints],
        "inference": {label: ev.tabicl_inference_settings(args, label, pretrained=label == "pretrained_tabicl_v2")
                      for label in MODEL_LABELS if label != "catboost"},
        "completed_checkpoints": [],
    }
    rows = []
    try:
        static = []
        for label in ("pretrained_tabicl_v2", "catboost"):
            static.extend(evaluate_model(args, tasks, label))
            gc.collect()
        for checkpoint in checkpoints:
            current = list(static)
            for local in checkpoint.local_specs:
                current.extend(evaluate_model(args, tasks, local.label, local.checkpoint_path))
                gc.collect()
            current = [{**row, "checkpoint_name": checkpoint.checkpoint_name,
                        "checkpoint_step": checkpoint.checkpoint_step} for row in current]
            summarize(current, expected_models)  # Require five folds for every scheduled model.
            rows.extend(current)
            metadata["completed_checkpoints"].append(checkpoint.checkpoint_name)
            write_results(output, rows, metadata)
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({**metadata, "error": str(exc)}, indent=2) + "\n")
        raise
    (output / "failure.json").unlink(missing_ok=True)
    print(f"Saved MPEA-only five-fold comparison to {output}")


if __name__ == "__main__":
    main()
