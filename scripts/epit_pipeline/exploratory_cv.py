"""Exploratory evaluation of existing EPIT checkpoints; no prior refitting.

Fold 1 preserves the frozen outer queries. Folds 2–5 partition development
rows, which have already influenced informed training. Their mean is not an
independent final-test estimate. The original split artifacts remain intact.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.epit_pipeline.evaluation_identity import frozen_membership

CV_SEED = 42
FOLDS = [1, 2, 3, 4, 5]


def fold_membership(manifest_path, fold):
    if fold not in FOLDS:
        raise ValueError("Exploratory CV fold must be between 1 and 5.")
    frozen = frozen_membership(manifest_path)
    development = np.asarray(frozen["context_task_row_indices"], dtype=int)
    final = frozen["query_task_row_indices"]
    if len(development) != 608 or len(final) != 152:
        raise ValueError("Exploratory EPIT CV requires the 608/152 split.")
    queries = [final] + [sorted(chunk.tolist()) for chunk in np.split(
        np.random.default_rng(CV_SEED).permutation(development), 4
    )]
    query = queries[fold - 1]
    context = sorted((set(development.tolist()) | set(final)) - set(query))
    return {"context_task_row_indices": context, "query_task_row_indices": query}


def apply_fold(tasks, args, ev):
    task, _, _, _ = ev._load_epit_pipeline_indices(tasks, manifest_path=args.epit_split_manifest)
    membership = fold_membership(args.epit_split_manifest, args.epit_exploratory_cv_fold)
    context = np.asarray(membership["context_task_row_indices"], dtype=int)
    task.fixed_split = ev.EvalSplit(
        train_index=context,
        test_index=np.asarray(membership["query_task_row_indices"], dtype=int),
        preprocessing_index=context.copy(),
        split_strategy=f"epit_pipeline_exploratory_cv_fold_{args.epit_exploratory_cv_fold}",
    )
    task.split_strategy = task.fixed_split.split_strategy


def child_arguments(argv):
    """Remove only options owned by the CV driver; preserve model options."""
    values = {"--exploratory-cv-folds", "--output-json", "--output-csv",
              "--output-wide-csv", "--output-summary-csv", "--output-plot-dir"}
    flags = {"--no-checkpoint-plots", "--save-task-tables"}
    result = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        option = arg.split("=", 1)[0]
        if option in values:
            index += 1 if "=" in arg else 2
        elif option in flags:
            index += 1
        else:
            result.append(arg)
            index += 1
    return result


def save_checkpoint_snapshot(specs, path):
    records = []
    for spec in specs:
        models = []
        for model in spec.local_specs:
            checkpoint = model.checkpoint_path.resolve()
            stat = checkpoint.stat()
            models.append({"label": model.label, "path": str(checkpoint),
                           "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        records.append({"name": spec.checkpoint_name, "step": spec.checkpoint_step, "models": models})
    path.write_text(json.dumps({"version": 1, "checkpoints": records}, indent=2) + "\n")


def load_checkpoint_snapshot(path, ev):
    payload = json.loads(Path(path).read_text())
    if payload.get("version") != 1 or not payload.get("checkpoints"):
        raise ValueError("Invalid CV checkpoint snapshot.")
    specs = []
    for record in payload["checkpoints"]:
        models = []
        for model in record["models"]:
            checkpoint = Path(model["path"])
            stat = checkpoint.stat()
            if (stat.st_size, stat.st_mtime_ns) != (model["size"], model["mtime_ns"]):
                raise RuntimeError(f"CV checkpoint changed since evaluation started: {checkpoint}")
            models.append(ev.LocalModelSpec(label=model["label"], checkpoint_path=checkpoint))
        specs.append(ev.CheckpointEvalSpec(checkpoint_name=record["name"],
                     checkpoint_step=record["step"], local_specs=tuple(models)))
    return specs


def complete_rows_only(rows, *, step_interval=0):
    """Recover only identical model/checkpoint tasks scored exactly once per fold."""
    groups = {}
    for row in rows:
        key = (row.get("checkpoint_name"), row["model"], row["task_id"])
        groups.setdefault(key, []).append(row)
    kept, dropped = [], []
    for (checkpoint, model, task), group in groups.items():
        folds = [row["cv_fold"] for row in group]
        if len(folds) != len(set(folds)):
            raise RuntimeError(f"Duplicate fold scores for {model}@{checkpoint}.")
        step = group[0].get("checkpoint_step")
        if step_interval and step is not None and int(step) % step_interval:
            dropped.append({"checkpoint_name": checkpoint, "model": model,
                            "task_id": task, "folds": sorted(folds), "reason": "outside_checkpoint_interval"})
            continue
        if set(folds) != set(FOLDS):
            dropped.append({"checkpoint_name": checkpoint, "model": model,
                            "task_id": task, "folds": sorted(folds)})
            continue
        if len({row.get("model_source") for row in group}) != 1:
            raise RuntimeError(f"Model source changed across folds for {model}@{checkpoint}.")
        kept.extend(group)
    return kept, dropped


def run_cv(args, ev, *, recover=False, membership_fn=fold_membership,
           membership_key="epit_reviewed_split", dataset_label="EPIT",
           split_strategy="epit_pipeline_exploratory_cv"):
    # Validate the split before launching any models.
    memberships = {fold: membership_fn(args.epit_split_manifest, fold) for fold in FOLDS}
    output_json, output_csv, output_wide, output_summary, plot_dir = ev.repeated_split_output_paths(args)
    for path in (output_json, output_csv, output_wide, output_summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    base = child_arguments(sys.argv[1:])
    snapshot_path = output_json.parent / "cv_checkpoints.json"
    if not recover:
        save_checkpoint_snapshot(ev.resolve_checkpoint_eval_specs(args), snapshot_path)
        base.extend(["--cv-checkpoint-manifest", str(snapshot_path)])
    rows, errors, outputs = [], [], []
    saved_settings = None
    for fold in FOLDS:
        directory = output_json.parent / f"cv_fold_{fold}"
        directory.mkdir(parents=True, exist_ok=True)
        paths = {"json": directory / "results.json", "rows_csv": directory / "rows.csv",
                 "wide_csv": directory / "wide.csv", "summary_csv": directory / "summary.csv"}
        command = [sys.executable, str(Path(ev.__file__).resolve()), *base,
                   "--epit-exploratory-cv-fold", str(fold),
                   "--output-plot-dir", str(directory / "plots")]
        if not args.checkpoint_plots:
            command.append("--no-checkpoint-plots")
        for flag, key in (("--output-json", "json"), ("--output-csv", "rows_csv"),
                          ("--output-wide-csv", "wide_csv"), ("--output-summary-csv", "summary_csv")):
            command.extend([flag, str(paths[key])])
        print(f"\n{'Recovering' if recover else 'Evaluating'} {dataset_label} CV fold {fold}/5", flush=True)
        if not recover:
            subprocess.run(command, cwd=ev.REPO_ROOT, check=True)
        payload = json.loads(paths["json"].read_text())
        settings = {key: payload.get(key) for key in (
            "target_binning", "regression_output", "regression_uncertainty", "n_estimators",
            "random_state", "tabicl_feat_shuffle_method", "catboost_settings", "missing_aware_models",
        )}
        if saved_settings is not None and settings != saved_settings:
            raise RuntimeError("CV folds have different evaluation settings; cannot average.")
        saved_settings = settings
        if payload.get(membership_key) != memberships[fold]:
            raise RuntimeError(f"CV fold {fold} changed row membership.")
        rows.extend({**row, "cv_fold": fold} for row in payload.get("rows", []))
        errors.extend({**row, "cv_fold": fold} for row in payload.get("errors", []))
        outputs.append({"cv_fold": fold, **{key: str(path) for key, path in paths.items()}})

    dropped = []
    if recover:
        if errors:
            raise RuntimeError("Saved folds contain model errors; refusing to hide them during recovery.")
        rows, dropped = complete_rows_only(rows, step_interval=args.checkpoint_step_interval)
        for item in dropped:
            print(f"Excluded {item['model']}@{item['checkpoint_name']}: "
                  f"{item.get('reason', 'incomplete_folds')}; present in folds {item['folds']}")

    # Reuse completeness and spread calculations without labeling CV as context-only evaluation.
    stats_rows = [{**row, "context_fold": row["cv_fold"]} for row in rows]
    stats_errors = [{**row, "context_fold": row["cv_fold"]} for row in errors]
    ev.require_complete_context_folds(stats_rows, stats_errors, FOLDS)
    if not rows:
        raise RuntimeError("Exploratory CV produced no evaluation rows.")
    summary = ev.print_summary(rows, errors, primary_metric=ev.primary_metric_for_args(args),
                               metric_sort_specs=ev.metric_sort_specs_for_args(args))
    summary = ev.add_context_fold_stats_to_summary(summary, stats_rows).rename(
        columns={"n_context_folds": "n_cv_folds"})
    wide = ev.make_repeated_split_wide_dataframe(rows, summary)
    if not wide.empty:
        wide["split_strategy"] = split_strategy
        wide["n_cv_folds"] = 5
    plots = []
    if args.checkpoint == "all" and args.checkpoint_plots:
        plots = ev.write_checkpoint_trend_plots(
            summary, plot_dir, list(args.plot_metric or ev.default_plot_metrics_for_args(args)))
    payload = {
        "exploratory_cv": True, "cv_seed": CV_SEED, "cv_folds": FOLDS,
        "cv_fold_outputs": outputs, "fold_membership": memberships,
        "original_final_test_fold": 1,
        "recovered_from_saved_folds": recover,
        "excluded_model_checkpoints": dropped,
        "excluded_incomplete_model_checkpoints": [item for item in dropped if "reason" not in item],
        "checkpoint_snapshot": str(snapshot_path) if snapshot_path.is_file() else None,
        "interpretation": "Exploratory: development query rows influenced informed prior training.",
        "metrics_averaged_across_cv_folds": True,
        "predictions_averaged_across_cv_folds": False,
        "rows": rows, "errors": errors, "summary": summary.to_dict(orient="records"),
        "output_files": {"json": str(output_json), "csv": str(output_csv),
                         "wide_csv": str(output_wide), "summary_csv": str(output_summary),
                         "plots": [str(path) for path in plots]},
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    wide.to_csv(output_wide, index=False)
    summary.to_csv(output_summary, index=False)
    print(f"\nSaved exploratory CV scores and fold standard deviations to {output_summary}")
