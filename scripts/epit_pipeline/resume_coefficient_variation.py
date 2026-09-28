#!/usr/bin/env python
"""Retry missing cells in an existing coefficient-variation Optuna study.

This recovery entry point deliberately lives outside the frozen experiment
fingerprint. It queues exact grid cells and delegates all planning, training,
evaluation, provenance checks, and summaries to run_coefficient_variation.
"""

from __future__ import annotations

from pathlib import Path

import optuna

from scripts.epit_pipeline import run_coefficient_variation as experiment
from scripts.epit_pipeline.artifact_hashes import sha256_file


def completed_cells(study):
    """Return the unique completed grid cells and reject ambiguous history."""
    completed = {}
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        key = (
            float(trial.params["coefficient_variation"]),
            int(trial.params["training_seed"]),
        )
        if key in completed:
            raise RuntimeError(f"Duplicate completed grid cell: {key}")
        completed[key] = trial
    return completed


def missing_cells(study, variations, seeds):
    """Put failed cells first, followed by never-completed cells in grid order."""
    grid = [(float(variation), int(seed)) for variation in variations for seed in seeds]
    grid_set = set(grid)
    completed = completed_cells(study)
    unexpected = set(completed) - grid_set
    if unexpected:
        raise RuntimeError(f"Completed trials lie outside the configured grid: {unexpected}")

    failed = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.FAIL:
            continue
        if not {"coefficient_variation", "training_seed"} <= trial.params.keys():
            continue
        key = (
            float(trial.params["coefficient_variation"]),
            int(trial.params["training_seed"]),
        )
        if key in grid_set and key not in completed and key not in failed:
            failed.append(key)
    return failed + [key for key in grid if key not in completed and key not in failed]


def open_existing_study(args, loaded):
    storage = args.storage or f"journal://{loaded.study_dir / 'study_journal.log'}"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        load_if_exists=True,
        storage=experiment.search.original.search_utils.build_optuna_storage(storage),
        # Every recovery trial has fixed queued parameters, so no parameter is
        # sampled. RandomSampler also avoids assigning a second GridSampler ID
        # to a retry of an already-visited failed cell.
        sampler=optuna.samplers.RandomSampler(seed=args.sampler_seed),
    )
    if study.direction != optuna.study.StudyDirection.MAXIMIZE:
        raise RuntimeError("Existing study has the wrong optimization direction.")
    saved = study.user_attrs.get(experiment.FINGERPRINT_KEY)
    if saved is None:
        raise RuntimeError("Recovery requires an existing fingerprinted study.")
    if saved != loaded.fingerprint:
        raise RuntimeError("Optuna study fingerprint differs; refusing recovery.")
    if any(
        trial.state in (optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.WAITING)
        for trial in study.trials
    ):
        raise RuntimeError("Study has running/waiting trials. Resolve them before recovery.")
    return study


def run(args):
    if args.dry_run:
        raise ValueError("Recovery operates only on an existing study; omit --dry-run.")
    loaded = experiment.load_experiment(args)
    with experiment.experiment_lock(loaded.study_dir):
        experiment.bind_directory(loaded)
        study = open_existing_study(args, loaded)
        cells = missing_cells(study, args.variations, args.seeds)
        cells = cells[: args.n_trials] if args.n_trials is not None else cells

        def objective(trial):
            strength = trial.suggest_categorical("coefficient_variation", args.variations)
            seed = trial.suggest_categorical("training_seed", args.seeds)
            trial.set_user_attr("training_seed_is_replicate_not_optimized", True)
            trial.set_user_attr("recovery_fixed_grid_cell", True)
            plan = experiment.build_trial_plan(
                args, loaded, trial.number, strength, seed
            )
            trial.set_user_attr("fingerprint_sha256", plan["fingerprint_sha256"])
            trial.set_user_attr("trial_dir", plan["trial_dir"])
            result = experiment.run_trial(plan, loaded)
            for evaluation in result["results"]:
                trial.set_user_attr(
                    f"spearman_step_{evaluation['step']}",
                    evaluation["mean_test_spearman"],
                )
            trial.set_user_attr(
                "result_sha256",
                sha256_file(Path(plan["trial_dir"]) / "trial_result.json"),
            )
            return float(result["objective"])

        try:
            for strength, seed in cells:
                print(
                    f"Recovery queues variation={strength:g}, seed={seed} "
                    f"as trial {len(study.trials)}.",
                    flush=True,
                )
                study.enqueue_trial(
                    {"coefficient_variation": strength, "training_seed": seed},
                    user_attrs={"recovery_fixed_grid_cell": True},
                    skip_if_exists=False,
                )
                study.optimize(objective, n_trials=1)
        finally:
            experiment.save_summary(study, args, loaded)
    return loaded.study_dir


if __name__ == "__main__":
    run(experiment.parse_args())
