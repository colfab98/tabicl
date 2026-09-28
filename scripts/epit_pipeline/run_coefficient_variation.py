#!/usr/bin/env python
"""Balanced Optuna experiment: coefficient variation at a fixed 2,000-step budget."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna

from scripts.epit_pipeline import run_optuna as search
from scripts.epit_pipeline.artifact_hashes import sha256_file
from scripts.epit_pipeline.diagnose_target_variation import DEFAULT_MANIFEST, load_settings

ROOT = search.REPO_ROOT
DEFAULT_STUDY = "epit_coefficient_variation_2k_v1"
WORK_ROOT = search.PIPELINE_ROOT / "coefficient_variation_search"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "epit_coefficient_variation"
FINGERPRINT_KEY = "coefficient_variation_experiment_v1"
MAX_STEPS = 2000
EVAL_STEPS = (1000, 2000)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-name", default=DEFAULT_STUDY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-manifest", type=Path, default=search.DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--target-rule-summary", type=Path, default=search.DEFAULT_TARGET_RULE_SUMMARY)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--storage", help="Optuna storage; defaults to a new journal inside this experiment directory.")
    parser.add_argument("--variations", type=float, nargs="+", default=[0, .3, .5, .8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, help="Limit new trials this invocation; default runs remaining grid cells.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Write commands and provenance only; no Optuna trials or training.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.study_name):
        parser.error("Study name must be a plain name, not a path.")
    if args.study_name == search.DEFAULT_STUDY_NAME:
        parser.error("The frozen v7 study must not be reused.")
    if any(not math.isfinite(v) or not 0 <= v < 1 for v in args.variations):
        parser.error("Variation strengths must be in [0, 1).")
    if len(set(args.variations)) != len(args.variations) or 0 not in args.variations:
        parser.error("Use distinct variation strengths and include the zero baseline.")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        parser.error("Training seeds must be distinct and in [0, 2**32).")
    if args.n_trials is not None and args.n_trials < 1:
        parser.error("--n-trials must be positive.")
    if not 0 <= args.sampler_seed < 2**32:
        parser.error("--sampler-seed must be in [0, 2**32).")
    args.variations = sorted(args.variations)
    args.seeds = sorted(args.seeds)
    for name in ("manifest", "split_manifest", "target_rule_summary", "work_root", "checkpoint_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def training_command(args, template, bounds, strength, seed, checkpoint_dir):
    start = next(i for i, value in enumerate(template) if value.endswith("/train/run.py")) + 1
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
        str(ROOT / "src/tabicl/train/run.py"), *template[start:],
    ]
    overrides = {
        "--max_steps": MAX_STEPS,
        "--scheduler_total_steps": 10000,
        "--np_seed": seed, "--torch_seed": seed, "--python_seed": seed,
        "--device": args.device,
        "--prior_n_jobs": 1,
        "--checkpoint_dir": checkpoint_dir,
        "--save_perm_every": 1000, "--save_temp_every": 2000,
        "--max_checkpoints": 1,
        "--wandb_log": "False", "--wandb_mode": "disabled",
        "--wandb_name": f"{args.study_name}_variation_{strength}_seed_{seed}",
        "--pitting_coefficient_variation": strength,
        "--pitting_coefficient_variation_seed": seed,
        "--pitting_coefficient_upper_bounds": json.dumps(bounds, sort_keys=True),
    }
    for option, value in overrides.items():
        if option in command:
            search._replace_option(command, option, str(value))
        else:
            command.extend([option, str(value)])
    return command


def load_experiment(args):
    fixed, bounds, config, rules = load_settings(args.manifest, args.target_rule_summary, args.split_manifest)
    manifest = json.loads(args.manifest.read_text())
    if args.study_name == manifest["study"]["name"]:
        raise ValueError("Use a new study name, not the source study.")
    if config.checkpoint_path or config.prior_dir or config.load_prior_start:
        raise ValueError("The source configuration must train from scratch on generated data.")
    if config.prior_n_jobs != 1:
        raise ValueError("The source experiment must use one prior job per worker.")
    template = manifest["training"]["train_command"]
    evaluation = manifest["evaluation_configuration"]
    eval_args = SimpleNamespace(
        split_manifest=args.split_manifest, device=args.device,
        eval_n_estimators=int(evaluation["n_estimators"]),
    )
    study_dir = args.work_root / args.study_name
    checkpoint_dir = args.checkpoint_root / args.study_name
    # Avoid introducing experiment directories inside frozen source artifacts.
    protected = [args.manifest.parent, args.target_rule_summary.parent, args.split_manifest.parent]
    protected.append(Path(manifest["selected_checkpoint"]["path"]).parent)
    for destination in (study_dir, checkpoint_dir):
        if any(destination == path or destination.is_relative_to(path) for path in protected):
            raise ValueError(f"Output directory lies inside a frozen artifact directory: {destination}")
    sources = [*sorted((ROOT / "src/tabicl").rglob("*.py")), Path(__file__),
               ROOT / "scripts/epit_pipeline/diagnose_target_variation.py",
               ROOT / "scripts/epit_pipeline/run_optuna.py",
               ROOT / "scripts/epit_pipeline/evaluate_optuna_folds.py",
               ROOT / "scripts/epit_pipeline/artifact_hashes.py",
               ROOT / "scripts/eval_corrosion_datasets.py",
               ROOT / "scripts/optuna_pitting_magpie_prior_search.py",
               ROOT / "scripts/optuna_pitting_fixed_pren_prior_search.py"]
    fingerprint = {
        "schema": FINGERPRINT_KEY, "study_name": args.study_name,
        "manifest_sha256": sha256_file(args.manifest),
        "artifacts": search.pipeline_artifact_identity(rules),
        "feature_profile": search.empirical_feature_profile_identity(),
        "coefficient_upper_bounds": bounds,
        "source_sha256s": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
        "grid": {"coefficient_variation": args.variations, "training_seed": args.seeds},
        "max_steps": MAX_STEPS, "evaluation_steps": list(EVAL_STEPS),
        "objective_step": 2000, "objective": "mean development-fold Spearman",
        "scheduler_total_steps": 10000, "sampler_seed": args.sampler_seed,
        "training_template": template, "device": args.device,
        "evaluation_n_estimators": eval_args.eval_n_estimators,
        "python_seed_policy": "explicit; same seed for Python, NumPy, Torch, and separate coefficient stream",
        "paths": {"study_dir": str(study_dir), "checkpoint_dir": str(checkpoint_dir)},
        "final_test_used": False,
        "optuna_version": optuna.__version__,
    }
    return SimpleNamespace(
        fixed=fixed, bounds=bounds, template=template, rules=rules, eval_args=eval_args,
        study_dir=study_dir, checkpoint_dir=checkpoint_dir, fingerprint=fingerprint,
    )


@contextmanager
def experiment_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".runner.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another runner is already using this experiment directory.") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def bind_directory(experiment):
    path = experiment.study_dir / "experiment.json"
    if path.exists():
        if json.loads(path.read_text()) != experiment.fingerprint:
            raise RuntimeError("Experiment configuration or source code changed; use a new study name.")
    else:
        if any(p.name != ".runner.lock" for p in experiment.study_dir.iterdir()):
            raise RuntimeError("Refusing to adopt a nonempty experiment directory without provenance.")
        search.write_json(path, experiment.fingerprint)


def build_trial_plan(args, experiment, trial_number, strength, seed):
    label = f"trial_{trial_number:04d}"
    trial_dir = experiment.study_dir / "trials" / label
    checkpoint_dir = experiment.checkpoint_dir / label
    evaluations = []
    for step in EVAL_STEPS:
        checkpoint = checkpoint_dir / f"step-{step}.ckpt"
        output = trial_dir / "evaluations" / f"step_{step}"
        model_label = f"{label}_step_{step}"
        evaluations.append({
            "step": step, "checkpoint": str(checkpoint), "output_dir": str(output),
            "model_label": model_label,
            "command": search.eval_command(
                experiment.eval_args, SimpleNamespace(use_magpie=False), checkpoint, model_label, output,
            ),
        })
    return {
        "trial_number": trial_number, "coefficient_variation": strength, "training_seed": seed,
        "trial_dir": str(trial_dir), "checkpoint_dir": str(checkpoint_dir),
        "fingerprint_sha256": search.canonical_json_sha256(experiment.fingerprint),
        "train_command": training_command(args, experiment.template, experiment.bounds, strength, seed, checkpoint_dir),
        "evaluations": evaluations,
    }


def run_trial(plan, experiment):
    trial_dir = Path(plan["trial_dir"])
    checkpoint_dir = Path(plan["checkpoint_dir"])
    trial_dir.mkdir(parents=True, exist_ok=False)
    if checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to reuse checkpoint directory: {checkpoint_dir}")
    search.write_json(trial_dir / "trial_config.json", plan)
    search.original.search_utils.run_command(plan["train_command"], cwd=ROOT, log_path=trial_dir / "train.log")
    results = []
    for evaluation in plan["evaluations"]:
        checkpoint = Path(evaluation["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing expected checkpoint: {checkpoint}")
        output = Path(evaluation["output_dir"])
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
        search.original.search_utils.run_command(
            evaluation["command"], cwd=ROOT, log_path=trial_dir / f"eval_{evaluation['step']}.log",
        )
        payload = search.verify_no_power_fold_evaluation(output)
        search.verify_trial_fold_evaluation(
            payload, rules=experiment.rules, checkpoint_path=checkpoint,
            params=SimpleNamespace(use_magpie=False),
        )
        if payload.get("validation_folds") != list(search.FIXED_VALIDATION_FOLDS) or payload.get("development_rows_only") is not True:
            raise RuntimeError("Evaluation did not use exactly the five development folds.")
        if payload.get("settings", {}).get("n_estimators") != experiment.eval_args.eval_n_estimators:
            raise RuntimeError("Evaluation changed the ensemble size.")
        summary = search.original.search_utils.load_eval_summary(output / "summary.csv", evaluation["model_label"])
        score = float(summary["mean_test_spearman"])
        if not math.isfinite(score):
            raise RuntimeError("Nonfinite development Spearman score.")
        results.append({
            "step": evaluation["step"], "mean_test_spearman": score,
            "checkpoint_sha256": sha256_file(checkpoint),
            "summary_csv": str(output / "summary.csv"),
            "summary_csv_sha256": sha256_file(output / "summary.csv"),
            "summary_json_sha256": sha256_file(output / "summary.json"),
            "rows_csv_sha256": sha256_file(output / "rows.csv"),
        })
    result = {**plan, "results": results, "objective": results[-1]["mean_test_spearman"]}
    search.write_json(trial_dir / "trial_result.json", result)
    return result


def summarize_study(study, variations, seeds):
    completed = {}
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            key = (trial.params["coefficient_variation"], trial.params["training_seed"])
            if key in completed:
                raise RuntimeError(f"Duplicate completed grid cell: {key}")
            completed[key] = trial
    rows = []
    for variation in variations:
        cells = [completed[(variation, seed)] for seed in seeds if (variation, seed) in completed]
        scores = [float(trial.value) for trial in cells]
        deltas = [
            float(trial.value) - float(completed[(0.0, trial.params["training_seed"])].value)
            for trial in cells if (0.0, trial.params["training_seed"]) in completed
        ]
        rows.append({
            "coefficient_variation": variation, "completed_seeds": len(cells), "expected_seeds": len(seeds),
            "complete": len(cells) == len(seeds),
            "mean_spearman_2000": float(np.mean(scores)) if scores else None,
            "std_across_seeds": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
            "paired_baseline_seeds": len(deltas),
            "mean_paired_gain": float(np.mean(deltas)) if deltas else None,
            "std_paired_gain": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None,
        })
    return rows


def save_summary(study, args, experiment):
    rows = summarize_study(study, args.variations, args.seeds)
    with (experiment.study_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    complete = all(row["complete"] for row in rows)
    search.write_json(experiment.study_dir / "summary.json", {
        "grid_complete": complete, "objective_step": 2000, "results": rows,
        "best_variation_by_mean": max(rows, key=lambda r: r["mean_spearman_2000"])["coefficient_variation"] if complete else None,
        "trial_states": {state.name: sum(t.state == state for t in study.trials) for state in optuna.trial.TrialState},
    })
    print(f"Summary: {experiment.study_dir / 'summary.csv'} (grid complete: {complete})", flush=True)


def run(args):
    experiment = load_experiment(args)
    with experiment_lock(experiment.study_dir):
        bind_directory(experiment)
        if args.dry_run:
            plans = [build_trial_plan(args, experiment, number, variation, seed)
                     for number, (variation, seed) in enumerate((v, s) for v in args.variations for s in args.seeds)]
            path = experiment.study_dir / "dry_run_plan.json"
            if path.exists() and json.loads(path.read_text()) != plans:
                raise RuntimeError("Existing dry-run plan differs.")
            search.write_json(path, plans)
            print(f"Dry run: {len(plans)} grid cells; no training or Optuna trials created.\nPlan: {path}")
            return path
        storage = args.storage or f"journal://{experiment.study_dir / 'study_journal.log'}"
        study = optuna.create_study(
            study_name=args.study_name, direction="maximize", load_if_exists=True,
            storage=search.original.search_utils.build_optuna_storage(storage),
            sampler=optuna.samplers.GridSampler(
                {"coefficient_variation": args.variations, "training_seed": args.seeds}, seed=args.sampler_seed,
            ),
        )
        if study.direction != optuna.study.StudyDirection.MAXIMIZE:
            raise RuntimeError("Existing study has the wrong optimization direction.")
        saved = study.user_attrs.get(FINGERPRINT_KEY)
        if saved is None:
            if study.trials:
                raise RuntimeError("Refusing to adopt a study containing unrelated trials.")
            study.set_user_attr(FINGERPRINT_KEY, experiment.fingerprint)
        elif saved != experiment.fingerprint:
            raise RuntimeError("Optuna study fingerprint differs; use a new study name.")
        if any(t.state in (optuna.trial.TrialState.RUNNING, optuna.trial.TrialState.WAITING) for t in study.trials):
            raise RuntimeError("Study has running/waiting trials. Resolve interrupted trials before restarting.")
        expected = len(args.variations) * len(args.seeds)
        remaining = max(0, expected - len(study.trials))

        def objective(trial):
            strength = trial.suggest_categorical("coefficient_variation", args.variations)
            seed = trial.suggest_categorical("training_seed", args.seeds)
            trial.set_user_attr("training_seed_is_replicate_not_optimized", True)
            plan = build_trial_plan(args, experiment, trial.number, strength, seed)
            trial.set_user_attr("fingerprint_sha256", plan["fingerprint_sha256"])
            trial.set_user_attr("trial_dir", plan["trial_dir"])
            result = run_trial(plan, experiment)
            for evaluation in result["results"]:
                trial.set_user_attr(f"spearman_step_{evaluation['step']}", evaluation["mean_test_spearman"])
            trial.set_user_attr("result_sha256", sha256_file(Path(plan["trial_dir"]) / "trial_result.json"))
            return result["objective"]

        try:
            if remaining:
                study.optimize(objective, n_trials=min(remaining, args.n_trials or remaining))
        finally:
            save_summary(study, args, experiment)
    return experiment.study_dir


if __name__ == "__main__":
    run(parse_args())
