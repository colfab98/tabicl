#!/usr/bin/env python
"""Compare Soccol checkpoints and baselines on the frozen outer queries."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline.artifact_hashes import load_frozen_final_model
from scripts.eval_corrosion_datasets import slugify
from scripts.soccol_pipeline.data import SPLIT_MANIFEST, reviewed_final_split_metadata
from tabicl.prior.soccol_reviewed_schema import (
    SOCCOL_REVIEWED_CATEGORICAL_COLUMNS, SOCCOL_REVIEWED_FEATURE_PROFILES,
)

SOCCOL_TASK_ID = "soccol_pitting_potential__event1__epit_mv_agagcl_3m_kcl"
DEFAULT_GENERIC_RUN = REPO_ROOT / "checkpoints/tabicl_s1_regression_baseline"
DEFAULT_SPLIT_MANIFEST = SPLIT_MANIFEST
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "corrosion_datasets/analysis/soccol_pipeline"
DEFAULT_FINAL_MODEL_MANIFEST = (
    DEFAULT_OUTPUT_ROOT / "final_v2/soccol_pipeline_optuna_empirical_features_scm_target_v8/final_model_manifest.json"
)
LEGACY_CANDIDATE_RUN = (
    REPO_ROOT / "checkpoints/soccol_pipeline_final_v1/final_soccol_pipeline_optuna_empirical_features_scm_target_v1_trial_0019"
)
CATBOOST_CATEGORICAL_COLUMNS = SOCCOL_REVIEWED_CATEGORICAL_COLUMNS
LEGACY_CATEGORICAL_COLUMNS = ("Prep_medium", "CP_aeration", "CP_agitation", "CP_anions_info")
REQUIRED_PLOTS = tuple(f"trend_mean_test_{metric}.svg" for metric in (
    "spearman", "mae", "nmae_iqr", "nrmse_iqr", "pearson", "r2",
))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-model-inputs", action="store_true", help="Evaluate historical missing-embedding checkpoints with their old input policy.")
    parser.add_argument("--legacy-data", action="store_true", help="Use the historical 37-column inputs and trial-19 default.")
    parser.add_argument("--context-folds", type=int, choices=[5], help="Use each accepted development fold as context; keep all 805 final queries.")
    parser.add_argument("--include-latest-checkpoint", action="store_true",
                        help="Also evaluate each run's latest checkpoint outside the requested interval.")
    parser.add_argument("--exploratory-cv-folds", type=int, choices=[5],
                        help="Original test fold plus four accepted-development folds; average scores.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--catboost-thread-count", type=int, default=8)
    parser.add_argument("--checkpoint-step-interval", type=int, default=500)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--candidate-run", type=Path, help="Explicit run for checkpoint trends; otherwise read the frozen final-model manifest.")
    source.add_argument("--final-model-manifest", type=Path)
    source.add_argument("--ratio-runs", type=Path, nargs=4, metavar=("RUN25", "RUN50", "RUN75", "RUN100"),
                        help="Checkpoint directories in 25/50/75/100 percent informed order.")
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--generic-run", type=Path, default=DEFAULT_GENERIC_RUN)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print the evaluation command without loading models or writing results.")
    args = parser.parse_args(argv)
    if args.exploratory_cv_folds is not None and (args.context_folds is not None or args.legacy_data):
        parser.error("--exploratory-cv-folds cannot be combined with --context-folds or --legacy-data.")
    if args.legacy_data and args.final_model_manifest is not None:
        parser.error("--final-model-manifest is for reviewed models; use --candidate-run for a legacy run.")
    if args.n_estimators <= 0 or args.checkpoint_step_interval < 0 or args.catboost_thread_count == 0:
        parser.error("Estimators must be positive, checkpoint interval nonnegative, and CatBoost threads nonzero.")
    if args.ratio_runs is not None:
        if args.legacy_data or args.candidate_label is not None:
            parser.error("--ratio-runs uses reviewed data and fixed ratio labels; omit --legacy-data and --candidate-label.")
        if len({path.expanduser().resolve() for path in args.ratio_runs}) != 4:
            parser.error("--ratio-runs requires four distinct checkpoint directories.")
    if args.candidate_label is None:
        args.candidate_label = "soccol_trial19" if args.legacy_data else "soccol_reviewed"
    args.candidate_label = slugify(args.candidate_label)
    if args.candidate_label in {"generic_baseline", "pretrained_tabicl_v2", "catboost"}:
        parser.error("Candidate label must differ from the three baseline labels.")
    return args


def candidate_run(args: argparse.Namespace) -> Path:
    if args.candidate_run is not None:
        return args.candidate_run.expanduser().resolve()
    if args.legacy_data:
        return LEGACY_CANDIDATE_RUN
    path = args.final_model_manifest or DEFAULT_FINAL_MODEL_MANIFEST
    if not path.expanduser().is_file():
        raise FileNotFoundError(
            f"Reviewed Soccol final model is not available yet: {path}. "
            "For a command preview, pass --candidate-run /path/to/new/run --dry-run. "
            "Development baselines can already run with scripts.soccol_pipeline.evaluate_baseline_folds."
        )
    frozen = load_frozen_final_model(path)
    config = frozen.manifest.get("evaluation_configuration", {})
    expected = reviewed_final_split_metadata(args.split_manifest)
    if (
        config.get("task_id") != SOCCOL_TASK_ID
        or not config.get("soccol_reviewed_data")
        or config.get("soccol_feature_profile") not in SOCCOL_REVIEWED_FEATURE_PROFILES
        or config.get("expected_n_features") != 58
        or bool(config.get("col_missing_value_embedding")) != args.legacy_model_inputs
        or any(frozen.manifest.get("split", {}).get(k) != v for k, v in expected.items())
    ):
        raise RuntimeError("Final manifest does not match reviewed Soccol inputs and frozen queries.")
    return frozen.checkpoint_path.parent


def candidate_models(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.ratio_runs is not None:
        return list(zip(("soccol_25pct", "soccol_50pct", "soccol_75pct", "soccol_100pct"),
                        (path.expanduser().resolve() for path in args.ratio_runs), strict=True))
    return [(args.candidate_label, candidate_run(args))]


def output_directory(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    mode = "context_folds" if args.context_folds is not None else "full_development"
    name = "exploratory_cv5_comparison" if args.exploratory_cv_folds is not None else f"final_test_{mode}_comparison"
    if args.ratio_runs is not None:
        name = "four_ratios_" + name
    return DEFAULT_OUTPUT_ROOT / (name if args.legacy_data else f"reviewed_{name}")


def evaluation_command(args: argparse.Namespace) -> list[str]:
    output = output_directory(args)
    models = candidate_models(args)
    model_arguments = []
    for label, path in [*models, ("generic_baseline", args.generic_run.expanduser().resolve())]:
        model_arguments.extend(["--run", str(path), "--local-model-label", label])
    command = [
        sys.executable, str(REPO_ROOT / "scripts/eval_corrosion_datasets.py"),
        *model_arguments,
        "--checkpoint", "all", "--min-checkpoint-step", "0",
        "--checkpoint-step-interval", str(args.checkpoint_step_interval),
        "--include-latest-checkpoint" if args.include_latest_checkpoint else "--no-include-latest-checkpoint",
        "--task", SOCCOL_TASK_ID, "--target-mode", "primary", "--target-binning", "continuous",
        "--max-samples-per-task", "0", "--epit-split-manifest", str(args.split_manifest.expanduser().resolve()),
        "--epit-final-test", "--device", args.device, "--n-estimators", str(args.n_estimators),
        "--random-state", str(args.random_state), "--tabicl-feat-shuffle-method", "none",
        "--tabicl-norm-methods", "none", "--regression-output", "median", "--no-regression-uncertainty",
        "--compare-pretrained-tabicl", "--pretrained-checkpoint-version", "tabicl-regressor-v2-20260212.ckpt",
        "--compare-catboost", "--catboost-iterations", "1000", "--catboost-depth", "6",
        "--catboost-learning-rate", "0.03", "--catboost-l2-leaf-reg", "3.0",
        "--catboost-thread-count", str(args.catboost_thread_count),
    ]
    if args.exploratory_cv_folds is not None:
        command.remove("--epit-final-test")
        command.extend(["--exploratory-cv-folds", "5"])
    if not args.legacy_data:
        command.append("--soccol-reviewed-data")
    command.append("--native-tabicl-baselines")
    if args.legacy_model_inputs or args.legacy_data:
        for label, _ in models:
            command.extend(["--fixed-schema-model", label])
            if not args.legacy_data:
                command.extend(["--missing-aware-model", label])
    categories = LEGACY_CATEGORICAL_COLUMNS if args.legacy_data else CATBOOST_CATEGORICAL_COLUMNS
    for column in categories:
        command.extend(["--catboost-categorical-column", column])
    for option, filename in (("--output-json", "results.json"), ("--output-csv", "rows.csv"),
                             ("--output-wide-csv", "wide.csv"), ("--output-summary-csv", "summary.csv"),
                             ("--output-plot-dir", "plots")):
        command.extend([option, str(output / filename)])
    if args.context_folds is not None:
        command.extend(["--context-folds", str(args.context_folds)])
    return command


def validate_results(args: argparse.Namespace, payload: dict, *, context_fold: int | None = None, cv_fold: int | None = None) -> None:
    if payload.get("errors"):
        raise RuntimeError(f"Comparison reported model errors: {payload['errors']}")
    if args.exploratory_cv_folds is not None and cv_fold is None:
        outputs = payload.get("cv_fold_outputs", [])
        if [item["cv_fold"] for item in outputs] != [1, 2, 3, 4, 5]:
            raise RuntimeError("Comparison did not evaluate all five CV folds.")
        for item in outputs:
            validate_results(args, json.loads(Path(item["json"]).read_text()), cv_fold=item["cv_fold"])
        return
    if args.context_folds is not None and context_fold is None:
        outputs = payload.get("context_fold_outputs", [])
        if [item["context_fold"] for item in outputs] != [1, 2, 3, 4, 5]:
            raise RuntimeError("Comparison did not evaluate all five context folds.")
        for item in outputs:
            validate_results(args, json.loads(Path(item["json"]).read_text()), context_fold=item["context_fold"])
        return
    if args.legacy_data:
        return
    metadata = reviewed_final_split_metadata(args.split_manifest)
    selection = metadata["soccol_development_selection"]
    context = [i for i, f in zip(selection["source_task_row_indices"], selection["validation_folds"], strict=True)
               if context_fold is None or f == context_fold]
    expected = {"soccol_development_selection": selection, "context_task_row_indices": context,
                "query_task_row_indices": metadata["final_query_task_row_indices"]}
    membership_key = "soccol_final_split"
    if cv_fold is not None:
        from scripts.soccol_pipeline.exploratory_cv import fold_membership
        expected = fold_membership(args.split_manifest, cv_fold)
        context = expected["context_task_row_indices"]
        membership_key = "soccol_exploratory_cv_split"
        if payload.get("soccol_development_selection") != selection:
            raise RuntimeError("Comparison changed accepted-development selection.")
    if payload.get(membership_key) != expected:
        raise RuntimeError("Comparison changed the accepted context or frozen query membership.")
    labels = {label for label, _ in candidate_models(args)}
    if set(payload.get("missing_aware_models", [])) != (labels if args.legacy_model_inputs else set()):
        raise RuntimeError("Comparison changed Soccol missing-value handling.")
    rows = payload.get("rows", [])
    if {row.get("model") for row in rows} != labels | {"generic_baseline", "pretrained_tabicl_v2", "catboost"}:
        raise RuntimeError("Comparison did not produce all expected models.")
    strategy = "epit_pipeline_final_test" + (f"_context_fold_{context_fold}" if context_fold is not None else "")
    if cv_fold is not None:
        strategy = f"soccol_pipeline_exploratory_cv_fold_{cv_fold}"
    for row in rows:
        if (row.get("task_id") != SOCCOL_TASK_ID or row.get("split_strategy") != strategy
                or row.get("n_train") != len(context) or row.get("n_test") != len(expected["query_task_row_indices"]) or row.get("n_features") != 58):
            raise RuntimeError("Comparison used the wrong dataset, schema or split.")


def run(args: argparse.Namespace) -> Path:
    command = evaluation_command(args)
    output = output_directory(args)
    if args.dry_run:
        print(shlex.join(command))
        return output
    if args.ratio_runs is not None:
        for label, path in candidate_models(args):
            if not path.is_dir() or not any(path.glob("step-*.ckpt")):
                raise FileNotFoundError(f"{label}: no checkpoint files found in {path}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Comparison output is not empty: {output}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    validate_results(args, json.loads((output / "results.json").read_text()))
    for filename in REQUIRED_PLOTS:
        path = output / "plots" / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required plot missing: {path}")
    return output


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
