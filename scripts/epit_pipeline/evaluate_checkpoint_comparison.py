#!/usr/bin/env python
"""Compare EPIT checkpoints and baselines on the frozen outer queries."""
from __future__ import annotations

import argparse
from copy import copy
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.epit_pipeline.artifact_hashes import load_frozen_final_model, load_frozen_split
from scripts.eval_corrosion_datasets import slugify
from scripts.epit_pipeline.run_optuna import DEFAULT_SPLIT_MANIFEST, DEFAULT_STUDY_NAME
from scripts.combined_corrosion.evaluate_final import add_evaluation_arguments, comparison_command
from tabicl.prior.epit_expanded_schema import EPIT_EXPANDED_CATEGORICAL_COLUMNS

EPIT_TASK_ID = "electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg"
PITTING_TASK_ID = EPIT_TASK_ID
DEFAULT_GENERIC_RUN = REPO_ROOT / "checkpoints/tabicl_s1_regression_baseline"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "corrosion_datasets/analysis/epit_pipeline"
DEFAULT_FINAL_MODEL_MANIFEST = (
    DEFAULT_OUTPUT_ROOT / "final_v2" / DEFAULT_STUDY_NAME / "final_model_manifest.json"
)
LEGACY_CANDIDATE_RUN = (
    REPO_ROOT / "checkpoints/epit_pipeline_final_v2/final_epit_pipeline_optuna_empirical_features_scm_target_v8_trial_0017"
)
CATBOOST_CATEGORICAL_COLUMNS = EPIT_EXPANDED_CATEGORICAL_COLUMNS
RATIO_LABELS = ("epit_0pct", "epit_25pct", "epit_50pct", "epit_75pct", "epit_100pct")
LEGACY_CATEGORICAL_COLUMNS = ("[Cl-] Test Method",)
REQUIRED_PLOTS = tuple(f"trend_mean_test_{metric}.svg" for metric in (
    "spearman", "mae", "nmae_iqr", "nrmse_iqr", "pearson", "r2",
))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-model-inputs", action="store_true", help="Evaluate historical v12 missing-embedding checkpoints with their old input policy.")
    parser.add_argument("--legacy-data", action="store_true", help="Use the historical 28-column inputs and v8 trial-17 default.")
    parser.add_argument("--context-folds", type=int, choices=[5], help="Use each accepted development fold as context; keep all 152 final queries.")
    add_evaluation_arguments(parser, default_output_dir=None, include_combined_data=False, default_device="cpu")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--candidate-run", "--trial-run", type=Path, help="Explicit run for checkpoint trends; otherwise read the frozen final-model manifest.")
    source.add_argument("--final-model-manifest", type=Path)
    source.add_argument("--ratio-runs", type=Path, nargs=5, metavar=("RUN0", "RUN25", "RUN50", "RUN75", "RUN100"),
                        help="Compare five informed ratios together, in 0/25/50/75/100 order, plus three baselines.")
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true", help="Print the evaluation command without loading models or writing results.")
    args = parser.parse_args(argv)
    if args.legacy_data and args.final_model_manifest is not None:
        parser.error("--final-model-manifest is for reviewed models; use --candidate-run for a legacy run.")
    if args.ratio_runs is not None:
        if args.legacy_data or args.candidate_label is not None:
            parser.error("--ratio-runs uses reviewed data and fixed ratio labels; omit --legacy-data and --candidate-label.")
        if len({path.expanduser().resolve() for path in args.ratio_runs}) != 5:
            parser.error("--ratio-runs requires five distinct checkpoint directories.")
    if args.n_estimators <= 0 or args.checkpoint_step_interval < 0 or args.catboost_thread_count == 0:
        parser.error("Estimators must be positive, checkpoint interval nonnegative, and CatBoost threads nonzero.")
    if args.max_checkpoint_step is not None:
        if args.checkpoint != "all":
            parser.error("--max-checkpoint-step requires --checkpoint all.")
        if args.max_checkpoint_step < 0 or args.max_checkpoint_step < args.min_checkpoint_step:
            parser.error("--max-checkpoint-step must be nonnegative and >= --min-checkpoint-step.")
    if args.candidate_label is None:
        args.candidate_label = "epit_legacy" if args.legacy_data else "epit_reviewed"
    args.candidate_label = slugify(args.candidate_label)
    if args.candidate_label in {"generic_baseline", "pretrained_tabicl_v2", "catboost"}:
        parser.error("Candidate label must differ from the three baseline labels.")
    return args


def candidate_models(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.ratio_runs is not None:
        return [(label, path.expanduser().resolve()) for label, path in zip(RATIO_LABELS, args.ratio_runs, strict=True)]
    return [(args.candidate_label, candidate_run(args))]


def candidate_labels(args: argparse.Namespace) -> tuple[str, ...]:
    return RATIO_LABELS if args.ratio_runs is not None else (args.candidate_label,)


def candidate_run(args: argparse.Namespace) -> Path:
    if args.candidate_run is not None:
        return args.candidate_run.expanduser().resolve()
    if args.legacy_data:
        return LEGACY_CANDIDATE_RUN
    path = args.final_model_manifest or DEFAULT_FINAL_MODEL_MANIFEST
    if not path.expanduser().is_file():
        raise FileNotFoundError(f"EPIT final manifest is not available: {path}. "
                                "Pass --candidate-run /path/to/new/run --dry-run for a preview.")
    frozen = load_frozen_final_model(path)
    split = load_frozen_split(args.split_manifest)
    config = frozen.manifest.get("evaluation_configuration", {})
    if (config.get("task_id") != EPIT_TASK_ID or not config.get("epit_expanded_features")
            or config.get("expected_n_features") != 70
            or frozen.manifest.get("split", {}).get("manifest_sha256") != split.manifest_sha256
            or frozen.manifest.get("split", {}).get("lock_sha256") != split.lock_sha256):
        raise RuntimeError("Final manifest does not match reviewed EPIT features and frozen split.")
    return frozen.checkpoint_path.parent



def output_directory(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    mode = "context_folds" if args.context_folds is not None else "full_development"
    name = f"{'five_ratios_' if args.ratio_runs is not None else ''}final_test_{mode}_comparison"
    if args.max_checkpoint_step is not None:
        name += f"_through_{args.max_checkpoint_step}"
    return DEFAULT_OUTPUT_ROOT / (name if args.legacy_data else f"reviewed_{name}")


def evaluation_command(args: argparse.Namespace) -> list[str]:
    options = copy(args)
    options.output_dir = output_directory(args)
    models = candidate_models(args)
    labels = candidate_labels(args)
    task_arguments = ["--epit-split-manifest", str(args.split_manifest.expanduser().resolve()), "--epit-final-test"]
    if not args.legacy_data:
        task_arguments.append("--epit-expanded-features")
    if args.context_folds is not None:
        task_arguments.extend(["--context-folds", str(args.context_folds)])
    command = comparison_command(
        options, [*models, ("generic_baseline", args.generic_run)],
        missing_aware_models=labels if args.legacy_model_inputs and not args.legacy_data else (),
        task_id=EPIT_TASK_ID, task_arguments=task_arguments,
        categorical_columns=LEGACY_CATEGORICAL_COLUMNS if args.legacy_data else CATBOOST_CATEGORICAL_COLUMNS,
    )
    if args.legacy_data:
        command.extend(["--fixed-schema-model", args.candidate_label])
    return command


def validate_results(args: argparse.Namespace, payload: dict, *, context_fold: int | None = None) -> None:
    if payload.get("errors"):
        raise RuntimeError(f"Comparison reported model errors: {payload['errors']}")
    if args.context_folds is not None and context_fold is None:
        outputs = payload.get("context_fold_outputs", [])
        if [item["context_fold"] for item in outputs] != [1, 2, 3, 4, 5]:
            raise RuntimeError("Comparison did not evaluate all five context folds.")
        for item in outputs:
            validate_results(args, json.loads(Path(item["json"]).read_text()), context_fold=item["context_fold"])
        return
    if args.legacy_data:
        return
    from scripts.epit_pipeline.evaluation_identity import frozen_membership
    expected = frozen_membership(args.split_manifest, context_fold=context_fold)
    if payload.get("epit_reviewed_split") != expected:
        raise RuntimeError("Comparison changed EPIT context or frozen final-query membership.")
    labels = set(candidate_labels(args))
    expected_missing = labels if args.legacy_model_inputs else set()
    if set(payload.get("missing_aware_models", [])) != expected_missing:
        raise RuntimeError("Comparison changed EPIT missing-value handling.")
    rows = payload.get("rows", [])
    if {row.get("model") for row in rows} != labels | {"generic_baseline", "pretrained_tabicl_v2", "catboost"}:
        raise RuntimeError("Comparison did not produce all requested models and three baselines.")
    strategy = "epit_pipeline_final_test" + (f"_context_fold_{context_fold}" if context_fold is not None else "")
    for row in rows:
        if args.max_checkpoint_step is not None and (
            row.get("checkpoint_step") is None or int(row["checkpoint_step"]) > args.max_checkpoint_step
        ):
            raise RuntimeError("Comparison exceeded the requested maximum checkpoint step.")
        if (row.get("task_id") != EPIT_TASK_ID or row.get("split_strategy") != strategy
                or row.get("n_train") != len(expected["context_task_row_indices"])
                or row.get("n_test") != len(expected["query_task_row_indices"]) or row.get("n_features") != 70):
            raise RuntimeError("Comparison used the wrong EPIT schema or split.")



def run(args: argparse.Namespace) -> Path:
    command = evaluation_command(args)
    output = output_directory(args)
    if args.dry_run:
        print(shlex.join(command))
        return output
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
