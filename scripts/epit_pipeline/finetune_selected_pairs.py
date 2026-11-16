"""Continue generic 10k weights with the retained-pair mixture and established S2 settings."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

from scripts.epit_pipeline import finetune, train_final
from scripts.epit_pipeline.artifact_hashes import sha256_file
from tabicl.prior.epit_selected_pairs import SelectedPairSampler

ROOT = train_final.REPO_ROOT
DEFAULT_NAME = "generic10k_selected_pairs_s2"
DEFAULT_INFORMED_ONLY_NAME = "generic10k_selected_pairs100_s2"
DEFAULT_RECIPE = ROOT / "corrosion_datasets/analysis/epit_pipeline/selected_pairs_all760_generic10k"
SCHEMA = "epit_selected_pair_finetuning_s2_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name")
    parser.add_argument("--informed-only", action="store_true",
                        help="Use 100% retained-pair tables; default keeps the retained-pair mixture.")
    parser.add_argument("--recipe-dir", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--finetune-from", type=Path, default=finetune.DEFAULT_PARENT)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints/epit_finetuning")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.run_name is None:
        args.run_name = DEFAULT_INFORMED_ONLY_NAME if args.informed_only else DEFAULT_NAME
    if not args.run_name or Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("--run-name must be a single directory name.")
    if args.nproc_per_node < 1 or args.workers < 0:
        parser.error("Processes must be positive; workers must be nonnegative.")
    if args.device == "cpu" and args.nproc_per_node != 1:
        parser.error("CPU training uses one process.")
    return args


def build_plan(args):
    sampler = SelectedPairSampler(args.recipe_dir)
    ratio = 1. if args.informed_only else sampler.ratio
    output = (args.output_dir or ROOT / "corrosion_datasets/analysis/epit_pipeline/selected_pair_finetuning"
              / args.run_name).resolve()
    # The old S2 planner supplies the architecture, optimizer, fresh seed,
    # weight-only initialization and step-zero checkpoint. Only plan it here.
    template = finetune.parse_args([
        "--finetune-from", str(args.finetune_from), "--override-informed-prior-ratio", "0",
        "--final-run-name", args.run_name, "--checkpoint-root", str(args.checkpoint_root),
        "--output-dir", str(output), "--device", args.device,
        "--nproc-per-node", str(args.nproc_per_node), "--dataloader-num-workers", str(args.workers),
        "--dry-run",
    ])
    captured = io.StringIO()
    with redirect_stdout(captured):
        train_final.run(template)
    stage = json.loads(captured.getvalue())["training"]
    command = stage["train_command"]
    if command[0] == "torchrun":
        command = [sys.executable, "-m", "torch.distributed.run", *command[1:]]
    else:
        command[0] = sys.executable
    for field, value in {
        "informed_prior_ratio": ratio,
        "informed_target_mix_weight": 1,
        "informed_target_mix_policy": "fixed",
        "pitting_coefficient_variation": 0,
        "epit_target_version": "v4",
    }.items():
        train_final.replace_option(command, f"--{field}", value)
    command.extend(["--epit_selected_pair_dir", str(sampler.path)])
    if args.informed_only:
        command.extend(["--epit_selected_pair_informed_only", "true"])
    stage["train_command"] = command
    params = stage["effective_trial_params"]
    params.update(informed_prior_ratio=ratio, informed_target_mix_weight=1.)
    stage["trial_params"] = deepcopy(params)
    stage["generic_stream_identity"] = stage.pop("manual_pipeline_identity")["fixed_prior"]["paired_prior_streams"]
    stage.pop("target_rule_summary_sha256")
    stage["configuration_source"] = SCHEMA
    stage["overrides"] = {
        "finetuning": stage["overrides"]["finetuning"],
        "manual_hyperparameters": deepcopy(params),
    }
    stage["selected_pair_finetuning"] = {
        "schema_version": SCHEMA, "recipe_dir": str(sampler.path), "recipe_identity": sampler.identity,
        "implementation_sha256": sha256_file(Path(__file__)),
        "all_data_recipe": True, "automatic_evaluation": False,
    }
    if args.informed_only:
        from tabicl.prior import epit_selected_pairs_informed_only
        stage["selected_pair_finetuning"].update(
            informed_only=True,
            informed_only_implementation_sha256=sha256_file(Path(epit_selected_pairs_informed_only.__file__)))
    validate_selected_stage(stage)
    return {"training": stage, "output_dir": str(output)}


def validate_selected_stage(stage):
    recipe = stage["selected_pair_finetuning"]
    if recipe["schema_version"] != SCHEMA or not recipe["all_data_recipe"]:
        raise ValueError("Unknown selected-pair fine-tuning recipe.")
    config = finetune.training_config(stage["train_command"])
    informed_only = recipe.get("informed_only", False)
    ratio = 1. if informed_only else recipe["recipe_identity"]["informed_ratio"]
    expected = {
        **finetune.S2_SETTINGS, **finetune.RETAINED_TABLE_SETTINGS,
        "informed_prior_ratio": ratio, "informed_target_mix_weight": 1.,
        "informed_target_mix_policy": "fixed", "pitting_coefficient_variation": 0.,
        "pitting_composition_perturb_strength": .3, "epit_target_version": "v4",
        "epit_selected_pair_dir": recipe["recipe_dir"], "soccol_selected_pair_dir": None,
        "epit_selected_pair_informed_only": informed_only,
        "np_seed": 43, "torch_seed": 43, "python_seed": 43, "prior_stream_seed": 43,
        "save_temp_every": 100, "save_perm_every": 100,
        "only_load_model": True, "save_initial_checkpoint": True,
        "original_prior_mix": True, "paired_prior_streams": True,
        "checkpoint_path": stage["overrides"]["finetuning"]["parent_checkpoint"],
    }
    for field, value in expected.items():
        if getattr(config, field) != value:
            raise ValueError(f"Selected-pair S2 requires {field}={value!r}.")
    if not 0 < ratio <= 1 or stage["effective_trial_params"]["informed_prior_ratio"] != ratio:
        raise ValueError("The S2 mixture must match the retained-pair proportion or explicit informed-only ablation.")


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    stage = plan["training"]
    directory, output = Path(stage["checkpoint_dir"]), Path(plan["output_dir"])
    ratio = stage["effective_trial_params"]["informed_prior_ratio"]
    print(f"Fine-tuning generic 10k: 2000 additional steps; {ratio:.2%} informed / {1-ratio:.2%} generic.", flush=True)
    print(f"Checkpoints: {directory}\nLog: {output / 'train.log'}", flush=True)
    if args.dry_run:
        print("Validated command; no training started.\n" + shlex.join(stage["train_command"]))
        return
    directory.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    train_final.write_json(directory / "finetuning.json", stage, exclusive=True)
    train_final.write_json(output / "training_plan.json", plan, exclusive=True)
    train_final.search.original.search_utils.run_command(
        stage["train_command"], cwd=ROOT, log_path=output / "train.log")
    endpoint = finetune.terminal_checkpoint(SimpleNamespace(max_steps=2000), directory)
    train_final.write_json(output / "terminal_checkpoint.json", endpoint, exclusive=True)
    print("Fine-tuning finished. No evaluation was launched.")


if __name__ == "__main__":
    main()
