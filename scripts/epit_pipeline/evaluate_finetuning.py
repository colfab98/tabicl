"""Compare fine-tuning stages from step zero on the frozen EPIT outer queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.epit_pipeline import evaluate_checkpoint_comparison as comparison
from scripts.epit_pipeline.finetune import training_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetune-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path)
    parser.add_argument("--split-manifest", type=Path, default=comparison.DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--checkpoint", default="all")
    parser.add_argument("--max-checkpoint-step", type=int)
    parser.add_argument("--checkpoint-step-interval", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def comparison_args(args):
    argv = ["--candidate-run", str(args.finetune_run), "--candidate-label", "epit_finetuned",
            "--split-manifest", str(args.split_manifest), "--output-dir", str(args.output_dir),
            "--device", args.device, "--n-estimators", str(args.n_estimators),
            "--checkpoint", args.checkpoint, "--min-checkpoint-step", "0",
            "--checkpoint-step-interval", str(args.checkpoint_step_interval)]
    if args.max_checkpoint_step is not None:
        argv.extend(["--max-checkpoint-step", str(args.max_checkpoint_step)])
    if args.dry_run:
        argv.append("--dry-run")
    options = comparison.parse_args(argv)
    # The unchanged generic parent is the actual step-zero checkpoint. Never
    # place a generic pretraining trajectory on the fine-tuning step axis.
    options._selected_model_runs = [("epit_finetuned", args.finetune_run.resolve())]
    if args.control_run is not None:
        if args.control_run.resolve() == args.finetune_run.resolve():
            raise ValueError("Control and informed stages must use distinct run directories.")
        options._selected_model_runs.append(("generic_continuation", args.control_run.resolve()))
    return options


def validate_stages(options):
    stages = {}
    frozen = comparison.load_frozen_split(options.split_manifest)
    for label, directory in options._selected_model_runs:
        stage = json.loads((directory / "finetuning.json").read_text())
        identity = stage["overrides"]["finetuning"]
        if (identity["schema_version"] != "epit_finetuning_s2_v1"
                or identity["parent_step"] != 10000 or identity["starting_step"] != 0):
            raise ValueError("Comparison requires stages initialized at generic 10k and numbered from zero.")
        if not (directory / "step-0.ckpt").is_file():
            raise FileNotFoundError(f"Missing unchanged step-zero baseline in {directory}")
        if (stage["split_manifest_sha256"] != frozen.manifest_sha256
                or stage["split_lock_sha256"] != frozen.lock_sha256):
            raise ValueError("Fine-tuning and evaluation must use the same frozen split.")
        stages[label] = stage
    if stages["epit_finetuned"]["effective_trial_params"]["informed_prior_ratio"] != 1:
        raise ValueError("The informed branch must use 100% informed data.")
    if "generic_continuation" in stages:
        control = stages["generic_continuation"]
        informed = stages["epit_finetuned"]
        if control["effective_trial_params"]["informed_prior_ratio"] != 0:
            raise ValueError("The control must continue on 0% informed data.")
        if control["overrides"]["finetuning"] != informed["overrides"]["finetuning"]:
            raise ValueError("Control and informed stages must share the parent, schedule, and sampling seed.")
        for field in ("nproc_per_node", "save_perm_every"):
            if control[field] != informed[field]:
                raise ValueError(f"Control and informed stages differ in {field}.")
        configs = [vars(training_config(stage["train_command"])) for stage in (control, informed)]
        for config in configs:
            for field in ("checkpoint_dir", "informed_prior_ratio"):
                config.pop(field)
        if configs[0] != configs[1]:
            raise ValueError("Control and informed training settings differ beyond the prior share.")
    return stages


def run(args):
    options = comparison_args(args)
    stages = {} if args.dry_run else validate_stages(options)
    output = comparison.run(options)
    if not args.dry_run:
        path = output / "results.json"
        payload = json.loads(path.read_text())
        payload["finetuning_stages"] = stages
        payload["checkpoint_step_semantics"] = "additional steps; step 0 is generic pretraining step 10000"
        payload["primary_comparison_step"] = 2000
        payload["primary_comparison_complete"] = all(
            any(row.get("model") == label and row.get("checkpoint_step") == 2000 for row in payload["rows"])
            for label in stages
        )
        if args.checkpoint == "all":
            for label in stages:
                if not any(row.get("model") == label and row.get("checkpoint_step") == 0
                           for row in payload["rows"]):
                    raise RuntimeError(f"Comparison omitted the step-zero baseline for {label}.")
        path.write_text(json.dumps(payload, indent=2) + "\n")
    return output


if __name__ == "__main__":
    run(parse_args())
