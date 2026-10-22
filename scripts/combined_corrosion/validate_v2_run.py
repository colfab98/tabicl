"""Validate a V2 fresh/resume launch against the exact shared trainer arguments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
from pathlib import Path
import re
import sys

import torch

from tabicl.train.missingness import MissingnessAugmenter
from tabicl.train.run import Trainer
from tabicl.train.train_config import build_parser


def validate_run(config: argparse.Namespace, *, mode: str, checkpoint_path: Path | None = None) -> Path | None:
    directory = Path(config.checkpoint_dir).resolve()
    if checkpoint_path is not None:
        if mode != "resume" or not checkpoint_path.is_file():
            raise ValueError("An explicit resume checkpoint must be an existing file.")
        paths = [checkpoint_path.resolve()]
    else:
        paths = list(directory.glob("step-*.ckpt"))
    if mode == "fresh":
        if paths:
            raise FileExistsError(f"Checkpoint directory already contains a run: {directory}. Use --resume.")
        print(f"Validated fresh V2 run: {directory}", file=sys.stderr)
        return None
    if mode != "resume":
        raise ValueError(f"Unknown run mode: {mode}")
    if not paths:
        raise FileNotFoundError(f"No checkpoint to resume in {directory}; refusing to start from scratch.")

    steps = {}
    for path in paths:
        match = re.fullmatch(r"step-(\d+)\.ckpt", path.name)
        if match is None:
            raise ValueError(f"Malformed checkpoint filename: {path.name}")
        steps[path] = int(match.group(1))
    path = max(paths, key=steps.get)
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    required = {"config", "state_dict", "optimizer_state", "scheduler_state", "curr_step",
                "combined_corrosion_prior", "missingness_augmentation"}
    if missing := required - checkpoint.keys():
        raise ValueError(f"Incomplete resume checkpoint; missing {sorted(missing)}")
    step = checkpoint["curr_step"]
    if not isinstance(step, int) or step != steps[path] or step <= 0:
        raise ValueError("Checkpoint step does not agree with its filename.")
    if config.only_load_model:
        raise ValueError("V2 resume must restore optimizer and scheduler, not only model weights.")
    if not step < config.max_steps <= config.scheduler_total_steps:
        raise ValueError("Resume requires checkpoint step < max_steps <= scheduler_total_steps.")
    # Historical V2 checkpoints do not store the LR lambda's horizon. Keep the
    # frozen launch schedule and also check that its LR agrees at the saved step.
    if (config.scheduler, config.scheduler_total_steps, config.warmup_proportion, config.lr) != (
        "cosine_warmup", 10000, 0.02, 1e-4
    ):
        raise ValueError("V2 resume requires the original 10,000-step cosine schedule, 0.02 warmup and 1e-4 LR.")

    prior_hash = hashlib.sha256(Path(config.combined_corrosion_prior_config).read_bytes()).hexdigest()
    prior = checkpoint["combined_corrosion_prior"] or {}
    if prior.get("config_sha256") != prior_hash:
        raise ValueError("Prior configuration differs from the checkpoint.")
    augmenter = MissingnessAugmenter(
        config.missingness_profile_path,
        augmentation_probability=config.missingness_augmentation_probability,
        min_observed_composition=config.missingness_min_observed_composition,
        query_column_probability=config.missingness_query_column_probability,
        preserve_observed_composition=config.missingness_preserve_observed_composition,
    )
    recorded_masks = dict(checkpoint["missingness_augmentation"] or {})
    current_masks = dict(augmenter.provenance)
    # A relocated checkout is valid when the profile's bytes and policy match.
    recorded_masks.pop("profile_path", None)
    current_masks.pop("profile_path", None)
    if recorded_masks != current_masks:
        raise ValueError("Missingness profile or policy differs from the checkpoint.")

    # Exercise the real model/optimizer/scheduler constructors and state loaders
    # on CPU. Do not initialize a data loader, WandB, distributed training or CUDA.
    trainer = Trainer.__new__(Trainer)
    trainer.config = copy.copy(config)
    trainer.config.device = "cpu"
    trainer.ddp = False
    trainer.master_process = False
    trainer.build_model()
    if checkpoint["config"] != trainer.model_config:
        raise ValueError("Model architecture differs from the checkpoint.")
    trainer.configure_optimizer()
    expected_lrs = [base * function(step) for base, function in zip(
        trainer.scheduler.base_lrs, trainer.scheduler.lr_lambdas, strict=True
    )]
    saved_scheduler = checkpoint["scheduler_state"]
    saved_optimizer = checkpoint["optimizer_state"]
    if saved_scheduler.get("last_epoch") != step or not saved_optimizer.get("state"):
        raise ValueError("Optimizer/scheduler state is missing or inconsistent with the saved step.")
    groups = saved_optimizer.get("param_groups", [])
    saved_lrs = saved_scheduler.get("_last_lr", [])
    if len(groups) != len(expected_lrs) or len(saved_lrs) != len(expected_lrs):
        raise ValueError("Optimizer/scheduler parameter groups do not match.")
    for group, saved_lr, expected_lr in zip(groups, saved_lrs, expected_lrs, strict=True):
        if not all(math.isclose(value, expected_lr, rel_tol=1e-9, abs_tol=1e-12)
                   for value in (group["lr"], saved_lr)):
            raise ValueError("Saved learning rate does not match the requested schedule.")
    trainer.raw_model.load_state_dict(checkpoint["state_dict"], strict=True)
    trainer.optimizer.load_state_dict(saved_optimizer)
    trainer.scheduler.load_state_dict(saved_scheduler)
    sampling = checkpoint.get("sampling_state")
    if sampling is not None:
        if sampling["version"] != 1 or len(sampling["ranks"]) != 1:
            raise ValueError("V2 resume requires a single-rank sampling checkpoint in the supported format.")
        state = sampling["ranks"][0]
        if state["loader"] is None:
            raise ValueError("V2 sampling checkpoint is missing the prior loader state.")
        trainer.ddp_rank = 0
        trainer.configure_missingness()
        trainer.configure_prior()
        trainer.dataloader.load_state_dict(state["loader"])
    print(f"Validated V2 resume: {path} (step {step} -> {config.max_steps}); "
          "model, prior, masking, optimizer and scheduler checked.", file=sys.stderr)
    if sampling is None:
        print("Historical checkpoints do not save RNG/data-loader state; this is not a "
              "bit-for-bit continuation of the synthetic data stream.", file=sys.stderr)
    else:
        print("Sampling state is present; worker count and prior stream settings checked.", file=sys.stderr)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("fresh", "resume"))
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    training_args = args.training_args
    if training_args[:1] == ["--"]:
        training_args = training_args[1:]
    config = build_parser().parse_args(training_args)
    torch.set_num_threads(2)
    path = validate_run(config, mode=args.mode)
    if path is not None:
        print(path)


if __name__ == "__main__":
    main()
