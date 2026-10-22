"""Validate a fork from EPIT V2, retaining optimizer/schedule and changing only prior weights."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

from scripts.combined_corrosion.train_specialist_shared_v2 import RECIPE_PROBABILITIES
from scripts.combined_corrosion.validate_v2_run import validate_run
from tabicl.train.train_config import build_parser


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_continuation(config, *, arm, mode, parent_checkpoint, dry_run=False):
    directory = Path(config.checkpoint_dir).resolve()
    parent_checkpoint = Path(parent_checkpoint).resolve()
    parent_prior = parent_checkpoint.parent / "prior_config.json"
    target_prior = Path(config.combined_corrosion_prior_config).resolve()
    source = json.loads(parent_prior.read_text())
    target = json.loads(target_prior.read_text())
    expected_weights = RECIPE_PROBABILITIES[{"control": "EPIT", "soccol": "Soccol"}[arm]]
    if {name: regime["probability"] for name, regime in source["regimes"].items()} != RECIPE_PROBABILITIES["EPIT"]:
        raise ValueError("The parent must use the 50% EPIT / 50% generic recipe.")
    # Policy/derived_from describe how the file was created; all generator and
    # feature definitions must remain identical to the actual parent's recipe.
    expected = copy.deepcopy(source)
    for name, weight in expected_weights.items():
        expected["regimes"][name]["probability"] = weight
    for metadata in ("policy", "derived_from"):
        expected.pop(metadata, None)
        target.pop(metadata, None)
    if target != expected:
        raise ValueError("Continuation may change only the specified prior weights, not generators or schema.")
    manifest = {
        "parent_checkpoint": str(parent_checkpoint),
        "parent_sha256": sha256(parent_checkpoint),
        "arm": arm,
        "prior_probabilities": expected_weights,
        "prior_sha256": sha256(target_prior),
        "restart_seeds": {name: getattr(config, name) for name in ("np_seed", "torch_seed", "python_seed")},
        "sampling_policy": "Restart sampling in both branches; the historical parent has no sampling state.",
    }
    manifest_path = directory / "continuation.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing continuation provenance differs; use a separate output directory.")
    if mode == "resume":
        if not manifest_path.is_file():
            raise FileNotFoundError("Continuation resume requires its continuation.json provenance.")
        return validate_run(config, mode="resume")

    if directory == parent_checkpoint.parent or directory == target_prior.parent:
        raise ValueError("Continuation requires a separate output directory from the source runs.")
    validate_run(config, mode="fresh")
    parent_config = copy.copy(config)
    parent_config.combined_corrosion_prior_config = str(parent_prior)
    validate_run(parent_config, mode="resume", checkpoint_path=parent_checkpoint)
    checkpoint = torch.load(parent_checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if checkpoint.get("sampling_state") is not None:
        raise ValueError("This experiment requires the historical EPIT parent without sampling state.")
    if not dry_run:
        directory.mkdir(parents=True, exist_ok=True)
        if not manifest_path.exists():
            with manifest_path.open("x") as handle:
                json.dump(manifest, handle, indent=2)
                handle.write("\n")
    print(f"Validated {arm} continuation: parent step {checkpoint['curr_step']} -> {config.max_steps}; "
          f"prior proportions {expected_weights}. Optimizer and schedule retained.", file=sys.stderr)
    return parent_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("control", "soccol"))
    parser.add_argument("--mode", required=True, choices=("fresh", "resume"))
    parser.add_argument("--parent-checkpoint", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    training_args = args.training_args[1:] if args.training_args[:1] == ["--"] else args.training_args
    config = build_parser().parse_args(training_args)
    torch.set_num_threads(2)
    print(prepare_continuation(config, arm=args.arm, mode=args.mode,
                               parent_checkpoint=args.parent_checkpoint, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
