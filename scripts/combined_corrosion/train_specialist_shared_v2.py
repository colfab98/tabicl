#!/usr/bin/env python
"""Launch one specialist recipe through the shared 50-column V2 training setup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIOR_CONFIG = (
    REPO_ROOT
    / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed"
    / "combined_prior_config.json"
)
RECIPE_PROBABILITIES = {
    "EPIT": {"generic": 0.5, "EPIT": 0.5, "Soccol": 0.0},
    "Soccol": {"generic": 0.0, "EPIT": 0.0, "Soccol": 1.0},
}


def build_specialist_config(base_path: Path, source: str) -> dict:
    """Retain the frozen generators and shared schema; change only their weights."""
    probabilities = RECIPE_PROBABILITIES[source]
    config_bytes = base_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema_version") != "combined_corrosion_prior_v1":
        raise ValueError("Expected a frozen combined corrosion prior configuration.")
    if set(config.get("regimes", {})) != set(probabilities):
        raise ValueError("The base configuration must define generic, EPIT, and Soccol regimes.")
    for name, probability in probabilities.items():
        config["regimes"][name]["probability"] = probability
    config["policy"]["source_model_probabilities"] = {
        name: float(name == source) for name in RECIPE_PROBABILITIES
    }
    config["policy"]["specialist_recipe"] = source
    config["derived_from"] = {
        "path": str(base_path.resolve()),
        "sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    return config


def prepare_run(source: str, environment: dict[str, str]) -> dict[str, str]:
    """Save the effective prior beside fresh checkpoints and return launch settings."""
    env = environment.copy()
    run_name = f"{source.lower()}_shared_v2_seed42"
    checkpoint_dir = (
        REPO_ROOT
        / Path(env.get("CHECKPOINT_DIR", f"checkpoints/combined_epit_soccol_mpea/{run_name}")).expanduser()
    ).resolve()
    if any(checkpoint_dir.glob("step-*.ckpt")):
        raise FileExistsError(f"Checkpoint directory already contains a run: {checkpoint_dir}")

    base_path = (REPO_ROOT / Path(env.get("PRIOR_CONFIG", str(DEFAULT_PRIOR_CONFIG))).expanduser()).resolve()
    config = build_specialist_config(base_path, source)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / "prior_config.json"
    try:
        with config_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(config, indent=2, sort_keys=True) + "\n")
    except FileExistsError:
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise FileExistsError(f"Refusing to replace a different prior configuration: {config_path}")

    env["PRIOR_CONFIG"] = str(config_path)
    env["CHECKPOINT_DIR"] = str(checkpoint_dir)
    env.setdefault("WANDB_NAME", f"combined_epit_soccol_mpea_{run_name}")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=RECIPE_PROBABILITIES)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Write the run configuration and print launch settings without starting training.",
    )
    args = parser.parse_args()
    env = prepare_run(args.source, dict(os.environ))
    command = ["bash", str(REPO_ROOT / "scripts/combined_corrosion/train_final_v2.sbatch")]
    print(f"{args.source} shared V2 recipe: {RECIPE_PROBABILITIES[args.source]}", flush=True)
    print(f"Prior configuration: {env['PRIOR_CONFIG']}", flush=True)
    print(f"Checkpoints: {env['CHECKPOINT_DIR']}", flush=True)
    if args.dry_run:
        print(shlex.join(command))
        return
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
