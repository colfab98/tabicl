"""Fine-tune generic 10k weights with the explicit EPIT v5 recipe, or a generic control."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

from scripts.epit_pipeline import train_final
from scripts.epit_pipeline.artifact_hashes import sha256_file

DEFAULT_PARENT = train_final.REPO_ROOT / "checkpoints/tabicl_s1_regression_baseline/step-10000.ckpt"
TRIAL27_TARGET_WEIGHT = 0.28659758621052106
S2_SETTINGS = {
    "max_steps": 2000, "scheduler_total_steps": 2000,
    "lr": 2e-5, "scheduler": "polynomial_decay_warmup",
    "warmup_proportion": 0.0, "poly_decay_lr_end": 5e-6, "poly_decay_power": 2.0,
    "batch_size": 512, "micro_batch_size": 1,
    "gradient_clipping": 1.0, "weight_decay": 0.0,
}
RETAINED_TABLE_SETTINGS = {
    "max_seq_len": 1024, "min_seq_len": None, "min_train_size": 0.1, "max_train_size": 0.9,
    "seq_len_per_gp": False, "log_seq_len": False, "batch_size_per_gp": 4,
}


def parse_args(argv=None):
    # Explicit current recipe; no new Optuna selection or journal dependency.
    defaults = [
        "--manual-hyperparameters", "--finetune-from", str(DEFAULT_PARENT),
        "--epit-target-version", "v5", "--informed-target-mix-policy", "uniform_per_table",
        "--override-mlp-prob", "0.7", "--override-informed-prior-ratio", "1",
        "--override-informed-target-mix-weight", str(TRIAL27_TARGET_WEIGHT),
        "--override-pitting-composition-perturb-strength", "0.3",
        "--max-steps", "2000", "--scheduler-total-steps", "2000",
        "--save-temp-every", "100", "--save-perm-every", "100",
        "--checkpoint-selection-min-step", "0", "--checkpoint-selection-interval", "100",
        "--np-seed", "43", "--torch-seed", "43", "--python-seed", "43",
        "--prior-stream-seed", "43",
        "--checkpoint-root", str(train_final.REPO_ROOT / "checkpoints/epit_finetuning"),
    ]
    return train_final.parse_args([*defaults, *(sys.argv[1:] if argv is None else argv)])


def validate_recipe(args):
    """Keep the agreed experiment fixed; only the informed share and resources vary."""
    expected = {
        "max_steps": 2000, "scheduler_total_steps": 2000,
        "np_seed": 43, "torch_seed": 43, "python_seed": 43, "prior_stream_seed": 43,
        "save_temp_every": 100, "save_perm_every": 100,
        "override_mlp_prob": 0.7,
        "override_informed_target_mix_weight": TRIAL27_TARGET_WEIGHT,
        "override_pitting_composition_perturb_strength": 0.3,
        "epit_target_version": "v5", "informed_target_mix_policy": "uniform_per_table",
    }
    for field, value in expected.items():
        if getattr(args, field) != value:
            raise ValueError(f"The predeclared fine-tuning recipe requires {field}={value!r}.")
    if args.override_informed_prior_ratio not in (0.0, 1.0):
        raise ValueError("Choose 100% informed (1) or the 100% generic control (0).")
    if args.override_use_magpie not in (None, False) or args.override_pitting_composition_family_probs is not None:
        raise ValueError("Fine-tuning retains the current EPIT feature and composition-family recipe.")


def parent_identity(args):
    import torch

    validate_recipe(args)
    path = args.finetune_from.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fine-tuning requires the completed generic 10k checkpoint: {path}")
    parent = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if parent.get("curr_step") != 10000 or path.name != "step-10000.ckpt":
        raise ValueError("Fine-tuning requires a generic checkpoint saved at step 10000.")
    stream = parent.get("paired_prior_stream") or {}
    # The original generic baseline predates sampler provenance. Accept its
    # explicitly supplied weights without pretending to recover its RNG state.
    if stream:
        groups = stream.get("group_counts", [])
        if (stream.get("version") != "original_tabicl_mixture_v1" or len(groups) != 2
                or groups[0] <= 0 or groups[1] != 0):
            raise ValueError("A parent with recorded prior provenance must be 0% informed.")
    if args.prior_stream_seed == stream.get("seed"):
        raise ValueError("Use a new --prior-stream-seed to avoid replaying the parent's synthetic tables.")
    if parent.get("missingness_augmentation") is not None:
        raise ValueError("The generic parent must use the original input policy without missingness augmentation.")
    if not parent.get("state_dict"):
        raise ValueError("The parent checkpoint contains no model weights.")
    if parent.get("combined_corrosion_prior") is not None:
        raise ValueError("The parent must be the generic baseline, not a combined corrosion model.")
    if parent.get("config", {}).get("max_classes") != 0:
        raise ValueError("Fine-tuning requires the baseline's regression head.")
    return {
        "schema_version": "epit_finetuning_s2_v1",
        "parent_checkpoint": str(path), "parent_sha256": sha256_file(path),
        "parent_step": 10000, "starting_step": 0,
        "step_semantics": "additional optimizer steps since generic pretraining",
        "initialization": "weights_only",
        "optimizer": "fresh_AdamW", "scaler": "fresh",
        "optimization": dict(S2_SETTINGS), "table_settings": dict(RETAINED_TABLE_SETTINGS),
        "primary_checkpoint_step": 2000,
        "additional_steps": args.max_steps, "prior_stream_seed": args.prior_stream_seed,
        "parent_prior_stream_seed": stream.get("seed"),
        "parent_sampler_provenance_available": bool(stream),
        "sampling": "fresh stream shared by control and informed branches",
        "implementation_sha256": {
            name: sha256_file(path) for name, path in {
                "finetune.py": Path(__file__),
                "run.py": train_final.REPO_ROOT / "src/tabicl/train/run.py",
                "optim.py": train_final.REPO_ROOT / "src/tabicl/train/optim.py",
            }.items()
        },
    }


def training_config(command):
    from tabicl.train.train_config import build_parser

    entry = next(i for i, token in enumerate(command) if token.endswith("/train/run.py"))
    return build_parser().parse_args(command[entry + 1:])


def normalized_model_config(config):
    from tabicl import TabICL

    defaults = {name: param.default for name, param in inspect.signature(TabICL).parameters.items()
                if param.default is not inspect.Parameter.empty}
    return {**defaults, **config}


def configure_initialization(args, command):
    import torch
    from tabicl.train.run import Trainer

    for field, value in S2_SETTINGS.items():
        option = f"--{field}"
        if option in command:
            train_final.replace_option(command, option, value)
        else:
            command.extend([option, str(value)])
    config = training_config(command)
    parent = torch.load(args.finetuning_identity["parent_checkpoint"], map_location="cpu",
                        weights_only=True, mmap=True)
    if normalized_model_config(parent["config"]) != normalized_model_config(Trainer.model_configuration(config)):
        raise ValueError("Fine-tuning must preserve the parent architecture and regression head exactly.")
    for field, value in RETAINED_TABLE_SETTINGS.items():
        if getattr(config, field) != value:
            raise ValueError(f"Fine-tuning must retain {field}={value!r}.")
    command.extend([
        "--checkpoint_path", args.finetuning_identity["parent_checkpoint"],
        "--only_load_model", "true", "--save_initial_checkpoint", "true",
    ])


def terminal_checkpoint(args, directory):
    """Freeze the a-priori endpoint without any development or test scoring."""
    import torch

    for step in (0, args.max_steps):
        path = directory / f"step-{step}.ckpt"
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if state.get("curr_step") != step:
            raise ValueError(f"Saved fine-tuning step does not match {path}.")
    return {"checkpoint_path": str(path), "checkpoint_sha256": sha256_file(path), "step": args.max_steps}


def main():
    train_final.run(parse_args())


if __name__ == "__main__":
    main()
