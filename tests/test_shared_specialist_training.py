from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.combined_corrosion.train_specialist_shared_v2 import (
    DEFAULT_PRIOR_CONFIG,
    build_specialist_config,
    prepare_run,
)


@pytest.mark.parametrize(
    ("source", "probabilities"),
    [
        ("EPIT", {"generic": 0.5, "EPIT": 0.5, "Soccol": 0.0}),
        ("Soccol", {"generic": 0.0, "EPIT": 0.0, "Soccol": 1.0}),
    ],
)
def test_specialist_recipe_preserves_frozen_generators(source, probabilities) -> None:
    original_bytes = DEFAULT_PRIOR_CONFIG.read_bytes()
    original = json.loads(original_bytes)
    config = build_specialist_config(DEFAULT_PRIOR_CONFIG, source)

    assert config["feature_columns_in_order"] == original["feature_columns_in_order"]
    assert config["source_commands"] == original["source_commands"]
    assert config["policy"]["mpea_used"] is False
    for name, regime in config["regimes"].items():
        expected = dict(original["regimes"][name], probability=probabilities[name])
        assert regime == expected
    command = original["source_commands"][source]["command"]
    original_informed_ratio = float(command[command.index("--informed_prior_ratio") + 1])
    assert probabilities[source] == original_informed_ratio
    assert probabilities["generic"] == 1.0 - original_informed_ratio
    assert config["derived_from"]["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert DEFAULT_PRIOR_CONFIG.read_bytes() == original_bytes


def test_specialist_launch_refuses_checkpoint_reuse(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step-500.ckpt"
    checkpoint.write_bytes(b"existing checkpoint")

    with pytest.raises(FileExistsError, match="already contains a run"):
        prepare_run("EPIT", {"CHECKPOINT_DIR": str(tmp_path)})

    assert checkpoint.read_bytes() == b"existing checkpoint"
    assert not (tmp_path / "prior_config.json").exists()


def test_specialist_launch_preserves_saved_config(tmp_path: Path) -> None:
    overrides = {"CHECKPOINT_DIR": str(tmp_path), "MAX_STEPS": "500"}
    env = prepare_run("EPIT", overrides)
    config_path = Path(env["PRIOR_CONFIG"])
    original_bytes = config_path.read_bytes()

    assert env["MAX_STEPS"] == "500"
    assert "epit_shared_v2" in env["WANDB_NAME"]
    assert prepare_run("EPIT", overrides) == env
    with pytest.raises(FileExistsError, match="different prior configuration"):
        prepare_run("Soccol", overrides)
    assert config_path.read_bytes() == original_bytes
