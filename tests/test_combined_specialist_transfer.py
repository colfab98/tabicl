from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.combined_corrosion import evaluate_specialist_transfer as specialist_eval
from tabicl.prior.epit_schema import EPIT_BASE_FEATURE_COUNT
from tabicl.prior.soccol_schema import SOCCOL_FEATURE_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    REPO_ROOT
    / "corrosion_datasets"
    / "datasets"
    / "combined_epit_soccol_mpea"
    / "processed"
)


def test_specialist_tasks_keep_frozen_query_and_native_schemas() -> None:
    tasks = specialist_eval.build_specialist_tasks(DATA_DIR)
    epit = tasks[specialist_eval.EPIT_MODEL_LABEL]
    soccol = tasks[specialist_eval.SOCCOL_MODEL_LABEL]

    assert epit.X.shape == (5054, EPIT_BASE_FEATURE_COUNT)
    assert soccol.X.shape == (5054, len(SOCCOL_FEATURE_COLUMNS))
    assert list(epit.X.columns) == list(specialist_eval.EPIT_NATIVE_COLUMNS)
    assert list(soccol.X.columns) == list(SOCCOL_FEATURE_COLUMNS)
    assert np.isfinite(epit.X.to_numpy(dtype=float)).all()
    assert np.isfinite(soccol.X.to_numpy(dtype=float)).all()
    assert epit.y.equals(soccol.y)
    for task in (epit, soccol):
        assert task.fixed_split is not None
        assert len(task.fixed_split.train_index) == 4787
        assert len(task.fixed_split.test_index) == 267
        assert task.fixed_split.test_index.tolist() == list(range(4787, 5054))


def test_checkpoint_selection_retains_each_specialists_last_step(tmp_path: Path) -> None:
    epit = tmp_path / "epit"
    soccol = tmp_path / "soccol"
    epit.mkdir()
    soccol.mkdir()
    for step in (500, 1000, 1400):
        (epit / f"step-{step}.ckpt").touch()
    for step in (500, 1000, 1200):
        (soccol / f"step-{step}.ckpt").touch()

    pairs = specialist_eval.select_checkpoint_pairs(
        epit,
        soccol,
        checkpoint="all",
        min_step=0,
        step_interval=1000,
    )
    assert pairs == [
        (1000, epit / "step-1000.ckpt", soccol / "step-1000.ckpt"),
        (1200, None, soccol / "step-1200.ckpt"),
        (1400, epit / "step-1400.ckpt", None),
    ]
    latest = specialist_eval.select_checkpoint_pairs(
        epit,
        soccol,
        checkpoint="latest",
        min_step=0,
        step_interval=0,
    )
    assert [step for step, _, _ in latest] == [1000]


def test_all_specialist_steps_need_no_shared_checkpoints(tmp_path: Path) -> None:
    epit, soccol = tmp_path / "epit", tmp_path / "soccol"
    epit.mkdir()
    soccol.mkdir()
    (epit / "step-500.ckpt").touch()
    (soccol / "step-1000.ckpt").touch()
    assert specialist_eval.select_checkpoint_pairs(
        epit, soccol, checkpoint="all", min_step=0, step_interval=0,
    ) == [
        (500, epit / "step-500.ckpt", None),
        (1000, None, soccol / "step-1000.ckpt"),
    ]


def test_specialist_driver_skips_missing_steps_and_keeps_outer_queries(tmp_path, monkeypatch):
    epit, soccol = tmp_path / "epit", tmp_path / "soccol"
    epit.mkdir()
    soccol.mkdir()
    (epit / "step-500.ckpt").touch()
    (soccol / "step-1000.ckpt").touch()
    output = tmp_path / "results"
    options = specialist_eval.parse_args([
        "--epit-run", str(epit), "--soccol-run", str(soccol),
        "--output-dir", str(output),
    ])
    monkeypatch.setattr(specialist_eval, "parse_args", lambda: options)
    calls = []

    def fake_evaluate(**kwargs):
        assert kwargs["task_split"].test_index.tolist() == list(range(4787, 5054))
        calls.append(kwargs["model_label"])
        return {
            "model": kwargs["model_label"], "model_kind": kwargs["model_kind"],
            "task_id": kwargs["task"].task_id, "test_spearman": .3,
        }

    monkeypatch.setattr(specialist_eval.corrosion_eval, "evaluate_estimator", fake_evaluate)
    specialist_eval.main()
    payload = json.loads((output / "results.json").read_text())
    assert not payload["errors"]
    assert calls == [specialist_eval.EPIT_MODEL_LABEL, specialist_eval.SOCCOL_MODEL_LABEL]
    assert [(row["checkpoint_step"], row["model"]) for row in payload["rows"]] == [
        (500, specialist_eval.EPIT_MODEL_LABEL), (1000, specialist_eval.SOCCOL_MODEL_LABEL),
    ]


def test_specialist_launcher_matches_checkpoint_comparison_convention() -> None:
    launcher = (
        REPO_ROOT
        / "scripts"
        / "combined_corrosion"
        / "evaluate_specialist_transfer.sbatch"
    ).read_text(encoding="utf-8")

    assert 'CHECKPOINT="${CHECKPOINT:-all}"' in launcher
    assert "python -m scripts.combined_corrosion.evaluate_specialist_transfer" in launcher
    assert "specialist_transfer_${RUN_TAG}" in launcher
