from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from corrosion_datasets.datasets.combined_epit_soccol_mpea.prepare_shared_tables import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
)
from scripts import eval_corrosion_datasets as corrosion_eval
from scripts.combined_corrosion import evaluate_final


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    REPO_ROOT
    / "corrosion_datasets"
    / "datasets"
    / "combined_epit_soccol_mpea"
    / "processed"
)


def test_combined_final_task_keeps_frozen_mpea_query_boundary() -> None:
    task = corrosion_eval._make_combined_pipeline_task(DATA_DIR)
    split = task.fixed_split

    assert split is not None
    assert task.X.shape == (5054, 50)
    assert list(task.X.columns) == list(FEATURE_COLUMNS)
    assert split.train_index.tolist() == list(range(4787))
    assert split.test_index.tolist() == list(range(4787, 5054))
    assert np.intersect1d(split.train_index, split.test_index).size == 0
    assert task.X.iloc[split.train_index]["reference_type"].isna().sum() == 3350
    assert task.X.iloc[split.test_index]["reference_type"].eq("SCE").all()
    assert task.split_strategy == "combined_full_context_frozen_mpea_queries"


def test_combined_candidate_and_generic_keep_separate_native_settings() -> None:
    args = SimpleNamespace(
        native_tabicl_baselines=True,
        fixed_schema_model=["combined_corrpfn"],
        missing_aware_model=["combined_corrpfn"],
        tabicl_feat_shuffle_method="none",
        tabicl_norm_methods=["none"],
    )

    assert corrosion_eval.tabicl_inference_settings(
        args, "combined_corrpfn"
    ) == ("none", ["none"])
    assert corrosion_eval.tabicl_inference_settings(
        args, "generic_tabicl"
    ) == ("latin", None)
    assert corrosion_eval.tabicl_inference_settings(
        args, "pretrained_tabicl_v2", pretrained=True
    ) == ("latin", None)
    assert corrosion_eval.expected_missing_value_mode(
        args, "combined_corrpfn"
    ) is True
    assert corrosion_eval.expected_missing_value_mode(
        args, "generic_tabicl"
    ) is False


def test_combined_final_wrapper_uses_shared_evaluator_and_all_baselines() -> None:
    args = evaluate_final.parse_args(["--device", "cpu"])
    command = evaluate_final.evaluation_command(args)

    assert command[1].endswith("scripts/eval_corrosion_datasets.py")
    assert command.count("--run") == 2
    assert command[command.index("--task") + 1] == corrosion_eval.COMBINED_PIPELINE_TASK_ID
    assert command[command.index("--fixed-schema-model") + 1] == "combined_corrpfn"
    assert command[command.index("--missing-aware-model") + 1] == "combined_corrpfn"
    assert command[command.index("--checkpoint") + 1] == "all"
    assert command[command.index("--checkpoint-step-interval") + 1] == "0"
    assert "--native-tabicl-baselines" in command
    assert "--compare-pretrained-tabicl" in command
    assert "--compare-catboost" in command
    assert command[command.index("--device") + 1] == "cpu"
    for column in CATEGORICAL_FEATURES:
        assert column in command


def test_combined_final_sbatch_matches_other_final_launchers() -> None:
    launcher = (
        REPO_ROOT / "scripts" / "combined_corrosion" / "evaluate_final.sbatch"
    ).read_text(encoding="utf-8")

    assert 'CHECKPOINT="${CHECKPOINT:-all}"' in launcher
    assert "python -m scripts.combined_corrosion.evaluate_final" in launcher
    assert "checkpoint_comparison_${RUN_TAG}" in launcher
