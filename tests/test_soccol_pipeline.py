from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import eval_corrosion_datasets as corrosion_eval
from scripts.eval_corrosion_datasets import preprocess_soccol_split
from tabicl.prior.dataset import (
    EPIT_TARGET_RULE_COEFFICIENTS,
    SOCCOL_TARGET_RULE_COEFFICIENTS,
    SCMPrior,
)
from tabicl.prior.prior_config import DEFAULT_FIXED_HP
from tabicl.prior.soccol_feature_generator import sample_soccol_feature_rows
from tabicl.prior.soccol_feature_profile import load_soccol_feature_profile
from tabicl.prior.soccol_schema import (
    SOCCOL_BASE_FEATURE_COUNT,
    SOCCOL_CATEGORICAL_COLUMNS,
    SOCCOL_COMPOSITION_COLUMNS,
    SOCCOL_CONTINUOUS_COLUMNS,
    SOCCOL_FEATURE_COLUMNS,
    SOCCOL_FEATURE_PROFILE,
    SOCCOL_FIXED_BLOCK_ALLOCATION,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RULE_ROOT = REPO_ROOT / "corrosion_datasets" / "analysis" / "soccol_pipeline" / "target_rules_v1"
SPLIT_MANIFEST = (
    REPO_ROOT
    / "corrosion_datasets"
    / "datasets"
    / "soccol_pitting_potential"
    / "processed"
    / "splits_v1"
    / "split_manifest.json"
)


def test_soccol_only_rules_are_not_exposed_to_legacy_epit_schema():
    soccol_only = {
        "pren_n_coupled_mns",
        "pren_n_coupled_mns_weak_anions",
    }

    assert soccol_only.isdisjoint(EPIT_TARGET_RULE_COEFFICIENTS)
    assert soccol_only <= SOCCOL_TARGET_RULE_COEFFICIENTS.keys()


def test_soccol_feature_profile_samples_finite_37_column_rows():
    profile = load_soccol_feature_profile()
    batch = sample_soccol_feature_rows(512, profile=profile, random_state=42)

    assert profile.n_rows == 4027
    assert batch.features.shape == (512, SOCCOL_BASE_FEATURE_COUNT)
    assert np.isfinite(batch.features).all()
    composition_sums = batch.features[:, : len(SOCCOL_COMPOSITION_COLUMNS)].sum(axis=1)
    assert np.all(np.isclose(composition_sums, 0.0) | np.isclose(composition_sums, 100.0))


def test_soccol_real_preprocessing_uses_context_only_state():
    train = pd.DataFrame(0.0, index=range(3), columns=SOCCOL_FEATURE_COLUMNS)
    test = pd.DataFrame(0.0, index=range(2), columns=SOCCOL_FEATURE_COLUMNS)
    for column in SOCCOL_CATEGORICAL_COLUMNS:
        train[column] = ["known", None, "known"]
        test[column] = ["known", "unseen"]
    train[SOCCOL_CONTINUOUS_COLUMNS[0]] = [1.0, np.nan, 3.0]
    test[SOCCOL_CONTINUOUS_COLUMNS[0]] = [np.nan, 9.0]

    encoded_train, encoded_test = preprocess_soccol_split(train, test)

    assert encoded_train.shape[1] == encoded_test.shape[1] == 37
    assert np.isfinite(encoded_train.to_numpy()).all()
    assert np.isfinite(encoded_test.to_numpy()).all()
    assert encoded_train.loc[1, SOCCOL_CONTINUOUS_COLUMNS[0]] == 2.0
    assert encoded_test.loc[0, SOCCOL_CONTINUOUS_COLUMNS[0]] == 2.0
    for column in SOCCOL_CATEGORICAL_COLUMNS:
        assert encoded_train.loc[1, column] == -1.0
        assert encoded_test.loc[1, column] == -1.0


def test_soccol_final_split_uses_3222_context_and_805_untouched_rows():
    task = corrosion_eval._make_soccol_pipeline_task(SPLIT_MANIFEST)
    corrosion_eval.apply_epit_pipeline_final_test(
        [task],
        manifest_path=SPLIT_MANIFEST,
    )

    split = task.fixed_split
    assert split is not None
    assert len(task.X) == 4027
    assert len(split.train_index) == 3222
    assert len(split.test_index) == 805
    assert not set(split.train_index) & set(split.test_index)
    assert split.split_strategy == "epit_pipeline_final_test"


def test_soccol_final_wrappers_reuse_epit_workflow_with_soccol_configuration():
    script = """
import json
from pathlib import Path

from scripts.epit_pipeline.artifact_hashes import FrozenFinalModel
from scripts.soccol_pipeline import evaluate_final, train_final

evaluate_final.configure()
train_base = train_final.base
eval_base = evaluate_final.base
args = train_base.parse_args(["--storage", "journal:///unused.log"])
split = train_base.load_frozen_split(args.split_manifest)
rules = train_base.search.load_target_rule_config(
    summary_path=args.target_rule_summary,
    split_manifest_path=args.split_manifest,
)
args.pitting_composition_mode = "empirical_features_scm_target"
params = train_base.search.TrialParams(
    use_magpie=False,
    informed_prior_ratio=0.5,
    mlp_prob=0.5,
    informed_feature_block_strength=0.0,
    informed_target_mix_weight=0.5,
    pitting_material_dirichlet_prob=0.0,
    pitting_material_dirichlet_concentration=None,
    pitting_material_dirichlet_active_prob=None,
    pitting_composition_perturb_strength=0.05,
)
train_command = train_base.search.training_command(
    args,
    params,
    Path("soccol_final_checkpoints").resolve(),
    rules,
)
fold_command = train_base.search.eval_command(
    args,
    params,
    Path("step-1000.ckpt").resolve(),
    "soccol_final_check",
    Path("soccol_fold_evaluation").resolve(),
)
configuration = {
    "task_id": eval_base.PITTING_TASK_ID,
    "device": "cpu",
    "random_state": 42,
    "n_estimators": 8,
    "tabicl_feat_shuffle_method": "none",
    "tabicl_norm_methods": ["none"],
    "regression_output": "median",
    "regression_uncertainty": False,
    "pitting_magpie_features": False,
    "expected_n_features": 37,
    "max_samples_per_task": 0,
}
frozen = FrozenFinalModel(
    manifest={
        "split": {
            "manifest": str(split.manifest_path),
            "manifest_sha256": split.manifest_sha256,
            "lock_sha256": split.lock_sha256,
        },
        "evaluation_configuration": configuration,
    },
    lock={},
    manifest_path=Path("final_model_manifest.json").resolve(),
    lock_path=Path("final_model_lock.json").resolve(),
    checkpoint_path=Path("step-10000.ckpt").resolve(),
    manifest_sha256="unused",
    lock_sha256="unused",
    checkpoint_sha256="unused",
)
command = eval_base.final_evaluation_command(
    frozen,
    output_dir=Path("final_test_evaluation").resolve(),
)
print(json.dumps({
    "study_name": args.study_name,
    "split_schema": split.manifest["schema_version"],
    "rule_count": len(rules.coefficients),
    "final_root": str(train_base.DEFAULT_FINAL_ROOT),
    "checkpoint_root": str(args.checkpoint_root),
    "train_task_id": train_base.PITTING_TASK_ID,
    "eval_task_id": eval_base.PITTING_TASK_ID,
    "development_rows": eval_base.DEVELOPMENT_ROWS,
    "final_test_rows": eval_base.FINAL_TEST_ROWS,
    "model_label": eval_base.FINAL_MODEL_LABEL,
    "expected_n_features": eval_base.validate_frozen_model(frozen)["expected_n_features"],
    "command_task": command[command.index("--task") + 1],
    "uses_final_test": "--epit-final-test" in command,
    "uses_validation_fold": "--epit-validation-fold" in command,
    "training_profile": train_command[
        train_command.index("--pitting_feature_profile") + 1
    ],
    "training_features": train_command[train_command.index("--min_features") + 1],
    "coefficient_variation": train_command[
        train_command.index("--pitting_coefficient_variation") + 1
    ],
    "soccol_fold_evaluator": (
        "scripts/soccol_pipeline/evaluate_optuna_folds.py" in " ".join(fold_command)
    ),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)

    assert payload == {
        "study_name": "soccol_pipeline_optuna_empirical_features_scm_target_v1",
        "split_schema": "soccol_composition_split_manifest_v1",
        "rule_count": 9,
        "final_root": str(
            REPO_ROOT
            / "corrosion_datasets"
            / "analysis"
            / "soccol_pipeline"
            / "final_v1"
        ),
        "checkpoint_root": str(
            REPO_ROOT / "checkpoints" / "soccol_pipeline_final_v1"
        ),
        "train_task_id": corrosion_eval.SOCCOL_PIPELINE_TASK_ID,
        "eval_task_id": corrosion_eval.SOCCOL_PIPELINE_TASK_ID,
        "development_rows": 3222,
        "final_test_rows": 805,
        "model_label": "final_soccol_model",
        "expected_n_features": 37,
        "command_task": corrosion_eval.SOCCOL_PIPELINE_TASK_ID,
        "uses_final_test": True,
        "uses_validation_fold": False,
        "training_profile": "soccol_pitting_features_v1",
        "training_features": "37",
        "coefficient_variation": "0.8",
        "soccol_fold_evaluator": True,
    }


def test_soccol_baseline_wrapper_uses_development_folds_and_fair_catboost():
    script = """
import json
from pathlib import Path

from scripts.soccol_pipeline import evaluate_baseline_folds

evaluate_baseline_folds.configure()
base = evaluate_baseline_folds.base
args = base.parse_args(["--device", "cpu"])
split = base.load_frozen_split(args.split_manifest)
command = base.fold_command(args, fold=3, fold_dir=Path("fold_3").resolve())
categorical_columns = [
    command[index + 1]
    for index, token in enumerate(command)
    if token == "--catboost-categorical-column"
]
print(json.dumps({
    "task_id": base.PITTING_TASK_ID,
    "split_schema": split.manifest["schema_version"],
    "development_rows": base.DEVELOPMENT_ROWS,
    "context_rows": sorted(base.CONTEXT_ROW_COUNTS),
    "validation_rows": sorted(base.VALIDATION_ROW_COUNTS),
    "output_dir": str(args.output_dir),
    "uses_validation_fold": "--epit-validation-fold" in command,
    "uses_final_test": "--epit-final-test" in command,
    "compares_pretrained": "--compare-pretrained-tabicl" in command,
    "compares_catboost": "--compare-catboost" in command,
    "categorical_columns": categorical_columns,
    "catboost_iterations": command[command.index("--catboost-iterations") + 1],
    "catboost_depth": command[command.index("--catboost-depth") + 1],
    "catboost_learning_rate": command[
        command.index("--catboost-learning-rate") + 1
    ],
    "catboost_l2_leaf_reg": command[
        command.index("--catboost-l2-leaf-reg") + 1
    ],
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)

    assert payload == {
        "task_id": corrosion_eval.SOCCOL_PIPELINE_TASK_ID,
        "split_schema": "soccol_composition_split_manifest_v1",
        "development_rows": 3222,
        "context_rows": [2577, 2578],
        "validation_rows": [644, 645],
        "output_dir": str(
            REPO_ROOT
            / "corrosion_datasets"
            / "analysis"
            / "soccol_pipeline"
            / "baseline_folds_v1"
        ),
        "uses_validation_fold": True,
        "uses_final_test": False,
        "compares_pretrained": True,
        "compares_catboost": True,
        "categorical_columns": list(SOCCOL_CATEGORICAL_COLUMNS),
        "catboost_iterations": "1000",
        "catboost_depth": "6",
        "catboost_learning_rate": "0.03",
        "catboost_l2_leaf_reg": "3.0",
    }


def test_soccol_empirical_scm_target_generates_finite_dataset():
    artifact = json.loads(
        (RULE_ROOT / "pren_n_coupled_mns_weak_anions.json").read_text(encoding="utf-8")
    )
    family = artifact["rule_family"]
    coefficients = artifact["final_development_calibration"]["coefficients"]
    fixed_hp = dict(DEFAULT_FIXED_HP)
    fixed_hp.update(
        {
            "pitting_composition_mode": "empirical_features_scm_target",
            "pitting_feature_profile": SOCCOL_FEATURE_PROFILE,
            "pitting_fixed_epit_schema": True,
            "informed_target_family": "pitting_potential",
            "informed_physical_marginal_profile": "pitting_potential_v1",
            "informed_physical_marginal_prob": 1.0,
            "informed_task_family_probs": (1.0, 0.0),
            "informed_normal_block_allocation": SOCCOL_FIXED_BLOCK_ALLOCATION,
            "informed_normal_block_allocation_min_counts": SOCCOL_FIXED_BLOCK_ALLOCATION,
            "informed_feature_block_strength": 0.0,
            "pitting_composition_perturb_strength": 0.05,
            "pitting_magpie_features": False,
            "cat_prob": 0.0,
            "permute_features": False,
            "permute_labels": False,
            "pitting_target_rule_scores": {family: 1.0},
            "pitting_target_rule_coefficients": [
                f"{family}.{name}={value}" for name, value in coefficients.items()
            ],
            "pitting_coefficient_variation": 0.0,
            "pitting_coefficient_upper_bounds": {
                family: artifact["coefficient_upper_bounds"]
            },
        }
    )
    prior = SCMPrior(
        batch_size=1,
        fixed_hp=fixed_hp,
        sampled_hp={},
        n_jobs=1,
        device="cpu",
    )
    params = {
        **fixed_hp,
        "seq_len": 64,
        "train_size": 40,
        "max_features": SOCCOL_BASE_FEATURE_COUNT,
        "num_features": SOCCOL_BASE_FEATURE_COUNT,
        "num_classes": 0,
        "num_layers": 2,
        "hidden_dim": 12,
        "noise_std": 0.0,
        "prior_type": "mlp_scm",
        "informed_mode": True,
        "device": "cpu",
    }
    np.random.seed(41)
    torch.manual_seed(41)

    X, y, feature_count = prior.generate_dataset(params)

    assert X.shape == (64, 37)
    assert y.shape == (64,)
    assert feature_count.item() == 37
    assert torch.isfinite(X).all()
    assert torch.isfinite(y).all()
    assert prior.last_pitting_target_rule["target_rule_family"] == family
    assert prior.last_pitting_target_rule["soccol_schema"] is True
