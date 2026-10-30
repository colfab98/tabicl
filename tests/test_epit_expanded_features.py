from dataclasses import replace
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.build_epit_expanded_feature_profile import build_assets
from scripts.epit_pipeline.composition_data import load_reviewed_compositions, PROCESSED_DIR
from scripts.epit_pipeline import run_optuna as search
from scripts.epit_pipeline import evaluate_optuna_folds
from scripts.eval_corrosion_datasets import (
    _make_expanded_epit_task, apply_epit_pipeline_final_test,
    preprocess_epit_split, augment_pitting_magpie_split,
)
from tabicl.prior.epit_expanded_schema import (
    EPIT_EXPANDED_FEATURE_COLUMNS as COLUMNS,
    EPIT_EXPANDED_CATEGORICAL_COLUMNS as CATEGORICAL,
    EPIT_EXPANDED_FEATURE_PROFILE as PROFILE,
    EPIT_EXPANDED_BLOCK_ALLOCATION as BLOCKS,
)
from tabicl.prior.epit_feature_profile import load_epit_feature_profile
from tabicl.prior.epit_feature_generator import sample_epit_feature_rows
from tabicl.prior.dataset import SCMPrior, EPIT_TARGET_RULE_COEFFICIENTS
from tabicl.prior.prior_config import DEFAULT_FIXED_HP


def test_all_reviewed_fields_and_missingness_stay_paired():
    profile = load_epit_feature_profile()
    registry = json.loads((PROCESSED_DIR / "feature_registry.json").read_text())
    assert profile.name == PROFILE
    assert list(COLUMNS) == registry["feature_columns"]
    assert len(COLUMNS) == 70
    data = load_reviewed_compositions().rows.set_index("task_row_index").loc[profile.source_task_row_indices]
    expected = data.loc[:, COLUMNS[24:]].copy()
    for column in CATEGORICAL:
        expected[column] = expected[column].map(profile.metadata["category_mappings"][column])
    np.testing.assert_allclose(profile.expanded_environment_values, expected.astype(float), equal_nan=True)
    first = sample_epit_feature_rows(512, random_state=97)
    second = sample_epit_feature_rows(512, random_state=97)
    np.testing.assert_array_equal(first.features, second.features)
    np.testing.assert_allclose(first.features[:, 24:], expected.to_numpy()[first.environment_indices], equal_nan=True)
    assert first.features.shape == (512, 70)
    assert np.isfinite(first.features[:, :24]).all()
    assert np.isfinite(first.latent_features).all()
    assert set(first.compositions.family_names) == {"fe_alloy", "nicrmo_alloy"}
    for old_index, column in enumerate(("CP_temp", "CP_Cl", "CP_pH"), 24):
        np.testing.assert_array_equal(first.formula_features[:, old_index], first.latent_features[:, COLUMNS.index(column)])
    np.testing.assert_array_equal(first.formula_features[:, 27], first.test_method_codes)
    assert "[Cl-] Test Method" not in COLUMNS
    for column in ("WO4_mol_L", "CP_SO4"):
        assert np.isnan(first.features[:, COLUMNS.index(column)]).all()
    observed = np.isfinite(first.features)
    np.testing.assert_array_equal(first.features[observed], first.latent_features[observed])


def test_final_queries_and_targets_cannot_change_expanded_sampling():
    reviewed = load_reviewed_compositions()
    expected = build_assets(reviewed)
    modified = reviewed.rows.copy(deep=True)
    ids = reviewed.assignments.loc[reviewed.assignments.outer_split.eq("final_test"), "row_id"]
    final = modified.row_id.isin(ids)
    for column in COLUMNS:
        modified.loc[final, column] = "query_only_category" if column in CATEGORICAL else 123456.0
    modified.loc[final, "[Cl-] Test Method"] = "query_only_method"
    modified.loc[final, "Material class"] = "query_only_family"
    modified["Epit, mV (SCE) Avg."] = -987654.0
    assert build_assets(replace(reviewed, rows=modified)) == expected


def _fixed_hp(rule="pren_n_linear", magpie=False):
    return {
        **DEFAULT_FIXED_HP,
        "pitting_composition_mode": "empirical_features_scm_target", "pitting_feature_profile": PROFILE,
        "pitting_fixed_epit_schema": True, "informed_target_family": "pitting_potential",
        "informed_physical_marginal_profile": "pitting_potential_v1", "informed_physical_marginal_prob": 1.,
        "informed_task_family_probs": (1., 0.), "informed_normal_block_allocation": BLOCKS,
        "informed_normal_block_allocation_min_counts": BLOCKS, "informed_target_mix_weight": 1.,
        "pitting_material_style_probs": (1., 0., 0., 0., 0.), "pitting_magpie_features": magpie,
        "pitting_target_rule_scores": {rule: 1.},
        "pitting_target_rule_coefficients": [f"{rule}.{k}={v}" for k, v in EPIT_TARGET_RULE_COEFFICIENTS[rule].items()],
        "cat_prob": 0., "permute_features": False, "permute_labels": False,
    }


def _generate(rule="pren_n_linear", kind="mlp_scm", magpie=False):
    hp = _fixed_hp(rule, magpie)
    prior = SCMPrior(batch_size=1, fixed_hp=hp, sampled_hp={}, n_jobs=1, device="cpu")
    width = 80 if magpie else 70
    params = {**hp, "seq_len": 96, "train_size": 64, "max_features": width, "num_features": width,
              "num_classes": 0, "num_layers": 2, "hidden_dim": 12, "noise_std": 0.,
              "prior_type": kind, "informed_mode": True, "device": "cpu"}
    np.random.seed(41)
    torch.manual_seed(41)
    return prior.generate_dataset(params), prior


@pytest.mark.parametrize("rule", search.PROMOTED_TARGET_RULE_FAMILIES)
def test_existing_formulas_still_use_their_original_inputs(rule):
    (X, y, d), prior = _generate(rule)
    batch = prior.last_pitting_feature_batch
    assert X.shape == (96, 70) and d.item() == 70
    assert prior.last_pitting_scm_target_input.shape == (96, 70)
    np.testing.assert_array_equal(torch.isnan(X).numpy(), np.isnan(batch.features))
    assert torch.isfinite(y).all()
    expected = SCMPrior.evaluate_fixed_epit_target_rule_numpy(
        batch.formula_features, prior.last_pitting_target_rule, environment_mode="raw",
    )
    np.testing.assert_allclose(y.numpy(), expected, atol=3e-5)
    assert prior.last_pitting_target_rule["target_rule_family"] == rule


@pytest.mark.parametrize("kind", ["mlp_scm", "tree_scm"])
@pytest.mark.parametrize("magpie", [False, True])
def test_scm_and_magpie_keep_new_columns_and_original_mask(kind, magpie):
    (X, y, d), prior = _generate(kind=kind, magpie=magpie)
    assert X.shape == (96, 80 if magpie else 70)
    assert d.item() == X.shape[1]
    assert torch.isfinite(y).all()
    assert not torch.isinf(X).any()
    assert torch.isfinite(X[:, :24]).all()
    np.testing.assert_array_equal(torch.isnan(X[:, :70]).numpy(), np.isnan(prior.last_pitting_feature_batch.features))
    if magpie:
        assert torch.isfinite(X[:, 70:]).all()


def test_real_evaluation_keeps_frozen_queries_and_same_encoding():
    task = _make_expanded_epit_task()
    assert tuple(task.X.columns) == COLUMNS and len(task.X) == 760
    apply_epit_pipeline_final_test([task], manifest_path=search.DEFAULT_SPLIT_MANIFEST)
    split = task.fixed_split
    assert len(split.train_index) == 608 and len(split.test_index) == 152
    reviewed = load_reviewed_compositions()
    expected_queries = reviewed.assignments.loc[reviewed.assignments.outer_split.eq("final_test"), "task_row_index"]
    assert set(split.test_index) == set(expected_queries)
    train, query = preprocess_epit_split(task.X.iloc[split.train_index], task.X.iloc[split.test_index])
    profile = load_epit_feature_profile()
    np.testing.assert_allclose(
        train.loc[profile.source_task_row_indices, COLUMNS[24:]], profile.expanded_environment_values, equal_nan=True,
    )
    augmented_train, augmented_query = augment_pitting_magpie_split(train, query)
    assert augmented_train.shape == (608, 80) and augmented_query.shape == (152, 80)
    pd.testing.assert_frame_equal(augmented_train.iloc[:, :70], train)
    altered = query.copy()
    altered["test_solution_type"] = "unseen_final_category"
    _, encoded = preprocess_epit_split(task.X.iloc[split.train_index], altered)
    assert encoded.test_solution_type.isna().all()


def _training_config(tmp_path):
    from tabicl.train.train_config import build_parser
    args = search.parse_args(["--device", "cpu", "--nproc-per-node", "1", "--prior-n-jobs", "1"])
    rules = search.load_target_rule_config(summary_path=args.target_rule_summary, split_manifest_path=args.split_manifest)
    params = search.trial_params_from_mapping({"informed_prior_ratio": 1., "mlp_prob": 1.,
        "pitting_composition_perturb_strength": 0.05, "informed_target_mix_weight": 0.5},
        composition_mode=args.pitting_composition_mode)
    command = search.training_command(args, params, tmp_path / "checkpoints", rules)
    entry = next(i for i, token in enumerate(command) if token.endswith("/train/run.py"))
    config = build_parser().parse_args(command[entry + 1:])
    assert config.min_features == config.max_features == 70
    assert config.pitting_feature_profile == PROFILE
    assert config.col_missing_value_embedding
    assert tuple(config.informed_normal_block_allocation) == BLOCKS
    return args, params, config


def test_new_training_and_fold_commands_use_matching_schema(tmp_path):
    args, params, _ = _training_config(tmp_path)
    command = search.eval_command(args, params, tmp_path / "checkpoint.ckpt", "new_epit", tmp_path / "eval")
    parsed = evaluate_optuna_folds.parse_args(command[2:])
    assert parsed.epit_expanded_features
    fold = evaluate_optuna_folds.fold_command(parsed, fold=1, fold_dir=tmp_path)
    assert "--epit-expanded-features" in fold
    assert fold[fold.index("--missing-aware-model") + 1] == "new_epit"
    assert "--epit-validation-fold" in fold


def test_one_cpu_training_step_preserves_sampled_missingness(tmp_path):
    from tabicl.train.run import Trainer
    _, _, config = _training_config(tmp_path)
    overrides = dict(device="cpu", prior_device="cpu", amp=False, python_seed=42, prior_n_jobs=1,
        dataloader_num_workers=0, batch_size=1, batch_size_per_gp=1, micro_batch_size=1, max_seq_len=32,
        num_quantiles=9, embed_dim=16, col_num_blocks=1, col_nhead=2, col_num_inds=4,
        row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2,
        warmup_proportion=0., max_steps=2, wandb_log=False)
    for key, value in overrides.items():
        setattr(config, key, value)
    trainer = Trainer(config)
    assert trainer.missingness_augmenter is None
    batch = trainer.dataloader.dataset.get_batch()
    assert batch[0].shape == (1, 32, 70)
    assert torch.isnan(batch[0]).any() and torch.isfinite(batch[0][..., :24]).all()
    seen = []
    handle = trainer.model.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    try:
        results = trainer.run_batch(batch)
    finally:
        handle.remove()
    torch.testing.assert_close(seen[0], batch[0], equal_nan=True)
    assert all(math.isfinite(value) for value in results.values())
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
    config.col_missing_value_embedding = False
    with pytest.raises(ValueError, match="require --col_missing_value_embedding"):
        trainer.configure_missingness()
