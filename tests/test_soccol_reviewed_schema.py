"""The reviewed bank, generator, trainer and evaluator share one feature schema."""

from copy import deepcopy
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import eval_corrosion_datasets as evaluation
from scripts.epit_pipeline import run_optuna as base, evaluate_optuna_folds
from scripts.soccol_pipeline import run_optuna as search
from scripts.soccol_pipeline.data import load_soccol_dataset, select_accepted_development, SPLIT_MANIFEST
from tabicl.prior.dataset import SCMPrior, SOCCOL_TARGET_RULE_COEFFICIENTS
from tabicl.prior.prior_config import DEFAULT_FIXED_HP
from tabicl.prior.soccol_schema import SOCCOL_FEATURE_COLUMNS as OLD_COLUMNS
from tabicl.prior.soccol_reviewed_schema import (
    SOCCOL_REVIEWED_FEATURE_PROFILE as PROFILE,
    SOCCOL_REVIEWED_FEATURE_COLUMNS as COLUMNS,
    SOCCOL_REVIEWED_CATEGORICAL_COLUMNS as CATEGORICAL,
    SOCCOL_REVIEWED_BLOCK_ALLOCATION as BLOCKS,
)
from tabicl.prior.soccol_feature_profile import load_soccol_feature_profile


def _generate(rule="pren_n_linear", kind="mlp_scm", mixing=1., profile=PROFILE):
    hp = {
        **DEFAULT_FIXED_HP,
        "pitting_composition_mode": "empirical_features_scm_target", "pitting_feature_profile": profile,
        "pitting_fixed_epit_schema": True, "informed_target_family": "pitting_potential",
        "informed_physical_marginal_profile": "pitting_potential_v1", "informed_physical_marginal_prob": 1.,
        "informed_task_family_probs": (1., 0.), "informed_normal_block_allocation": BLOCKS,
        "informed_normal_block_allocation_min_counts": BLOCKS, "informed_target_mix_weight": mixing,
        "pitting_target_rule_scores": {rule: 1.},
        "pitting_target_rule_coefficients": [f"{rule}.{k}={v}" for k, v in SOCCOL_TARGET_RULE_COEFFICIENTS[rule].items()],
        "cat_prob": 0., "permute_features": False, "permute_labels": False,
    }
    prior = SCMPrior(batch_size=1, fixed_hp=hp, sampled_hp={}, n_jobs=1, device="cpu")
    params = {**hp, "seq_len": 64, "train_size": 40, "max_features": 58, "num_features": 58,
              "num_classes": 0, "num_layers": 2, "hidden_dim": 12, "noise_std": 0.,
              "prior_type": kind, "informed_mode": True, "device": "cpu"}
    np.random.seed(41)
    torch.manual_seed(41)
    return prior.generate_dataset(params), prior


@pytest.mark.parametrize("rule", search.PROMOTED_RULES)
def test_all_nine_formulas_read_the_same_named_values_after_reordering(rule):
    (X, y, d), prior = _generate(rule)
    batch = prior.last_pitting_feature_batch
    assert X.shape == (64, 58) and d.item() == 58
    assert prior.last_pitting_scm_target_input.shape == (64, 58)
    assert prior.last_pitting_target_rule["soccol_feature_columns"] == list(COLUMNS)
    np.testing.assert_array_equal(torch.isnan(X).numpy(), np.isnan(batch.features))
    # Independent legacy-order projection: the excluded text/other-ion fields
    # never enter these formulas, so their arbitrary placeholders cannot matter.
    old = np.column_stack([
        batch.latent_features[:, COLUMNS.index(c)] if c in COLUMNS else np.full(64, 123456.)
        for c in OLD_COLUMNS
    ])
    old_rule = dict(prior.last_pitting_target_rule)
    old_rule.pop("soccol_feature_columns")
    _, old_drive = prior._evaluate_soccol_target_rule_tensor(torch.as_tensor(old, dtype=torch.float32), old_rule)
    expected = prior._standardize_signal(old_drive)
    torch.testing.assert_close(y, expected, atol=3e-5, rtol=1e-5)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("kind", ["mlp_scm", "tree_scm"])
@pytest.mark.parametrize("mixing", [0., 0.5])
def test_both_scm_targets_receive_all_58_fields_before_restoring_missingness(kind, mixing):
    (X, y, d), prior = _generate(kind=kind, mixing=mixing)
    assert X.shape == (64, 58) and d.item() == 58
    assert torch.isfinite(y).all() and not torch.isinf(X).any()
    assert torch.isfinite(X[:, :24]).all()
    np.testing.assert_array_equal(torch.isnan(X), np.isnan(prior.last_pitting_feature_batch.features))
    assert prior.last_pitting_scm_target_input.shape == (64, 58)


def test_explicit_v2_still_generates_reviewed_columns_and_missingness():
    (X, y, d), prior = _generate(profile="soccol_pitting_features_v2")
    assert X.shape == (64, 58) and d.item() == 58
    assert torch.isfinite(y).all()
    assert prior.last_pitting_feature_batch.profile_name == "soccol_pitting_features_v2"
    np.testing.assert_array_equal(torch.isnan(X), np.isnan(prior.last_pitting_feature_batch.features))


def test_real_inputs_share_bank_encoding_types_order_and_missingness():
    reviewed = load_soccol_dataset()
    selected = select_accepted_development(reviewed)
    profile = load_soccol_feature_profile(PROFILE)
    task = evaluation._make_soccol_pipeline_task(SPLIT_MANIFEST, reviewed_data=True)
    evaluation.apply_epit_pipeline_final_test([task], manifest_path=SPLIT_MANIFEST)
    split = task.fixed_split
    assert len(split.train_index) == 2950 and len(split.test_index) == 805
    train, query = evaluation.preprocess_soccol_split(task.X.iloc[split.train_index], task.X.iloc[split.test_index])
    assert tuple(train.columns) == tuple(query.columns) == COLUMNS
    assert all(dtype == float for dtype in train.dtypes)
    np.testing.assert_array_equal(train.to_numpy(), profile.feature_values)
    assert query[list(COLUMNS[:24])].isna().any(axis=1).sum() == 10
    # No vocabulary is fitted on supplied query or context subsets.
    mappings = profile.metadata["preprocessing"]["category_mappings"]
    assert set(mappings) == set(CATEGORICAL) and len(mappings) == 9
    for column in CATEGORICAL:
        unknown = selected.features.iloc[:3].copy()
        mapping = mappings[column]
        known = next(iter(mapping), np.nan)
        unknown[column] = [known, "query-only-value", np.nan]
        _, encoded = evaluation.preprocess_soccol_split(selected.features.iloc[:3], unknown)
        assert encoded[column].iloc[1:].isna().all()
        if mapping:
            # A legitimate category code of zero is not the missing-value code.
            assert encoded[column].iloc[0] == 0.0
        else:
            # An entirely unreported field has no invented category vocabulary.
            assert encoded[column].isna().all()
    with pytest.raises(ValueError, match="58-column order"):
        evaluation.preprocess_soccol_split(selected.features, unknown[list(reversed(COLUMNS))])
    changed = task.X.iloc[split.train_index].copy()
    changed.attrs = deepcopy(changed.attrs)
    changed.attrs["soccol_development_selection"]["selection_sha256"] = "different"
    with pytest.raises(RuntimeError, match="different development data"):
        evaluation.preprocess_soccol_split(changed, unknown)


def _training_configuration(tmp_path):
    """Construct commands without changing the EPIT module for other tests."""
    from tabicl.train.train_config import build_parser

    original = vars(base).copy()
    try:
        search.configure(reviewed_data=True)
        args = base.parse_args(["--device", "cpu", "--nproc-per-node", "1", "--prior-n-jobs", "1"])
        rules = base.load_target_rule_config(summary_path=args.target_rule_summary, split_manifest_path=args.split_manifest)
        params = base.trial_params_from_mapping({"informed_prior_ratio": 1., "mlp_prob": 1.,
            "pitting_composition_perturb_strength": 0.05, "informed_target_mix_weight": 0.5},
            composition_mode=args.pitting_composition_mode)
        command = search.training_command(args, params, tmp_path / "checkpoints", rules)
        entry = next(i for i, token in enumerate(command) if token.endswith("/train/run.py"))
        config = build_parser().parse_args(command[entry + 1:])
        fold_command = search.eval_command(args, params, tmp_path / "checkpoint.ckpt", "new_soccol", tmp_path / "eval")
        identity = search.empirical_feature_profile_identity()
        assert args.target_rule_summary.parent.name == "target_rules_v2"
        assert args.study_name.endswith("_v7")
        return config, fold_command, identity
    finally:
        for name, value in original.items():
            if vars(base).get(name) is not value:
                setattr(base, name, value)


def test_reviewed_training_and_fold_commands_use_the_same_schema(tmp_path):
    config, command, identity = _training_configuration(tmp_path)
    assert config.min_features == config.max_features == 58
    assert config.pitting_feature_profile == PROFILE
    assert tuple(config.informed_normal_block_allocation) == BLOCKS
    assert not config.col_missing_value_embedding
    assert config.original_prior_mix and config.permute_features
    assert identity["feature_columns"] == list(COLUMNS)
    assert identity["development_selection"]["source_rows"] == 2950
    parsed = evaluate_optuna_folds.parse_args(command[2:])
    assert parsed.soccol_reviewed_data
    fold = evaluate_optuna_folds.fold_command(parsed, fold=1, fold_dir=tmp_path)
    assert "--soccol-reviewed-data" in fold
    assert "--missing-aware-model" not in fold
    assert "--native-tabicl-baselines" in fold
    assert "--fixed-schema-model" not in fold
    assert "--epit-final-test" not in fold


def test_reviewed_optuna_refuses_missing_or_historical_calibration(tmp_path):
    with pytest.raises(RuntimeError, match="calibration is not ready"):
        search.load_reviewed_target_rule_config(summary_path=tmp_path / "absent.json", split_manifest_path=SPLIT_MANIFEST)
    old = search.PIPELINE_ROOT / "target_rules_v1" / "calibration_summary.json"
    with pytest.raises(RuntimeError, match="historical calibration cannot be reused"):
        search.load_reviewed_target_rule_config(summary_path=old, split_manifest_path=SPLIT_MANIFEST)


def test_one_cpu_training_step_and_local_checkpoint_inference(tmp_path):
    from tabicl.train.run import Trainer
    from tabicl import TabICLRegressor

    config, _, _ = _training_configuration(tmp_path)
    for key, value in dict(device="cpu", prior_device="cpu", amp=False, python_seed=42, prior_n_jobs=1,
        dataloader_num_workers=0, batch_size=1, batch_size_per_gp=1, micro_batch_size=1, max_seq_len=32,
        num_quantiles=9, embed_dim=16, col_num_blocks=1, col_nhead=2, col_num_inds=4,
        row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2,
        warmup_proportion=0., max_steps=2, wandb_log=False).items():
        setattr(config, key, value)
    trainer = Trainer(config)
    assert trainer.missingness_augmenter is None
    batch = trainer.dataloader.dataset.get_batch()
    assert batch[0].shape == (1, 32, 58)
    assert torch.isfinite(batch[0]).all()
    assert trainer.model.col_embedder.in_linear.weight.shape[1] == 3
    seen = []
    handle = trainer.model.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    try:
        result = trainer.run_batch(batch)
    finally:
        handle.remove()
    torch.testing.assert_close(seen[0], batch[0][..., :int(batch[2].max())])
    assert all(math.isfinite(value) for value in result.values())
    checkpoint = tmp_path / "schema_smoke.ckpt"
    torch.save({"config": trainer.model_config, "state_dict": trainer.model.state_dict()}, checkpoint)
    selected = select_accepted_development()
    # A tiny development-only smoke check; no final-model scores are produced.
    train, query = evaluation.preprocess_soccol_split(selected.features.iloc[:24], selected.features.iloc[24:28])
    estimator = TabICLRegressor(model_path=checkpoint, allow_auto_download=False, device="cpu",
        use_amp=False, n_estimators=8, norm_methods=["none", "power"], feat_shuffle_method="latin")
    estimator.fit(train, selected.targets.iloc[:24])
    assert not estimator.preserve_missing_values_
    prediction = estimator.predict(query)
    assert prediction.shape == (4,) and np.isfinite(prediction).all()
    assert 0 < estimator.ensemble_generator_.n_features_in_ <= 58
    assert sum(map(len, estimator.ensemble_generator_.ensemble_configs_.values())) == min(
        8, 2 * estimator.ensemble_generator_.n_features_in_)  # Original Latin permutations do not repeat views.
    encoded = estimator.X_encoder_.transform(query)
    assert np.isfinite(encoded).all()
    config.original_prior_mix = False  # Historical NaN-preserving mode still needs its encoder.
    with pytest.raises(ValueError, match="require --col_missing_value_embedding"):
        trainer.configure_missingness()
