from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tabicl.prior.dataset import SCMPrior
from tabicl.prior.source_ids import (
    MISSINGNESS_SOURCE_EPIT,
    MISSINGNESS_SOURCE_GENERIC,
    MISSINGNESS_SOURCE_SOCCOL,
)
from tabicl.prior.epit_feature_profile import EPIT_FEATURE_PROFILE
from tabicl.prior.soccol_schema import SOCCOL_FEATURE_PROFILE
from tabicl.train.missingness import MissingnessAugmenter
from tabicl.train.train_config import build_parser


def _write_profile(path: Path, pattern: str, *, count: int = 1) -> Path:
    payload = {
        "schema_version": "test_v1",
        "source": "context only",
        "feature_columns_in_order": ["A", "B", "C", "D"],
        "composition_feature_count": 2,
        "patterns_by_source": {
            "EPIT": {
                "row_count": count,
                "patterns": [{"missing_mask": pattern, "count": count}],
            },
            "Soccol": {
                "row_count": count,
                "patterns": [{"missing_mask": pattern, "count": count}],
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_masks_observed_zero_and_noncomposition_from_joint_pattern(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path / "profile.json", "1001")
    augmenter = MissingnessAugmenter(
        profile_path,
        augmentation_probability=1.0,
        min_observed_composition=1,
    )
    X = torch.tensor([[[0.0, 2.0, 3.0, 4.0]]])
    y = torch.tensor([[7.0]])

    augmented, stats = augmenter.augment(
        X,
        torch.tensor([4]),
        generator=torch.Generator().manual_seed(3),
    )

    assert torch.isnan(augmented[0, 0, 0])
    assert augmented[0, 0, 1:3].tolist() == [2.0, 3.0]
    assert torch.isnan(augmented[0, 0, 3])
    assert y.tolist() == [[7.0]]
    assert stats["missing_mask_fraction"] == pytest.approx(0.5)


def test_retains_minimum_observed_composition_and_never_fills_original_nan(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path / "profile.json", "1100")
    augmenter = MissingnessAugmenter(
        profile_path,
        augmentation_probability=1.0,
        min_observed_composition=2,
    )
    X = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [float("nan"), 2.0, 3.0, 4.0],
            ]
        ]
    )

    augmented, _ = augmenter.augment(
        X,
        torch.tensor([4]),
        generator=torch.Generator().manual_seed(5),
    )

    assert torch.isfinite(augmented[0, 0, :2]).sum().item() == 2
    assert torch.isnan(augmented[0, 1, 0])
    assert torch.isfinite(augmented[0, 1, 1])


def test_does_not_mask_inactive_padding_columns(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path / "profile.json", "1111")
    augmenter = MissingnessAugmenter(
        profile_path,
        augmentation_probability=1.0,
        min_observed_composition=0,
    )
    X = torch.tensor([[[1.0, 2.0, 30.0, 40.0]]])

    augmented, _ = augmenter.augment(X, torch.tensor([2]))

    assert torch.isnan(augmented[0, 0, :2]).all()
    assert augmented[0, 0, 2:].tolist() == [30.0, 40.0]


def test_informed_tables_receive_source_matched_masks(tmp_path: Path) -> None:
    payload = {
        "schema_version": "test_v1",
        "source": "context only",
        "feature_columns_in_order": ["A", "B", "C", "D"],
        "composition_feature_count": 2,
        "patterns_by_source": {
            "EPIT": {"row_count": 1, "patterns": [{"missing_mask": "1000", "count": 1}]},
            "Soccol": {"row_count": 1, "patterns": [{"missing_mask": "0100", "count": 1}]},
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    augmenter = MissingnessAugmenter(
        profile_path,
        augmentation_probability=1.0,
        min_observed_composition=1,
    )
    X = torch.ones(2, 1, 4)

    augmented, _ = augmenter.augment(
        X,
        torch.tensor([4, 4]),
        source_ids=torch.tensor([MISSINGNESS_SOURCE_EPIT, MISSINGNESS_SOURCE_SOCCOL]),
        generator=torch.Generator().manual_seed(9),
    )

    assert torch.isnan(augmented[0, 0, 0])
    assert torch.isfinite(augmented[0, 0, 1])
    assert torch.isfinite(augmented[1, 0, 0])
    assert torch.isnan(augmented[1, 0, 1])


def test_prior_source_ids_distinguish_generic_epit_and_soccol() -> None:
    assert SCMPrior._missingness_source_id({"informed_mode": False}) == MISSINGNESS_SOURCE_GENERIC
    assert SCMPrior._missingness_source_id(
        {"informed_mode": True, "pitting_feature_profile": EPIT_FEATURE_PROFILE}
    ) == MISSINGNESS_SOURCE_EPIT
    assert SCMPrior._missingness_source_id(
        {"informed_mode": True, "pitting_feature_profile": SOCCOL_FEATURE_PROFILE}
    ) == MISSINGNESS_SOURCE_SOCCOL


def test_training_flags_are_opt_in() -> None:
    parser = build_parser()
    defaults = parser.parse_args([])
    enabled = parser.parse_args(
        [
            "--col_missing_value_embedding",
            "true",
            "--missingness_profile_path",
            "profile.json",
        ]
    )

    assert defaults.col_missing_value_embedding is False
    assert defaults.missingness_profile_path is None
    assert defaults.combined_corrosion_prior_config is None
    assert defaults.missingness_query_column_probability == 0.0
    assert defaults.missingness_preserve_observed_composition is False
    assert enabled.col_missing_value_embedding is True
    assert enabled.missingness_profile_path == "profile.json"


def test_query_masking_protects_composition_and_leaves_context_unchanged(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path / "profile.json", "1101")
    baseline = MissingnessAugmenter(
        profile_path, augmentation_probability=1.0, preserve_observed_composition=True
    )
    augmenter = MissingnessAugmenter(
        profile_path,
        augmentation_probability=1.0,
        query_column_probability=1.0,
        preserve_observed_composition=True,
    )
    X = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    X[0, 4, 0] = torch.nan
    original = X.clone()
    expected, _ = baseline.augment(X, torch.tensor([4]), generator=torch.Generator().manual_seed(3))

    augmented, stats = augmenter.augment(
        X, torch.tensor([4]), train_size=3, generator=torch.Generator().manual_seed(3)
    )

    torch.testing.assert_close(X, original, equal_nan=True)
    torch.testing.assert_close(augmented[:, :3], expected[:, :3], equal_nan=True)
    torch.testing.assert_close(augmented[..., :2], original[..., :2], equal_nan=True)
    assert torch.isnan(augmented[:, 3:, 2:]).all()
    assert stats["composition_missing_mask_fraction"] == 0.0
    assert stats["query_missing_mask_fraction"] > stats["context_missing_mask_fraction"]
    assert augmenter.provenance["query_column_probability"] == 1.0
    assert augmenter.provenance["preserve_observed_composition"] is True


def test_query_column_masks_are_shared_across_rows_and_reproducible(tmp_path: Path) -> None:
    augmenter = MissingnessAugmenter(
        _write_profile(tmp_path / "profile.json", "0000"),
        query_column_probability=0.8,
        preserve_observed_composition=True,
    )
    X = torch.ones(128, 8, 4)
    d = torch.full((128,), 4)
    first, _ = augmenter.augment(X, d, train_size=3, generator=torch.Generator().manual_seed(42))
    second, _ = augmenter.augment(X, d, train_size=3, generator=torch.Generator().manual_seed(42))

    torch.testing.assert_close(first, second, equal_nan=True)
    assert torch.isfinite(first[:, :3]).all()
    assert torch.isfinite(first[..., :2]).all()
    query_mask = torch.isnan(first[:, 3:, 2:])
    assert torch.equal(query_mask, query_mask[:, :1].expand_as(query_mask))
    assert 0.7 < query_mask.float().mean().item() < 0.9


def test_query_masking_respects_inactive_padding(tmp_path: Path) -> None:
    augmenter = MissingnessAugmenter(
        _write_profile(tmp_path / "profile.json", "0000"), query_column_probability=1.0
    )
    X = torch.ones(2, 5, 4)
    augmented, _ = augmenter.augment(X, torch.tensor([2, 3]), train_size=2)

    torch.testing.assert_close(augmented[0], X[0])
    assert torch.isnan(augmented[1, 2:, 2]).all()
    assert torch.isfinite(augmented[1, :, 3]).all()
    assert torch.isfinite(augmented[:, :2]).all()


def test_disabled_query_masking_preserves_output_and_rng_state(tmp_path: Path) -> None:
    augmenter = MissingnessAugmenter(
        _write_profile(tmp_path / "profile.json", "1001"), min_observed_composition=1
    )
    X = torch.ones(3, 5, 4)
    d = torch.full((3,), 4)
    legacy_rng = torch.Generator().manual_seed(42)
    split_rng = torch.Generator().manual_seed(42)
    legacy, _ = augmenter.augment(X, d, generator=legacy_rng)
    with_split, _ = augmenter.augment(X, d, train_size=2, generator=split_rng)

    torch.testing.assert_close(legacy, with_split, equal_nan=True)
    assert torch.equal(legacy_rng.get_state(), split_rng.get_state())


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_query_mask_probability(tmp_path: Path, probability: float) -> None:
    with pytest.raises(ValueError, match="query_column_probability"):
        MissingnessAugmenter(
            _write_profile(tmp_path / "profile.json", "0000"), query_column_probability=probability
        )


@pytest.mark.parametrize("train_size", [None, 0, 5, -1, 1.5, True])
def test_query_masking_requires_valid_context_boundary(tmp_path: Path, train_size) -> None:
    augmenter = MissingnessAugmenter(
        _write_profile(tmp_path / "profile.json", "0000"), query_column_probability=0.8
    )
    with pytest.raises(ValueError, match="train_size"):
        augmenter.augment(torch.ones(1, 5, 4), torch.tensor([4]), train_size=train_size)


def test_trainer_configures_query_masking_and_requires_missing_embedding(tmp_path: Path) -> None:
    from tabicl.train.run import Trainer

    profile_path = _write_profile(tmp_path / "profile.json", "0000")
    config = build_parser().parse_args(
        [
            "--col_missing_value_embedding", "true",
            "--missingness_profile_path", str(profile_path),
            "--max_features", "4",
            "--missingness_query_column_probability", "0.8",
            "--missingness_preserve_observed_composition", "true",
        ]
    )
    trainer = Trainer.__new__(Trainer)
    trainer.config = config
    trainer.configure_missingness()
    assert trainer.missingness_augmenter.query_column_probability == 0.8
    assert trainer.missingness_augmenter.preserve_observed_composition is True

    config.col_missing_value_embedding = False
    with pytest.raises(ValueError, match="require --col_missing_value_embedding"):
        trainer.configure_missingness()


@pytest.mark.parametrize(
    ("source", "expected_counts"),
    [(None, [1, 1, 2]), ("EPIT", [2, 2, 0]), ("Soccol", [0, 0, 4])],
)
def test_combined_training_step_with_query_masking(tmp_path: Path, source, expected_counts) -> None:
    from scripts.combined_corrosion.train_specialist_shared_v2 import build_specialist_config
    from tabicl.train.run import Trainer

    processed = (
        Path(__file__).resolve().parents[1]
        / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed"
    )
    prior_config = processed / "combined_prior_config.json"
    if source is not None:
        specialist_config = build_specialist_config(prior_config, source)
        prior_config = tmp_path / "prior_config.json"
        prior_config.write_text(json.dumps(specialist_config), encoding="utf-8")
    config = build_parser().parse_args(
        [
            "--device", "cpu", "--prior_device", "cpu", "--amp", "false",
            "--python_seed", "42", "--prior_n_jobs", "1", "--dataloader_num_workers", "0",
            "--batch_size", "4", "--micro_batch_size", "2", "--max_seq_len", "32",
            "--min_features", "50", "--max_features", "50", "--max_classes", "0",
            "--num_quantiles", "9", "--embed_dim", "16",
            "--col_num_blocks", "1", "--col_nhead", "2", "--col_num_inds", "4",
            "--row_num_blocks", "1", "--row_nhead", "2", "--row_num_cls", "2",
            "--icl_num_blocks", "1", "--icl_nhead", "2",
            "--warmup_proportion", "0", "--max_steps", "2",
            "--checkpoint_dir", str(tmp_path / "checkpoints"),
            "--combined_corrosion_prior_config", str(prior_config),
            "--col_missing_value_embedding", "true",
            "--missingness_profile_path", str(processed / "context_missingness_profile.json"),
            "--missingness_query_column_probability", "0.8",
            "--missingness_preserve_observed_composition", "true",
        ]
    )
    trainer = Trainer(config)
    batch = trainer.dataloader.dataset.get_batch()
    assert batch[0].shape == (4, 32, 50)
    assert batch[5].bincount(minlength=3).tolist() == expected_counts
    original_X, original_y = batch[0].clone(), batch[1].clone()
    boundary = int(batch[4][0])
    composition_count = trainer.missingness_augmenter.composition_feature_count
    model_inputs = []

    def capture_input(module, args):
        model_inputs.append((args[0].clone(), args[1].clone()))

    handle = trainer.model.register_forward_pre_hook(capture_input)
    try:
        results = trainer.run_batch(batch)
    finally:
        handle.remove()

    seen_X = torch.cat([X for X, _ in model_inputs])
    seen_y = torch.cat([y for _, y in model_inputs])
    torch.testing.assert_close(batch[0], original_X, equal_nan=True)
    torch.testing.assert_close(batch[1], original_y)
    torch.testing.assert_close(seen_y, original_y[:, :boundary])
    torch.testing.assert_close(
        seen_X[..., :composition_count], original_X[..., :composition_count], equal_nan=True
    )
    assert results["query_missing_mask_fraction"] > results["context_missing_mask_fraction"]
    assert results["composition_missing_mask_fraction"] == 0.0
    assert all(math.isfinite(value) for value in results.values())
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())

    trainer.curr_step = 1
    trainer.save_checkpoint("step-1.ckpt")
    checkpoint = torch.load(tmp_path / "checkpoints/step-1.ckpt", weights_only=False)
    assert checkpoint["curr_step"] == 1
    assert checkpoint["missingness_augmentation"]["query_column_probability"] == 0.8
    assert checkpoint["missingness_augmentation"]["preserve_observed_composition"] is True
    assert checkpoint["combined_corrosion_prior"]["regime_probabilities"] == {
        name: count / 4
        for name, count in zip(("generic", "EPIT", "Soccol"), expected_counts)
    }
