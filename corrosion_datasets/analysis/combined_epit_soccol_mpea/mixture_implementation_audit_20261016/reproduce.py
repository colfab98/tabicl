"""CPU checks of source preservation and batch independence in mixed-prior training."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

from tabicl.model import TabICL
from tabicl.prior.combined_corrosion import (
    COMBINED_FEATURE_COLUMNS,
    EPIT_FEATURE_COLUMNS,
    CombinedCorrosionPriorDataset,
)
from tabicl.prior.soccol_schema import SOCCOL_FEATURE_COLUMNS
from tabicl.train.missingness import MissingnessAugmenter

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed"
RUNS = ROOT / "checkpoints/combined_epit_soccol_mpea"
NAMES = {0: "generic", 1: "EPIT", 2: "Soccol"}


def make_prior(config_path, batch_size=4, seq_len=32):
    return CombinedCorrosionPriorDataset(
        config_path=config_path, batch_size=batch_size, batch_size_per_gp=4,
        max_classes=0, min_seq_len=None, max_seq_len=seq_len,
        min_train_size=0.4, max_train_size=0.6, n_jobs=1, device="cpu",
    )


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state()


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def main():
    torch.set_num_threads(2)
    random.seed(716)
    np.random.seed(716)
    torch.manual_seed(716)
    prior = make_prior(DATA / "combined_prior_config.json")
    captured = {}
    for name, source in prior.priors.items():
        original = source.get_batch

        def capture(count, name=name, source=source, original=original):
            before = rng_state()
            coefficient_state = (
                copy.deepcopy(source._coefficient_rng.bit_generator.state)
                if source._coefficient_rng is not None else None
            )
            batch = original(count)
            captured[name] = (before, coefficient_state, tuple(t.clone() for t in batch))
            return batch

        source.get_batch = capture

    batch = prior.get_batch()
    X, y, d, _, sizes, source_ids = batch
    for index, source_id in enumerate(source_ids.tolist()):
        name = NAMES[source_id]
        raw_X, raw_y, *_ = captured[name][2]
        matches = [j for j in range(len(raw_y)) if torch.equal(y[index], raw_y[j])]
        assert len(matches) == 1
        native = raw_X[matches[0]]
        columns = {
            "generic": COMBINED_FEATURE_COLUMNS, "EPIT": EPIT_FEATURE_COLUMNS,
            "Soccol": SOCCOL_FEATURE_COLUMNS,
        }[name]
        for destination, column in enumerate(COMBINED_FEATURE_COLUMNS):
            if column in columns:
                torch.testing.assert_close(
                    X[index, :, destination], native[:, columns.index(column)],
                    rtol=0, atol=0, equal_nan=True,
                )
            else:
                assert torch.isnan(X[index, :, destination]).all()

    replay_checks = {}
    for source_name, run in (("EPIT", "epit_shared_v2_seed42"), ("Soccol", "soccol_shared_v2_seed42")):
        standalone = make_prior(RUNS / run / "prior_config.json")
        before, coefficient_state, expected = captured[source_name]
        restore_rng(before)
        standalone.priors[source_name]._coefficient_rng.bit_generator.state = coefficient_state
        actual = standalone.priors[source_name].get_batch(len(expected[0]))
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
        replay_checks[source_name] = "exactly identical features, targets, widths, and boundaries"

    augmenter = MissingnessAugmenter(
        DATA / "context_missingness_profile.json", query_column_probability=0.8,
        preserve_observed_composition=True,
    )
    boundary = int(sizes[0])
    X, _ = augmenter.augment(
        X, d, train_size=boundary, source_ids=source_ids,
        generator=torch.Generator().manual_seed(917),
    )
    ckpt = torch.load(RUNS / "final_v2_seed42/step-1000.ckpt", map_location="cpu", mmap=True, weights_only=False)
    model = TabICL(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.train()
    del ckpt
    model.zero_grad(set_to_none=True)
    mixed_predictions = model(X, y[:, :boundary], d)
    mixed_loss = model.quantile_dist(mixed_predictions).crps(y[:, boundary:]).mean()
    mixed_loss.backward()
    mixed_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    mixed_predictions = mixed_predictions.detach()
    model.zero_grad(set_to_none=True)
    separate_predictions = []
    separate_loss = 0.0
    for index in range(len(X)):
        prediction = model(X[index:index+1], y[index:index+1, :boundary], d[index:index+1])
        loss = model.quantile_dist(prediction).crps(y[index:index+1, boundary:]).mean() / len(X)
        loss.backward()
        separate_loss += float(loss.detach())
        separate_predictions.append(prediction.detach())
    separate_predictions = torch.cat(separate_predictions)
    difference = (mixed_predictions - separate_predictions).abs()
    gradient_error_squared = gradient_squared = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            assert name not in mixed_gradients
            continue
        gradient_error_squared += float(((parameter.grad - mixed_gradients[name]).double() ** 2).sum())
        gradient_squared += float((mixed_gradients[name].double() ** 2).sum())
    relative_gradient_error = (gradient_error_squared / max(gradient_squared, 1e-30)) ** 0.5
    torch.testing.assert_close(mixed_predictions, separate_predictions, rtol=1e-4, atol=1e-5)
    assert relative_gradient_error < 1e-4
    result = {
        "scope": "CPU float32, actual combined V2 step-1000 weights, four fresh 32-row synthetic tables; no real query rows or targets",
        "feature_target_source_pairing": "exact after mapping and shuffle",
        "source_generator_replay": replay_checks,
        "source_counts": source_ids.bincount(minlength=3).tolist(),
        "mixed_batch_vs_individual_tables": {
            "max_prediction_difference": float(difference.max()),
            "mixed_loss": float(mixed_loss.detach()),
            "individual_loss_sum": separate_loss,
            "relative_gradient_l2_difference": relative_gradient_error,
        },
        "passed": True,
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
