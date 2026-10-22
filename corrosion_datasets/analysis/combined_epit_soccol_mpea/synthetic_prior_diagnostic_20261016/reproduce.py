"""Compare saved models on identical fresh synthetic tasks; never reads MPEA labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from tabicl.model import TabICL
from tabicl.prior.combined_corrosion import CombinedCorrosionPriorDataset
from tabicl.train.missingness import MissingnessAugmenter

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed"
RUNS = ROOT / "checkpoints/combined_epit_soccol_mpea"
MODELS = {
    "epit_shared_1000": RUNS / "epit_shared_v2_seed42/step-1000.ckpt",
    "soccol_shared_1000": RUNS / "soccol_shared_v2_seed42/step-1000.ckpt",
    "combined_v2_1000": RUNS / "final_v2_seed42/step-1000.ckpt",
    "combined_v2_3000": RUNS / "final_v2_seed42/step-3000.ckpt",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-per-source", type=int, default=16)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1700629)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    boundary = args.rows // 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = CombinedCorrosionPriorDataset(
        config_path=DATA / "combined_prior_config.json",
        batch_size=args.tasks_per_source, batch_size_per_gp=1,
        max_classes=0, min_seq_len=None, max_seq_len=args.rows,
        min_train_size=boundary, max_train_size=boundary + 1,
        n_jobs=1, device="cpu", seed_offset=args.seed,
    )
    augmenter = MissingnessAugmenter(
        DATA / "context_missingness_profile.json", query_column_probability=0.8,
        preserve_observed_composition=True,
    )
    tasks = {}
    dataset_hash = hashlib.sha256()
    for source_id, source in enumerate(generator.REGIME_ORDER):
        native, y, _, _, sizes = generator.priors[source].get_batch(args.tasks_per_source)
        assert sizes.eq(boundary).all()
        X = generator._map_regime(source, native)
        d = torch.full((len(X),), 50, dtype=torch.long)
        X, stats = augmenter.augment(
            X, d, train_size=boundary,
            source_ids=torch.full((len(X),), source_id, dtype=torch.long),
            generator=torch.Generator().manual_seed(args.seed + 100 + source_id),
        )
        assert torch.isfinite(y).all()
        for tensor in (X, y):
            dataset_hash.update(tensor.numpy().tobytes())
        tasks[source] = (X, y, d)
        print(f"Generated {source}: {len(X)} tasks, {args.rows} rows, masks {stats}", flush=True)

    rows = []
    predictions = {}
    for name, path in MODELS.items():
        started = time.monotonic()
        ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        model = TabICL(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        assert model.col_embedder.missing_value_embedding
        assert ckpt["config"]["dropout"] == 0.0
        del ckpt
        # Exactly the raw training path; dropout is zero and gradients are disabled.
        model.train()
        with torch.no_grad():
            for source, (X, y, d) in tasks.items():
                medians = []
                for start in range(0, len(X), args.batch_size):
                    stop = min(len(X), start + args.batch_size)
                    target = y[start:stop, boundary:]
                    output = model(X[start:stop], y[start:stop, :boundary], d[start:stop])
                    dist = model.quantile_dist(output)
                    median = dist.icdf(torch.tensor(0.5))
                    crps = dist.crps(target).mean(dim=1)
                    assert torch.isfinite(median).all() and torch.isfinite(crps).all()
                    medians.append(median.numpy())
                    for offset in range(stop - start):
                        i = start + offset
                        pred = median[offset]
                        truth = target[offset]
                        context = y[i, :boundary]
                        baseline_crps = (
                            (truth[:, None] - context[None, :]).abs().mean()
                            - 0.5 * (context[:, None] - context[None, :]).abs().mean()
                        )
                        rows.append({
                            "model": name, "source": source, "task": i,
                            "spearman": float(spearmanr(pred.numpy(), truth.numpy()).statistic),
                            "crps": float(crps[offset]),
                            "rmse": float(((pred - truth) ** 2).mean().sqrt()),
                            "context_only_crps": float(baseline_crps),
                            "context_median_rmse": float(((context.median() - truth) ** 2).mean().sqrt()),
                        })
                predictions[f"{name}__{source}"] = np.concatenate(medians)
                current = pd.DataFrame(rows)
                summary = current[(current.model == name) & (current.source == source)]
                print(name, source, summary[["spearman", "crps", "rmse"]].mean().round(5).to_dict(), flush=True)
                current.to_csv(args.output_dir / "rows.csv", index=False)
        del model
        print(f"Completed {name} in {time.monotonic() - started:.1f}s", flush=True)

    frame = pd.DataFrame(rows)
    summary = frame.groupby(["model", "source"])[
        ["spearman", "crps", "rmse", "context_only_crps", "context_median_rmse"]
    ].mean().reset_index()
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    for source, (_, y, _) in tasks.items():
        predictions[f"target__{source}"] = y[:, boundary:].numpy()
    np.savez_compressed(args.output_dir / "predictions.npz", **predictions)
    metadata = {
        "diagnostic_only": True,
        "real_evaluation_rows_or_targets_used": False,
        "seed": args.seed, "tasks_per_source": args.tasks_per_source,
        "rows_per_task": args.rows, "context_rows_per_task": boundary,
        "independent_hyperparameter_draws_per_table": True,
        "data_sha256": dataset_hash.hexdigest(),
        "model_checkpoints": {name: str(path) for name, path in MODELS.items()},
        "missingness": augmenter.provenance,
        "limitation": "Synthetic diagnostic at a fixed context length, not a replacement for frozen MPEA evaluation.",
        "summary": summary.to_dict(orient="records"),
    }
    (args.output_dir / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
