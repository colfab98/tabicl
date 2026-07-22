#!/usr/bin/env python
"""Validate the standalone empirical EPIT composition profile and sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tabicl.prior.epit_composition_profile import (
    EPIT_COMPOSITION_DEFAULT_PERTURB_STRENGTH,
    EPIT_COMPOSITION_FAMILY_PROBS,
    EPIT_COMPOSITION_PROFILE,
    load_epit_composition_profile,
    sample_epit_compositions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=EPIT_COMPOSITION_PROFILE)
    parser.add_argument("--n-samples", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--family-probs",
        type=float,
        nargs=5,
        default=list(EPIT_COMPOSITION_FAMILY_PROBS),
        metavar=("FE", "AL", "HEA", "NICRMO", "OTHER"),
    )
    parser.add_argument(
        "--perturb-strength",
        type=float,
        default=EPIT_COMPOSITION_DEFAULT_PERTURB_STRENGTH,
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def summarize(args: argparse.Namespace) -> dict[str, object]:
    profile = load_epit_composition_profile(args.profile)
    batch = sample_epit_compositions(
        args.n_samples,
        profile=profile,
        family_probabilities=args.family_probs,
        perturb_strength=args.perturb_strength,
        random_state=args.random_state,
    )
    counts = np.bincount(batch.family_indices, minlength=len(profile.families))
    full_active = np.sum(batch.full_compositions > 0.0, axis=1)
    observed_active = np.sum(batch.observed_compositions > 0.0, axis=1)
    observed_sums = batch.observed_compositions.sum(axis=1)

    return {
        "profile": profile.name,
        "n_templates": profile.n_templates,
        "n_samples": int(args.n_samples),
        "random_state": int(args.random_state),
        "perturb_strength": float(args.perturb_strength),
        "family_counts": {
            family: int(count) for family, count in zip(profile.families, counts, strict=True)
        },
        "family_frequencies": {
            family: float(count / args.n_samples)
            for family, count in zip(profile.families, counts, strict=True)
        },
        "full_sum": {
            "minimum": float(batch.full_compositions.sum(axis=1).min()),
            "maximum": float(batch.full_compositions.sum(axis=1).max()),
        },
        "observed_sum": {
            "mean": float(observed_sums.mean()),
            "minimum": float(observed_sums.min()),
            "median": float(np.median(observed_sums)),
            "maximum": float(observed_sums.max()),
        },
        "active_elements": {
            "full_median": float(np.median(full_active)),
            "full_maximum": int(full_active.max()),
            "observed_median": float(np.median(observed_active)),
            "observed_maximum": int(observed_active.max()),
        },
        "observed_element_positive_frequencies": {
            element: float(np.mean(batch.observed_compositions[:, index] > 0.0))
            for index, element in enumerate(profile.observed_elements)
        },
    }


def main() -> None:
    args = parse_args()
    summary = summarize(args)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
