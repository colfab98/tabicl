"""Deterministic local improvement for the exact-size Optuna folds."""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np

from scripts.epit_pipeline.split_data import (
    INNER_FOLD_COUNT,
    CompositionGrouping,
    EpitDataset,
    OptimizerRecord,
    SplitAssignment,
    build_balance_features,
    validate_split_assignment,
)


DEFAULT_REFINEMENT_SEED = 12_345
DEFAULT_MAX_PROPOSALS = 300_000
DEFAULT_STALE_PROPOSALS = 50_000


def _fold_bundles(
    fold_by_group: np.ndarray,
    group_sizes: np.ndarray,
    fold: int,
) -> dict[int, list[tuple[int, ...]]]:
    """Return one- and two-group bundles indexed by their total row count."""
    groups = list(np.flatnonzero(fold_by_group == fold))
    bundles: dict[int, list[tuple[int, ...]]] = {}
    for group in groups:
        bundles.setdefault(int(group_sizes[group]), []).append((int(group),))
    for first_index, first in enumerate(groups):
        for second in groups[first_index + 1 :]:
            size = int(group_sizes[first] + group_sizes[second])
            bundles.setdefault(size, []).append((int(first), int(second)))
    return bundles


def refine_inner_folds(
    dataset: EpitDataset,
    grouping: CompositionGrouping,
    assignment: SplitAssignment,
    labels: dict[str, np.ndarray],
    *,
    random_seed: int = DEFAULT_REFINEMENT_SEED,
    max_proposals: int = DEFAULT_MAX_PROPOSALS,
    stale_proposals: int = DEFAULT_STALE_PROPOSALS,
) -> SplitAssignment:
    """Improve balance without changing any fold's exact number of rows.

    Swaps contain one or two complete composition groups on each side. Only
    equal-row-count bundles are exchanged, so composition separation and the
    122/122/122/121/121 validation sizes remain fixed.
    """
    features = build_balance_features(labels, grouping)
    feature_counts = np.asarray([feature.group_counts for feature in features])
    feature_weights = np.asarray([feature.weight for feature in features])
    group_sizes = grouping.sizes
    fold_by_group = assignment.validation_fold_by_group.copy()
    development_groups = np.flatnonzero(fold_by_group >= 0)
    fold_rows = np.asarray(
        [group_sizes[fold_by_group == fold].sum() for fold in range(INNER_FOLD_COUNT)]
    )
    development_rows = int(fold_rows.sum())
    desired = np.asarray(
        [
            feature_counts[:, development_groups].sum(axis=1)
            * fold_row_count
            / development_rows
            for fold_row_count in fold_rows
        ]
    )
    scaled_weights = feature_weights / int(fold_rows.max())
    fold_counts = np.asarray(
        [
            feature_counts[:, fold_by_group == fold].sum(axis=1)
            for fold in range(INNER_FOLD_COUNT)
        ]
    )

    def fold_score(fold: int, counts: np.ndarray) -> float:
        return float(np.sum(np.abs(counts - desired[fold]) * scaled_weights))

    scores = np.asarray(
        [fold_score(fold, fold_counts[fold]) for fold in range(INNER_FOLD_COUNT)]
    )
    bundles = [
        _fold_bundles(fold_by_group, group_sizes, fold)
        for fold in range(INNER_FOLD_COUNT)
    ]
    rng = random.Random(random_seed)
    last_improvement = 0
    accepted = 0
    proposals_run = 0

    for proposal in range(max_proposals):
        proposals_run = proposal + 1
        first_fold, second_fold = rng.sample(range(INNER_FOLD_COUNT), 2)
        common_sizes = sorted(
            set(bundles[first_fold]).intersection(bundles[second_fold])
        )
        if not common_sizes:
            continue
        bundle_size = rng.choice(common_sizes)
        first_bundle = rng.choice(bundles[first_fold][bundle_size])
        second_bundle = rng.choice(bundles[second_fold][bundle_size])
        first_indices = list(first_bundle)
        second_indices = list(second_bundle)
        new_first_counts = (
            fold_counts[first_fold]
            - feature_counts[:, first_indices].sum(axis=1)
            + feature_counts[:, second_indices].sum(axis=1)
        )
        new_second_counts = (
            fold_counts[second_fold]
            - feature_counts[:, second_indices].sum(axis=1)
            + feature_counts[:, first_indices].sum(axis=1)
        )
        new_first_score = fold_score(first_fold, new_first_counts)
        new_second_score = fold_score(second_fold, new_second_counts)
        change = (
            new_first_score
            + new_second_score
            - scores[first_fold]
            - scores[second_fold]
        )
        if change < -1e-12:
            fold_by_group[first_indices] = second_fold
            fold_by_group[second_indices] = first_fold
            fold_counts[first_fold] = new_first_counts
            fold_counts[second_fold] = new_second_counts
            scores[first_fold] = new_first_score
            scores[second_fold] = new_second_score
            bundles[first_fold] = _fold_bundles(
                fold_by_group, group_sizes, first_fold
            )
            bundles[second_fold] = _fold_bundles(
                fold_by_group, group_sizes, second_fold
            )
            last_improvement = proposal
            accepted += 1
        if proposal - last_improvement >= stale_proposals:
            break

    refined_record = OptimizerRecord(
        message=(
            f"{assignment.inner_optimizer.message}; deterministic bundle refinement "
            f"accepted {accepted} of {proposals_run} proposals"
        ),
        objective=float(scores.sum()),
        mip_gap=None,
        node_count=assignment.inner_optimizer.node_count,
        time_limit_seconds=assignment.inner_optimizer.time_limit_seconds,
    )
    refined = replace(
        assignment,
        validation_fold_by_group=fold_by_group,
        inner_optimizer=refined_record,
    )
    validate_split_assignment(dataset, grouping, refined)
    return refined
