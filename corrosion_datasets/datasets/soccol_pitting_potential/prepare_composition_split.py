#!/usr/bin/env python3
"""Build and freeze the Soccol composition-separated regression split.

This adapts the established EPIT split methodology:
- zero-filled composition values rounded to 0.01 wt.% for grouping only;
- L1 links at <= 1.0 wt.% and connected-component composition groups;
- exact 80/20 development/final-test row counts;
- five exact-size composition-grouped development folds;
- balanced target, material, isolation, environment, and source blocks.

The prior test-method balance block is omitted because Soccol has no comparable
field, by explicit user decision. The input regression table is not modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUT_CSV = HERE / "processed" / "soccol_regression_event1.csv"
DEFAULT_OUTPUT_DIR = HERE / "processed" / "splits_v1"
MANIFEST_NAME = "split_manifest.json"
ASSIGNMENTS_NAME = "split_assignments.csv"
REPORT_NAME = "split_report.html"
LOCK_NAME = "split_lock.json"

COMPOSITION_COLUMNS = ["Fe", "C", "N", "Si", "P", "S", "Ti", "V", "Cr", "Mn", "Ni", "Nb", "Mo"]
EXPECTED_ROWS = 4027
FINAL_TEST_ROWS = round(EXPECTED_ROWS * 0.20)
INNER_FOLD_COUNT = 5
ROUND_DECIMALS = 2
DISTANCE_THRESHOLD = 1.0
REFINEMENT_SEED = 12_345
MAX_REFINEMENT_PROPOSALS = 300_000
STALE_REFINEMENT_PROPOSALS = 50_000

BALANCE_BLOCK_WEIGHTS = {
    "target_decile": 3.0,
    "material_family": 2.0,
    "composition_isolation": 2.0,
    "temperature": 1.0,
    "chloride": 1.0,
    "pH": 1.0,
    "source": 0.5,
}


@dataclass(frozen=True)
class CompositionGrouping:
    group_ids: list[str]
    group_rows: list[list[int]]
    row_group_index: np.ndarray
    centroids: np.ndarray
    diameters: np.ndarray
    isolation_distances: np.ndarray
    rounded_key_count: int

    @property
    def n_groups(self) -> int:
        return len(self.group_rows)

    @property
    def sizes(self) -> np.ndarray:
        return np.asarray([len(rows) for rows in self.group_rows], dtype=int)


@dataclass(frozen=True)
class BalanceFeature:
    block: str
    category: str
    weight: float
    group_counts: np.ndarray

    @property
    def total(self) -> float:
        return float(self.group_counts.sum())


@dataclass(frozen=True)
class OptimizerRecord:
    message: str
    objective: float
    mip_gap: float | None
    node_count: int | None
    time_limit_seconds: float


@dataclass(frozen=True)
class SplitAssignment:
    outer_split_by_group: np.ndarray
    validation_fold_by_group: np.ndarray
    outer_optimizer: OptimizerRecord
    inner_optimizer: OptimizerRecord

    @property
    def final_test_groups(self) -> np.ndarray:
        return np.flatnonzero(self.outer_split_by_group == "final_test")

    @property
    def development_groups(self) -> np.ndarray:
        return np.flatnonzero(self.outer_split_by_group == "development")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outer-time-limit", type=float, default=20.0)
    parser.add_argument("--inner-time-limit", type=float, default=60.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric(row: dict[str, str], column: str) -> float:
    value = row.get(column, "").strip()
    if not value:
        return math.nan
    try:
        result = float(value)
    except ValueError:
        return math.nan
    return result if math.isfinite(result) else math.nan


def load_rows() -> list[dict[str, str]]:
    with INPUT_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} regression rows, found {len(rows)}")
    if any(row.get("regression_eligible") != "1" for row in rows):
        raise RuntimeError("Input contains a row outside the event=1 regression task")
    if any(not math.isfinite(numeric(row, "E_pit")) for row in rows):
        raise RuntimeError("Regression target contains a nonnumeric value")
    for column in COMPOSITION_COLUMNS:
        if any(not math.isfinite(numeric(row, column)) for row in rows):
            raise RuntimeError(f"Processed composition column {column!r} contains missing values")
    return rows


def cluster_vectors(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 1:
        return np.zeros(1, dtype=int)
    parent = np.arange(len(vectors), dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    distances = squareform(pdist(vectors, metric="cityblock"))
    for left, right in np.argwhere(np.triu(distances <= DISTANCE_THRESHOLD + 1e-12, k=1)):
        union(int(left), int(right))
    remap: dict[int, int] = {}
    return np.asarray([remap.setdefault(find(index), len(remap)) for index in range(len(vectors))])


def build_composition_groups(rows: list[dict[str, str]]) -> CompositionGrouping:
    key_rows: dict[tuple[float, ...], list[int]] = {}
    for row_index, row in enumerate(rows):
        key = tuple(round(numeric(row, column), ROUND_DECIMALS) for column in COMPOSITION_COLUMNS)
        key_rows.setdefault(key, []).append(row_index)

    keys = list(key_rows)
    vectors = np.asarray(keys, dtype=float)
    key_clusters = cluster_vectors(vectors)
    raw_group_rows: dict[int, list[int]] = defaultdict(list)
    raw_group_vectors: dict[int, list[np.ndarray]] = defaultdict(list)
    for cluster_index, key, vector in zip(key_clusters, keys, vectors, strict=True):
        raw_group_rows[int(cluster_index)].extend(key_rows[key])
        raw_group_vectors[int(cluster_index)].append(vector)

    ordered_groups = sorted(raw_group_rows, key=lambda group: min(raw_group_rows[group]))
    group_rows = [sorted(raw_group_rows[group]) for group in ordered_groups]
    group_ids = [f"composition_{index + 1:04d}" for index in range(len(group_rows))]
    row_group_index = np.empty(len(rows), dtype=int)
    centroids: list[np.ndarray] = []
    diameters: list[float] = []
    group_vector_arrays: list[np.ndarray] = []
    for group_index, raw_group in enumerate(ordered_groups):
        row_group_index[group_rows[group_index]] = group_index
        group_vectors = np.asarray(raw_group_vectors[raw_group], dtype=float)
        group_vector_arrays.append(group_vectors)
        centroids.append(group_vectors.mean(axis=0))
        diameters.append(0.0 if len(group_vectors) < 2 else float(pdist(group_vectors, metric="cityblock").max()))

    isolation = np.full(len(group_vector_arrays), np.inf, dtype=float)
    for left in range(len(group_vector_arrays)):
        for right in range(left + 1, len(group_vector_arrays)):
            distances = np.abs(
                group_vector_arrays[left][:, None, :] - group_vector_arrays[right][None, :, :]
            ).sum(axis=2)
            minimum = float(distances.min())
            isolation[left] = min(isolation[left], minimum)
            isolation[right] = min(isolation[right], minimum)
    isolation[~np.isfinite(isolation)] = 0.0

    grouping = CompositionGrouping(
        group_ids=group_ids,
        group_rows=group_rows,
        row_group_index=row_group_index,
        centroids=np.asarray(centroids, dtype=float),
        diameters=np.asarray(diameters, dtype=float),
        isolation_distances=isolation,
        rounded_key_count=len(keys),
    )
    observed = sorted(index for group in group_rows for index in group)
    if observed != list(range(len(rows))):
        raise RuntimeError("Composition groups do not cover each row exactly once")
    key_distances = squareform(pdist(vectors, metric="cityblock"))
    key_group = np.asarray([row_group_index[key_rows[key][0]] for key in keys])
    crossing = np.argwhere(
        np.triu((key_distances <= DISTANCE_THRESHOLD + 1e-12) & (key_group[:, None] != key_group[None, :]), k=1)
    )
    if len(crossing):
        raise RuntimeError("A close composition pair crosses grouping boundaries")
    return grouping


def quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.arange(1, n_bins) / n_bins))
    return np.digitize(values, edges, right=True).astype(int)


def numeric_bin(value: float, edges: list[float], *, separate_zero: bool = False) -> str:
    if not math.isfinite(value):
        return "missing"
    if separate_zero and value == 0.0:
        return "zero"
    return f"bin_{int(np.digitize([value], edges, right=True)[0])}"


def build_balance_labels(rows: list[dict[str, str]], grouping: CompositionGrouping) -> dict[str, np.ndarray]:
    targets = np.asarray([numeric(row, "E_pit") for row in rows], dtype=float)
    isolation_by_group = quantile_bins(grouping.isolation_distances, 5)
    sources = np.asarray([row["source"].strip() or "<missing>" for row in rows])
    common_sources = {source for source, _ in Counter(sources).most_common(10)}
    source_labels = np.asarray([source if source in common_sources else "other" for source in sources])
    return {
        "target_decile": quantile_bins(targets, 10).astype(str),
        "material_family": np.asarray([row["row_material_family"].strip() or "<missing>" for row in rows]),
        "composition_isolation": isolation_by_group[grouping.row_group_index].astype(str),
        "temperature": np.asarray([numeric_bin(numeric(row, "CP_temp"), [10.0, 35.0, 70.0]) for row in rows]),
        "chloride": np.asarray([
            numeric_bin(numeric(row, "CP_Cl"), [0.01, 0.1, 0.6, 1.0], separate_zero=True)
            for row in rows
        ]),
        "pH": np.asarray([numeric_bin(numeric(row, "CP_pH"), [3.0, 6.0, 8.0]) for row in rows]),
        "source": source_labels,
    }


def build_balance_features(labels: dict[str, np.ndarray], grouping: CompositionGrouping) -> list[BalanceFeature]:
    features: list[BalanceFeature] = []
    for block, row_labels in labels.items():
        for category in sorted(set(row_labels)):
            group_counts = np.bincount(
                grouping.row_group_index,
                weights=(row_labels == category).astype(float),
                minlength=grouping.n_groups,
            )
            features.append(BalanceFeature(block, str(category), BALANCE_BLOCK_WEIGHTS[block], group_counts))
    return features


def optimizer_record(result: Any, limit: float) -> OptimizerRecord:
    return OptimizerRecord(
        message=str(result.message),
        objective=float(result.fun),
        mip_gap=None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        node_count=None if getattr(result, "mip_node_count", None) is None else int(result.mip_node_count),
        time_limit_seconds=float(limit),
    )


def require_solution(result: Any, name: str) -> np.ndarray:
    if result.x is None or not np.isfinite(result.x).all():
        raise RuntimeError(f"{name} optimizer found no feasible split: {result.message}")
    return np.asarray(result.x, dtype=float)


def make_bounds(binary_count: int, variable_count: int) -> Bounds:
    return Bounds(
        np.zeros(variable_count),
        np.concatenate([np.ones(binary_count), np.full(variable_count - binary_count, np.inf)]),
    )


def optimize_outer(grouping: CompositionGrouping, features: list[BalanceFeature], limit: float) -> tuple[np.ndarray, OptimizerRecord]:
    n_groups = grouping.n_groups
    n_variables = n_groups + len(features)
    sizes = grouping.sizes.astype(float)
    fraction = FINAL_TEST_ROWS / sizes.sum()
    final_group_count = round(n_groups * fraction)
    objective = np.zeros(n_variables)
    objective[:n_groups] = np.arange(n_groups) * 1e-10 / max(n_groups, 1)
    objective[n_groups:] = np.asarray([feature.weight for feature in features]) / FINAL_TEST_ROWS
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for coefficients, target in ((sizes, FINAL_TEST_ROWS), (np.ones(n_groups), final_group_count)):
        constraint = np.zeros(n_variables); constraint[:n_groups] = coefficients
        rows.append(constraint); lower.append(float(target)); upper.append(float(target))
    for feature_index, feature in enumerate(features):
        desired = feature.total * fraction
        for sign in (1.0, -1.0):
            constraint = np.zeros(n_variables)
            constraint[:n_groups] = sign * feature.group_counts
            constraint[n_groups + feature_index] = -1.0
            rows.append(constraint); lower.append(-np.inf); upper.append(sign * desired)
    result = milp(
        objective,
        integrality=np.concatenate([np.ones(n_groups), np.zeros(len(features))]),
        bounds=make_bounds(n_groups, n_variables),
        constraints=LinearConstraint(coo_matrix(np.asarray(rows)), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": limit, "mip_rel_gap": 0.01},
    )
    solution = require_solution(result, "Outer split")
    split = np.full(n_groups, "development", dtype=object)
    split[np.flatnonzero(solution[:n_groups] > 0.5)] = "final_test"
    return split, optimizer_record(result, limit)


def balanced_targets(total: int, parts: int) -> list[int]:
    quotient, remainder = divmod(total, parts)
    return [quotient + (index < remainder) for index in range(parts)]


def optimize_inner(
    grouping: CompositionGrouping,
    features: list[BalanceFeature],
    outer_split: np.ndarray,
    limit: float,
) -> tuple[np.ndarray, OptimizerRecord]:
    development_groups = np.flatnonzero(outer_split == "development")
    development_rows = int(grouping.sizes[development_groups].sum())
    expected = EXPECTED_ROWS - FINAL_TEST_ROWS
    if development_rows != expected:
        raise RuntimeError(f"Expected {expected} development rows, found {development_rows}")
    fold_rows = balanced_targets(development_rows, INNER_FOLD_COUNT)
    fold_groups = balanced_targets(len(development_groups), INNER_FOLD_COUNT)
    n_development_groups = len(development_groups)
    n_binary = INNER_FOLD_COUNT * n_development_groups
    n_slack = INNER_FOLD_COUNT * len(features)
    n_variables = n_binary + n_slack
    development_sizes = grouping.sizes[development_groups].astype(float)
    objective = np.zeros(n_variables)
    objective[:n_binary] = np.arange(n_binary) * 1e-11 / max(n_binary, 1)
    objective[n_binary:] = np.tile([feature.weight for feature in features], INNER_FOLD_COUNT) / max(fold_rows)
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for local_group in range(n_development_groups):
        constraint = np.zeros(n_variables)
        constraint[np.arange(INNER_FOLD_COUNT) * n_development_groups + local_group] = 1.0
        rows.append(constraint); lower.append(1.0); upper.append(1.0)
    for fold in range(INNER_FOLD_COUNT):
        start, stop = fold * n_development_groups, (fold + 1) * n_development_groups
        for coefficients, target in ((development_sizes, fold_rows[fold]), (np.ones(n_development_groups), fold_groups[fold])):
            constraint = np.zeros(n_variables); constraint[start:stop] = coefficients
            rows.append(constraint); lower.append(float(target)); upper.append(float(target))
        fraction = fold_rows[fold] / development_rows
        for feature_index, feature in enumerate(features):
            counts = feature.group_counts[development_groups]
            desired = float(counts.sum()) * fraction
            slack = n_binary + fold * len(features) + feature_index
            for sign in (1.0, -1.0):
                constraint = np.zeros(n_variables)
                constraint[start:stop] = sign * counts
                constraint[slack] = -1.0
                rows.append(constraint); lower.append(-np.inf); upper.append(sign * desired)
    result = milp(
        objective,
        integrality=np.concatenate([np.ones(n_binary), np.zeros(n_slack)]),
        bounds=make_bounds(n_binary, n_variables),
        constraints=LinearConstraint(coo_matrix(np.asarray(rows)), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": limit, "mip_rel_gap": 0.01},
    )
    solution = require_solution(result, "Inner folds")
    validation_fold = np.full(grouping.n_groups, -1, dtype=int)
    for fold in range(INNER_FOLD_COUNT):
        start, stop = fold * n_development_groups, (fold + 1) * n_development_groups
        selected = np.flatnonzero(solution[start:stop] > 0.5)
        validation_fold[development_groups[selected]] = fold
    return validation_fold, optimizer_record(result, limit)


def fold_bundles(folds: np.ndarray, sizes: np.ndarray, fold: int) -> dict[int, list[tuple[int, ...]]]:
    groups = list(np.flatnonzero(folds == fold))
    bundles: dict[int, list[tuple[int, ...]]] = {}
    for group in groups:
        bundles.setdefault(int(sizes[group]), []).append((int(group),))
    for first_index, first in enumerate(groups):
        for second in groups[first_index + 1:]:
            bundles.setdefault(int(sizes[first] + sizes[second]), []).append((int(first), int(second)))
    return bundles


def refine_inner(
    grouping: CompositionGrouping,
    assignment: SplitAssignment,
    features: list[BalanceFeature],
) -> SplitAssignment:
    feature_counts = np.asarray([feature.group_counts for feature in features])
    feature_weights = np.asarray([feature.weight for feature in features])
    sizes = grouping.sizes
    folds = assignment.validation_fold_by_group.copy()
    development_groups = np.flatnonzero(folds >= 0)
    fold_rows = np.asarray([sizes[folds == fold].sum() for fold in range(INNER_FOLD_COUNT)])
    development_rows = int(fold_rows.sum())
    desired = np.asarray([
        feature_counts[:, development_groups].sum(axis=1) * rows / development_rows
        for rows in fold_rows
    ])
    scaled_weights = feature_weights / int(fold_rows.max())
    fold_counts = np.asarray([feature_counts[:, folds == fold].sum(axis=1) for fold in range(INNER_FOLD_COUNT)])

    def score(fold: int, counts: np.ndarray) -> float:
        return float(np.sum(np.abs(counts - desired[fold]) * scaled_weights))

    scores = np.asarray([score(fold, fold_counts[fold]) for fold in range(INNER_FOLD_COUNT)])
    bundles = [fold_bundles(folds, sizes, fold) for fold in range(INNER_FOLD_COUNT)]
    rng = random.Random(REFINEMENT_SEED)
    last_improvement = accepted = proposals_run = 0
    for proposal in range(MAX_REFINEMENT_PROPOSALS):
        proposals_run = proposal + 1
        first_fold, second_fold = rng.sample(range(INNER_FOLD_COUNT), 2)
        common_sizes = sorted(set(bundles[first_fold]).intersection(bundles[second_fold]))
        if not common_sizes:
            continue
        bundle_size = rng.choice(common_sizes)
        first = list(rng.choice(bundles[first_fold][bundle_size]))
        second = list(rng.choice(bundles[second_fold][bundle_size]))
        new_first = fold_counts[first_fold] - feature_counts[:, first].sum(axis=1) + feature_counts[:, second].sum(axis=1)
        new_second = fold_counts[second_fold] - feature_counts[:, second].sum(axis=1) + feature_counts[:, first].sum(axis=1)
        first_score, second_score = score(first_fold, new_first), score(second_fold, new_second)
        change = first_score + second_score - scores[first_fold] - scores[second_fold]
        if change < -1e-12:
            folds[first] = second_fold; folds[second] = first_fold
            fold_counts[first_fold], fold_counts[second_fold] = new_first, new_second
            scores[first_fold], scores[second_fold] = first_score, second_score
            bundles[first_fold] = fold_bundles(folds, sizes, first_fold)
            bundles[second_fold] = fold_bundles(folds, sizes, second_fold)
            last_improvement = proposal; accepted += 1
        if proposal - last_improvement >= STALE_REFINEMENT_PROPOSALS:
            break
    record = OptimizerRecord(
        message=f"{assignment.inner_optimizer.message}; deterministic bundle refinement accepted {accepted} of {proposals_run} proposals",
        objective=float(scores.sum()),
        mip_gap=None,
        node_count=assignment.inner_optimizer.node_count,
        time_limit_seconds=assignment.inner_optimizer.time_limit_seconds,
    )
    return replace(assignment, validation_fold_by_group=folds, inner_optimizer=record)


def validate_assignment(grouping: CompositionGrouping, assignment: SplitAssignment) -> dict[str, Any]:
    sizes = grouping.sizes
    final_groups, development_groups = assignment.final_test_groups, assignment.development_groups
    if int(sizes[final_groups].sum()) != FINAL_TEST_ROWS:
        raise RuntimeError("Final-test row count is not exact")
    if int(sizes[development_groups].sum()) != EXPECTED_ROWS - FINAL_TEST_ROWS:
        raise RuntimeError("Development row count is not exact")
    if np.any(assignment.validation_fold_by_group[final_groups] != -1):
        raise RuntimeError("Final-test groups received development folds")
    observed = [int(sizes[assignment.validation_fold_by_group == fold].sum()) for fold in range(INNER_FOLD_COUNT)]
    expected = balanced_targets(EXPECTED_ROWS - FINAL_TEST_ROWS, INNER_FOLD_COUNT)
    if observed != expected:
        raise RuntimeError(f"Development fold sizes {observed} differ from {expected}")
    return {
        "development_row_count": EXPECTED_ROWS - FINAL_TEST_ROWS,
        "final_test_row_count": FINAL_TEST_ROWS,
        "development_validation_row_counts": {str(index + 1): count for index, count in enumerate(observed)},
        "outer_composition_overlap": 0,
        "inner_composition_overlap_each_fold": [0] * INNER_FOLD_COUNT,
        "outer_close_composition_pair_crossings": 0,
        "inner_close_composition_pair_crossings_each_fold": [0] * INNER_FOLD_COUNT,
        "close_composition_pair_distance_threshold": DISTANCE_THRESHOLD,
    }


def row_assignment(row_index: int, grouping: CompositionGrouping, assignment: SplitAssignment) -> tuple[int, str, str, int | None]:
    group_index = int(grouping.row_group_index[row_index])
    outer = str(assignment.outer_split_by_group[group_index])
    fold = int(assignment.validation_fold_by_group[group_index])
    return group_index, grouping.group_ids[group_index], outer, None if fold < 0 else fold + 1


def source_overlap(rows: list[dict[str, str]], grouping: CompositionGrouping, assignment: SplitAssignment) -> dict[str, Any]:
    development = {row["source"] for i, row in enumerate(rows) if row_assignment(i, grouping, assignment)[2] == "development"}
    final_test = {row["source"] for i, row in enumerate(rows) if row_assignment(i, grouping, assignment)[2] == "final_test"}
    return {
        "development_sources": len(development),
        "final_test_sources": len(final_test),
        "shared_sources": len(development & final_test),
        "development_only_sources": len(development - final_test),
        "final_test_only_sources": len(final_test - development),
    }


def build_manifest(
    rows: list[dict[str, str]],
    grouping: CompositionGrouping,
    assignment: SplitAssignment,
    labels: dict[str, np.ndarray],
    checks: dict[str, Any],
) -> dict[str, Any]:
    row_records = []
    for index, row in enumerate(rows):
        _, group_id, outer, fold = row_assignment(index, grouping, assignment)
        row_records.append({
            "task_row_index": index,
            "raw_workbook_row": int(row["raw_workbook_row"]),
            "source": row["source"],
            "composition_group": group_id,
            "outer_split": outer,
            "development_validation_fold": fold,
        })
    group_records = []
    for group_index, group_id in enumerate(grouping.group_ids):
        fold = int(assignment.validation_fold_by_group[group_index])
        group_records.append({
            "composition_group": group_id,
            "row_count": len(grouping.group_rows[group_index]),
            "task_row_indices": grouping.group_rows[group_index],
            "outer_split": str(assignment.outer_split_by_group[group_index]),
            "development_validation_fold": None if fold < 0 else fold + 1,
            "maximum_within_group_l1_wt_percent": float(grouping.diameters[group_index]),
            "nearest_other_group_l1_wt_percent": float(grouping.isolation_distances[group_index]),
        })
    return {
        "schema_version": "soccol_composition_split_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "dataset_id": "soccol_pitting_potential",
            "task": "numeric E_pit with event=1",
            "source_file": str(INPUT_CSV.relative_to(ROOT)),
            "source_sha256": sha256(INPUT_CSV),
            "usable_rows": len(rows),
            "target_column": "E_pit",
            "target_unit": "mV_vs_AgAgCl_3M_KCl",
            "model_visible_composition_columns": COMPOSITION_COLUMNS,
        },
        "composition_grouping": {
            "normalization": "none",
            "round_decimal_places": ROUND_DECIMALS,
            "blank_handling_for_distance": "processed composition zero encoding",
            "missingness_indicators_used_for_distance": False,
            "distance": "sum_absolute_difference_wt_percent",
            "clustering": "threshold_connected_components",
            "pair_separation_rule": "Every pair at or below the distance threshold belongs to the same atomic composition component.",
            "maximum_link_distance": DISTANCE_THRESHOLD,
            "maximum_component_diameter": float(grouping.diameters.max(initial=0.0)),
            "minimum_between_component_distance": float(grouping.isolation_distances.min(initial=np.inf)),
            "rounded_composition_keys": grouping.rounded_key_count,
            "composition_groups": grouping.n_groups,
        },
        "split_design": {
            "development_rows": EXPECTED_ROWS - FINAL_TEST_ROWS,
            "final_test_rows": FINAL_TEST_ROWS,
            "development_fraction": (EXPECTED_ROWS - FINAL_TEST_ROWS) / EXPECTED_ROWS,
            "final_test_fraction": FINAL_TEST_ROWS / EXPECTED_ROWS,
            "development_validation_folds": INNER_FOLD_COUNT,
            "fold_meaning": "Rows assigned to fold N are validation for fold N; the other development folds are context.",
            "balance_block_weights": BALANCE_BLOCK_WEIGHTS,
            "target_balance": "global E_pit deciles",
            "environment_balance_bins": {
                "CP_temp_celsius": [10.0, 35.0, 70.0],
                "CP_Cl_molar": [0.0, 0.01, 0.1, 0.6, 1.0],
                "CP_pH": [3.0, 6.0, 8.0],
            },
            "omitted_balance_blocks": {"test_method_family": "no comparable Soccol field; omitted by user decision"},
            "source_role": "balance and audit only; source does not define atomic groups",
        },
        "optimizer": {"outer": asdict(assignment.outer_optimizer), "inner": asdict(assignment.inner_optimizer)},
        "checks": {**checks, "source_overlap": source_overlap(rows, grouping, assignment)},
        "groups": group_records,
        "rows": row_records,
    }


def write_assignments(
    path: Path,
    rows: list[dict[str, str]],
    grouping: CompositionGrouping,
    assignment: SplitAssignment,
    labels: dict[str, np.ndarray],
) -> None:
    fields = [
        "task_row_index", "raw_workbook_row", "label", "source", "composition_group",
        "composition_group_rows", "outer_split", "development_validation_fold",
        "row_material_family", "material_scope_primary", "target_decile",
        "composition_isolation_quintile", "temperature_bin", "chloride_bin", "pH_bin",
        "source_balance_category", "E_pit_development_only", *COMPOSITION_COLUMNS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            group_index, group_id, outer, fold = row_assignment(index, grouping, assignment)
            record = {
                "task_row_index": index,
                "raw_workbook_row": row["raw_workbook_row"],
                "label": row["label"],
                "source": row["source"],
                "composition_group": group_id,
                "composition_group_rows": len(grouping.group_rows[group_index]),
                "outer_split": outer,
                "development_validation_fold": "" if fold is None else fold,
                "row_material_family": row["row_material_family"],
                "material_scope_primary": row["material_scope_primary"],
                "target_decile": int(labels["target_decile"][index]) + 1,
                "composition_isolation_quintile": int(labels["composition_isolation"][index]) + 1,
                "temperature_bin": labels["temperature"][index],
                "chloride_bin": labels["chloride"][index],
                "pH_bin": labels["pH"][index],
                "source_balance_category": labels["source"][index],
                "E_pit_development_only": row["E_pit"] if outer == "development" else "",
            }
            record.update({column: row[column] for column in COMPOSITION_COLUMNS})
            writer.writerow(record)


def count_table(values: np.ndarray, partitions: np.ndarray) -> dict[str, dict[str, int]]:
    return {
        partition: dict(sorted(Counter(values[partitions == partition]).items()))
        for partition in ("development", "final_test")
    }


def write_report(
    path: Path,
    rows: list[dict[str, str]],
    grouping: CompositionGrouping,
    assignment: SplitAssignment,
    labels: dict[str, np.ndarray],
    checks: dict[str, Any],
) -> None:
    partitions = assignment.outer_split_by_group[grouping.row_group_index]
    summary = {
        "rows": len(rows),
        "composition_groups": grouping.n_groups,
        "largest_group_rows": int(grouping.sizes.max()),
        "development_rows": int(np.sum(partitions == "development")),
        "final_test_rows": int(np.sum(partitions == "final_test")),
        "development_fold_rows": checks["development_validation_row_counts"],
        "source_overlap": source_overlap(rows, grouping, assignment),
        "balance_counts": {block: count_table(values, partitions) for block, values in labels.items()},
    }
    payload = html.escape(json.dumps(summary, indent=2, sort_keys=True))
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Soccol split report</title>"
        "<style>body{font-family:system-ui;max-width:1100px;margin:2rem auto}pre{background:#f4f4f4;padding:1rem;overflow:auto}</style>"
        "</head><body><h1>Soccol composition-separated split v1</h1>"
        "<p>Previous EPIT methodology: L1 composition components, exact 80/20 outer split, and five grouped development folds. The unavailable test-method block is omitted.</p>"
        f"<pre>{payload}</pre></body></html>\n",
        encoding="utf-8",
    )


def ensure_paths(output_dir: Path, force: bool) -> tuple[Path, Path, Path, Path]:
    manifest = output_dir / MANIFEST_NAME
    assignments = output_dir / ASSIGNMENTS_NAME
    report = output_dir / REPORT_NAME
    lock = output_dir / LOCK_NAME
    if lock.exists():
        raise SystemExit(f"Refusing to replace frozen split {output_dir}; use a new versioned directory")
    existing = [path for path in (manifest, assignments, report) if path.exists()]
    if existing and not force:
        raise SystemExit("Existing split artifacts found; pass --force intentionally")
    output_dir.mkdir(parents=True, exist_ok=True)
    return manifest, assignments, report, lock


def main() -> None:
    args = parse_args()
    if args.outer_time_limit <= 0 or args.inner_time_limit <= 0:
        raise SystemExit("Optimizer time limits must be positive")
    manifest_path, assignments_path, report_path, lock_path = ensure_paths(args.output_dir.resolve(), args.force)
    rows = load_rows()
    grouping = build_composition_groups(rows)
    labels = build_balance_labels(rows, grouping)
    features = build_balance_features(labels, grouping)
    outer, outer_record = optimize_outer(grouping, features, args.outer_time_limit)
    folds, inner_record = optimize_inner(grouping, features, outer, args.inner_time_limit)
    assignment = SplitAssignment(outer, folds, outer_record, inner_record)
    assignment = refine_inner(grouping, assignment, features)
    checks = validate_assignment(grouping, assignment)
    manifest = build_manifest(rows, grouping, assignment, labels, checks)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_assignments(assignments_path, rows, grouping, assignment, labels)
    write_report(report_path, rows, grouping, assignment, labels, checks)
    lock = {
        "schema_version": "epit_split_lock_v1",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(INPUT_CSV),
        "manifest": {"file": manifest_path.name, "sha256": sha256(manifest_path)},
        "assignments": {"file": assignments_path.name, "sha256": sha256(assignments_path)},
        "report": {"file": report_path.name, "sha256": sha256(report_path)},
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Created frozen split: {args.output_dir.resolve()}")
    print(f"Rows/groups: {len(rows)}/{grouping.n_groups}")
    print(f"Development/final: {EXPECTED_ROWS - FINAL_TEST_ROWS}/{FINAL_TEST_ROWS}")
    print(f"Fold rows: {checks['development_validation_row_counts']}")


if __name__ == "__main__":
    main()
