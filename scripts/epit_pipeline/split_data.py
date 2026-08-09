"""Data loading, composition grouping, and balanced split construction."""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_SCRIPT_DIR = REPO_ROOT / "corrosion_datasets" / "analysis" / "scripts"
if str(ANALYSIS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT_DIR))

from analyze_structure import Table, load_electrochemical_metrics  # noqa: E402
from scripts.eval_corrosion_datasets import eval_to_float  # noqa: E402


DATASET_ID = "electrochemical_metrics_alloys"
TABLE_NAME = "Pitting Potential"
TASK_ID = "electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg"
TARGET_COLUMN = "Epit, mV (SCE) Avg."
SOURCE_FILE = (
    REPO_ROOT
    / "corrosion_datasets"
    / "datasets"
    / DATASET_ID
    / "raw"
    / "CRA_database_Scientific_Data_Publication_12102020.xlsx"
)
EXPECTED_ROW_COUNT = 760
OUTER_TEST_ROWS = 152
INNER_FOLD_COUNT = 5
COMPOSITION_ROUND_DECIMALS = 2
COMPOSITION_DISTANCE_THRESHOLD = 1.0

OMITTED_COMPOSITION_COLUMNS = {
    "Composition, wt.% N",
    "Composition, wt.% C",
    "Composition, wt.% Si",
    "Composition, wt.% Mn",
    "Composition, wt.% Cu",
    "Composition, wt.% P",
    "Composition, wt.% S",
}

BALANCE_BLOCK_WEIGHTS = {
    "target_decile": 3.0,
    "material_class": 2.0,
    "composition_isolation": 2.0,
    "temperature": 1.0,
    "chloride": 1.0,
    "pH": 1.0,
    "test_method_family": 1.0,
    "reference": 0.5,
}


@dataclass(frozen=True)
class EpitDataset:
    table: Table
    rows: list[dict[str, Any]]
    target: np.ndarray
    composition_columns: list[str]

    @property
    def n_rows(self) -> int:
        return len(self.rows)


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
        return np.asarray([len(indices) for indices in self.group_rows], dtype=int)


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


def _numeric(table: Table, row: dict[str, Any], column: str) -> float:
    return eval_to_float(
        row.get(column),
        column=column,
        group=table.groups.get(column, ""),
    )


def load_epit_dataset() -> EpitDataset:
    tables = [table for table in load_electrochemical_metrics() if table.table == TABLE_NAME]
    if len(tables) != 1:
        raise RuntimeError(f"Expected one {TABLE_NAME!r} table, found {len(tables)}.")
    table = tables[0]
    rows = [
        row
        for row in table.rows
        if math.isfinite(_numeric(table, row, TARGET_COLUMN))
    ]
    if len(rows) != EXPECTED_ROW_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_ROW_COUNT} usable EPIT rows, found {len(rows)}."
        )

    composition_columns = [
        column
        for column in table.columns
        if table.groups.get(column) == "material"
        and column not in OMITTED_COMPOSITION_COLUMNS
    ]
    if len(composition_columns) != 17:
        raise RuntimeError(
            f"Expected 17 model-visible composition columns, found {len(composition_columns)}."
        )

    target = np.asarray(
        [_numeric(table, row, TARGET_COLUMN) for row in rows],
        dtype=float,
    )
    if not np.isfinite(target).all():
        raise RuntimeError("Filtered EPIT target still contains non-finite values.")
    return EpitDataset(
        table=table,
        rows=rows,
        target=target,
        composition_columns=composition_columns,
    )


def _rounded_composition_key(
    dataset: EpitDataset,
    row: dict[str, Any],
) -> tuple[float | None, ...]:
    values: list[float | None] = []
    for column in dataset.composition_columns:
        value = _numeric(dataset.table, row, column)
        values.append(
            None
            if not math.isfinite(value)
            else round(value, COMPOSITION_ROUND_DECIMALS)
        )
    return tuple(values)


def cluster_composition_vectors(
    vectors: np.ndarray,
    *,
    distance_threshold: float = COMPOSITION_DISTANCE_THRESHOLD,
) -> np.ndarray:
    """Connected components for pairs within the absolute wt.% threshold."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("Composition vectors must be a non-empty 2D array.")
    if len(vectors) == 1:
        return np.zeros(1, dtype=int)

    parent = np.arange(len(vectors), dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    distances = squareform(pdist(vectors, metric="cityblock"))
    close_pairs = np.argwhere(
        np.triu(distances <= distance_threshold + 1e-12, k=1)
    )
    for left, right in close_pairs:
        union(int(left), int(right))

    remap: dict[int, int] = {}
    stable = np.empty(len(vectors), dtype=int)
    for index in range(len(vectors)):
        stable[index] = remap.setdefault(find(index), len(remap))
    return stable


def build_composition_groups(dataset: EpitDataset) -> CompositionGrouping:
    key_rows: dict[tuple[float | None, ...], list[int]] = {}
    for row_index, row in enumerate(dataset.rows):
        key_rows.setdefault(_rounded_composition_key(dataset, row), []).append(row_index)

    keys = list(key_rows)
    # Blank and explicit zero are treated alike only for conservative grouping.
    vectors = np.asarray(
        [[0.0 if value is None else value for value in key] for key in keys],
        dtype=float,
    )
    key_cluster = cluster_composition_vectors(vectors)
    raw_group_rows: dict[int, list[int]] = defaultdict(list)
    raw_group_vectors: dict[int, list[np.ndarray]] = defaultdict(list)
    for cluster_index, key, vector in zip(key_cluster, keys, vectors, strict=True):
        raw_group_rows[int(cluster_index)].extend(key_rows[key])
        raw_group_vectors[int(cluster_index)].append(vector)

    ordered_raw_groups = sorted(
        raw_group_rows,
        key=lambda group: min(raw_group_rows[group]),
    )
    group_rows = [sorted(raw_group_rows[group]) for group in ordered_raw_groups]
    group_ids = [f"composition_{index + 1:04d}" for index in range(len(group_rows))]
    row_group_index = np.empty(dataset.n_rows, dtype=int)
    centroids: list[np.ndarray] = []
    diameters: list[float] = []
    for group_index, raw_group in enumerate(ordered_raw_groups):
        row_group_index[group_rows[group_index]] = group_index
        group_vectors = np.asarray(raw_group_vectors[raw_group], dtype=float)
        centroids.append(group_vectors.mean(axis=0))
        diameter = (
            0.0
            if len(group_vectors) < 2
            else float(pdist(group_vectors, metric="cityblock").max())
        )
        diameters.append(diameter)

    centroid_array = np.asarray(centroids, dtype=float)
    group_vector_arrays = [
        np.asarray(raw_group_vectors[raw_group], dtype=float)
        for raw_group in ordered_raw_groups
    ]
    isolation = np.full(len(group_vector_arrays), np.inf, dtype=float)
    for left in range(len(group_vector_arrays)):
        for right in range(left + 1, len(group_vector_arrays)):
            distances = np.abs(
                group_vector_arrays[left][:, None, :]
                - group_vector_arrays[right][None, :, :]
            ).sum(axis=2)
            minimum = float(distances.min())
            isolation[left] = min(isolation[left], minimum)
            isolation[right] = min(isolation[right], minimum)
    isolation[~np.isfinite(isolation)] = 0.0

    grouping = CompositionGrouping(
        group_ids=group_ids,
        group_rows=group_rows,
        row_group_index=row_group_index,
        centroids=centroid_array,
        diameters=np.asarray(diameters, dtype=float),
        isolation_distances=isolation,
        rounded_key_count=len(keys),
    )
    validate_composition_groups(dataset, grouping)
    return grouping


def validate_composition_groups(
    dataset: EpitDataset,
    grouping: CompositionGrouping,
) -> None:
    observed_rows = sorted(index for indices in grouping.group_rows for index in indices)
    if observed_rows != list(range(dataset.n_rows)):
        raise RuntimeError("Composition groups do not cover every row exactly once.")
    row_vectors = np.asarray(
        [
            [0.0 if value is None else value for value in _rounded_composition_key(dataset, row)]
            for row in dataset.rows
        ],
        dtype=float,
    )
    distances = squareform(pdist(row_vectors, metric="cityblock"))
    close_pairs = np.argwhere(
        np.triu(distances <= COMPOSITION_DISTANCE_THRESHOLD + 1e-12, k=1)
    )
    crossing_pairs = [
        (int(left), int(right))
        for left, right in close_pairs
        if grouping.row_group_index[left] != grouping.row_group_index[right]
    ]
    if crossing_pairs:
        raise RuntimeError(
            "Composition pairs within the approved distance threshold cross groups: "
            f"{crossing_pairs[:5]}"
        )
    for group_id, indices in zip(grouping.group_ids, grouping.group_rows, strict=True):
        classes = {
            str(dataset.rows[index].get("Material class") or "<missing>").strip()
            for index in indices
        }
        if len(classes) > 1:
            raise RuntimeError(f"{group_id} mixes material classes: {sorted(classes)}")


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.arange(1, n_bins) / n_bins))
    return np.digitize(values, edges, right=True).astype(int)


def _numeric_bin(
    value: float,
    edges: list[float],
    *,
    separate_zero: bool = False,
) -> str:
    if not math.isfinite(value):
        return "missing"
    if separate_zero and value == 0.0:
        return "zero"
    return f"bin_{int(np.digitize([value], edges, right=True)[0])}"


def _method_family(value: Any) -> str:
    text = str(value or "").lower()
    dynamic = "potentiodynamic" in text or "potcntiodynamic" in text
    static = "potentiostatic" in text
    if "scratch" in text:
        return "scratch"
    if dynamic and static:
        return "mixed_dynamic_static"
    if dynamic:
        return "potentiodynamic"
    if static:
        return "potentiostatic"
    return "other"


def build_balance_labels(
    dataset: EpitDataset,
    grouping: CompositionGrouping,
) -> dict[str, np.ndarray]:
    target_decile = _quantile_bins(dataset.target, 10).astype(str)
    isolation_quintile_by_group = _quantile_bins(grouping.isolation_distances, 5)
    isolation_quintile = isolation_quintile_by_group[grouping.row_group_index].astype(str)

    references = np.asarray(
        [str(row.get("Reference") or "<missing>").strip() for row in dataset.rows]
    )
    common_references = {
        reference for reference, _ in Counter(references).most_common(10)
    }
    references = np.asarray(
        [reference if reference in common_references else "other" for reference in references]
    )

    return {
        "target_decile": target_decile,
        "material_class": np.asarray(
            [str(row.get("Material class") or "<missing>").strip() for row in dataset.rows]
        ),
        "composition_isolation": isolation_quintile,
        "temperature": np.asarray(
            [
                _numeric_bin(_numeric(dataset.table, row, "Test Temp. oC"), [10.0, 35.0, 70.0])
                for row in dataset.rows
            ]
        ),
        "chloride": np.asarray(
            [
                _numeric_bin(
                    _numeric(dataset.table, row, "[Cl-] M"),
                    [0.01, 0.1, 0.6, 1.0],
                    separate_zero=True,
                )
                for row in dataset.rows
            ]
        ),
        "pH": np.asarray(
            [
                _numeric_bin(_numeric(dataset.table, row, "[Cl-] pH"), [3.0, 6.0, 8.0])
                for row in dataset.rows
            ]
        ),
        "test_method_family": np.asarray(
            [_method_family(row.get("[Cl-] Test Method")) for row in dataset.rows]
        ),
        "reference": references,
    }


def build_balance_features(
    labels: dict[str, np.ndarray],
    grouping: CompositionGrouping,
) -> list[BalanceFeature]:
    features: list[BalanceFeature] = []
    for block, row_labels in labels.items():
        if block not in BALANCE_BLOCK_WEIGHTS:
            raise RuntimeError(f"Missing balance weight for block {block!r}.")
        for category in sorted(set(row_labels)):
            group_counts = np.bincount(
                grouping.row_group_index,
                weights=(row_labels == category).astype(float),
                minlength=grouping.n_groups,
            )
            features.append(
                BalanceFeature(
                    block=block,
                    category=str(category),
                    weight=BALANCE_BLOCK_WEIGHTS[block],
                    group_counts=group_counts,
                )
            )
    return features


def _optimizer_record(result: Any, time_limit_seconds: float) -> OptimizerRecord:
    return OptimizerRecord(
        message=str(result.message),
        objective=float(result.fun),
        mip_gap=(
            None
            if getattr(result, "mip_gap", None) is None
            else float(result.mip_gap)
        ),
        node_count=(
            None
            if getattr(result, "mip_node_count", None) is None
            else int(result.mip_node_count)
        ),
        time_limit_seconds=float(time_limit_seconds),
    )


def _require_feasible_result(result: Any, name: str) -> np.ndarray:
    if result.x is None or not np.isfinite(result.x).all():
        raise RuntimeError(f"{name} optimizer did not find a feasible split: {result.message}")
    return np.asarray(result.x, dtype=float)


def _make_bounds(binary_count: int, variable_count: int) -> Bounds:
    return Bounds(
        np.zeros(variable_count, dtype=float),
        np.concatenate(
            [
                np.ones(binary_count, dtype=float),
                np.full(variable_count - binary_count, np.inf),
            ]
        ),
    )


def optimize_outer_split(
    grouping: CompositionGrouping,
    features: list[BalanceFeature],
    *,
    time_limit_seconds: float,
) -> tuple[np.ndarray, OptimizerRecord]:
    n_groups = grouping.n_groups
    n_slack = len(features)
    n_variables = n_groups + n_slack
    sizes = grouping.sizes.astype(float)
    final_fraction = OUTER_TEST_ROWS / sizes.sum()
    final_group_count = round(n_groups * final_fraction)

    objective = np.zeros(n_variables, dtype=float)
    objective[:n_groups] = np.arange(n_groups) * 1e-10 / max(n_groups, 1)
    objective[n_groups:] = np.asarray([feature.weight for feature in features]) / OUTER_TEST_ROWS

    constraint_rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for coefficients, target in (
        (sizes, OUTER_TEST_ROWS),
        (np.ones(n_groups), final_group_count),
    ):
        row = np.zeros(n_variables, dtype=float)
        row[:n_groups] = coefficients
        constraint_rows.append(row)
        lower.append(float(target))
        upper.append(float(target))

    for feature_index, feature in enumerate(features):
        desired = feature.total * final_fraction
        for sign in (1.0, -1.0):
            row = np.zeros(n_variables, dtype=float)
            row[:n_groups] = sign * feature.group_counts
            row[n_groups + feature_index] = -1.0
            constraint_rows.append(row)
            lower.append(-np.inf)
            upper.append(sign * desired)

    result = milp(
        objective,
        integrality=np.concatenate([np.ones(n_groups), np.zeros(n_slack)]),
        bounds=_make_bounds(n_groups, n_variables),
        constraints=LinearConstraint(
            coo_matrix(np.asarray(constraint_rows)),
            np.asarray(lower),
            np.asarray(upper),
        ),
        options={"time_limit": time_limit_seconds, "mip_rel_gap": 0.01},
    )
    solution = _require_feasible_result(result, "Outer split")
    final_groups = np.flatnonzero(solution[:n_groups] > 0.5)
    outer_split = np.full(n_groups, "development", dtype=object)
    outer_split[final_groups] = "final_test"
    return outer_split, _optimizer_record(result, time_limit_seconds)


def _balanced_integer_targets(total: int, parts: int) -> list[int]:
    quotient, remainder = divmod(total, parts)
    return [quotient + (index < remainder) for index in range(parts)]


def optimize_inner_folds(
    grouping: CompositionGrouping,
    features: list[BalanceFeature],
    outer_split_by_group: np.ndarray,
    *,
    time_limit_seconds: float,
) -> tuple[np.ndarray, OptimizerRecord]:
    development_groups = np.flatnonzero(outer_split_by_group == "development")
    development_rows = int(grouping.sizes[development_groups].sum())
    if development_rows != EXPECTED_ROW_COUNT - OUTER_TEST_ROWS:
        raise RuntimeError(f"Expected 608 development rows, found {development_rows}.")

    fold_rows = _balanced_integer_targets(development_rows, INNER_FOLD_COUNT)
    fold_groups = _balanced_integer_targets(len(development_groups), INNER_FOLD_COUNT)
    n_development_groups = len(development_groups)
    n_binary = INNER_FOLD_COUNT * n_development_groups
    n_slack = INNER_FOLD_COUNT * len(features)
    n_variables = n_binary + n_slack
    development_sizes = grouping.sizes[development_groups].astype(float)

    objective = np.zeros(n_variables, dtype=float)
    objective[:n_binary] = np.arange(n_binary) * 1e-11 / max(n_binary, 1)
    feature_weights = np.asarray([feature.weight for feature in features])
    objective[n_binary:] = np.tile(feature_weights, INNER_FOLD_COUNT) / max(fold_rows)

    constraint_rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    for local_group in range(n_development_groups):
        row = np.zeros(n_variables, dtype=float)
        row[np.arange(INNER_FOLD_COUNT) * n_development_groups + local_group] = 1.0
        constraint_rows.append(row)
        lower.append(1.0)
        upper.append(1.0)

    for fold_index in range(INNER_FOLD_COUNT):
        start = fold_index * n_development_groups
        stop = start + n_development_groups
        for coefficients, target in (
            (development_sizes, fold_rows[fold_index]),
            (np.ones(n_development_groups), fold_groups[fold_index]),
        ):
            row = np.zeros(n_variables, dtype=float)
            row[start:stop] = coefficients
            constraint_rows.append(row)
            lower.append(float(target))
            upper.append(float(target))

        fold_fraction = fold_rows[fold_index] / development_rows
        for feature_index, feature in enumerate(features):
            development_counts = feature.group_counts[development_groups]
            desired = float(development_counts.sum()) * fold_fraction
            slack_index = n_binary + fold_index * len(features) + feature_index
            for sign in (1.0, -1.0):
                row = np.zeros(n_variables, dtype=float)
                row[start:stop] = sign * development_counts
                row[slack_index] = -1.0
                constraint_rows.append(row)
                lower.append(-np.inf)
                upper.append(sign * desired)

    result = milp(
        objective,
        integrality=np.concatenate([np.ones(n_binary), np.zeros(n_slack)]),
        bounds=_make_bounds(n_binary, n_variables),
        constraints=LinearConstraint(
            coo_matrix(np.asarray(constraint_rows)),
            np.asarray(lower),
            np.asarray(upper),
        ),
        options={"time_limit": time_limit_seconds, "mip_rel_gap": 0.01},
    )
    solution = _require_feasible_result(result, "Inner folds")
    validation_fold = np.full(grouping.n_groups, -1, dtype=int)
    for fold_index in range(INNER_FOLD_COUNT):
        start = fold_index * n_development_groups
        stop = start + n_development_groups
        selected_local = np.flatnonzero(solution[start:stop] > 0.5)
        validation_fold[development_groups[selected_local]] = fold_index
    return validation_fold, _optimizer_record(result, time_limit_seconds)


def build_split_assignment(
    dataset: EpitDataset,
    grouping: CompositionGrouping,
    *,
    outer_time_limit_seconds: float,
    inner_time_limit_seconds: float,
) -> tuple[SplitAssignment, dict[str, np.ndarray]]:
    labels = build_balance_labels(dataset, grouping)
    features = build_balance_features(labels, grouping)
    outer_split, outer_record = optimize_outer_split(
        grouping,
        features,
        time_limit_seconds=outer_time_limit_seconds,
    )
    validation_fold, inner_record = optimize_inner_folds(
        grouping,
        features,
        outer_split,
        time_limit_seconds=inner_time_limit_seconds,
    )
    assignment = SplitAssignment(
        outer_split_by_group=outer_split,
        validation_fold_by_group=validation_fold,
        outer_optimizer=outer_record,
        inner_optimizer=inner_record,
    )
    validate_split_assignment(dataset, grouping, assignment)
    return assignment, labels


def validate_split_assignment(
    dataset: EpitDataset,
    grouping: CompositionGrouping,
    assignment: SplitAssignment,
) -> None:
    sizes = grouping.sizes
    final_groups = assignment.final_test_groups
    development_groups = assignment.development_groups
    if int(sizes[final_groups].sum()) != OUTER_TEST_ROWS:
        raise RuntimeError("Final-test split does not contain exactly 152 rows.")
    if int(sizes[development_groups].sum()) != dataset.n_rows - OUTER_TEST_ROWS:
        raise RuntimeError("Development split does not contain exactly 608 rows.")
    if set(final_groups) & set(development_groups):
        raise RuntimeError("A composition group crosses the outer split boundary.")
    if np.any(assignment.validation_fold_by_group[final_groups] != -1):
        raise RuntimeError("Final-test groups were assigned to Optuna folds.")
    development_folds = assignment.validation_fold_by_group[development_groups]
    if set(development_folds) != set(range(INNER_FOLD_COUNT)):
        raise RuntimeError("Development groups are not assigned to all five Optuna folds.")
    expected_rows = sorted(_balanced_integer_targets(dataset.n_rows - OUTER_TEST_ROWS, INNER_FOLD_COUNT))
    observed_rows = sorted(
        int(sizes[assignment.validation_fold_by_group == fold].sum())
        for fold in range(INNER_FOLD_COUNT)
    )
    if observed_rows != expected_rows:
        raise RuntimeError(
            f"Optuna fold row counts are {observed_rows}, expected {expected_rows}."
        )
