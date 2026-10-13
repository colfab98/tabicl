"""Name-based alignment to the combined corrosion model feature schema."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd


DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = DATASET_DIR / "processed" / "shared_feature_registry.json"


@dataclass(frozen=True)
class FeatureAlignmentReport:
    """Audit record for a name-based feature alignment."""

    schema_version: str
    matched_columns: dict[str, str]
    aliases_applied: dict[str, str]
    missing_columns_filled_with_nan: tuple[str, ...]
    ignored_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_feature_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    """Load the frozen shared-feature registry."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalized_name(name: Any) -> str:
    return str(name).strip().casefold()


def _name_lookup(
    registry: Mapping[str, Any],
    additional_aliases: Mapping[str, str] | None,
) -> dict[str, str]:
    canonical_columns = list(registry["feature_columns_in_order"])
    lookup: dict[str, str] = {}

    def add(name: str, canonical: str) -> None:
        normalized = _normalized_name(name)
        existing = lookup.get(normalized)
        if existing is not None and existing != canonical:
            raise ValueError(
                f"Ambiguous feature name {name!r}: maps to both {existing!r} and {canonical!r}."
            )
        lookup[normalized] = canonical

    for canonical in canonical_columns:
        add(canonical, canonical)
    for canonical, aliases in registry.get("feature_aliases", {}).items():
        if canonical not in canonical_columns:
            raise ValueError(f"Alias registry refers to unknown canonical feature {canonical!r}.")
        for alias in aliases:
            add(alias, canonical)
    for alias, canonical in (additional_aliases or {}).items():
        if canonical not in canonical_columns:
            raise ValueError(f"Additional alias {alias!r} maps to unknown feature {canonical!r}.")
        add(alias, canonical)
    return lookup


def align_feature_frame(
    frame: pd.DataFrame,
    *,
    registry: Mapping[str, Any] | None = None,
    additional_aliases: Mapping[str, str] | None = None,
    unknown_columns: Literal["error", "ignore"] = "error",
) -> tuple[pd.DataFrame, FeatureAlignmentReport]:
    """Align a named table to the canonical model schema.

    Incoming order is ignored. Recognized names and approved aliases are mapped
    to canonical names, absent features are filled with NaN, and the returned
    frame always follows ``feature_columns_in_order``. No unit conversion or
    fuzzy name matching is performed.
    """
    if unknown_columns not in {"error", "ignore"}:
        raise ValueError("unknown_columns must be either 'error' or 'ignore'.")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("align_feature_frame requires a pandas DataFrame with named columns.")
    duplicated = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Duplicate input columns are not allowed: {duplicated}")

    registry = load_feature_registry() if registry is None else dict(registry)
    canonical_columns = list(registry["feature_columns_in_order"])
    lookup = _name_lookup(registry, additional_aliases)

    source_by_canonical: dict[str, Any] = {}
    ignored: list[str] = []
    aliases_applied: dict[str, str] = {}
    for source in frame.columns:
        canonical = lookup.get(_normalized_name(source))
        if canonical is None:
            ignored.append(str(source))
            continue
        previous = source_by_canonical.get(canonical)
        if previous is not None:
            raise ValueError(
                f"Columns {previous!r} and {source!r} both map to canonical feature {canonical!r}."
            )
        source_by_canonical[canonical] = source
        if str(source) != canonical:
            aliases_applied[str(source)] = canonical

    if ignored and unknown_columns == "error":
        raise ValueError(
            "Unrecognized columns are not model features: "
            + ", ".join(repr(name) for name in ignored)
        )

    aligned_data = {
        canonical: (
            frame[source_by_canonical[canonical]]
            if canonical in source_by_canonical
            else pd.Series(np.nan, index=frame.index)
        )
        for canonical in canonical_columns
    }
    aligned = pd.DataFrame(aligned_data, index=frame.index)
    missing = tuple(name for name in canonical_columns if name not in source_by_canonical)
    report = FeatureAlignmentReport(
        schema_version=str(registry["schema_version"]),
        matched_columns={
            canonical: str(source_by_canonical[canonical])
            for canonical in canonical_columns
            if canonical in source_by_canonical
        },
        aliases_applied=aliases_applied,
        missing_columns_filled_with_nan=missing,
        ignored_columns=tuple(ignored),
    )
    return aligned, report
