#!/usr/bin/env python
"""Analyze the static EPIT composition families without using target values.

The report is intended to support a later family-conditioned composition
generator.  It summarizes the exact, versioned composition profile used by the
project; it does not read EPIT targets or fit target rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tabicl.prior.epit_composition_profile import (
    EPIT_COMPOSITION_FAMILY_COUNTS,
    EPIT_COMPOSITION_PROFILE,
    EpitCompositionProfile,
    load_epit_composition_profile,
)


DEFAULT_OUTPUT_DIR = Path(
    "corrosion_datasets/analysis/epit_composition_families/epit_dataset_v1"
)
FAMILY_EXPECTED_BASE = {
    "fe_alloy": "Fe",
    "al_alloy": "Al",
    "nicrmo_alloy": "Ni",
}
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=EPIT_COMPOSITION_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--robust-family-min",
        type=int,
        default=50,
        help="Minimum templates for the report's 'supported' descriptive band.",
    )
    parser.add_argument(
        "--limited-family-min",
        type=int,
        default=15,
        help="Minimum templates for the report's 'limited' descriptive band.",
    )
    parser.add_argument(
        "--min-correlation-samples",
        type=int,
        default=8,
        help="Minimum positive co-occurrences for a positive-only log correlation.",
    )
    return parser.parse_args()


def _optional_float(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _quantile(values: np.ndarray, probability: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values, probability))


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "minimum": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "minimum": float(np.min(values)),
        "p05": _quantile(values, 0.05),
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p95": _quantile(values, 0.95),
        "maximum": float(np.max(values)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic one-based average ranks, including exact ties."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        sorted_ranks[start:stop] = 0.5 * ((start + 1) + stop)
        start = stop
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    return _optional_float(np.corrcoef(left, right)[0, 1])


def _family_support(
    count: int, *, robust_family_min: int, limited_family_min: int
) -> str:
    if count >= robust_family_min:
        return "supported"
    if count >= limited_family_min:
        return "limited"
    return "insufficient"


def _family_summary(
    profile: EpitCompositionProfile,
    family_index: int,
    values: np.ndarray,
    *,
    robust_family_min: int,
    limited_family_min: int,
) -> dict[str, Any]:
    family = profile.families[family_index]
    count = int(values.shape[0])
    filled = np.nan_to_num(values, nan=0.0)
    observed = filled[:, profile.observed_element_indices]
    full_sums = filled.sum(axis=1)
    observed_sums = observed.sum(axis=1)
    full_active = np.sum(filled > 0.0, axis=1)
    observed_active = np.sum(observed > 0.0, axis=1)
    dominant_indices = np.argmax(filled, axis=1)
    dominant_counts = np.bincount(dominant_indices, minlength=len(profile.elements))
    dominant = {
        element: int(dominant_counts[index])
        for index, element in enumerate(profile.elements)
        if dominant_counts[index] > 0
    }
    maximum_shares = 100.0 * np.max(filled, axis=1) / full_sums
    expected_base = FAMILY_EXPECTED_BASE.get(family)
    expected_base_dominant_fraction = None
    if expected_base is not None:
        base_index = profile.elements.index(expected_base)
        expected_base_dominant_fraction = float(np.mean(dominant_indices == base_index))

    return {
        "family": family,
        "templates": count,
        "profile_probability": float(profile.family_probabilities[family_index]),
        "support_band": _family_support(
            count,
            robust_family_min=robust_family_min,
            limited_family_min=limited_family_min,
        ),
        "expected_base_element": expected_base,
        "expected_base_dominant_fraction": expected_base_dominant_fraction,
        "dominant_element_counts": dominant,
        "unique_zero_filled_compositions": int(np.unique(filled, axis=0).shape[0]),
        "unique_observed_active_masks": int(np.unique(observed > 0.0, axis=0).shape[0]),
        "full_sum_wt_percent": _distribution(full_sums),
        "observed_17_sum_wt_percent": _distribution(observed_sums),
        "full_active_elements": _distribution(full_active),
        "observed_17_active_elements": _distribution(observed_active),
        "largest_element_share_percent": _distribution(maximum_shares),
    }


def _element_rows(
    profile: EpitCompositionProfile,
    family_index: int,
    values: np.ndarray,
    support_band: str,
) -> list[dict[str, Any]]:
    family = profile.families[family_index]
    count = int(values.shape[0])
    observed = set(profile.observed_elements)
    rows: list[dict[str, Any]] = []
    for element_index, element in enumerate(profile.elements):
        raw = values[:, element_index]
        reported = np.isfinite(raw)
        filled = np.nan_to_num(raw, nan=0.0)
        positive = filled[filled > 0.0]
        all_distribution = _distribution(filled)
        positive_distribution = _distribution(positive)
        row = {
            "family": family,
            "family_templates": count,
            "support_band": support_band,
            "element": element,
            "observed_by_model": element in observed,
            "reported_count": int(np.sum(reported)),
            "missing_count": int(np.sum(~reported)),
            "positive_count": int(positive.size),
            "positive_fraction": float(positive.size / count),
        }
        row.update({f"all_{key}": value for key, value in all_distribution.items()})
        row.update(
            {f"positive_{key}": value for key, value in positive_distribution.items()}
        )
        rows.append(row)
    return rows


def _correlation_rows(
    profile: EpitCompositionProfile,
    family_index: int,
    values: np.ndarray,
    *,
    min_correlation_samples: int,
) -> list[dict[str, Any]]:
    family = profile.families[family_index]
    observed = np.nan_to_num(
        values[:, profile.observed_element_indices], nan=0.0
    )
    rows: list[dict[str, Any]] = []
    for left_index, left_element in enumerate(profile.observed_elements):
        for right_index in range(left_index + 1, len(profile.observed_elements)):
            right_element = profile.observed_elements[right_index]
            left = observed[:, left_index]
            right = observed[:, right_index]
            both_positive = (left > 0.0) & (right > 0.0)
            cooccurrences = int(np.sum(both_positive))
            log_positive_correlation = None
            if cooccurrences >= min_correlation_samples:
                log_positive_correlation = _correlation(
                    np.log(left[both_positive]), np.log(right[both_positive])
                )
            rows.append(
                {
                    "family": family,
                    "family_templates": int(values.shape[0]),
                    "left_element": left_element,
                    "right_element": right_element,
                    "cooccurrence_count": cooccurrences,
                    "cooccurrence_fraction": float(cooccurrences / values.shape[0]),
                    "pearson_zero_filled": _correlation(left, right),
                    "spearman_zero_filled": _correlation(
                        _average_ranks(left), _average_ranks(right)
                    ),
                    "pearson_log_both_positive": log_positive_correlation,
                }
            )
    return rows


def analyze_profile(
    profile: EpitCompositionProfile,
    *,
    robust_family_min: int = 50,
    limited_family_min: int = 15,
    min_correlation_samples: int = 8,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if robust_family_min <= limited_family_min:
        raise ValueError("robust_family_min must exceed limited_family_min.")
    if limited_family_min <= 0 or min_correlation_samples < 2:
        raise ValueError("Analysis sample thresholds are invalid.")

    family_summaries: list[dict[str, Any]] = []
    element_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    for family_index, _family in enumerate(profile.families):
        mask = profile.template_family_indices == family_index
        values = profile.template_values[mask]
        family_summary = _family_summary(
            profile,
            family_index,
            values,
            robust_family_min=robust_family_min,
            limited_family_min=limited_family_min,
        )
        family_summaries.append(family_summary)
        element_rows.extend(
            _element_rows(
                profile,
                family_index,
                values,
                str(family_summary["support_band"]),
            )
        )
        correlation_rows.extend(
            _correlation_rows(
                profile,
                family_index,
                values,
                min_correlation_samples=min_correlation_samples,
            )
        )

    summary = {
        "profile": profile.name,
        "profile_metadata": profile.metadata,
        "analysis_scope": {
            "uses_composition_features_only": True,
            "uses_epit_targets": False,
            "templates": profile.n_templates,
            "elements": len(profile.elements),
            "model_observed_elements": len(profile.observed_elements),
            "positive_core_range": "5th--95th percentiles among positive entries",
            "missing_value_treatment": (
                "Missing entries are counted separately and treated as zero for "
                "active-count, dominance, sum, and correlation summaries, matching "
                "the empirical runtime generator."
            ),
            "support_bands": {
                "supported": f">={robust_family_min} templates",
                "limited": (
                    f">={limited_family_min} and <{robust_family_min} templates"
                ),
                "insufficient": f"<{limited_family_min} templates",
            },
            "min_positive_cooccurrences_for_log_correlation": min_correlation_samples,
        },
        "families": family_summaries,
    }
    validate_analysis(profile, summary, element_rows, correlation_rows)
    return summary, element_rows, correlation_rows


def validate_analysis(
    profile: EpitCompositionProfile,
    summary: dict[str, Any],
    element_rows: Sequence[dict[str, Any]],
    correlation_rows: Sequence[dict[str, Any]],
) -> None:
    family_summaries = summary["families"]
    counts = tuple(int(row["templates"]) for row in family_summaries)
    if counts != EPIT_COMPOSITION_FAMILY_COUNTS:
        raise ValueError(f"Unexpected family counts in analysis: {counts}")
    if sum(counts) != profile.n_templates:
        raise ValueError("Family template counts do not sum to the profile size.")
    if len(element_rows) != len(profile.families) * len(profile.elements):
        raise ValueError("Element-statistic row count is incomplete.")
    expected_pairs = math.comb(len(profile.observed_elements), 2)
    if len(correlation_rows) != len(profile.families) * expected_pairs:
        raise ValueError("Correlation row count is incomplete.")
    for row in family_summaries:
        full_sum = row["full_sum_wt_percent"]
        if float(full_sum["minimum"]) < 99.0 or float(full_sum["maximum"]) > 101.0:
            raise ValueError(f"Family {row['family']} contains a non-closing composition.")
        if sum(row["dominant_element_counts"].values()) != row["templates"]:
            raise ValueError(f"Dominant-element counts are incomplete for {row['family']}.")
    for row in element_rows:
        if row["reported_count"] + row["missing_count"] != row["family_templates"]:
            raise ValueError("Reported and missing element counts are inconsistent.")
        if not 0.0 <= float(row["positive_fraction"]) <= 1.0:
            raise ValueError("Element positive fraction is outside [0, 1].")
    for row in correlation_rows:
        for key in (
            "pearson_zero_filled",
            "spearman_zero_filled",
            "pearson_log_both_positive",
        ):
            value = row[key]
            if value is not None and not -1.0000001 <= float(value) <= 1.0000001:
                raise ValueError(f"Invalid correlation in {key}: {value}")


def _format_number(value: float | int | None, decimals: int = 3) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{decimals}f}"


def _major_element_rows(
    element_rows: Sequence[dict[str, Any]], family: str
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in element_rows
        if row["family"] == family
        and row["observed_by_model"]
        and row["positive_count"] >= 5
        and row["positive_fraction"] >= 0.10
    ]
    return sorted(
        selected,
        key=lambda row: (
            -float(row["positive_fraction"]),
            -float(row["positive_median"] or 0.0),
            str(row["element"]),
        ),
    )


def _strong_correlation_rows(
    correlation_rows: Sequence[dict[str, Any]], family: str
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in correlation_rows
        if row["family"] == family
        and row["pearson_log_both_positive"] is not None
    ]
    return sorted(
        eligible,
        key=lambda row: -abs(float(row["pearson_log_both_positive"])),
    )[:5]


def render_markdown(
    summary: dict[str, Any],
    element_rows: Sequence[dict[str, Any]],
    correlation_rows: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# EPIT composition-family analysis",
        "",
        (
            "This target-free report analyzes the 403 exact composition templates "
            "in `epit_dataset_v1`, matching the static empirical composition profile "
            "documented in `main2.tex`."
        ),
        "",
        "The reported core range is P05--P95 among templates where the element is "
        "positive. Presence must be used alongside that range; the range alone does "
        "not describe sparsity.",
        "",
        "## Family coverage",
        "",
        "| Family | Templates | Support | Expected base dominant | Median active (17) | Median largest share |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for family in summary["families"]:
        base_fraction = family["expected_base_dominant_fraction"]
        lines.append(
            "| {family} | {templates} | {support} | {base} | {active} | {share}% |".format(
                family=family["family"],
                templates=family["templates"],
                support=family["support_band"],
                base=(
                    "n/a"
                    if base_fraction is None
                    else f"{100.0 * float(base_fraction):.1f}%"
                ),
                active=_format_number(
                    family["observed_17_active_elements"]["median"], 1
                ),
                share=_format_number(
                    family["largest_element_share_percent"]["median"], 1
                ),
            )
        )

    for family in summary["families"]:
        family_name = str(family["family"])
        dominance = sorted(
            family["dominant_element_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        )
        dominance_text = ", ".join(
            f"{element} {count}/{family['templates']}"
            for element, count in dominance
        )
        lines.extend(
            [
                "",
                f"## {family_name}",
                "",
                f"Dominant elements: {dominance_text}.",
                "",
                (
                    "Visible 17-element sum P05/median/P95: "
                    f"{_format_number(family['observed_17_sum_wt_percent']['p05'], 2)} / "
                    f"{_format_number(family['observed_17_sum_wt_percent']['median'], 2)} / "
                    f"{_format_number(family['observed_17_sum_wt_percent']['p95'], 2)} wt.%."
                ),
                "",
                "| Element | Presence | Positive P05 | Positive median | Positive P95 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in _major_element_rows(element_rows, family_name):
            lines.append(
                "| {element} | {presence:.1f}% | {p05} | {median} | {p95} |".format(
                    element=row["element"],
                    presence=100.0 * float(row["positive_fraction"]),
                    p05=_format_number(row["positive_p05"]),
                    median=_format_number(row["positive_median"]),
                    p95=_format_number(row["positive_p95"]),
                )
            )
        strong_correlations = _strong_correlation_rows(
            correlation_rows, family_name
        )
        if strong_correlations:
            lines.extend(["", "Strongest positive-only log correlations:", ""])
            for row in strong_correlations:
                lines.append(
                    "- {left}--{right}: r={correlation}, n={count}".format(
                        left=row["left_element"],
                        right=row["right_element"],
                        correlation=_format_number(
                            row["pearson_log_both_positive"]
                        ),
                        count=row["cooccurrence_count"],
                    )
                )

    lines.extend(
        [
            "",
            "## Generator interpretation",
            "",
            "- Fe and Al have enough distinct templates for dataset-derived family ranges.",
            "- HEA and Ni--Cr--Mo can be included as limited-data pilot families, but their ranges should not be treated as precise population bounds.",
            "- `other` is too small and chemically mixed for one learned family distribution; exclude it initially or split it only after adding data.",
            "- A future generator should model element presence separately from positive amount. Sampling only continuous ranges would make alloys too dense.",
            "- Dedicated Al/HEA EPIT target rules are not justified by this composition-only analysis; this report supports composition generation only.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    element_rows: Sequence[dict[str, Any]],
    correlation_rows: Sequence[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    family_csv_rows = []
    for family in summary["families"]:
        family_csv_rows.append(
            {
                "family": family["family"],
                "templates": family["templates"],
                "profile_probability": family["profile_probability"],
                "support_band": family["support_band"],
                "expected_base_element": family["expected_base_element"],
                "expected_base_dominant_fraction": family[
                    "expected_base_dominant_fraction"
                ],
                "unique_zero_filled_compositions": family[
                    "unique_zero_filled_compositions"
                ],
                "unique_observed_active_masks": family[
                    "unique_observed_active_masks"
                ],
                "observed_sum_p05": family["observed_17_sum_wt_percent"]["p05"],
                "observed_sum_median": family["observed_17_sum_wt_percent"][
                    "median"
                ],
                "observed_sum_p95": family["observed_17_sum_wt_percent"]["p95"],
                "observed_active_p05": family["observed_17_active_elements"]["p05"],
                "observed_active_median": family["observed_17_active_elements"][
                    "median"
                ],
                "observed_active_p95": family["observed_17_active_elements"]["p95"],
                "largest_share_p05": family["largest_element_share_percent"]["p05"],
                "largest_share_median": family["largest_element_share_percent"][
                    "median"
                ],
                "largest_share_p95": family["largest_element_share_percent"]["p95"],
                "dominant_element_counts": json.dumps(
                    family["dominant_element_counts"], sort_keys=True
                ),
            }
        )
    _write_csv(output_dir / "family_summary.csv", family_csv_rows)
    _write_csv(output_dir / "element_ranges.csv", element_rows)
    _write_csv(output_dir / "element_correlations.csv", correlation_rows)
    (output_dir / "report.md").write_text(
        render_markdown(summary, element_rows, correlation_rows), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    profile = load_epit_composition_profile(args.profile)
    summary, element_rows, correlation_rows = analyze_profile(
        profile,
        robust_family_min=args.robust_family_min,
        limited_family_min=args.limited_family_min,
        min_correlation_samples=args.min_correlation_samples,
    )
    write_outputs(args.output_dir, summary, element_rows, correlation_rows)
    print(f"Analyzed {profile.n_templates} templates from {profile.name}.")
    print(f"Wrote family report to {args.output_dir / 'report.md'}.")


if __name__ == "__main__":
    main()
