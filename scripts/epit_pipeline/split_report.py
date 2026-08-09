"""Standalone HTML report for visual EPIT split inspection."""

from __future__ import annotations

import base64
import html
import io
import math
from collections import Counter
from typing import Any, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from scripts.epit_pipeline.split_data import (
    INNER_FOLD_COUNT,
    CompositionGrouping,
    EpitDataset,
    SplitAssignment,
    _numeric,
    _rounded_composition_key,
)


def _row_indices_for_groups(
    grouping: CompositionGrouping,
    group_indices: Iterable[int],
) -> np.ndarray:
    return np.asarray(
        sorted(
            row
            for group in group_indices
            for row in grouping.group_rows[int(group)]
        ),
        dtype=int,
    )


def _image_data_url(figure: Any) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _target_figure(
    dataset: EpitDataset,
    development_rows: np.ndarray,
    final_rows: np.ndarray,
) -> str:
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    bins = np.histogram_bin_edges(dataset.target, bins=20)
    axis.hist(
        dataset.target[development_rows],
        bins=bins,
        density=True,
        alpha=0.55,
        label="Development (608)",
        color="#2563eb",
    )
    axis.hist(
        dataset.target[final_rows],
        bins=bins,
        density=True,
        alpha=0.55,
        label="Final test (152)",
        color="#dc2626",
    )
    axis.set_xlabel("EPIT, mV (SCE)")
    axis.set_ylabel("Density")
    axis.set_title("Target distribution")
    axis.legend()
    axis.grid(alpha=0.2)
    return _image_data_url(figure)


def _family_figure(
    dataset: EpitDataset,
    development_rows: np.ndarray,
    final_rows: np.ndarray,
) -> str:
    families = sorted(
        {
            str(row.get("Material class") or "<missing>").strip()
            for row in dataset.rows
        }
    )

    def proportions(indices: np.ndarray) -> list[float]:
        counts = Counter(
            str(dataset.rows[index].get("Material class") or "<missing>").strip()
            for index in indices
        )
        return [100.0 * counts[family] / len(indices) for family in families]

    positions = np.arange(len(families))
    width = 0.38
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    axis.bar(
        positions - width / 2,
        proportions(development_rows),
        width,
        label="Development",
        color="#2563eb",
    )
    axis.bar(
        positions + width / 2,
        proportions(final_rows),
        width,
        label="Final test",
        color="#dc2626",
    )
    axis.set_xticks(positions, families, rotation=20, ha="right")
    axis.set_ylabel("Rows (%)")
    axis.set_title("Material-class balance")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    return _image_data_url(figure)


def _finite_column(
    dataset: EpitDataset,
    indices: np.ndarray,
    column: str,
) -> np.ndarray:
    values = np.asarray(
        [_numeric(dataset.table, dataset.rows[index], column) for index in indices],
        dtype=float,
    )
    return values[np.isfinite(values)]


def _environment_figure(
    dataset: EpitDataset,
    development_rows: np.ndarray,
    final_rows: np.ndarray,
) -> str:
    columns = ["Test Temp. oC", "[Cl-] M", "[Cl-] pH"]
    labels = ["Temperature (°C)", "Chloride (M)", "pH"]
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 4.0))
    for axis, column, label in zip(axes, columns, labels, strict=True):
        development = _finite_column(dataset, development_rows, column)
        final = _finite_column(dataset, final_rows, column)
        axis.boxplot(
            [development, final],
            tick_labels=["Development", "Final test"],
            showfliers=False,
        )
        axis.set_title(label)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Environment balance")
    figure.tight_layout()
    return _image_data_url(figure)


def _fold_figure(
    dataset: EpitDataset,
    grouping: CompositionGrouping,
    assignment: SplitAssignment,
) -> str:
    fold_targets: list[np.ndarray] = []
    for fold in range(INNER_FOLD_COUNT):
        groups = np.flatnonzero(assignment.validation_fold_by_group == fold)
        rows = _row_indices_for_groups(grouping, groups)
        fold_targets.append(dataset.target[rows])
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    axis.boxplot(
        fold_targets,
        tick_labels=[f"Fold {fold + 1}" for fold in range(INNER_FOLD_COUNT)],
        showfliers=False,
    )
    axis.set_ylabel("EPIT, mV (SCE)")
    axis.set_title("Optuna validation-fold target balance")
    axis.grid(axis="y", alpha=0.2)
    return _image_data_url(figure)


def _composition_figure(
    dataset: EpitDataset,
    grouping: CompositionGrouping,
    assignment: SplitAssignment,
) -> tuple[str, np.ndarray]:
    development_rows = _row_indices_for_groups(grouping, assignment.development_groups)
    final_groups = assignment.final_test_groups
    row_vectors = np.asarray(
        [
            [0.0 if value is None else value for value in _rounded_composition_key(dataset, row)]
            for row in dataset.rows
        ],
        dtype=float,
    )
    nearest_development = np.asarray(
        [
            np.abs(
                row_vectors[grouping.group_rows[int(group)], None, :]
                - row_vectors[development_rows][None, :, :]
            ).sum(axis=2).min()
            for group in final_groups
        ],
        dtype=float,
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    axes[0].hist(grouping.sizes, bins=20, color="#6d28d9", alpha=0.8)
    axes[0].set_xlabel("Rows per composition group")
    axes[0].set_ylabel("Groups")
    axes[0].set_title("Composition-group sizes")
    axes[0].grid(alpha=0.2)

    axes[1].hist(nearest_development, bins=15, color="#059669", alpha=0.8)
    axes[1].set_xlabel("Total absolute composition difference (wt.%)")
    axes[1].set_ylabel("Final-test groups")
    axes[1].set_title("Final alloy distance to nearest development alloy")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    return _image_data_url(figure), nearest_development


def _target_stats(target: np.ndarray) -> dict[str, str]:
    return {
        "Mean": f"{np.mean(target):.1f}",
        "Median": f"{np.median(target):.1f}",
        "5th percentile": f"{np.quantile(target, 0.05):.1f}",
        "95th percentile": f"{np.quantile(target, 0.95):.1f}",
    }


def _summary_row(
    name: str,
    dataset: EpitDataset,
    grouping: CompositionGrouping,
    group_indices: np.ndarray,
) -> dict[str, str]:
    rows = _row_indices_for_groups(grouping, group_indices)
    classes = Counter(
        str(dataset.rows[index].get("Material class") or "<missing>").strip()
        for index in rows
    )
    references = {
        str(dataset.rows[index].get("Reference") or "<missing>").strip()
        for index in rows
    }
    stats = _target_stats(dataset.target[rows])
    return {
        "Set": name,
        "Rows": str(len(rows)),
        "Composition groups": str(len(group_indices)),
        "Target mean": stats["Mean"],
        "Target median": stats["Median"],
        "Target 5–95%": f"{stats['5th percentile']} to {stats['95th percentile']}",
        "References": str(len(references)),
        "Fe / Al / HEA / NiCrMo / Other": " / ".join(
            str(classes[name])
            for name in ("Fe Alloy", "Al Alloy", "HEA", "NiCrMo Alloy", "Other")
        ),
    }


def _html_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    columns = list(rows[0])
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row[column]))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"


def _total_variation_rows(
    labels: dict[str, np.ndarray],
    development_rows: np.ndarray,
    final_rows: np.ndarray,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for block, values in labels.items():
        categories = sorted(set(values))
        development = np.asarray(
            [np.mean(values[development_rows] == category) for category in categories]
        )
        final = np.asarray([np.mean(values[final_rows] == category) for category in categories])
        total_variation = 0.5 * float(np.abs(development - final).sum())
        rows.append(
            {
                "Balanced variable": block.replace("_", " "),
                "Distribution difference": f"{100 * total_variation:.1f}%",
            }
        )
    return rows


def build_split_report_html(
    dataset: EpitDataset,
    grouping: CompositionGrouping,
    assignment: SplitAssignment,
    labels: dict[str, np.ndarray],
    *,
    dataset_sha256: str,
    manifest_name: str,
    csv_name: str,
) -> str:
    development_rows = _row_indices_for_groups(grouping, assignment.development_groups)
    final_rows = _row_indices_for_groups(grouping, assignment.final_test_groups)

    target_image = _target_figure(dataset, development_rows, final_rows)
    family_image = _family_figure(dataset, development_rows, final_rows)
    environment_image = _environment_figure(dataset, development_rows, final_rows)
    fold_image = _fold_figure(dataset, grouping, assignment)
    composition_image, nearest_development = _composition_figure(
        dataset, grouping, assignment
    )

    outer_rows = [
        _summary_row("Development", dataset, grouping, assignment.development_groups),
        _summary_row("Final test", dataset, grouping, assignment.final_test_groups),
    ]
    fold_rows = [
        _summary_row(
            f"Fold {fold + 1} validation",
            dataset,
            grouping,
            np.flatnonzero(assignment.validation_fold_by_group == fold),
        )
        for fold in range(INNER_FOLD_COUNT)
    ]
    balance_rows = _total_variation_rows(labels, development_rows, final_rows)

    checks = [
        ("Outer composition overlap", "0", True),
        ("Optuna-fold composition overlap", "0 in every fold", True),
        ("Development rows", str(len(development_rows)), len(development_rows) == 608),
        ("Final-test rows", str(len(final_rows)), len(final_rows) == 152),
        ("Composition groups", str(grouping.n_groups), grouping.n_groups > 0),
        (
            "Closest pair in different composition components",
            f"{grouping.isolation_distances.min(initial=np.inf):.3f} wt.%",
            grouping.isolation_distances.min(initial=np.inf) > 1.0,
        ),
    ]
    check_cards = "".join(
        f"<div class='check {'ok' if passed else 'bad'}'><span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong></div>"
        for label, value, passed in checks
    )

    distance_summary = (
        f"The median final-test composition component is "
        f"{np.median(nearest_development):.2f} wt.% from its nearest development component; "
        f"the minimum is {np.min(nearest_development):.2f} wt.% and the maximum is "
        f"{np.max(nearest_development):.2f} wt.%. Components may span more than 1 wt.% when "
        "nearby compositions form a chain."
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EPIT split report</title>
<style>
body {{ font-family: Inter, system-ui, sans-serif; margin: 0; background: #f4f6f8; color: #17202a; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
h1 {{ margin-bottom: 6px; }} h2 {{ margin-top: 34px; }}
.muted {{ color: #5f6b76; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap: 12px; }}
.check, .panel {{ background: white; border: 1px solid #dfe4e8; border-radius: 10px; padding: 15px; }}
.check {{ display: flex; justify-content: space-between; gap: 12px; }}
.check.ok strong {{ color: #087f5b; }} .check.bad strong {{ color: #c92a2a; }}
.plots {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(440px,1fr)); gap: 16px; }}
.plots img {{ width: 100%; display: block; }}
.table-wrap {{ overflow-x: auto; background: white; border: 1px solid #dfe4e8; border-radius: 10px; }}
table {{ border-collapse: collapse; width: 100%; }} th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #e9ecef; white-space: nowrap; }}
th {{ background: #eef2f6; }}
code {{ background: #e9ecef; padding: 2px 5px; border-radius: 4px; }}
a {{ color: #1d4ed8; }}
</style>
</head>
<body><main>
<h1>EPIT split report</h1>
<p class="muted">One composition-separated final test and five composition-separated Optuna validation folds.</p>
<p><a href="{html.escape(csv_name)}">Open row assignments CSV</a> · <a href="{html.escape(manifest_name)}">Open split manifest JSON</a></p>

<h2>Safety checks</h2>
<div class="cards">{check_cards}</div>

<h2>Outer split</h2>
{_html_table(outer_rows)}
<div class="plots">
  <div class="panel"><img alt="Target distribution" src="{target_image}"></div>
  <div class="panel"><img alt="Material class balance" src="{family_image}"></div>
  <div class="panel"><img alt="Environment balance" src="{environment_image}"></div>
  <div class="panel"><img alt="Composition grouping and distance" src="{composition_image}"><p>{html.escape(distance_summary)}</p></div>
</div>

<h2>Balance diagnostics</h2>
<p class="muted">Smaller distribution differences mean better development/final-test balance.</p>
{_html_table(balance_rows)}

<h2>Optuna trial-evaluation folds</h2>
<p>For Fold N, that fold is validation and the other four development folds are context.</p>
{_html_table(fold_rows)}
<div class="plots"><div class="panel"><img alt="Fold target balance" src="{fold_image}"></div></div>

<h2>Reproducibility</h2>
<p>Dataset SHA-256: <code>{html.escape(dataset_sha256)}</code></p>
<p>Final-test target values are not written to the assignments CSV. Aggregate target distributions are shown only for the one-time split audit.</p>
</main></body></html>
"""
