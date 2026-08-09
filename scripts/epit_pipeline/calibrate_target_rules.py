#!/usr/bin/env python
"""Stage 2: calibrate and directly evaluate EPIT target-rule families.

Each rule is calibrated on the context rows of each saved Optuna fold and
evaluated on that fold's held-out development rows. A final coefficient set is
also fitted on all development rows for later final training. Final-test target
values are never used by the calibration or direct-evaluation calculations.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from scripts.epit_pipeline.artifact_hashes import (
    FrozenSplit,
    load_frozen_split,
    sha256_file,
)
from scripts.epit_pipeline.prepare_splits import DEFAULT_OUTPUT_DIR as DEFAULT_SPLIT_DIR
from scripts.epit_pipeline.split_data import (
    INNER_FOLD_COUNT,
    REPO_ROOT,
    SOURCE_FILE,
    EpitDataset,
    load_epit_dataset,
)
from scripts.epit_pipeline.target_rules import TargetRuleFamily, get_rule_families


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "corrosion_datasets" / "analysis" / "epit_pipeline" / "target_rules_v2"
)
SUMMARY_NAME = "calibration_summary.json"
PREDICTIONS_NAME = "direct_evaluation_predictions.csv"
REPORT_NAME = "direct_evaluation_report.html"


@dataclass(frozen=True)
class CoefficientFit:
    coefficients: np.ndarray
    objective: float


@dataclass(frozen=True)
class ManifestRows:
    development: np.ndarray
    final_test: np.ndarray
    validation_by_fold: dict[int, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=DEFAULT_SPLIT_DIR / "split_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--rule",
        action="append",
        dest="rules",
        help="Registered rule family to evaluate; repeat as needed (default: all).",
    )
    parser.add_argument(
        "--anchor-strength",
        type=float,
        default=0.1,
        help="Regularization toward the rule's physics anchor (default: 0.1).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace this stage's existing output files.",
    )
    return parser.parse_args()


def _load_manifest(
    path: Path,
    dataset: EpitDataset,
) -> tuple[FrozenSplit, ManifestRows]:
    frozen = load_frozen_split(path)
    manifest = frozen.manifest
    metadata = manifest.get("dataset", {})
    if int(metadata.get("usable_rows", -1)) != dataset.n_rows:
        raise RuntimeError("Split manifest row count does not match the EPIT dataset.")
    if metadata.get("model_visible_composition_columns") != dataset.composition_columns:
        raise RuntimeError("Split manifest composition columns do not match the dataset.")
    if str(metadata.get("source_sha256", "")) != sha256_file(SOURCE_FILE):
        raise RuntimeError("EPIT source file changed after the split was created.")

    records = manifest.get("rows", [])
    if len(records) != dataset.n_rows:
        raise RuntimeError("Split manifest needs one assignment per EPIT row.")
    by_index = {int(record["task_row_index"]): record for record in records}
    if sorted(by_index) != list(range(dataset.n_rows)):
        raise RuntimeError("Split row indices do not cover the dataset exactly once.")

    development = np.asarray(
        [
            index
            for index, row in by_index.items()
            if row["outer_split"] == "development"
        ],
        dtype=int,
    )
    final_test = np.asarray(
        [
            index
            for index, row in by_index.items()
            if row["outer_split"] == "final_test"
        ],
        dtype=int,
    )
    validation_by_fold = {
        fold: np.asarray(
            [
                int(index)
                for index in development
                if int(by_index[int(index)]["optuna_validation_fold"]) == fold
            ],
            dtype=int,
        )
        for fold in range(1, INNER_FOLD_COUNT + 1)
    }
    expected_development = int(manifest["split_design"]["development_rows"])
    expected_final = int(manifest["split_design"]["final_test_rows"])
    if len(development) != expected_development or len(final_test) != expected_final:
        raise RuntimeError("Split development/final row counts are inconsistent.")
    covered = np.concatenate(list(validation_by_fold.values()))
    if sorted(covered.tolist()) != sorted(development.tolist()):
        raise RuntimeError("Validation folds do not cover development rows exactly once.")
    return frozen, ManifestRows(development, final_test, validation_by_fold)


def _rank_standardize(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=float), method="average")
    std = float(np.std(ranks, ddof=0))
    if std <= 1e-12:
        raise RuntimeError("Cannot calibrate against a constant target.")
    return (ranks - float(np.mean(ranks))) / std


def fit_coefficients(
    terms: np.ndarray,
    target: np.ndarray,
    anchor: np.ndarray,
    *,
    anchor_strength: float,
    upper_bounds: np.ndarray | None = None,
) -> CoefficientFit:
    """Fit non-negative relative coefficients that sum to one."""
    terms = np.asarray(terms, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    anchor = np.asarray(anchor, dtype=float).reshape(-1)
    if terms.ndim != 2 or terms.shape[0] != target.size:
        raise ValueError("Rule terms and target have incompatible shapes.")
    if terms.shape[1] != anchor.size:
        raise ValueError("Coefficient anchor width does not match rule terms.")
    if not np.isfinite(terms).all() or not np.isfinite(target).all():
        raise ValueError("Calibration inputs must be finite.")
    if anchor_strength < 0 or not np.isfinite(anchor_strength):
        raise ValueError("Anchor strength must be finite and non-negative.")
    if upper_bounds is None:
        upper_bounds = np.ones(anchor.size, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float).reshape(-1)
    if upper_bounds.shape != anchor.shape:
        raise ValueError("Coefficient upper bounds do not match the anchor.")
    if (
        not np.isfinite(upper_bounds).all()
        or np.any(upper_bounds <= 0.0)
        or float(upper_bounds.sum()) < 1.0
    ):
        raise ValueError("Coefficient upper bounds cannot satisfy the simplex.")

    anchor = np.clip(anchor, 0.0, None)
    if float(anchor.sum()) <= 0.0:
        raise ValueError("Coefficient anchor must contain a positive value.")
    anchor = anchor / anchor.sum()
    if np.any(anchor > upper_bounds + 1e-12):
        raise ValueError("Coefficient anchor exceeds a configured upper bound.")
    target_rank = _rank_standardize(target)

    def objective(weights: np.ndarray) -> float:
        residual = target_rank - terms @ weights
        return float(
            np.mean(residual**2)
            + float(anchor_strength) * np.sum((weights - anchor) ** 2)
        )

    result = minimize(
        objective,
        x0=anchor,
        method="SLSQP",
        bounds=[(0.0, float(bound)) for bound in upper_bounds],
        constraints=[
            {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}
        ],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Coefficient calibration failed: {result.message}")
    coefficients = np.asarray(result.x, dtype=float)
    coefficients[np.abs(coefficients) < 1e-12] = 0.0
    coefficients /= coefficients.sum()
    return CoefficientFit(
        coefficients=coefficients,
        objective=objective(coefficients),
    )


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values, ddof=0))
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def _spearman(target: np.ndarray, prediction: np.ndarray) -> float:
    if (
        np.unique(np.asarray(target, dtype=float)).size < 2
        or np.unique(np.asarray(prediction, dtype=float)).size < 2
    ):
        return 0.0
    value = float(spearmanr(target, prediction).correlation)
    return value if np.isfinite(value) else 0.0


def _score(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target_z = _standardize(target)
    prediction_z = _standardize(prediction)
    residual = prediction_z - target_z
    return {
        "spearman": _spearman(target, prediction),
        "standardized_mae": float(np.mean(np.abs(residual))),
        "standardized_rmse": float(np.sqrt(np.mean(residual**2))),
    }


def _coefficient_dict(
    family: TargetRuleFamily,
    values: np.ndarray,
) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(family.term_names, values, strict=True)
    }


def calibrate_family(
    family: TargetRuleFamily,
    dataset: EpitDataset,
    rows: ManifestRows,
    *,
    anchor_strength: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_results: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    oof_standardized = np.full(dataset.n_rows, np.nan, dtype=float)
    eligible_development = family.eligible_rows(dataset, rows.development)
    if len(eligible_development) < 10:
        raise RuntimeError(
            f"{family.name} has only {len(eligible_development)} eligible "
            "development rows."
        )

    for fold, saved_validation_rows in rows.validation_by_fold.items():
        validation_rows = family.eligible_rows(dataset, saved_validation_rows)
        if len(validation_rows) < 3:
            raise RuntimeError(
                f"{family.name} has only {len(validation_rows)} eligible rows "
                f"in validation fold {fold}."
            )
        context_rows = np.setdiff1d(
            eligible_development,
            validation_rows,
            assume_unique=True,
        )
        prepared = family.fit_terms(dataset, context_rows)
        fit = fit_coefficients(
            prepared.values,
            dataset.target[context_rows],
            family.coefficient_anchor,
            anchor_strength=anchor_strength,
            upper_bounds=family.coefficient_upper_bounds,
        )
        validation_terms = family.transform_terms(
            dataset,
            validation_rows,
            prepared.state,
        )
        predictions = validation_terms @ fit.coefficients
        metrics = _score(dataset.target[validation_rows], predictions)
        prediction_z = _standardize(predictions)
        oof_standardized[validation_rows] = prediction_z
        observed_ranks = rankdata(
            dataset.target[validation_rows],
            method="average",
        )
        predicted_ranks = rankdata(predictions, method="average")
        denominator = float(len(validation_rows))
        for position, row_index in enumerate(validation_rows):
            prediction_rows.append(
                {
                    "rule_family": family.name,
                    "task_row_index": int(row_index),
                    "optuna_validation_fold": fold,
                    "observed_epit_mV_SCE": float(dataset.target[row_index]),
                    "out_of_fold_rule_score": float(predictions[position]),
                    "out_of_fold_rule_score_standardized": float(
                        prediction_z[position]
                    ),
                    "observed_rank_percentile_within_fold": float(
                        (observed_ranks[position] - 0.5) / denominator
                    ),
                    "predicted_rank_percentile_within_fold": float(
                        (predicted_ranks[position] - 0.5) / denominator
                    ),
                }
            )
        fold_results.append(
            {
                "fold": fold,
                "context_rows": int(len(context_rows)),
                "validation_rows": int(len(validation_rows)),
                "coefficients": _coefficient_dict(
                    family,
                    fit.coefficients,
                ),
                "objective": fit.objective,
                **metrics,
                "feature_state": prepared.state,
            }
        )

    if not np.isfinite(oof_standardized[eligible_development]).all():
        raise RuntimeError(f"Incomplete out-of-fold predictions for {family.name}.")

    final_prepared = family.fit_terms(dataset, eligible_development)
    final_fit = fit_coefficients(
        final_prepared.values,
        dataset.target[eligible_development],
        family.coefficient_anchor,
        anchor_strength=anchor_strength,
        upper_bounds=family.coefficient_upper_bounds,
    )
    fitted_development_prediction = (
        final_prepared.values @ final_fit.coefficients
    )
    fold_spearman = np.asarray(
        [row["spearman"] for row in fold_results],
        dtype=float,
    )
    result = {
        "schema_version": "epit_target_rule_v2",
        "rule_family": family.name,
        "description": family.description,
        "evaluation_role": family.evaluation_role,
        "applicability": family.applicability,
        "formula": family.metadata,
        "term_names": list(family.term_names),
        "coefficient_constraints": "nonnegative; sum to one",
        "coefficient_anchor": _coefficient_dict(
            family,
            family.coefficient_anchor,
        ),
        "coefficient_upper_bounds": (
            None
            if family.coefficient_upper_bounds is None
            else _coefficient_dict(
                family,
                family.coefficient_upper_bounds,
            )
        ),
        "anchor_strength": float(anchor_strength),
        "calibration_target": "standardized EPIT ranks",
        "fold_calibrations": fold_results,
        "direct_evaluation": {
            "scope": (
                "out-of-fold eligible development rows only; applicability="
                f"{family.applicability}"
            ),
            "rows": int(len(eligible_development)),
            "minimum_validation_rows": int(
                min(row["validation_rows"] for row in fold_results)
            ),
            "small_sample_warning": bool(len(eligible_development) < 50),
            "mean_fold_spearman": float(np.mean(fold_spearman)),
            "standard_deviation_fold_spearman": float(
                np.std(fold_spearman, ddof=0)
            ),
            "minimum_fold_spearman": float(np.min(fold_spearman)),
            "maximum_fold_spearman": float(np.max(fold_spearman)),
            "pooled_out_of_fold_spearman": _spearman(
                dataset.target[eligible_development],
                oof_standardized[eligible_development],
            ),
        },
        "final_development_calibration": {
            "purpose": "coefficient center for final training; not final evaluation",
            "rows": int(len(eligible_development)),
            "coefficients": _coefficient_dict(
                family,
                final_fit.coefficients,
            ),
            "objective": final_fit.objective,
            "in_sample_spearman_diagnostic": _spearman(
                dataset.target[eligible_development],
                fitted_development_prediction,
            ),
            "feature_state": final_prepared.state,
        },
        "final_test_targets_used": False,
    }
    return result, prediction_rows


def _image_data_url(figure: Any) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def _family_plots(
    result: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> tuple[str, str, str]:
    observed = np.asarray(
        [row["observed_epit_mV_SCE"] for row in predictions],
    )
    observed_rank = np.asarray(
        [row["observed_rank_percentile_within_fold"] for row in predictions],
    )
    predicted_rank = np.asarray(
        [row["predicted_rank_percentile_within_fold"] for row in predictions],
    )

    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    scatter = axis.scatter(
        observed,
        predicted_rank,
        c=[row["optuna_validation_fold"] for row in predictions],
        cmap="viridis",
        s=18,
        alpha=0.65,
    )
    axis.set_xlabel("Observed EPIT, mV (SCE)")
    axis.set_ylabel("Rule-predicted rank within held-out fold")
    axis.set_title("Out-of-fold direct rule predictions")
    axis.grid(alpha=0.2)
    figure.colorbar(scatter, ax=axis, label="Optuna validation fold")
    scatter_image = _image_data_url(figure)

    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    axis.scatter(
        observed_rank,
        predicted_rank,
        s=18,
        alpha=0.65,
        color="#2563eb",
    )
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#dc2626",
        linewidth=1.2,
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Observed rank within held-out fold")
    axis.set_ylabel("Predicted rank within held-out fold")
    axis.set_title("Observed versus predicted ordering")
    axis.grid(alpha=0.2)
    rank_image = _image_data_url(figure)

    folds = result["fold_calibrations"]
    term_names = result["term_names"]
    positions = np.arange(len(term_names))
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for fold in folds:
        axis.scatter(
            positions,
            [fold["coefficients"][name] for name in term_names],
            alpha=0.55,
            color="#64748b",
            s=28,
        )
    final = result["final_development_calibration"]["coefficients"]
    axis.scatter(
        positions,
        [final[name] for name in term_names],
        color="#dc2626",
        marker="D",
        s=65,
        label="All-development calibration",
    )
    axis.set_xticks(
        positions,
        [name.replace("_", " ") for name in term_names],
        rotation=15,
    )
    axis.set_ylabel("Relative coefficient")
    axis.set_ylim(bottom=0)
    axis.set_title("Coefficient stability across folds")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    coefficient_image = _image_data_url(figure)
    return scatter_image, rank_image, coefficient_image


def _table(records: list[dict[str, Any]]) -> str:
    if not records:
        return "<p>No results.</p>"
    columns = list(records[0])
    header = "".join(
        f"<th>{html.escape(str(column))}</th>"
        for column in columns
    )
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(record[column]))}</td>"
            for column in columns
        )
        + "</tr>"
        for record in records
    )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
    )


def build_report(
    results: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    manifest_path: Path,
) -> str:
    summary_rows = []
    sections = []
    for result in results:
        family_predictions = [
            row
            for row in predictions
            if row["rule_family"] == result["rule_family"]
        ]
        direct = result["direct_evaluation"]
        summary_rows.append(
            {
                "Rule": result["rule_family"],
                "Role": result["evaluation_role"],
                "Applicable alloys": result["applicability"],
                "Rows": direct["rows"],
                "Mean fold Spearman": f"{direct['mean_fold_spearman']:.3f}",
                "Fold range": (
                    f"{direct['minimum_fold_spearman']:.3f} to "
                    f"{direct['maximum_fold_spearman']:.3f}"
                ),
                "Pooled OOF Spearman": (
                    f"{direct['pooled_out_of_fold_spearman']:.3f}"
                ),
            }
        )
        fold_rows = [
            {
                "Fold": fold["fold"],
                "Context rows": fold["context_rows"],
                "Validation rows": fold["validation_rows"],
                "Spearman": f"{fold['spearman']:.3f}",
                "Standardized MAE": f"{fold['standardized_mae']:.3f}",
            }
            for fold in result["fold_calibrations"]
        ]
        scatter, ranks, coefficients = _family_plots(
            result,
            family_predictions,
        )
        sections.append(
            f"""
            <section>
              <h2>{html.escape(result['rule_family'])}</h2>
              <p>{html.escape(result['description'])}</p>
              <p><strong>Applicable alloys:</strong>
              {html.escape(result['applicability'])};
              <strong>development rows:</strong> {direct['rows']}.</p>
              {
                  '<p class="warning"><strong>Small-sample warning:</strong> '
                  'treat this direct score as exploratory.</p>'
                  if direct['small_sample_warning']
                  else ''
              }
              {_table(fold_rows)}
              <div class="plots"><img src="{scatter}"><img src="{ranks}"></div>
              <img class="wide" src="{coefficients}">
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EPIT target-rule direct evaluation</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 18px;color:#172033}}
h1,h2{{color:#0f172a}} .note{{background:#eff6ff;border-left:4px solid #2563eb;padding:12px 16px}}
.warning{{background:#fff7ed;border-left:4px solid #ea580c;padding:10px 14px}}
.table-wrap{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;margin:14px 0 24px}}
th,td{{border:1px solid #dbe3ef;padding:8px 10px;text-align:left}} th{{background:#f1f5f9}}
.plots{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} img{{width:100%;height:auto}}
.wide{{max-width:760px}} section{{border-top:1px solid #dbe3ef;margin-top:30px;padding-top:18px}}
code{{background:#f1f5f9;padding:2px 5px}} @media(max-width:760px){{.plots{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>EPIT target-rule direct evaluation</h1>
<p class="note"><strong>This is not the final model evaluation.</strong>
Every score below uses only held-out folds inside the 608-row development set.
Alloy-specific rules use only their applicable rows. The 152 final-test
targets were not used.</p>
<p>Split manifest: <code>{html.escape(str(manifest_path))}</code></p>
<h2>Rule comparison</h2>
<p><strong>Compare scores only when the applicable rows are the same.</strong>
The Fe/Ni candidates use the same 452 rows and matched current-PREN baseline.
The all-alloy historical baseline is a separate diagnostic.</p>
{_table(summary_rows)}
{''.join(sections)}
</body></html>"""


def _ensure_outputs(
    output_dir: Path,
    rule_names: list[str],
    *,
    force: bool,
) -> tuple[Path, Path, Path, list[Path]]:
    summary = output_dir / SUMMARY_NAME
    predictions = output_dir / PREDICTIONS_NAME
    report = output_dir / REPORT_NAME
    rule_paths = [output_dir / f"{name}.json" for name in rule_names]
    existing = [
        path
        for path in [summary, predictions, report, *rule_paths]
        if path.exists()
    ]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(
            f"Refusing to replace existing artifacts ({names}). "
            "Use --force intentionally."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return summary, predictions, report, rule_paths


def _write_predictions(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError("No direct-evaluation predictions were generated.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.anchor_strength < 0 or not np.isfinite(args.anchor_strength):
        raise SystemExit("--anchor-strength must be finite and non-negative.")
    try:
        families = get_rule_families(args.rules)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    manifest_path = args.split_manifest.resolve()
    output_dir = args.output_dir.resolve()
    summary_path, predictions_path, report_path, rule_paths = _ensure_outputs(
        output_dir,
        [family.name for family in families],
        force=args.force,
    )
    dataset = load_epit_dataset()
    frozen_split, rows = _load_manifest(manifest_path, dataset)
    manifest = frozen_split.manifest
    masked_target = dataset.target.copy()
    masked_target[rows.final_test] = np.nan
    dataset = EpitDataset(
        table=dataset.table,
        rows=dataset.rows,
        target=masked_target,
        composition_columns=dataset.composition_columns,
    )

    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for family, rule_path in zip(families, rule_paths, strict=True):
        result, family_predictions = calibrate_family(
            family,
            dataset,
            rows,
            anchor_strength=float(args.anchor_strength),
        )
        result.update(
            {
                "split_manifest": str(frozen_split.manifest_path),
                "split_manifest_sha256": frozen_split.manifest_sha256,
                "split_lock": str(frozen_split.lock_path),
                "split_lock_sha256": frozen_split.lock_sha256,
                "source_sha256": manifest["dataset"]["source_sha256"],
            }
        )
        rule_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        results.append(result)
        predictions.extend(family_predictions)

    rule_artifact_sha256 = {
        result["rule_family"]: sha256_file(rule_path)
        for result, rule_path in zip(results, rule_paths, strict=True)
    }
    summary = {
        "schema_version": "epit_target_rule_calibration_summary_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_manifest": str(frozen_split.manifest_path),
        "split_manifest_schema": manifest["schema_version"],
        "split_manifest_sha256": frozen_split.manifest_sha256,
        "split_lock": str(frozen_split.lock_path),
        "split_lock_sha256": frozen_split.lock_sha256,
        "source_sha256": manifest["dataset"]["source_sha256"],
        "development_rows": int(len(rows.development)),
        "final_test_rows_excluded": int(len(rows.final_test)),
        "final_test_targets_masked_before_calibration": True,
        "final_test_targets_used": False,
        "rules": [
            {
                "rule_family": result["rule_family"],
                "evaluation_role": result["evaluation_role"],
                "applicability": result["applicability"],
                **result["direct_evaluation"],
                "artifact": f"{result['rule_family']}.json",
                "artifact_sha256": rule_artifact_sha256[result["rule_family"]],
            }
            for result in results
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_predictions(predictions_path, predictions)
    report_path.write_text(
        build_report(
            results,
            predictions,
            manifest_path=manifest_path,
        ),
        encoding="utf-8",
    )

    print(f"Created {summary_path}")
    for rule_path in rule_paths:
        print(f"Created {rule_path}")
    print(f"Created {predictions_path}")
    print(f"Created {report_path}")
    for result in results:
        direct = result["direct_evaluation"]
        print(
            f"{result['rule_family']}: mean fold Spearman="
            f"{direct['mean_fold_spearman']:.4f}; "
            "final-test targets used=False"
        )


if __name__ == "__main__":
    main()
