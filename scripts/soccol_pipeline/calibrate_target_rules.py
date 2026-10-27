"""Refit the existing nine Soccol formulas on reviewed frozen development rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from corrosion_datasets.datasets.soccol_pitting_potential import compare_target_rules as formulas
from scripts.epit_pipeline.artifact_hashes import load_frozen_split, sha256_file
from scripts.soccol_pipeline.data import REPO_ROOT, SPLIT_MANIFEST
from scripts.soccol_pipeline.formula_data import load_formula_data
from scripts.soccol_pipeline.target_rules import RULE_SELECTION, rename_terms


DEFAULT_OUTPUT = REPO_ROOT / "corrosion_datasets/analysis/soccol_pipeline/target_rules_v2"


def calibrate(development: pd.DataFrame, *, anchor_strength: float = 0.1):
    """Return full fold fits, all-development fits and held-out predictions."""
    selection = development.attrs["soccol_development_selection"]
    if (
        not development.outer_split.eq("development").all()
        or not development.composition_status.eq("accepted").all()
        or development.row_id.tolist() != selection["source_row_ids"]
        or development.task_row_index.tolist() != selection["source_task_row_indices"]
        or development.development_validation_fold.astype(int).tolist() != selection["validation_folds"]
    ):
        raise RuntimeError("Soccol calibration requires the unchanged accepted development selection.")
    if not np.isfinite(anchor_strength) or anchor_strength < 0:
        raise ValueError("anchor_strength must be finite and nonnegative.")
    specs = {spec.name: spec for spec in formulas.RULES}
    artifacts, predictions = {}, []
    for family, source_name in RULE_SELECTION:
        spec = specs[source_name]
        result, rows = formulas._evaluate_rule(development, spec, anchor_strength)
        folds = [{**fold, "coefficients": rename_terms(fold["coefficients"])} for fold in result["fold_results"]]
        fitted = result["all_development_calibration"]
        artifacts[family] = {
            "schema_version": "soccol_target_rule_v2",
            "rule_family": family, "source_rule_name": source_name,
            "description": spec.description, "evaluation_role": "candidate",
            "development_selection": selection,
            "source_sha256": selection["reviewed_data"]["files_sha256"]["soccol_processed.csv"],
            "split_manifest_sha256": selection["reviewed_data"]["split_manifest_sha256"],
            "split_lock_sha256": selection["reviewed_data"]["split_lock_sha256"],
            "development_rows": len(development), "final_test_targets_used": False,
            "formula": {
                "material": result["material_formula"], "halide": result["halide_formula"],
                "terms": list(rename_terms(dict.fromkeys(spec.term_names, 0.0))),
                "source_term_names": list(spec.term_names),
                "feature_state_term_names": "unchanged source formula names",
            },
            "coefficient_anchor": rename_terms(formulas._coefficient_dict(spec, spec.anchor)),
            "coefficient_upper_bounds": rename_terms(formulas._coefficient_dict(spec, spec.upper_bounds)),
            "anchor_strength": float(anchor_strength),
            "calibration_target": "standardized E_pit ranks",
            "coefficient_constraints": "nonnegative, sum to one, unchanged per-term upper bounds",
            "fold_calibrations": folds,
            "direct_evaluation": result["direct_evaluation"],
            "final_development_calibration": {
                **fitted, "coefficients": rename_terms(fitted["coefficients"]),
                "rows": len(development), "purpose": "synthetic-training coefficient center; not a final-test score",
            },
        }
        predictions.extend({"rule_family": family, **row} for row in rows)
        print(f"{family}: mean fold Spearman {result['direct_evaluation']['mean_fold_spearman']:.6f}", flush=True)
    return artifacts, pd.DataFrame(predictions)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_outputs(output_dir: Path, artifacts, predictions, development, frozen, anchor_strength):
    selection = development.attrs["soccol_development_selection"]
    records, metrics = [], []
    for family, artifact in artifacts.items():
        path = output_dir / f"{family}.json"
        _write_json(path, artifact)
        direct = artifact["direct_evaluation"]
        records.append({
            "rule_family": family, "source_rule_name": artifact["source_rule_name"],
            "evaluation_role": "candidate", "artifact": path.name, "artifact_sha256": sha256_file(path),
            "rows": len(development), "mean_fold_spearman": direct["mean_fold_spearman"],
            "standard_deviation_fold_spearman": direct["std_fold_spearman"],
            "minimum_fold_spearman": direct["minimum_fold_spearman"],
            "maximum_fold_spearman": direct["maximum_fold_spearman"],
            "pooled_out_of_fold_spearman": direct["pooled_oof_spearman"],
        })
        metrics.extend({"rule_family": family, **{k: v for k, v in fold.items() if k not in ("feature_state", "coefficients")}}
                       for fold in artifact["fold_calibrations"])
    score_sum = sum(record["mean_fold_spearman"] for record in records)
    for record in records:
        record["sampling_probability"] = record["mean_fold_spearman"] / score_sum
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(records).to_csv(output_dir / "rule_scores.csv", index=False)
    summary = {
        "schema_version": "soccol_target_rule_calibration_summary_v2",
        "development_selection": selection,
        "source_sha256": selection["reviewed_data"]["files_sha256"]["soccol_processed.csv"],
        "frozen_original_source_sha256": frozen.manifest["dataset"]["source_sha256"],
        "split_manifest_sha256": frozen.manifest_sha256, "split_lock_sha256": frozen.lock_sha256,
        "development_rows": len(development), "withheld_development_rows_excluded": selection["withheld_development_rows"],
        "final_test_rows_excluded": selection["final_query_rows_excluded"],
        "final_test_targets_used": False, "final_test_rows_excluded_before_calibration": True,
        "method": {
            "anchor_strength": anchor_strength, "coefficient_fit": "SLSQP on context target ranks; same nine formulas, anchors and bounds",
            "preprocessing": "context-only filling values and scaling for each validation fold; full accepted development for final coefficient centers",
            "missing_values": "context mean internally, zero only when a context field is entirely unobserved; saved inputs remain NaN",
            "scores": "unweighted mean of five validation-fold Spearman correlations",
        },
        "fold_assignments": [{
            "fold": fold,
            "context_task_row_indices": development.loc[development.development_validation_fold.ne(fold), "task_row_index"].tolist(),
            "validation_task_row_indices": development.loc[development.development_validation_fold.eq(fold), "task_row_index"].tolist(),
        } for fold in range(1, 6)],
        "rules": records,
        "output_sha256": {name: sha256_file(output_dir / name) for name in ("oof_predictions.csv", "fold_metrics.csv", "rule_scores.csv")},
    }
    _write_json(output_dir / "calibration_summary.json", summary)
    lines = ["# Reviewed Soccol formula calibration", "",
             "The same nine formulas were refitted on 2,950 accepted development rows. All 805 frozen final queries and 272 withheld development rows were excluded.", "",
             "Each fold fitted filling values, scaling and coefficients using its context only. Final coefficient centers used all accepted development rows. Formula definitions, anchors and bounds were unchanged. The scores below are development formula scores, not final-model scores.", "",
             "| Formula | Mean fold Spearman | Minimum fold | Maximum fold |", "|---|---:|---:|---:|"]
    lines.extend(f"| {r['rule_family']} | {r['mean_fold_spearman']:.6f} | {r['minimum_fold_spearman']:.6f} | {r['maximum_fold_spearman']:.6f} |" for r in records)
    lines.extend(["", "Per-fold filling/scaling states and coefficients were saved in each rule JSON. Held-out row IDs and predictions were saved in `oof_predictions.csv`.", ""])
    (output_dir / "README.md").write_text("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anchor-strength", type=float, default=0.1)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite existing calibration; use a new output directory.")
    development = load_formula_data()
    frozen = load_frozen_split(SPLIT_MANIFEST, expected_manifest_schema="soccol_composition_split_manifest_v1")
    artifacts, predictions = calibrate(development, anchor_strength=args.anchor_strength)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(args.output_dir, artifacts, predictions, development, frozen, args.anchor_strength)
    print(f"Saved reviewed coefficients and scores to {args.output_dir}")


if __name__ == "__main__":
    main()
