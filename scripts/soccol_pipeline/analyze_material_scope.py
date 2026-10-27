"""Describe Soccol's accepted material families and saved development scores.

This reproduces the v2 observation-weighted sampling analysis. It does not
change family labels, sampling, rules, coefficients or frozen partitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.epit_pipeline.artifact_hashes import load_frozen_split, sha256_file
from scripts.epit_pipeline.formula_data import load_formula_dataset
from scripts.soccol_pipeline.data import REPO_ROOT, PROCESSED_DIR, load_soccol_dataset, select_accepted_development
from tabicl.prior.epit_composition_profile import EPIT_COMPOSITION_FAMILY_PROBS
from tabicl.prior.epit_feature_profile import load_epit_feature_profile
from tabicl.prior.soccol_feature_profile import load_soccol_feature_profile


ANALYSIS = REPO_ROOT / "corrosion_datasets/analysis"
SOCCOL_RESULTS = ANALYSIS / "soccol_pipeline/target_rules_v2"
EPIT_RESULTS = ANALYSIS / "epit_pipeline/target_rules_v4"
DEFAULT_OUTPUT = ANALYSIS / "soccol_pipeline/material_scope_audit_v1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def family_scores(predictions, dataset):
    """Average within-family fold correlations; retain small/absent-fold status."""
    folds, scores = [], []
    for (rule, family), group in predictions.groupby(["rule_family", "material_family"]):
        valid = []
        for fold in range(1, 6):
            rows = group.loc[group.fold.eq(fold)]
            status = ("fewer_than_three_rows" if len(rows) < 3 else
                      "constant_target" if rows.target.nunique() < 2 else
                      "constant_prediction" if rows.prediction.nunique() < 2 else "scored")
            score = float(spearmanr(rows.target, rows.prediction).statistic) if status == "scored" else np.nan
            if status == "scored":
                valid.append(score)
            folds.append({"dataset": dataset, "rule_family": rule, "material_family": family,
                          "fold": fold, "rows": len(rows), "status": status, "spearman": score})
        scores.append({"dataset": dataset, "rule_family": rule, "material_family": family,
                       "rows": len(group), "valid_folds": len(valid),
                       "mean_valid_fold_spearman": float(np.mean(valid)) if valid else np.nan,
                       "minimum_valid_fold_spearman": min(valid) if valid else np.nan,
                       "maximum_valid_fold_spearman": max(valid) if valid else np.nan})
    return pd.DataFrame(scores), pd.DataFrame(folds)


def analyze():
    dataset = load_soccol_dataset()
    development = select_accepted_development(dataset)
    profile = load_soccol_feature_profile("soccol_pitting_features_v2")
    require(profile.source_task_row_indices.tolist() == development.task_row_indices.tolist(),
            "Soccol sampling bank and accepted development rows differ.")
    rows = development.rows.copy()
    elements = dataset.registry["composition_columns"]
    amounts = rows[elements].to_numpy()
    require(np.all((amounts == amounts.max(axis=1, keepdims=True)).sum(axis=1) == 1),
            "Largest-element diagnostic needs explicit handling for tied amounts.")
    rows["largest_element"] = rows[elements].idxmax(axis=1)
    rows["task_row_index"] = development.task_row_indices
    counts = []
    for family, group in rows.groupby("row_material_family"):
        counts.append({
            "material_family": family, "development_rows": len(group),
            "distinct_compositions": len(group[elements].drop_duplicates()), "source_count": group.source.nunique(),
            "composition_draw_probability": len(group) / len(rows),
            **{f"largest_element_{e}_rows": int(group.largest_element.eq(e).sum()) for e in ("Fe", "Ni", "Cr")},
            **{f"fold_{f}_rows": int(group.development_validation_fold.eq(f).sum()) for f in range(1, 6)},
        })
    counts = pd.DataFrame(counts).sort_values("development_rows", ascending=False)
    partitions = dataset.rows.groupby(["outer_split", "composition_status", "row_material_family"]).size().rename("rows").reset_index()
    exceptions = rows.loc[rows.row_material_family.str.startswith("fe_based") & rows.largest_element.ne("Fe"),
                          ["row_id", "task_row_index", "source", "row_material_family", "Fe", "Cr", "Ni", "Mo", "largest_element"]]

    summary = json.loads((SOCCOL_RESULTS / "calibration_summary.json").read_text())
    require(summary["development_selection"] == development.identity, "Saved Soccol formula selection differs.")
    predictions_path = SOCCOL_RESULTS / "oof_predictions.csv"
    require(sha256_file(predictions_path) == summary["output_sha256"][predictions_path.name], "Soccol predictions changed.")
    predictions = pd.read_csv(predictions_path, float_precision="round_trip")
    expected = rows.set_index("row_id")
    for rule in summary["rules"]:
        predicted = predictions.loc[predictions.rule_family.eq(rule["rule_family"])]
        require(predicted.row_id.is_unique and set(predicted.row_id) == set(expected.index), "Soccol predicted row membership differs.")
        source = expected.loc[predicted.row_id]
        require(np.array_equal(predicted.fold, source.development_validation_fold), "Soccol prediction folds differ.")
        require(np.array_equal(predicted.task_row_index, source.task_row_index), "Soccol prediction indices differ.")
        require(np.array_equal(predicted.observed_E_pit, source.E_pit), "Soccol targets differ.")
        mean = np.mean([spearmanr(g.observed_E_pit, g.oof_rule_score).statistic for _, g in predicted.groupby("fold")])
        require(abs(mean - rule["mean_fold_spearman"]) < 1e-12, "Soccol saved score did not reproduce.")
    predictions["material_family"] = predictions.row_id.map(expected.row_material_family)
    soccol_scores, soccol_folds = family_scores(predictions.rename(columns={"observed_E_pit": "target", "oof_rule_score": "prediction"}), "soccol")

    epit_data, _ = load_formula_dataset()
    epit_summary = json.loads((EPIT_RESULTS / "calibration_summary.json").read_text())
    frozen = load_frozen_split(ANALYSIS / "epit_pipeline/splits_v2/split_manifest.json")
    require(frozen.manifest_sha256 == epit_summary["split_manifest_sha256"] and frozen.lock_sha256 == epit_summary["split_lock_sha256"],
            "EPIT formula results use a different frozen split.")
    records = {r["task_row_index"]: r for r in frozen.manifest["rows"]}
    eligible = {i for i, r in records.items() if r["outer_split"] == "development"
                and epit_data.rows[i]["Material class"] in ("Fe Alloy", "NiCrMo Alloy")}
    epit_predictions = pd.read_csv(EPIT_RESULTS / "direct_evaluation_predictions.csv", float_precision="round_trip")
    for rule in epit_summary["rules"]:
        group = epit_predictions.loc[epit_predictions.rule_family.eq(rule["rule_family"])]
        require(group.task_row_index.is_unique and set(group.task_row_index) == eligible, "Unexpected EPIT formula scoring rows.")
        require(group.optuna_validation_fold.tolist() == [records[i]["optuna_validation_fold"] for i in group.task_row_index],
                "EPIT formula folds differ from the frozen development folds.")
        mean = np.mean([spearmanr(g.observed_epit_mV_SCE, g.out_of_fold_rule_score).statistic
                        for _, g in group.groupby("optuna_validation_fold")])
        require(abs(mean - rule["mean_fold_spearman"]) < 1e-12, "EPIT saved score did not reproduce.")
    epit_predictions["material_family"] = [epit_data.rows[i]["Material class"] for i in epit_predictions.task_row_index]
    epit_scores, epit_folds = family_scores(epit_predictions.rename(columns={
        "optuna_validation_fold": "fold", "observed_epit_mV_SCE": "target", "out_of_fold_rule_score": "prediction"}), "epit")
    epit_profile = load_epit_feature_profile()
    active = np.asarray([EPIT_COMPOSITION_FAMILY_PROBS[epit_profile.composition_profile.families.index(f)] for f in epit_profile.families])
    active /= active.sum()
    input_paths = [Path(__file__), PROCESSED_DIR / "soccol_processed.csv", PROCESSED_DIR / "feature_registry.json",
                   SOCCOL_RESULTS / "calibration_summary.json", predictions_path,
                   EPIT_RESULTS / "calibration_summary.json", EPIT_RESULTS / "direct_evaluation_predictions.csv",
                   frozen.manifest_path, frozen.lock_path,
                   REPO_ROOT / "src/tabicl/prior/soccol_feature_generator.py",
                   REPO_ROOT / "src/tabicl/prior/epit_composition_profile.py",
                   REPO_ROOT / "src/tabicl/prior/epit_feature_generator.py",
                   REPO_ROOT / "src/tabicl/prior/dataset.py"]
    info = {
        "schema_version": "soccol_material_scope_audit_v1", "accepted_development_rows": len(rows),
        "withheld_development_rows": development.identity["withheld_development_rows"],
        "frozen_final_queries_excluded_from_scores": development.identity["final_query_rows_excluded"],
        "distinct_compositions": len(rows[elements].drop_duplicates()),
        "largest_element_counts": rows.largest_element.value_counts().to_dict(),
        "stored_family_label_caveats": len(exceptions),
        "historical_scope_flags": rows.material_scope_primary.value_counts().to_dict(),
        "soccol_selection": "All accepted development observations, regardless of historical include/review/exclude metadata.",
        "soccol_sampling": "Uniform composition observation draws; independent uniform joint environment observation draws; no material-family routing.",
        "soccol_nickel_label_probability": float(rows.row_material_family.str.startswith("ni_based").mean()),
        "epit_composition_family_probabilities": dict(zip(epit_profile.families, active.tolist())),
        "epit_distinct_composition_templates_by_family": dict(zip(epit_profile.families, np.bincount(epit_profile.composition_family_indices).tolist())),
        "epit_environment_rows_by_family": epit_profile.metadata["family_environment_counts"],
        "score_definition": "Unweighted mean of within-family Spearman over valid development folds. Folds with fewer than three rows or constant targets/predictions were not scored. Existing globally fitted coefficients were retained.",
        "comparison_limit": "EPIT and Soccol subgroup rows and contexts differ. Their scores describe each workflow, not a matched experiment isolating a cause.",
        "recommendation": "Retain current accepted-material scope for this run. Do not exclude nickel or automatically apply historical exclusion flags. Small nickel groups warrant formula-specific investigation, not an unsupported scope change.",
        "final_query_predictions_used": False, "refitting_performed": False,
        "source_sha256": {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in input_paths},
    }
    return info, {
        "accepted_family_counts.csv": counts, "partition_family_counts.csv": partitions,
        "family_label_caveats.csv": exceptions,
        "formula_family_scores.csv": pd.concat([soccol_scores, epit_scores], ignore_index=True),
        "formula_family_fold_scores.csv": pd.concat([soccol_folds, epit_folds], ignore_index=True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), "Choose a new --output-dir; existing analyses are not overwritten.")
    summary, tables = analyze()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(args.output_dir / name, index=False)
    summary["output_sha256"] = {name: sha256_file(args.output_dir / name) for name in tables}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(tables["accepted_family_counts.csv"].to_string(index=False))
    print(f"Saved material-scope analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
