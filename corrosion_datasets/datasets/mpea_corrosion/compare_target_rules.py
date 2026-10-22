#!/usr/bin/env python3
"""Audit and fit MPEA rule candidates on the frozen development folds only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from corrosion_datasets.datasets.mpea_corrosion.prepare_composition_split import (
    DEFAULT_OUTPUT_DIR as SPLIT_DIR, ROOT, TARGET, load_frozen_mpea_split,
)
from scripts.epit_pipeline.artifact_hashes import sha256_file
from scripts.epit_pipeline.calibrate_target_rules import ManifestRows, _spearman, calibrate_family
from scripts.epit_pipeline.split_data import EpitDataset, Table
from scripts.mpea_pipeline.target_rules import MpeaReducedRule, get_mpea_rules


DEFAULT_OUTPUT = ROOT / "corrosion_datasets/analysis/mpea_target_rules_v1"
SOURCE_CONFIG = ROOT / "corrosion_datasets/datasets/combined_epit_soccol_mpea/processed/combined_prior_config.json"
ROW_AUDIT = ROOT / "corrosion_datasets/datasets/mpea_corrosion/processed/mpea_row_audit.csv"


def load_development(split_dir: Path = SPLIT_DIR) -> tuple[EpitDataset, ManifestRows, pd.DataFrame, Any]:
    frozen = load_frozen_mpea_split(split_dir)
    # Only parse the labeled development export. Final queries/targets are hash
    # checked by the frozen-split loader but never parsed into the fitting data.
    data = pd.read_csv(split_dir / "development.csv")
    assignments = pd.DataFrame(frozen.manifest["rows"])
    assignments = assignments.loc[assignments.outer_split.eq("development")]
    if data.row_id.tolist() != assignments.row_id.tolist():
        raise ValueError("Development export and frozen split rows differ.")
    if len(data) != frozen.manifest["split_design"]["development_rows"]:
        raise ValueError("Development row count differs from the frozen split.")
    for column in ("task_row_index", "composition_group", "optuna_validation_fold"):
        data[column] = assignments[column].to_numpy()
    features = frozen.manifest["dataset"]["feature_columns_in_order"]
    composition = frozen.manifest["dataset"]["model_visible_composition_columns"]
    for column in features:
        if column not in data:
            raise ValueError(f"Missing feature {column}")
    target = data[TARGET].to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("Development targets must be finite.")
    # Keep targets separate from the records accessed by formula preprocessing.
    records = data[features].to_dict("records")
    table = Table("mpea_corrosion", "frozen_development", features, {c: "material" for c in composition}, records)
    matrix = data[composition].to_numpy(dtype=float)
    dataset = EpitDataset(
        table, records, target, composition, np.nansum(matrix, axis=1),
        np.isnan(matrix).sum(axis=1), np.zeros(len(data), dtype=int), np.zeros(len(data), dtype=bool),
    )
    rows = ManifestRows(
        development=np.arange(len(data)), final_test=np.empty(0, dtype=int),
        validation_by_fold={fold: np.flatnonzero(data.optuna_validation_fold.to_numpy() == fold) for fold in range(1, 6)},
    )
    if sorted(np.concatenate(list(rows.validation_by_fold.values())).tolist()) != list(range(len(data))):
        raise ValueError("Five development folds must cover each development row exactly once.")
    return dataset, rows, data, frozen


def feature_coverage(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    records = []
    for column in features:
        values = data[column]
        record = {"feature": column, "observed_rows": int(values.notna().sum()), "missing_rows": int(values.isna().sum()), "distinct_observed_values": int(values.nunique())}
        for fold in range(1, 6):
            context = values[data.optuna_validation_fold.ne(fold)]
            record[f"fold_{fold}_context_observed"] = int(context.notna().sum())
            record[f"fold_{fold}_context_distinct"] = int(context.nunique())
        records.append(record)
    return pd.DataFrame(records)


def original_rule_audit(data: pd.DataFrame, rules: list[MpeaReducedRule]) -> list[dict[str, Any]]:
    config = json.loads(SOURCE_CONFIG.read_text())
    records = []
    for source in ("EPIT", "Soccol"):
        hp = config["regimes"][source]["fixed_hp"]
        for entry in hp["pitting_target_rule_scores"]:
            name = entry.split("=", 1)[0]
            required = ["Cr", "Mo", "CP_temp", "CP_Cl", "CP_pH"]
            if source == "EPIT":
                required.append("W")
            if name not in ("threshold_saturation", "coupled_breakdown", "fe_ni_cr_threshold"):
                required.append("N")
            if "mns" in name:
                required += ["Mn", "S"]
            if name == "pren_n_coupled_mns_weak_anions":
                required += ["CP_Br", "CP_SO4", "CP_NO3"]
            if name == "method_aware_pren_n":
                required.append("test_method")
            if name == "fe_ni_cr_threshold":
                required.append("Material class")
            missing = [column for column in required if column not in data or not data[column].notna().any()]
            adapted = [rule.name for rule in rules if rule.material.source == source.lower() and name in rule.source_rules]
            records.append({
                "source": source, "original_rule": name, "required_columns": required,
                "entirely_unrecorded_columns": missing,
                "directly_evaluable_unchanged": not missing,
                "temperature_observations": int(data.CP_temp.notna().sum()),
                "reduced_candidates": adapted,
                "decision": "Test reduced hypothesis; do not invent missing values." if adapted else "Omit class-specific threshold: MPEA has no comparable Fe/Ni class labels.",
            })
    return records


def evaluate_rule(rule: MpeaReducedRule, dataset: EpitDataset, rows: ManifestRows, data: pd.DataFrame, anchor_strength: float):
    result, predictions = calibrate_family(rule, dataset, rows, anchor_strength=anchor_strength)
    result["schema_version"] = "mpea_target_rule_screening_v1"
    anchor_metrics = []
    for fold_result in result["fold_calibrations"]:
        fold = fold_result["fold"]
        indices = rows.validation_by_fold[fold]
        terms = rule.transform_terms(dataset, indices, fold_result["feature_state"])
        anchor_prediction = terms @ rule.coefficient_anchor
        score = _spearman(dataset.target[indices], anchor_prediction)
        anchor_metrics.append(score)
        fold_result["anchor_spearman_before_coefficient_fit"] = score
    result["direct_evaluation"]["mean_fold_anchor_spearman"] = float(np.mean(anchor_metrics))
    for prediction in predictions:
        local = prediction["task_row_index"]
        prediction["development_row_index"] = local
        prediction["task_row_index"] = int(data.iloc[local].task_row_index)
        prediction["row_id"] = data.iloc[local].row_id
        prediction["composition_group"] = data.iloc[local].composition_group
        prediction["chloride_observed"] = bool(pd.notna(data.iloc[local].CP_Cl))
    by_fold = []
    for fold in range(1, 6):
        subset = [row for row in predictions if row["optuna_validation_fold"] == fold and row["chloride_observed"]]
        by_fold.append({
            "fold": fold, "rows": len(subset),
            "spearman": _spearman(np.asarray([row["observed_epit_mV_SCE"] for row in subset]), np.asarray([row["out_of_fold_rule_score"] for row in subset])),
        })
    result["observed_chloride_subset_diagnostic"] = {
        "scope": "Same full-development OOF fits; score only rows with recorded chloride, without refitting or resplitting.",
        "rows": int(data.CP_Cl.notna().sum()), "folds": by_fold,
        "mean_fold_spearman": float(np.mean([row["spearman"] for row in by_fold])),
    }
    result["production_registry_changed"] = False
    return result, predictions


def assess(results: list[dict[str, Any]], rules: list[MpeaReducedRule]) -> dict[str, Any]:
    by_name = {result["rule_family"]: result for result in results}
    comparisons = []
    for rule in rules:
        if rule.environment == "material_only":
            continue
        candidate = by_name[rule.name]
        baseline = by_name[f"{rule.material.name}_material_only"]
        differences = [a["spearman"] - b["spearman"] for a, b in zip(candidate["fold_calibrations"], baseline["fold_calibrations"], strict=True)]
        comparisons.append({
            "rule": rule.name, "composition_only_control": baseline["rule_family"],
            "mean_fold_gain_over_composition_only": float(np.mean(differences)),
            "folds_improved": sum(value > 1e-12 for value in differences), "fold_gains": differences,
        })
    candidates = [result for result in results if result["evaluation_role"] == "MPEA development candidate"]
    best = max(candidates, key=lambda result: result["direct_evaluation"]["mean_fold_spearman"])
    return {
        "best_reduced_candidate": best["rule_family"],
        "selection_metric": "mean of the five development-fold Spearman correlations",
        "comparisons_to_composition_only": comparisons,
        "production_promotion": "None in this screening step; reduced formulas and their full-development coefficient centers are saved for review.",
        "unavailable_effects": "Do not include N/S/pH/method/extra-ion effects as independently fitted MPEA terms; those fields are unrecorded.",
        "sparse_temperature": "Four development observations, with only one observed context value in fold 2; omitted from this first screen.",
        "deduplication": "Original rules that differ only in omitted terms collapse to the same reduced candidate; they are not counted as independent successes.",
        "recommendation": "Keep the simple Cr-Mo-W/chloride rule as a reference. Investigate material effects outside Cr/Mo/W before promoting an MPEA prior; fitting these inherited terms alone gives weak and unstable development rankings.",
    }


def diagnose_best_rule(predictions: list[dict[str, Any]], data: pd.DataFrame, name: str):
    selected = pd.DataFrame([row for row in predictions if row["rule_family"] == name])
    selected = selected.merge(data[["row_id", "Cr", "Mo", "W", "Ti", "Nb", "Al"]], on="row_id", validate="one_to_one")
    # Read explanatory metadata only, never the raw-audit response columns.
    audit = pd.read_csv(ROW_AUDIT, usecols=["row_id", "alloy_name_raw", "electrolyte_raw", "test_environment_raw", "source_dois"])
    selected = selected.merge(audit, on="row_id", validate="one_to_one")
    subgroup_rows = []
    for (fold, observed), subset in selected.groupby(["optuna_validation_fold", "chloride_observed"]):
        subgroup_rows.append({
            "fold": int(fold), "chloride_observed": bool(observed), "rows": len(subset),
            "spearman": _spearman(subset.observed_epit_mV_SCE.to_numpy(), subset.out_of_fold_rule_score.to_numpy()),
            "median_target_mV_SCE": float(subset.observed_epit_mV_SCE.median()),
            "median_rule_score": float(subset.out_of_fold_rule_score.median()),
        })
    groups = []
    for (fold, group), subset in selected.groupby(["optuna_validation_fold", "composition_group"]):
        groups.append({
            "fold": int(fold), "composition_group": group, "rows": len(subset),
            "alloy_names": sorted(set(subset.alloy_name_raw.astype(str).str.strip())),
            "electrolytes": sorted(set(subset.electrolyte_raw.astype(str).str.strip())),
            "median_target_mV_SCE": float(subset.observed_epit_mV_SCE.median()),
            "median_rule_score": float(subset.out_of_fold_rule_score.median()),
            "mean_Cr_Mo_W_wt_percent": float(subset[["Cr", "Mo", "W"]].sum(axis=1).mean()),
            "mean_Ti_Nb_wt_percent": float(subset[["Ti", "Nb"]].sum(axis=1).mean()),
        })
    fold_scores = {int(f): _spearman(g.observed_epit_mV_SCE.to_numpy(), g.out_of_fold_rule_score.to_numpy()) for f, g in selected.groupby("optuna_validation_fold")}
    worst_fold = min(fold_scores, key=fold_scores.get)
    largest_group = max((group for group in groups if group["fold"] == worst_fold), key=lambda group: group["rows"])
    return selected, pd.DataFrame(subgroup_rows), {
        "rule": name, "fold_spearman": fold_scores, "worst_fold": worst_fold,
        "largest_composition_group_in_worst_fold": largest_group,
        "scope": "Descriptive analysis of saved development predictions only. No rows excluded, no folds changed, no new fits.",
        "interpretation": "Cr/Mo/W-only material scores cannot distinguish protective effects of other alloy components. Missing chloride and different electrolytes also coincide with these groups; this does not isolate their separate causal effects or prove the cause of a trained model's performance.",
        "groups": groups,
    }


def write_report(output: Path, summary: pd.DataFrame, assessment: dict[str, Any], diagnosis: dict[str, Any]) -> None:
    best = summary.loc[summary.rule_family.eq(assessment["best_reduced_candidate"])].iloc[0]
    text = [
        "# MPEA formula investigation\n",
        "Uses only the 214 frozen development rows, with five composition-grouped folds. The 53 final-test targets are not parsed, fitted or scored.\n",
        "## Method\n",
        "The EPIT `calibrate_family` and coefficient fitter are reused: fit imputation, all scaling and constrained coefficients on four development folds; score the fifth. Coefficients are nonnegative, sum to one, use inherited bounds, and are regularized toward the retained physics anchor with strength 0.1. Full-development coefficient centers are fitted only after collecting fold results. These are dimensionless ranking scores, not predictions in mV.\n",
        "All 18 production source/rule combinations need inputs absent from MPEA. Reduced candidates explicitly remove unavailable terms. N or S is not assumed physically zero; pH and other ions are not invented. The common Fe/Ni hypotheses are being screened on MPEA, not declared chemically validated for every alloy.\n",
        "Temperature has 4 observations; pH, nitrogen, sulfur, scan rate and the extra anions are unrecorded. Chloride is recorded for 151/214 rows, at 3 distinct concentrations. Unknown chloride uses the current context mean, as in EPIT; it does not mean zero chloride. A separate score on recorded-chloride rows checks sensitivity without changing the folds.\n",
        "The eleven reduced candidates and eight matching composition-only controls were defined before their scores were inspected. EPIT and Soccol variants retain their respective material expressions (including EPIT's W term). Unsupported Mn-S, Mo-N-acid, method and anion additions collapse to simpler rules rather than receiving artificial fitted coefficients.\n",
        "## Development results\n",
        f"Best reduced candidate: **{best.rule_family}**, mean fold Spearman **{best.mean_fold_spearman:.4f}**. This is formula screening, not trained-TabICL performance or a final-test result.\n",
        "| Rule | Mean fold Spearman | Fold SD | Before coefficient fit | Recorded chloride only |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        text.append(f"| {row.rule_family} | {row.mean_fold_spearman:.4f} | {row.std_fold_spearman:.4f} | {row.anchor_mean_fold_spearman:.4f} | {row.observed_chloride_mean_fold_spearman:.4f} |")
    text += ["\n## Does chloride add useful information?\n", "Each candidate is compared with its own material formula alone on exactly the same validation rows.\n", "| Rule | Mean gain over composition only | Folds improved |", "| --- | ---: | ---: |"]
    for comparison in assessment["comparisons_to_composition_only"]:
        text.append(f"| {comparison['rule']} | {comparison['mean_fold_gain_over_composition_only']:+.4f} | {comparison['folds_improved']}/5 |")
    group = diagnosis["largest_composition_group_in_worst_fold"]
    text += [
        "\n## Where the inherited formulas fail\n",
        f"The best rule's weakest fold is fold {diagnosis['worst_fold']} (Spearman {diagnosis['fold_spearman'][diagnosis['worst_fold']]:.4f}). Its largest composition group contains {group['rows']} rows: {', '.join(group['alloy_names'])}, tested in {', '.join(group['electrolytes'])}. Their median measured potential is {group['median_target_mV_SCE']:.0f} mV SCE, but their median formula score is {group['median_rule_score']:.3f}. They contain {group['mean_Cr_Mo_W_wt_percent']:.2f} wt.% Cr+Mo+W, versus {group['mean_Ti_Nb_wt_percent']:.2f} wt.% Ti+Nb. The inherited material formula responds only to Cr/Mo/W, so these other composition differences have no effect on its material score.\n",
        "Recorded-chloride rows give a much stronger ranking than the full mixture for the leading rules, but this is a different subset, not proof that filling chloride differently would fix the problem. Alloy chemistry and electrolyte differ together. Composition-only controls also fail in the weakest fold, so missing chloride alone cannot explain this failure. All original development rows remain in the main scores.\n",
        "## Decision from this screen\n",
        assessment["recommendation"] + "\n",
        "Do not add the unavailable N, Mn-S, acidic-pH or extra-ion effects to the MPEA fitting stage. Do not count their reduced copies as additional independent prior families. The simple EPIT-derived Cr-Mo-W/chloride rule is a useful reference, but its chloride addition improves only two of five folds. Threshold and interaction complexity do not establish a reliable improvement. This screen does not yet justify launching Optuna with the inherited formulas unchanged.\n",
    ]
    text += [
        "\n## Saved artifacts and interpretation\n",
        "`feature_coverage.csv` and `original_rule_audit.json` document missing inputs. `summary.csv`, `fold_metrics.csv` and `oof_predictions.csv` contain the development comparisons. Each file in `rules/` includes the formula, five fitted coefficient vectors, context preprocessing states, and a coefficient center fitted on all 214 development rows. `comparison_summary.json` records provenance, settings and recommendations; `artifact_lock.json` hashes the outputs.\n",
        "A higher mean here supports further development, not a guarantee that the informed transformer will beat its baselines. Coefficient fitting optimizes regularized squared error against standardized target ranks, as in EPIT/Soccol; it does not directly maximize held-out Spearman and can lower it. The best candidate was selected among several hypotheses on these same development folds. Sparse effects and absent features cannot be validated by this comparison.\n",
        "Existing EPIT/Soccol formulas, production registries and MPEA split artifacts are unchanged. No training or Optuna is launched.\n",
        "Reproduce from the repository root (use a new output folder, or `--force` to replace this report):\n",
        "```bash\n.venv/bin/python -m corrosion_datasets.datasets.mpea_corrosion.compare_target_rules\n```\n",
    ]
    (output / "README.md").write_text("\n".join(text))


def run(output: Path, *, split_dir: Path = SPLIT_DIR, force: bool = False) -> pd.DataFrame:
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"Output already exists: {output}; use a new folder or --force.")
    dataset, rows, data, frozen = load_development(split_dir)
    rules = get_mpea_rules()
    coverage = feature_coverage(data, frozen.manifest["dataset"]["feature_columns_in_order"])
    audit = original_rule_audit(data, rules)
    results, predictions, fold_metrics = [], [], []
    for rule in rules:
        result, rule_predictions = evaluate_rule(rule, dataset, rows, data, 0.1)
        results.append(result)
        predictions.extend(rule_predictions)
        for fold in result["fold_calibrations"]:
            fold_metrics.append({"rule_family": rule.name, **{k: v for k, v in fold.items() if k not in ("feature_state", "coefficients")}})
        print(f"{rule.name}: {result['direct_evaluation']['mean_fold_spearman']:.4f}", flush=True)
    summary = pd.DataFrame([{
        "rule_family": result["rule_family"], "role": result["evaluation_role"],
        "mean_fold_spearman": result["direct_evaluation"]["mean_fold_spearman"],
        "std_fold_spearman": result["direct_evaluation"]["standard_deviation_fold_spearman"],
        "minimum_fold_spearman": result["direct_evaluation"]["minimum_fold_spearman"],
        "pooled_oof_spearman": result["direct_evaluation"]["pooled_out_of_fold_spearman"],
        "anchor_mean_fold_spearman": result["direct_evaluation"]["mean_fold_anchor_spearman"],
        "observed_chloride_mean_fold_spearman": result["observed_chloride_subset_diagnostic"]["mean_fold_spearman"],
    } for result in results]).sort_values("mean_fold_spearman", ascending=False)
    assessment = assess(results, rules)
    diagnostic_rows, subgroup_metrics, diagnosis = diagnose_best_rule(predictions, data, assessment["best_reduced_candidate"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "rules").mkdir(exist_ok=True)
    json_write = lambda path, value: path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    for result in results:
        json_write(output / "rules" / f"{result['rule_family']}.json", result)
    coverage.to_csv(output / "feature_coverage.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "oof_predictions.csv", index=False)
    diagnostic_rows.to_csv(output / "best_rule_predictions_with_metadata.csv", index=False)
    subgroup_metrics.to_csv(output / "best_rule_chloride_subgroups.csv", index=False)
    json_write(output / "failure_diagnosis.json", diagnosis)
    json_write(output / "original_rule_audit.json", audit)
    code_paths = [Path(__file__), ROOT / "scripts/mpea_pipeline/target_rules.py", ROOT / "scripts/epit_pipeline/calibrate_target_rules.py", ROOT / "scripts/epit_pipeline/target_rules.py", ROOT / "corrosion_datasets/datasets/mpea_corrosion/prepare_composition_split.py"]
    json_write(output / "comparison_summary.json", {
        "schema_version": "mpea_target_rules_comparison_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_rows": len(data), "final_test_rows_used": 0, "fold_sizes": {str(k): len(v) for k, v in rows.validation_by_fold.items()},
        "split_manifest_sha256": frozen.manifest_sha256, "split_lock_sha256": frozen.lock_sha256,
        "development_sha256": sha256_file(split_dir / "development.csv"), "source_config_sha256": sha256_file(SOURCE_CONFIG),
        "row_audit_sha256": sha256_file(ROW_AUDIT),
        "implementation_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in code_paths},
        "anchor_strength": 0.1, "coefficient_fit": "Unchanged EPIT fitter: nonnegative simplex with inherited caps; rank-target MSE plus anchor penalty.",
        "candidate_count": 11, "composition_only_controls": 8, "production_registry_changed": False,
        "results": results, "assessment": assessment,
    })
    write_report(output, summary, assessment, diagnosis)
    artifacts = [path for path in output.rglob("*") if path.is_file() and path.name != "artifact_lock.json"]
    json_write(output / "artifact_lock.json", {str(path.relative_to(output)): sha256_file(path) for path in sorted(artifacts)})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, split_dir=args.split_dir, force=args.force)


if __name__ == "__main__":
    main()
