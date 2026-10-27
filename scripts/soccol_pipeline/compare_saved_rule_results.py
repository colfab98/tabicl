"""Compare saved Soccol formula results, including identical validation rows.

No coefficients or models are fitted. The old predictions retain their old
context fits; restricting the scored rows does not refit those contexts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from scripts.epit_pipeline.artifact_hashes import sha256_file
from scripts.soccol_pipeline.data import (
    REPO_ROOT, SPLIT_MANIFEST, load_soccol_dataset, select_accepted_development,
)
from scripts.soccol_pipeline.target_rules import RULE_SELECTION


ANALYSIS = REPO_ROOT / "corrosion_datasets/analysis"
OLD = ANALYSIS / "soccol_pipeline/target_rules_v1"
NEW = ANALYSIS / "soccol_pipeline/target_rules_v2"
HISTORICAL = ANALYSIS / "soccol_target_rules_v1"
DEFAULT_OUTPUT = ANALYSIS / "soccol_pipeline/target_rule_comparison_v1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def compare():
    dataset = load_soccol_dataset()
    accepted = select_accepted_development(dataset)
    old = json.loads((OLD / "calibration_summary.json").read_text())
    new = json.loads((NEW / "calibration_summary.json").read_text())
    historical = json.loads((HISTORICAL / "comparison_summary.json").read_text())
    require(old["final_test_targets_used"] is False and new["final_test_targets_used"] is False,
            "Expected development-only formula results.")
    require(new["development_selection"] == accepted.identity, "Reviewed row selection changed.")
    for key in ("split_manifest_sha256", "split_lock_sha256"):
        require(old[key] == new[key] == dataset.identity[key], f"Different frozen split: {key}")
    require(old["source_sha256"] == new["frozen_original_source_sha256"] == historical["inputs"]["data_sha256"],
            "Historical results refer to different source files.")
    require(new["source_sha256"] == dataset.identity["files_sha256"]["soccol_processed.csv"],
            "Reviewed results refer to different processed inputs.")
    source_paths = [OLD / "calibration_summary.json", NEW / "calibration_summary.json",
                    HISTORICAL / "comparison_summary.json", HISTORICAL / "oof_predictions.csv",
                    NEW / "oof_predictions.csv", SPLIT_MANIFEST,
                    SPLIT_MANIFEST.parent / "split_assignments.csv", Path(__file__)]
    require(sha256_file(NEW / "oof_predictions.csv") == new["output_sha256"]["oof_predictions.csv"],
            "Reviewed prediction file hash differs from its saved summary.")
    a = pd.read_csv(HISTORICAL / "oof_predictions.csv", float_precision="round_trip")
    b = pd.read_csv(NEW / "oof_predictions.csv", float_precision="round_trip")
    assignments = dataset.assignments.set_index("raw_workbook_row")
    old_ids = set(assignments.index[assignments.outer_split.eq("development")])
    new_ids = set(accepted.rows.raw_workbook_row)
    old_rules = {r["rule_family"]: r for r in old["rules"]}
    new_rules = {r["rule_family"]: r for r in new["rules"]}
    historical_rules = {r["rule"]: r for r in historical["rules"]}
    records, folds = [], []
    max_target_roundoff = 0.
    for family, source_name in RULE_SELECTION:
        for directory, record in ((OLD, old_rules[family]), (NEW, new_rules[family])):
            path = directory / record["artifact"]
            require(sha256_file(path) == record["artifact_sha256"], f"Changed formula artifact: {path}")
            source_paths.append(path)
        old_artifact = json.loads((OLD / old_rules[family]["artifact"]).read_text())
        new_artifact = json.loads((NEW / new_rules[family]["artifact"]).read_text())
        require(all(old_artifact["formula"][k] == new_artifact["formula"][k]
                    for k in ("material", "halide", "terms")), f"Formula definition changed: {family}")
        previous = a.loc[a.rule.eq(source_name)].copy()
        reviewed = b.loc[b.rule_family.eq(family)].copy()
        for frame, ids in ((previous, old_ids), (reviewed, new_ids)):
            require(frame.raw_workbook_row.is_unique and set(frame.raw_workbook_row) == ids,
                    f"Missing, repeated or unexpected development predictions: {family}")
            expected = assignments.loc[frame.raw_workbook_row]
            require(np.array_equal(frame.fold, expected.development_validation_fold), f"Changed folds: {family}")
            require(np.array_equal(frame.composition_group, expected.composition_group), f"Changed groups: {family}")
        require(np.array_equal(reviewed.task_row_index, assignments.loc[reviewed.raw_workbook_row, "task_row_index"]),
                f"Changed task indices: {family}")
        require(reviewed.row_id.tolist() == [f"soccol_r{int(i):06d}" for i in reviewed.raw_workbook_row],
                f"Changed reviewed row IDs: {family}")
        paired = previous.merge(reviewed, on="raw_workbook_row", suffixes=("_old", "_new"), validate="one_to_one")
        require(len(paired) == len(new_ids), f"Incomplete matched validation rows: {family}")
        require(np.allclose(paired.observed_E_pit_old, paired.observed_E_pit_new, rtol=0., atol=1e-10),
                f"Targets changed: {family}")
        max_target_roundoff = max(max_target_roundoff, float(np.max(np.abs(paired.observed_E_pit_old - paired.observed_E_pit_new))))
        family_folds = []
        for fold in range(1, 6):
            full = previous.loc[previous.fold.eq(fold)]
            common = paired.loc[paired.fold_new.eq(fold)]
            require(np.array_equal(rankdata(common.observed_E_pit_old), rankdata(common.observed_E_pit_new)),
                    f"Target ranks changed: {family}, fold {fold}")
            old_score = float(spearmanr(full.observed_E_pit, full.oof_rule_score).statistic)
            # Both paired scores use exactly the same target values and row IDs.
            old_common = float(spearmanr(common.observed_E_pit_new, common.oof_rule_score_old).statistic)
            new_score = float(spearmanr(common.observed_E_pit_new, common.oof_rule_score_new).statistic)
            saved_old = next(f for f in historical_rules[source_name]["fold_results"] if f["fold"] == fold)
            saved_new = next(f for f in new_artifact["fold_calibrations"] if f["fold"] == fold)
            require(abs(old_score - saved_old["spearman"]) < 1e-12 and abs(new_score - saved_new["spearman"]) < 1e-12,
                    f"Saved fold scores did not reproduce: {family}, fold {fold}")
            row = {
                "rule_family": family, "fold": fold,
                "old_context_rows": len(old_ids) - len(full), "reviewed_context_rows": len(new_ids) - len(common),
                "old_validation_rows": len(full), "matched_validation_rows": len(common),
                "old_full_spearman": old_score, "old_matched_spearman": old_common,
                "reviewed_spearman": new_score,
                "delta_original_populations": new_score - old_score,
                "delta_matched_validation_rows": new_score - old_common,
            }
            family_folds.append(row)
            folds.append(row)
        means = pd.DataFrame(family_folds).mean(numeric_only=True)
        for key, value in (("old_full_spearman", old_rules[family]["mean_fold_spearman"]),
                           ("reviewed_spearman", new_rules[family]["mean_fold_spearman"])):
            require(abs(means[key] - value) < 1e-12, f"Saved mean score did not reproduce: {family}")
        records.append({
            "rule_family": family, "source_rule_name": source_name,
            "old_rows": len(old_ids), "matched_rows": len(new_ids),
            **{k: float(means[k]) for k in ("old_full_spearman", "old_matched_spearman", "reviewed_spearman",
                                           "delta_original_populations", "delta_matched_validation_rows")},
        })
    scores = pd.DataFrame(records)
    summary = {
        "schema_version": "soccol_saved_formula_comparison_v1",
        "metric": "unweighted mean of five development-fold Spearman correlations",
        "old_development_rows": len(old_ids), "reviewed_development_rows": len(new_ids),
        "frozen_final_queries_excluded": int(dataset.assignments.outer_split.eq("final_test").sum()),
        "refitting_performed": False, "final_query_predictions_used": False,
        "matched_comparison": "Saved old predictions re-scored on exactly the reviewed validation rows. Old contexts and fitted coefficients were retained.",
        "interpretation_limit": "The matched comparison controls validation membership, but context membership, preprocessing, scaling and fitted coefficients differ. It does not isolate the effect of cleaning alone or establish a trained-model improvement.",
        "target_csv_max_roundoff": max_target_roundoff, "matched_target_ranks_identical": True,
        "all_45_old_and_45_new_saved_fold_scores_reproduced": True,
        "mean_across_nine_formulas": {k: float(scores[k].mean()) for k in (
            "old_full_spearman", "old_matched_spearman", "reviewed_spearman", "delta_matched_validation_rows")},
        "improved_on_matched_rows": int(scores.delta_matched_validation_rows.gt(0).sum()),
        "declined_on_matched_rows": int(scores.delta_matched_validation_rows.lt(0).sum()),
        "source_sha256": {str(p.relative_to(REPO_ROOT)): sha256_file(p) for p in source_paths},
    }
    return scores, pd.DataFrame(folds), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()),
            "Refusing to overwrite an existing comparison; choose a new --output-dir.")
    scores, folds, summary = compare()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "comparison_vs_v1.csv", index=False)
    folds.to_csv(args.output_dir / "fold_comparison_vs_v1.csv", index=False)
    summary["output_sha256"] = {name: sha256_file(args.output_dir / name)
                                for name in ("comparison_vs_v1.csv", "fold_comparison_vs_v1.csv")}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    lines = [
        "# Soccol: previous and reviewed formula scores", "",
        "Reused existing saved results and predictions; no training or coefficient fitting was run. All scores below were unweighted means of five development-fold Spearman correlations. All 805 frozen final queries were excluded.", "",
        "| Formula | Old: 3,222 rows | Old: same 2,950 rows | Reviewed: 2,950 rows | Change on same rows |",
        "|---|---:|---:|---:|---:|",
        *(f"| {r.rule_family} | {r.old_full_spearman:.6f} | {r.old_matched_spearman:.6f} | {r.reviewed_spearman:.6f} | {r.delta_matched_validation_rows:+.6f} |"
          for r in scores.itertuples()), "",
        "Across nine formulas, the mean changed from 0.497655 on the original old population to 0.480534 when old predictions were scored on the accepted rows, versus 0.477531 for the reviewed fits. This average described the formulas; it was not an ensemble score.", "",
        "Three formulas improved and six declined on matched validation rows. The changes ranged from -0.010448 to +0.007483. The best remained `pren_n_coupled_mns_weak_anions`: 0.579633 old versus 0.571158 reviewed on the same rows. `pren_n_coupled_mns` remained second at 0.560894 reviewed.", "",
        "The smaller matched-row difference showed that much of the apparent drop came from changing which validation rows were scored. It did not establish that cleaning alone helped or hurt: the old predictions retained their original larger context fits, and preprocessing, scaling and coefficients also differed. These were formula scores, not trained TabICL scores.", "",
        "Verified frozen split identities, original/reviewed source identities, formula artifact hashes and unchanged definitions, row IDs, folds and targets. Recomputed all 45 historical and 45 reviewed fold scores and both sets of nine saved means. Old/new CSV target round-off was at most 1.14e-13; target ranks were identical in every matched fold.", "",
        "Historical predictions came from `analysis/soccol_target_rules_v1/oof_predictions.csv`, with rule names matched to the promoted `soccol_pipeline/target_rules_v1` artifacts. Reviewed predictions came from `soccol_pipeline/target_rules_v2/oof_predictions.csv`. Their source hashes were saved in `summary.json`.", "",
        "Reproduce in a fresh directory:", "", "```bash",
        ".venv/bin/python -m scripts.soccol_pipeline.compare_saved_rule_results --output-dir /tmp/soccol_rule_comparison", "```", "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(lines))
    print(scores.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
