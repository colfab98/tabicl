# Informed-SCM Parameter Decision

This note translates the external corrosion dataset analysis into a practical next training plan.

## Current Status

We have now completed a first quantitative structural analysis, not just dataset collection.

The analysis is in:

- `structural_analysis_results.md`
- `structural_analysis_results.json`
- `scripts/analyze_structure.py`
- `column_group_map.md`

This analysis estimates broad structure from external corrosion datasets only. It does not use DatacorTech test performance.

## What The Evidence Says

| Parameter | Current v11/v12 Value | Evidence-Based Reading |
|---|---:|---|
| `informed_feature_block_strength` | `0.30` | Plausible. Within-block dependence is usually moderate. Do not raise aggressively. |
| `informed_interaction_strength` | `0.35` | Plausible but not strongly proven by the simple interaction probes. Keep moderate. |
| `informed_history_strength` | `0.70` | Corrosion logic supports path dependence, but downloaded time-series evidence is narrow and mixed. Test a weaker value. |
| `informed_intervention_strength` | `0.20` | Plausible and conservative. Processing/intervention labels are informative but not enough to justify a large value. |
| `informed_prior_ratio` | `0.50` | Not directly estimable from external datasets. This is the most important ablation/control knob. |

The key conclusion is that the current settings are **defensible heuristic values**, but they are not yet optimized or proven. Since the informed prior has not clearly beaten the generic prior on every criterion, the next experiment should mostly test whether the informed structure is too strong or too frequent, rather than blindly increasing all strengths.

## Recommended Small Ablation

Use v11/v12 as the current-control point, then run a small grid around it.

| Proposed Run | Purpose | `informed_prior_ratio` | `feature_block` | `interaction` | `history` | `intervention` |
|---|---|---:|---:|---:|---:|---:|
| current control | Reproduce v11/v12 behavior | `0.50` | `0.30` | `0.35` | `0.70` | `0.20` |
| conservative informed | Test whether informed structure helps when less dominant | `0.25` | `0.30` | `0.35` | `0.50` | `0.20` |
| weaker structure | Test whether the structural transform is too strong | `0.50` | `0.20` | `0.25` | `0.50` | `0.10` |
| stronger blocks only | Test the strongest dataset-supported signal without over-boosting everything | `0.50` | `0.45` | `0.35` | `0.50` | `0.20` |
| higher informed mix, optional | Only if compute allows; tests whether more domain-shaped tasks help | `0.75` | `0.30` | `0.35` | `0.50` | `0.20` |

The first three new runs are the most important. The optional higher-ratio run should come later, because the current evidence does not strongly suggest that simply adding more informed tasks is the right fix.

## Why This Avoids Leakage

The ablation values are chosen from external dataset structure:

- within-block correlations,
- material/environment grouping,
- simple interaction probes,
- processing/intervention target association,
- limited time-series autocorrelation.

They are not chosen because a DatacorTech test metric improved.

DatacorTech should be used for a pre-specified comparison after the run definitions are fixed.

## Interpretation Rule For The Next Results

Do not select the final model using only one test metric.

Recommended reading:

- If generic still has best AUROC but informed has better balanced accuracy/MCC, report that as a ranking-vs-classification tradeoff.
- If conservative informed beats current informed, the prior was probably too strong or too frequent.
- If weaker structure beats current informed, reduce strengths before adding physical range priors.
- If current control remains best informed variant but still does not beat generic AUROC, the next improvement is probably physical feature marginals, not stronger structural coupling.

## Next Implementation Step

Before launching many jobs, add a small run-plan table to the stage-1 sbatch notes or commands file so these settings are pre-registered. Then train only the first three new informed variants plus the current control.
