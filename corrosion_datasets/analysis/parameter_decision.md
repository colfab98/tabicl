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

## What The Stricter Evidence Says

The analysis now excludes text identifiers, alloy/formula strings, processing
labels, prose methods, and categorical environment labels from numeric Spearman
and interaction evidence. Those fields can still contribute through categorical
association checks, but they are not treated as scalar physical measurements.

| Parameter | Current v11/v12 Value | Evidence-Based Reading |
|---|---:|---|
| `informed_feature_block_strength` | `0.30` | Still plausible, but the corrected within-block evidence is softer: mean `0.343`, median `0.302`, `n=18`. Do not raise aggressively. |
| `informed_interaction_strength` | `0.35` | Material-environment coupling is scientifically sensible, but the corrected interaction probe median is approximately `0.000`. Keep moderate or slightly weaker. |
| `informed_history_strength` | `0.70` | Not supported as a universal default by the downloaded data. The time-series evidence is narrow and mixed, with median absolute lag-1 Spearman `0.068`. |
| `informed_intervention_strength` | `0.20` | Evidence is sparse and categorical after removing numeric parsing artifacts. Keep weak and conditional. |
| `informed_prior_ratio` | `0.50` | Not directly estimable from external datasets. This is the most important ablation/control knob. |

The key conclusion is that the informed-prior direction is defensible, but the
old v11/v12 strengths are not calibrated by these datasets. The stricter
analysis argues against a strong universal history term and against increasing
the material-environment interaction strength. Since the informed prior has not
clearly beaten the generic prior on every criterion, the next experiment should
test softer and better matched informed structure rather than stronger structure.

## Recommended Small Ablation

Use v11/v12 as the current-control point, then run a small grid around it.

| Proposed Run | Purpose | `informed_prior_ratio` | `feature_block` | `interaction` | `history` | `intervention` |
|---|---|---:|---:|---:|---:|---:|
| current control | Reproduce v11/v12 behavior | `0.50` | `0.30` | `0.35` | `0.70` | `0.20` |
| corrected soft structure | Test the stricter-analysis default | `0.50` | `0.25` | `0.25` | `0.25` | `0.10` |
| low-history structure | Test whether row-order dependence hurts mostly static tables | `0.50` | `0.25` | `0.25` | `0.00` | `0.10` |
| lower informed mix | Test whether informed tasks should be rarer | `0.25` | `0.25` | `0.25` | `0.25` | `0.10` |
| current ratio, old blocks only | Isolate whether block correlation helped while history/intervention were too strong | `0.50` | `0.30` | `0.25` | `0.00` | `0.05` |

The first two new runs are the most important. The current evidence does not
suggest that simply adding more informed tasks, stronger interaction, or stronger
history is the right fix.

## Why This Avoids Leakage

The ablation values are chosen from external dataset structure:

- within-block correlations,
- material/environment grouping,
- simple interaction probes,
- processing/intervention target association,
- limited time-series autocorrelation.

The stricter parser is also part of the leakage control: identifiers and prose
fields are not converted into accidental numeric variables.

They are not chosen because a DatacorTech test metric improved.

DatacorTech should be used for a pre-specified comparison after the run definitions are fixed.

## Interpretation Rule For The Next Results

Do not select the final model using only one test metric.

Recommended reading:

- If generic still has best AUROC but informed has better balanced accuracy/MCC, report that as a ranking-vs-classification tradeoff.
- If corrected soft structure beats current informed, the old structural prior was probably too strong.
- If low-history structure beats corrected soft structure, row-order dependence should be disabled by default for mostly static corrosion tables.
- If lower informed mix wins, the issue is likely frequency of informed tasks rather than only strength.
- If current control remains best informed variant but still does not beat generic AUROC, the next improvement is likely better feature-block composition or marginal distributions, not stronger structural coupling.

## Next Implementation Step

Before launching many jobs, decide whether the next training run should test
these stricter structural settings directly or first change the synthetic block
allocation to match the material-heavy evaluation tasks.
