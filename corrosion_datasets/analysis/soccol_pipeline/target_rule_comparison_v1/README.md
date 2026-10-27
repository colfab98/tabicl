# Soccol: previous and reviewed formula scores

Reused existing saved results and predictions; no training or coefficient fitting was run. All scores below were unweighted means of five development-fold Spearman correlations. All 805 frozen final queries were excluded.

| Formula | Old: 3,222 rows | Old: same 2,950 rows | Reviewed: 2,950 rows | Change on same rows |
|---|---:|---:|---:|---:|
| pren_n_linear | 0.463315 | 0.441652 | 0.431203 | -0.010448 |
| cr_mow_n_synergy | 0.494717 | 0.472136 | 0.479619 | +0.007483 |
| threshold_saturation | 0.459631 | 0.434369 | 0.439298 | +0.004930 |
| pren_n_improved_environment | 0.459677 | 0.436797 | 0.438341 | +0.001544 |
| mo_n_acid_repassivation | 0.459924 | 0.436328 | 0.433876 | -0.002452 |
| mns_inclusion_penalty | 0.498030 | 0.477475 | 0.467508 | -0.009967 |
| coupled_breakdown | 0.502793 | 0.482026 | 0.475886 | -0.006141 |
| pren_n_coupled_mns | 0.554193 | 0.564396 | 0.560894 | -0.003502 |
| pren_n_coupled_mns_weak_anions | 0.586618 | 0.579633 | 0.571158 | -0.008475 |

Across nine formulas, the mean changed from 0.497655 on the original old population to 0.480534 when old predictions were scored on the accepted rows, versus 0.477531 for the reviewed fits. This average described the formulas; it was not an ensemble score.

Three formulas improved and six declined on matched validation rows. The changes ranged from -0.010448 to +0.007483. The best remained `pren_n_coupled_mns_weak_anions`: 0.579633 old versus 0.571158 reviewed on the same rows. `pren_n_coupled_mns` remained second at 0.560894 reviewed.

The smaller matched-row difference showed that much of the apparent drop came from changing which validation rows were scored. It did not establish that cleaning alone helped or hurt: the old predictions retained their original larger context fits, and preprocessing, scaling and coefficients also differed. These were formula scores, not trained TabICL scores.

Verified frozen split identities, original/reviewed source identities, formula artifact hashes and unchanged definitions, row IDs, folds and targets. Recomputed all 45 historical and 45 reviewed fold scores and both sets of nine saved means. Old/new CSV target round-off was at most 1.14e-13; target ranks were identical in every matched fold.

Historical predictions came from `analysis/soccol_target_rules_v1/oof_predictions.csv`, with rule names matched to the promoted `soccol_pipeline/target_rules_v1` artifacts. Reviewed predictions came from `soccol_pipeline/target_rules_v2/oof_predictions.csv`. Their source hashes were saved in `summary.json`.

Reproduce in a fresh directory:

```bash
.venv/bin/python -m scripts.soccol_pipeline.compare_saved_rule_results --output-dir /tmp/soccol_rule_comparison
```
