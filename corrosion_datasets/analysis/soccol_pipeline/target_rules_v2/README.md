# Reviewed Soccol formula calibration

The same nine formulas were refitted on 2,950 accepted development rows. All 805 frozen final queries and 272 withheld development rows were excluded.

Each fold fitted filling values, scaling and coefficients using its context only. Final coefficient centers used all accepted development rows. Formula definitions, anchors and bounds were unchanged. The scores below are development formula scores, not final-model scores.

| Formula | Mean fold Spearman | Minimum fold | Maximum fold |
|---|---:|---:|---:|
| pren_n_linear | 0.431203 | 0.270351 | 0.560568 |
| cr_mow_n_synergy | 0.479619 | 0.251769 | 0.638362 |
| threshold_saturation | 0.439298 | 0.209321 | 0.637983 |
| pren_n_improved_environment | 0.438341 | 0.238026 | 0.645227 |
| mo_n_acid_repassivation | 0.433876 | 0.205696 | 0.651045 |
| mns_inclusion_penalty | 0.467508 | 0.285060 | 0.711165 |
| coupled_breakdown | 0.475886 | 0.230192 | 0.734326 |
| pren_n_coupled_mns | 0.560894 | 0.457106 | 0.770954 |
| pren_n_coupled_mns_weak_anions | 0.571158 | 0.319911 | 0.717882 |

Per-fold filling/scaling states and coefficients were saved in each rule JSON. Held-out row IDs and predictions were saved in `oof_predictions.csv`.
