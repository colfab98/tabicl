# MPEA formula investigation

Uses only the 214 frozen development rows, with five composition-grouped folds. The 53 final-test targets are not parsed, fitted or scored.

## Method

The EPIT `calibrate_family` and coefficient fitter are reused: fit imputation, all scaling and constrained coefficients on four development folds; score the fifth. Coefficients are nonnegative, sum to one, use inherited bounds, and are regularized toward the retained physics anchor with strength 0.1. Full-development coefficient centers are fitted only after collecting fold results. These are dimensionless ranking scores, not predictions in mV.

All 18 production source/rule combinations need inputs absent from MPEA. Reduced candidates explicitly remove unavailable terms. N or S is not assumed physically zero; pH and other ions are not invented. The common Fe/Ni hypotheses are being screened on MPEA, not declared chemically validated for every alloy.

Temperature has 4 observations; pH, nitrogen, sulfur, scan rate and the extra anions are unrecorded. Chloride is recorded for 151/214 rows, at 3 distinct concentrations. Unknown chloride uses the current context mean, as in EPIT; it does not mean zero chloride. A separate score on recorded-chloride rows checks sensitivity without changing the folds.

The eleven reduced candidates and eight matching composition-only controls were defined before their scores were inspected. EPIT and Soccol variants retain their respective material expressions (including EPIT's W term). Unsupported Mn-S, Mo-N-acid, method and anion additions collapse to simpler rules rather than receiving artificial fitted coefficients.

## Development results

Best reduced candidate: **epit_pren_chloride**, mean fold Spearman **0.2334**. This is formula screening, not trained-TabICL performance or a final-test result.

| Rule | Mean fold Spearman | Fold SD | Before coefficient fit | Recorded chloride only |
| --- | ---: | ---: | ---: | ---: |
| epit_pren_chloride | 0.2334 | 0.3367 | 0.2424 | 0.5320 |
| epit_synergy_interaction | 0.2314 | 0.3406 | 0.2205 | 0.5326 |
| epit_pren_interaction | 0.2205 | 0.3330 | 0.2110 | 0.5072 |
| epit_coupled_pren_coupled | 0.2174 | 0.3432 | 0.2129 | 0.5559 |
| soccol_synergy_interaction | 0.2101 | 0.3280 | 0.2187 | 0.4875 |
| epit_coupled_pren_material_only | 0.2064 | 0.3856 | 0.2064 | 0.4650 |
| soccol_pren_chloride | 0.2058 | 0.3259 | 0.2242 | 0.4805 |
| epit_pren_material_only | 0.2042 | 0.3837 | 0.2042 | 0.4650 |
| epit_synergy_material_only | 0.2037 | 0.3839 | 0.2037 | 0.4654 |
| epit_threshold_material_only | 0.2024 | 0.3822 | 0.2024 | 0.4656 |
| soccol_pren_interaction | 0.2011 | 0.3273 | 0.2058 | 0.4576 |
| epit_threshold_interaction | 0.1997 | 0.3284 | 0.2327 | 0.4878 |
| soccol_pren_coupled | 0.1904 | 0.3283 | 0.1932 | 0.5075 |
| soccol_coupled_pren_coupled | 0.1900 | 0.3280 | 0.1930 | 0.5063 |
| soccol_synergy_material_only | 0.1766 | 0.3704 | 0.1766 | 0.4135 |
| soccol_threshold_material_only | 0.1764 | 0.3698 | 0.1764 | 0.4137 |
| soccol_pren_material_only | 0.1764 | 0.3700 | 0.1764 | 0.4131 |
| soccol_coupled_pren_material_only | 0.1764 | 0.3700 | 0.1764 | 0.4131 |
| soccol_threshold_interaction | 0.1594 | 0.3249 | 0.2231 | 0.4393 |

## Does chloride add useful information?

Each candidate is compared with its own material formula alone on exactly the same validation rows.

| Rule | Mean gain over composition only | Folds improved |
| --- | ---: | ---: |
| epit_pren_interaction | +0.0163 | 2/5 |
| epit_synergy_interaction | +0.0278 | 2/5 |
| epit_threshold_interaction | -0.0027 | 2/5 |
| epit_pren_chloride | +0.0292 | 2/5 |
| epit_coupled_pren_coupled | +0.0110 | 3/5 |
| soccol_pren_interaction | +0.0247 | 2/5 |
| soccol_synergy_interaction | +0.0335 | 2/5 |
| soccol_threshold_interaction | -0.0170 | 2/5 |
| soccol_pren_chloride | +0.0293 | 2/5 |
| soccol_coupled_pren_coupled | +0.0135 | 2/5 |
| soccol_pren_coupled | +0.0139 | 2/5 |

## Where the inherited formulas fail

The best rule's weakest fold is fold 5 (Spearman -0.4288). Its largest composition group contains 13 rows: Al2NbTi3V2Zr, tested in HNO3. Their median measured potential is 1856 mV SCE, but their median formula score is -0.835. They contain 0.00 wt.% Cr+Mo+W, versus 48.91 wt.% Ti+Nb. The inherited material formula responds only to Cr/Mo/W, so these other composition differences have no effect on its material score.

Recorded-chloride rows give a much stronger ranking than the full mixture for the leading rules, but this is a different subset, not proof that filling chloride differently would fix the problem. Alloy chemistry and electrolyte differ together. Composition-only controls also fail in the weakest fold, so missing chloride alone cannot explain this failure. All original development rows remain in the main scores.

## Decision from this screen

Keep the simple Cr-Mo-W/chloride rule as a reference. Investigate material effects outside Cr/Mo/W before promoting an MPEA prior; fitting these inherited terms alone gives weak and unstable development rankings.

Do not add the unavailable N, Mn-S, acidic-pH or extra-ion effects to the MPEA fitting stage. Do not count their reduced copies as additional independent prior families. The simple EPIT-derived Cr-Mo-W/chloride rule is a useful reference, but its chloride addition improves only two of five folds. Threshold and interaction complexity do not establish a reliable improvement. This screen does not yet justify launching Optuna with the inherited formulas unchanged.


## Saved artifacts and interpretation

`feature_coverage.csv` and `original_rule_audit.json` document missing inputs. `summary.csv`, `fold_metrics.csv` and `oof_predictions.csv` contain the development comparisons. Each file in `rules/` includes the formula, five fitted coefficient vectors, context preprocessing states, and a coefficient center fitted on all 214 development rows. `comparison_summary.json` records provenance, settings and recommendations; `artifact_lock.json` hashes the outputs.

A higher mean here supports further development, not a guarantee that the informed transformer will beat its baselines. Coefficient fitting optimizes regularized squared error against standardized target ranks, as in EPIT/Soccol; it does not directly maximize held-out Spearman and can lower it. The best candidate was selected among several hypotheses on these same development folds. Sparse effects and absent features cannot be validated by this comparison.

Existing EPIT/Soccol formulas, production registries and MPEA split artifacts are unchanged. No training or Optuna is launched.

Reproduce from the repository root (use a new output folder, or `--force` to replace this report):

```bash
.venv/bin/python -m corrosion_datasets.datasets.mpea_corrosion.compare_target_rules
```
