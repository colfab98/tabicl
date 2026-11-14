# Combined formulas on real EPiT samples

Each sample receives ONE formula, with quotas matching training probabilities to integer rounding. No per-sample averaging.
MAE and main Spearman use development-calibrated mV predictions. raw_score_spearman uses the uncalibrated mixed formula scores.
Final-test results are held out. Development and pooled results are descriptive and partly in-sample.
The 5th–95th percentile range measures random assignment variation, not statistical confidence.
Fixed fitted formulas only: saved development preprocessing, no coefficient variation or SCM mixing. This is not a full training-target replay.
All alloy families are retained, including those outside the Fe/Ni formula fitting domain. Threshold targets retain the existing numeric-threshold convention.

```
                scope group   n  repeats  mae_mV_mean  mae_mV_p05  mae_mV_p95  spearman_mean  spearman_p05  spearman_p95  raw_score_spearman_mean
 all_rows_descriptive   all 760     1000   287.050814  283.654532  290.627147       0.518138      0.504874      0.532046                 0.511942
development_in_sample   all 608     1000   287.636743  283.476818  291.675463       0.547566      0.533034      0.562400                 0.541650
           final_test   all 152     1000   284.707097  276.788510  293.105132       0.417127      0.380879      0.452426                 0.408286
```

See assigned_predictions.csv for every sample, formula, prediction and error; summary.csv also separates alloy families.
