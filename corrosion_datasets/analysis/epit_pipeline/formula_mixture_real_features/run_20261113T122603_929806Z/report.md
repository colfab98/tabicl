# Combined formulas on real EPiT samples

Each sample receives ONE formula, with quotas matching training probabilities to integer rounding. No per-sample averaging.
MAE and main Spearman use development-calibrated mV predictions. raw_score_spearman uses the uncalibrated mixed formula scores.
Final-test results are held out. Development and pooled results are descriptive and partly in-sample.
The 5th–95th percentile range measures random assignment variation, not statistical confidence.
Fixed fitted formulas only: saved development preprocessing, no coefficient variation or SCM mixing. This is not a full training-target replay.
All alloy families are retained, including those outside the Fe/Ni formula fitting domain. Threshold targets retain the existing numeric-threshold convention.

```
                scope group   n  repeats  mae_mV_mean  mae_mV_p05  mae_mV_p95  spearman_mean  spearman_p05  spearman_p95  raw_score_spearman_mean
 all_rows_descriptive   all 760      100   287.243099  284.091958  291.135421       0.517756      0.503306      0.531122                 0.511081
development_in_sample   all 608      100   287.742561  284.150688  291.689024       0.547452      0.532194      0.565452                 0.541338
           final_test   all 152      100   285.245251  278.876603  292.547406       0.415967      0.384320      0.449855                 0.405262
```

See assigned_predictions.csv for every sample, formula, prediction and error; summary.csv also separates alloy families.
