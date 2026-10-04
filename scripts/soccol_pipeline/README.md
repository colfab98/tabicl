# Soccol pitting-potential pipeline

This directory reruns the previous empirical-feature SCM-target Optuna method
on the frozen Soccol composition split.

- Inputs: 37 columns in the order defined by `soccol_schema.py`.
- Rules: the nine candidates frozen in
  `corrosion_datasets/analysis/soccol_pipeline/target_rules_v1/`.
- Search: 50 trials, 10 startup trials, 1,000 training steps, 10,000 scheduler
  steps, five saved development folds, eight estimators, and coefficient
  variation `0.8` with seed `42`.
- Final-test targets remain unused during search.

Launch with:

```bash
sbatch scripts/soccol_pipeline/run_optuna.sbatch
```
