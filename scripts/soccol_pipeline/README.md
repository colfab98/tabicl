# Soccol pitting-potential pipeline

This directory reruns the previous empirical-feature SCM-target Optuna method
on the frozen Soccol composition split.

- Inputs: 37 columns in the order defined by `soccol_schema.py`.
- Rules: the nine candidates frozen in
  `corrosion_datasets/analysis/soccol_pipeline/target_rules_v1/`.
- Search: 50 trials, 10 startup trials, 1,000 training steps, 10,000 scheduler
  steps, five saved development folds, eight estimators, and coefficient
  variation `0.8` with seed `42`.
- Final training: reuse the selected Optuna configuration for 10,000 steps and
  select a checkpoint using only the five development folds.
- Final evaluation: use all 3,222 development rows as context and evaluate the
  untouched 805-row final-test set once.
- Final-test targets remain unused until that one final evaluation.

Launch the stages in order:

```bash
sbatch scripts/soccol_pipeline/run_optuna.sbatch
sbatch scripts/soccol_pipeline/train_final.sbatch
sbatch scripts/soccol_pipeline/evaluate_final.sbatch
```

The final stages reuse the EPIT implementation through Soccol-only wrappers.
Their output roots are
`corrosion_datasets/analysis/soccol_pipeline/final_v1/` and
`checkpoints/soccol_pipeline_final_v1/`.
