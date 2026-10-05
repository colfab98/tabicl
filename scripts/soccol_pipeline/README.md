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

Evaluate fixed development-fold references without touching the final test:

```bash
sbatch scripts/soccol_pipeline/evaluate_baseline_folds.sbatch
```

This follows the old EPIT development checkpoint comparison: it evaluates the
generic run at 500-step intervals on all five frozen development folds and adds
pretrained TabICL v2 and fixed CatBoost at every checkpoint step. The fair
reference for the 1,000-step Soccol Optuna trials is the step-1,000 row. CatBoost
is not tuned on the validation targets and receives the same 37 context-only
encoded inputs, with the four schema-declared procedure columns marked as
categorical. The unchanged EPIT SVG trend writer produces per-fold and combined
plots; missing plots fail the run. Outputs are written under
`corrosion_datasets/analysis/soccol_pipeline/baseline_folds_v1/`, with a fresh
suffix for repeated runs.

The final stages reuse the EPIT implementation through Soccol-only wrappers.
Their output roots are
`corrosion_datasets/analysis/soccol_pipeline/final_v1/` and
`checkpoints/soccol_pipeline_final_v1/`.
