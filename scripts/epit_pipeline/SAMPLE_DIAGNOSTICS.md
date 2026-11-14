# Paired EPIT sample diagnostics

This workflow identifies samples and categories where an informed checkpoint has
higher absolute error than the generic checkpoint at the same training step and
on the same query rows. It does not refit or change the target formulas.

The usual evaluation saves aggregate metrics, not individual predictions. The
exporter therefore requires model inference. It evaluates only generic and the
requested informed models; it does not run CatBoost or pretrained baselines.

From the repository root, inspect the plan without loading models:

```bash
PYTHONPATH=src:. .venv/bin/python -m scripts.epit_pipeline.diagnose_cv_samples --dry-run
```

Run on CPU when ready:

```bash
PYTHONPATH=src:. .venv/bin/python -m scripts.epit_pipeline.diagnose_cv_samples --device cpu
```

Defaults:

- Source: `corrosion_datasets/analysis/epit_pipeline/targetv5_four_ratios_through_8000_cv5_cpu`.
- Models: `epit_100pct` and `generic_baseline`.
- Steps: all checkpoints shared across the requested models and all five folds.
- Output: `corrosion_datasets/analysis/epit_pipeline/targetv5_cv5_sample_diagnostic`.
- A 1 mV absolute-error difference is treated as a practical tie; adjust with `--tie-mV`.

Use `--steps 1000 2000 4500 5500` for four contrasting exploratory snapshots
instead of every available checkpoint. Use `--models epit_25pct epit_50pct
epit_75pct epit_100pct` to include other ratios; available shared steps will then
be limited by the shortest run. This is diagnostic comparison, not checkpoint
selection on final-test labels.

The exporter uses the existing evaluator, preprocessing and native TabICL
settings, and captures the same predictions used to calculate its metrics.
MAE, RMSE, Spearman, Pearson and R² must reproduce the saved model/fold/checkpoint
metrics within the declared numerical tolerance. A mismatch stops extraction;
do not increase tolerances merely to conceal an unexplained discrepancy.

Completed jobs are saved separately with checkpoint and prediction SHA-256
checksums. Repeating the same command resumes verified jobs. Changing the model
set, step set, device, inputs or inference implementation requires a fresh output
directory. The checkpoint snapshot provides size verification; matching aggregate
metrics is an additional reproduction check, not proof that the original binary
had the same hash (the old snapshot did not store checkpoint hashes).

Once extraction is complete, this recomputes the analysis without model inference:

```bash
PYTHONPATH=src:. .venv/bin/python -m scripts.epit_pipeline.diagnose_cv_samples --analyze-only
```

Repeat any custom extraction arguments, such as `--steps` and `--models`, when
resuming or using `--analyze-only`.

Outputs:

- `paired_sample_errors.csv`: per-checkpoint informed/generic predictions, signed
  residuals, absolute-error differences, relative rank errors, sample IDs and metadata.
- `sample_consistency.csv`: sample-level errors and fraction of selected
  checkpoints where informed loses; checkpoints are not counted as extra samples.
- `persistently_worse_samples.csv`: samples worse in at least 75% of checkpoints.
- `persistently_not_better_samples.csv`: samples not better beyond the tie band in
  at least 75% of checkpoints.
- `category_errors.csv` and `category_errors_by_step.csv`: family, composition
  ranges, environment, heat treatment, preparation, phase missingness, source and
  family/treatment intersections, with support counts and publication-cluster
  descriptive intervals where at least five publications are represented.
- `*_paired_errors.png`: paired error scatterplots by family, with fold 1 separate.
- `metric_reproduction_deltas.csv`: numerical agreement with the original scores.
- `report.md`: category shortlist and links to the sample diagnostics.

Fold 1 remains separate from folds 2–5 because the latter include rows used to
fit the informed prior. Small or source-specific groups are flagged. Categories
overlap; pointwise bootstrap intervals are not multiple-testing-corrected claims
of statistical significance. Rank error is a within-fold percentile diagnostic,
not a per-sample Spearman contribution. Any proposed formula repair should be
tested on development-only source/group-held-out ablations before being claimed
as an improvement.
