# Soccol pitting-potential pipeline

Current searches use **v8** with the [same original TabICL policy as EPIT v14](../epit_pipeline/README.md#current-experiment-v14).
The reviewed feature profile remains v3; formula and data artifacts were retained.

Both streams use grouping size 3 and feature shuffling. The missingness embedding
is disabled. Informed NaNs are filled from context means before original feature
preprocessing. The generic generator retains its original full-batch sampling.
Optuna, final checkpoint selection and final evaluation use the common eight-view
TabICL ensemble. Frozen development/final-query membership is unchanged.

v8 fixes grouping when training tables have different real feature counts.
Use fresh studies/checkpoints; v7 training cannot be resumed across this fix.
The final-training launcher now forwards command-line ratio/resume arguments
and derives its process count from the GPU allocation, matching EPIT.

Reviewed Soccol final training now defaults to **manual settings, without an
Optuna study**: **MLP 0.70 / tree 0.30**, **composition perturbation 0.30**, and
**formula/SCM target weight 0.28** (28% formula / 72% SCM per informed table).
Soccol uses its existing calibrated formulas and fixed target weight. These
settings are independent of the generic/informed table ratio. Explicit options
still take precedence. Optuna search settings and `--legacy-data` are unchanged.

```bash
sbatch --gpus=1 scripts/soccol_pipeline/train_final.sbatch --override-informed-prior-ratio 0.0
sbatch --gpus=1 scripts/soccol_pipeline/train_final.sbatch --override-informed-prior-ratio 0.25
sbatch --gpus=1 scripts/soccol_pipeline/train_final.sbatch --override-informed-prior-ratio 0.5
sbatch --gpus=1 scripts/soccol_pipeline/train_final.sbatch --override-informed-prior-ratio 0.75
sbatch --gpus=1 scripts/soccol_pipeline/train_final.sbatch --override-informed-prior-ratio 1.0
```

No study, journal or trial number is required. Checkpoints use
`checkpoints/soccol_pipeline_final_v2/soccol_manual_v1_ratio_<ratio>_override_<hash>/`;
analysis/manifest files use `analysis/soccol_pipeline/final_v2/manual/<run>/`.
Use the final manifest printed by the run for evaluation. To resume interrupted
manual training, repeat the same command with `--resume-training`; no trial
number is needed. To explicitly use a study instead, pass
`--no-manual-hyperparameters --storage <journal-url> --study-name <name>`.

Sync `scripts/soccol_pipeline/train_final.py`, `train_final.sbatch`, and the shared
`scripts/epit_pipeline/train_final.py` to each training node before launching.

```bash
sbatch --gpus=1 \
  --export=ALL,STUDY_NAME=soccol_pipeline_optuna_empirical_features_scm_target_v8,N_TRIALS=50 \
  scripts/soccol_pipeline/run_optuna.sbatch
```

Use fresh studies and checkpoints. The sections below document historical
versions; v8 supersedes their missingness, grouping and evaluation settings.

## Reviewed-data migration: 58-column sampling, training and evaluation connected

`data.py:load_soccol_dataset()` reads `processed/soccol_reference_v1` directly.
It loads the 58 registered features, all 24 composition columns, unchanged
`E_pit` targets, composition acceptance audit and frozen row assignments.
It preserves the saved amounts, zeros and NaNs without rebalancing Fe,
normalizing compositions or fitting any filling values. File checksums,
feature/reference equality, row identities, units and frozen assignments are
validated. The loader does not read `soccol_regression_event1.csv`.

The loader retains all 4,027 rows in their original frozen order, including
272 unresolved development rows and ten unresolved final queries. It exposes
the saved eligibility flags. `select_accepted_development()` supplies the same
2,950 accepted development rows to reviewed formula inputs, sampling input
preparation and evaluation folds. The reference dataset and frozen split
remain unchanged; no rows were reassigned.

The accepted validation folds contain **622, 615, 638, 445 and 630** rows.
Each fold's context contains the other accepted development rows. Reviewed
final-test selection uses 2,950 context rows and all 805 original final queries;
reducing context to one fold never removes or substitutes query rows. The ten
unresolved queries remain present. Selection records include original row IDs,
original task indices, fold labels and a common selection checksum.

The general evaluator's explicit `--soccol-reviewed-data` route uses this
loader and accepted selection for both inspection and model evaluation. It
uses the exact order in `soccol_reviewed_schema.py`: 24 composition columns and
34 environment/history/reference fields (49 numeric fields and nine categories
overall). Real inputs use the bank's development-only category mappings.
Unknown categories and missing numeric measurements remain NaN; the historical
37-column imputer is bypassed. Inspect the reviewed development task with:

```bash
.venv/bin/python -m scripts.eval_corrosion_datasets \
  --soccol-reviewed-data --target-binning continuous \
  --epit-split-manifest corrosion_datasets/datasets/soccol_pitting_potential/processed/splits_v1/split_manifest.json \
  --epit-validation-fold 1 --list-tasks
```

`formula_data.py` supplies only accepted development rows before formula
preprocessing. The formula comparison driver accepts `--reviewed-data` and
records the shared selection. It preserves complete compositions and uses
context-only means for internal missing-ion calculations; the input tables
retain their NaNs. Its reviewed output default is separate from the historical
comparison. Inspect its inputs without fitting coefficients:

```bash
.venv/bin/python -m corrosion_datasets.datasets.soccol_pitting_potential.compare_target_rules \
  --reviewed-data --inspect-inputs
```

The sampling preparation entry point exposes `load_reviewed_feature_source()`:
2,950 target-free rows with all 58 saved fields, original IDs and task indices.
The frozen `src/tabicl/prior/assets/soccol_pitting_features_v3.csv` and `.json`
store the replacement sampling bank and its development-only statistics.
All accepted materials are retained without EPIT's family filter. Composition
sampling gives each of the **349 distinct saved compositions** equal probability,
matching EPIT's template weighting. Joint environment/history sampling gives
each of the **2,950 accepted observations** equal probability. These two draws
remain independent. The first accepted row with each composition supplies its
template ID; its environment is not attached to that template.
Saved real compositions are unchanged; only generated compositions close to
100 wt.%. Missing inputs remain NaN. Development means and category modes are
stored separately for internal calculations, as in the reviewed EPIT bank.

The loader and sampler default to `soccol_pitting_features_v3`; the current
training runtime selects that same version explicitly. Missing reviewed bank
files cause an error instead of falling back to v1.
The reviewed training allocation is `(24, 34, 0, 0, 0, 0, 0, 0, 0)`; category
roles follow their saved positions within the context block. Training uses
`empirical_features_scm_target` and requires `--col_missing_value_embedding true`.
All 58 fields reach the SCM. The nine existing formulas read their original
named inputs from the new order. After finite internal calculations, the
sampled row's exact missing-value mask is restored for the model.

The historical `v1` and `v2` profiles require explicit names and preserve their
original observation-weighted composition draws. The asset builder also
defaults to the reviewed v3 bank; `--reviewed-data` remains an optional explicit
alias, while `--legacy-data` selects historical v1 generation. Inspect the
reviewed source without rebuilding the bank:

```bash
.venv/bin/python -m corrosion_datasets.datasets.soccol_pitting_potential.prepare_pipeline_assets \
  --inspect-inputs
```

The [before/after input audit](../../corrosion_datasets/analysis/soccol_pipeline/input_audit_v1/README.md)
records changed cells, schema additions/removals, fold-context filling changes
and checks against EPIT. It explicitly uses the historical reviewed v2 bank.
Reproduce it without training or refitting rules:

```bash
.venv/bin/python -m scripts.soccol_pipeline.audit_input_changes --output-dir /tmp/soccol_input_audit
```

The audit refuses to overwrite an existing nonempty directory. Its formula
dependency check identifies an existing difference: Soccol's
`cr_mow_n_synergy` formula does not use W, while EPIT's corresponding rule does.

The [previous/reviewed formula comparison](../../corrosion_datasets/analysis/soccol_pipeline/target_rule_comparison_v1/README.md)
also re-scores saved old predictions on the same 2,950 accepted validation
rows. It retains the old context fits, so it does not isolate preprocessing
alone. Reproduce it without training or coefficient fitting:

```bash
.venv/bin/python -m scripts.soccol_pipeline.compare_saved_rule_results --output-dir /tmp/soccol_rule_comparison
```

The [material-family analysis](../../corrosion_datasets/analysis/soccol_pipeline/material_scope_audit_v1/README.md)
records the six stored family groups, their v2 sampling shares and each formula's
within-family development scores. Historical `material_scope_primary` flags
do not filter the reviewed workflow; all composition-accepted development
rows are used. The analysis also documents EPIT's different composition
weighting and Fe/Ni-specific rule. Reproduce the numerical outputs with:

```bash
.venv/bin/python -m scripts.soccol_pipeline.analyze_material_scope --output-dir /tmp/soccol_material_scope
```

The [synthetic-sampling audit](../../corrosion_datasets/analysis/soccol_pipeline/synthetic_sampling_audit_v1/README.md)
checks joint context sampling, category codes, missingness, composition noise
and the training input path against EPIT. It quantifies Soccol's historical v2
observation weighting: its top 20 compositions received 52.71% of draws.
The approved v3 sampler now draws distinct compositions uniformly, so those
same 20 compositions receive **5.73%** of draws. The bank values, statistics,
category mappings and joint context sampling are unchanged. See the
[implementation report](../../corrosion_datasets/DATASET_PROCESSING.md#soccol-pipeline-uniform-composition-sampling-and-fixed-coefficient-variation).
Reproduce the original v2 numerical audit with:

```bash
.venv/bin/python -m scripts.soccol_pipeline.audit_synthetic_sampling --output-dir /tmp/soccol_sampling_audit
```

Verify the frozen bank exactly matches the reviewed development source:

```bash
.venv/bin/python -m corrosion_datasets.datasets.soccol_pitting_potential.prepare_pipeline_assets \
  --check
```

Without `--check`, this command builds the two bank files and refuses to
overwrite existing versioned files. It does not rebuild formula coefficients.

The Optuna fold wrapper forwards `--soccol-reviewed-data` with fixed-column and
missing-aware model flags. The search command now defaults to reviewed data;
the same flag makes that choice explicit. It uses 58-column training and matching
evaluation, with separate `optuna_v4` outputs and a fingerprint recording the
schema, accepted-development bank identity, composition weighting and search
policy. A study using the previous weighting cannot silently resume with v3.
This reviewed route requires `target_rules_v2/calibration_summary.json` fitted
on the same accepted rows. It rejects missing or historical calibration.

The nine existing formulas have now been refitted in
`corrosion_datasets/analysis/soccol_pipeline/target_rules_v2/`. Each validation
fold fitted its filling values, scaling and coefficients using only its context
rows. All-development coefficient centers used the same 2,950 accepted rows.
Definitions, coefficient anchors, upper bounds and anchor strength (0.1) were
retained. Per-fold states, coefficients, scores and held-out predictions are
saved alongside the summary, and the reviewed search loader uses these files.

Reproduce the calibration into a new directory with:

```bash
.venv/bin/python -m scripts.soccol_pipeline.calibrate_target_rules \
  --output-dir /tmp/soccol_formula_refit
```

Existing nonempty output directories are not overwritten. The highest mean
development-fold Spearman was **0.571158** for
`pren_n_coupled_mns_weak_anions`; this was a formula score, not a final-model
evaluation. All nine rules remained in the training mixture.

## Current Optuna search

The reviewed Soccol search follows EPIT's setup, with its own composition-noise
range based on the historical Soccol results:

- Informed ratio: fixed **100%**, excluded from Optuna's searched parameters.
- MLP share: continuous **0–1**; the remaining share uses the tree SCM.
- Formula contribution: continuous **0–1**.
- Composition perturbation strength: continuous **0–0.15** (EPIT uses **0–0.30**).
- Budget: **50 trials**, **10 startup trials**, **1,000 training steps** per
  trial and a **10,000-step scheduler**.
- Evaluation: all **five frozen development folds**, eight estimators and
  mean development-fold Spearman as the objective.
- Coefficient variation: fixed a priori at **0.8**, with seed **42**, exactly
  as in EPIT. This is not an Optuna parameter or a value selected from Soccol
  scores. The same setting applies to final training at every informed ratio.

The new study is `soccol_pipeline_optuna_empirical_features_scm_target_v6`.
It uses the reviewed `soccol_pitting_features_v3` bank and `target_rules_v2`
coefficients. Search outputs go to `analysis/soccol_pipeline/optuna_v4/`.
Coefficient sampling multiplies the fitted coefficients by independent factors
between 0.2 and 1.8, then normalizes and enforces the existing bounds. Zero
coefficients stay zero. Both pipelines use the same sampling implementation;
0.8 describes the initial multiplier range, not a variance or a guaranteed
final percentage change after normalization.
The sbatch launcher stores its journal under the existing shared
`/home/fcolanto/RZ-Dienste/hpc-user/fcolanto/optuna/<study>/` location, so workers
on different nodes can join the same study. Each worker requests 50 trials by
default, matching EPIT. A mismatched study fingerprint is rejected.

```bash
sbatch scripts/soccol_pipeline/run_optuna.sbatch
```

The update was checked using focused tests and command-only previews; no
Optuna training job was launched.

The [historical search review](../../corrosion_datasets/analysis/soccol_pipeline/optuna_range_review_v1/README.md)
checked all 50 completed old trials. The approved change kept the full 0–1 MLP
and formula ranges but reduced composition perturbation to 0–0.15: Soccol's
old top ten used 0.019–0.045, unlike EPIT's preference near its old upper limit.
The sampler, study fingerprint and trial metadata use the same 0–0.15 range.
A study already bound to 0–0.30 is rejected rather than silently resumed.

## Current final training and evaluation

Final training now defaults to the **best completed trial** in the reviewed
v6 study; the launcher no longer pins old trial 19. It validates the selected
trial's saved results, reviewed data, sampling bank and refitted formulas.
When another node's proxy files are missing, automatic selection uses the
validated shared-journal record, matching EPIT. Files available locally still
require matching hashes, and incomplete journal records are rejected. The final
manifest records the verification method and missing files. A trial-number
override is not required for this fallback.
It trains for **10,000 steps**, then chooses and freezes a checkpoint using
only the five accepted development folds.

```bash
sbatch scripts/soccol_pipeline/train_final.sbatch
```

The saved Optuna trial has no searched informed-ratio parameter; final training
recovers its fixed **100%** value from the study fingerprint. The existing
`--override-informed-prior-ratio` option and `OVERRIDE_INFORMED_PRIOR_RATIO`
launcher variable continue to accept **0.25, 0.50, 0.75 and 1.00**. Overrides
retain the selected formula/sampling settings and receive separate run names
and output directories by default. For example:

```bash
sbatch --export=ALL,OVERRIDE_INFORMED_PRIOR_RATIO=0.25 scripts/soccol_pipeline/train_final.sbatch
```

Reviewed final outputs use `analysis/soccol_pipeline/final_v2/` and
`checkpoints/soccol_pipeline_final_v2/`. The default final manifest is
`analysis/soccol_pipeline/final_v2/soccol_pipeline_optuna_empirical_features_scm_target_v6/final_model_manifest.json`.
After final training has written that manifest, evaluate it with:

```bash
sbatch scripts/soccol_pipeline/evaluate_final.sbatch
```

For an override run, set `FINAL_MODEL_MANIFEST` to that run's own frozen
manifest. Final evaluation uses **2,950 accepted development context rows**
and the exact **805 original final queries**, including the ten unresolved
compositions. It uses the reviewed **58-column** schema, shared category codes
and missing-aware input handling. The frozen manifest records accepted context
IDs, all final-query IDs and the reviewed input identity; validation checks
membership as well as counts.

Optional `--context-mode development-folds` diagnostics reduce only context
to **622, 615, 638, 445 or 630** accepted rows. Every diagnostic still evaluates
the same 805 outer queries. They are never replaced with development queries.

## Reviewed baseline and checkpoint comparisons

Development baselines can be evaluated before training a new Soccol model:

```bash
.venv/bin/python -m scripts.soccol_pipeline.evaluate_baseline_folds --device cpu
```

This uses the generic `step-1000.ckpt`, pretrained TabICL v2 and fixed CatBoost
on the same five accepted development folds as Optuna. Validation counts are
622, 615, 638, 445 and 630; the remaining accepted rows supply each context.
The 805 final queries are excluded. Results and their mean across the five
folds go to `analysis/soccol_pipeline/baseline_folds_reviewed_v1/`. The summary
records reviewed input hashes, original row membership and model settings.
Use `--dry-run` to print commands, or `--auto-output-dir` to preserve previous
results in a separate numbered directory.

After final training, compare the Soccol and generic checkpoint curves with
pretrained TabICL v2 and fixed CatBoost:

```bash
sbatch scripts/soccol_pipeline/evaluate_checkpoint_comparison.sbatch
```

The launcher calls the same Python entry point as local execution. Its default
candidate comes from the frozen reviewed v6 final-model manifest. There is no
fallback to old trial 19 when the new model is missing. Use
`FINAL_MODEL_MANIFEST` for a specific informed-ratio run, or `CANDIDATE_RUN`
for an explicit checkpoint directory. Local equivalents are
`--final-model-manifest` and `--candidate-run`.

Every model receives the reviewed 58-column inputs and the same 2,950 accepted
development context rows and 805 frozen final queries. Soccol uses its fixed
column order and learned missing-value handling. Generic and pretrained models
retain their existing handling; CatBoost receives all nine reviewed category
columns, including the reference type. Numeric `CP_agitation` is not marked as
a category. Category codes and source NaNs come from the reviewed input path.

The comparison retains every run's available interval-filtered checkpoints
and latest endpoint, including steps present in only one run. It requires all
four models, exact context/query membership and all six metric SVGs. Output
directories must be empty; reviewed defaults are separate from old results.
Optional `--context-folds 5` / `CONTEXT_FOLDS=5` changes only the context and
averages its five evaluations; all 805 final queries remain fixed.

Preview before the new run exists:

```bash
.venv/bin/python -m scripts.soccol_pipeline.evaluate_checkpoint_comparison \
  --candidate-run /path/to/future/soccol/run --device cpu --dry-run
```

Checkpoint curves on final queries remain diagnostics. They must not change
the checkpoint already selected using development folds. These commands were
verified with mock model scores; no new baseline or final-model scores were
computed during implementation.

## Historical v1 workflow

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

The historical search remains available explicitly:

```bash
.venv/bin/python -m scripts.soccol_pipeline.run_optuna --legacy-data \
  --storage journal:///path/to/historical_soccol/study_journal_v1.log
```

Reproduce the historical final stages explicitly (the sbatch launchers now
default to reviewed data):

```bash
.venv/bin/python -m scripts.soccol_pipeline.train_final --legacy-data \
  --storage journal:///path/to/historical_soccol/study_journal_v1.log
.venv/bin/python -m scripts.soccol_pipeline.evaluate_final --legacy-data \
  --final-model-manifest /path/to/historical_soccol/final_model_manifest.json
```

Run the old-EPIT-style checkpoint comparison on the held-out final test:

```bash
sbatch --export=ALL,LEGACY_DATA=true,OUTPUT_ROOT=corrosion_datasets/analysis/soccol_pipeline/legacy_checkpoint_comparison scripts/soccol_pipeline/evaluate_checkpoint_comparison.sbatch
```

This uses all 3,222 development rows as context and the 805 untouched rows as
queries, with no development-fold cross-validation. It evaluates all checkpoint
steps available in either the generic or Soccol run, plus pretrained TabICL v2 and fixed
CatBoost. The unchanged EPIT SVG trend writer is mandatory: missing plots fail
the run. These all-checkpoint final-test results are diagnostics and must not be
used to change the development-selected checkpoint.

The historical final stages reuse the EPIT implementation through Soccol-only
wrappers. Their output roots remain
`corrosion_datasets/analysis/soccol_pipeline/final_v1/` and
`checkpoints/soccol_pipeline_final_v1/`.
