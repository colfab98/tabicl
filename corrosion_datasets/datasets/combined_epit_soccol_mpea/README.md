# Combined EPIT/Soccol context and MPEA queries

This directory contains only derived artifacts. It does not modify the EPIT,
Soccol, or MPEA source datasets.

Run:

```bash
.venv/bin/python -m corrosion_datasets.datasets.combined_epit_soccol_mpea.prepare_shared_tables
```

Generated under `processed/`:

- `epit_soccol_context.csv`: 760 EPIT and 4,027 Soccol labeled rows.
- `mpea_queries.csv`: 267 source-screened MPEA rows with no target column.
- `mpea_scoring_targets.csv`: the row-aligned MPEA SCE targets for final scoring.
- `shared_feature_registry.json`: fixed predictor order, units, and source mappings.
- `context_missingness_profile.json`: counted EPIT/Soccol context masks for training augmentation.
- `combined_prior_config.json`: frozen generic/EPIT/Soccol prior regimes and source-command hashes.
- `preparation_manifest.json`: input/output hashes and row counts.

The target is the reported pitting potential in mV. `reference_type` identifies
whether that value is relative to verified SCE or Ag/AgCl. Soccol sources
without primary-source verification have `reference_type=NaN`; the reason is
preserved in `reference_type_evidence`, and their reported values are not
silently converted.

Missing predictors remain `NaN`. In particular, zero-filled Soccol composition
cells are restored to `NaN` when the retained audit flag says the original cell
was unreported.

Freeze the selected EPIT and Soccol source-model settings with:

```bash
.venv/bin/python -m corrosion_datasets.datasets.combined_epit_soccol_mpea.prepare_prior_config --overwrite
```

Train the combined missing-aware model with
`--combined_corrosion_prior_config
corrosion_datasets/datasets/combined_epit_soccol_mpea/processed/combined_prior_config.json
--col_missing_value_embedding true --missingness_profile_path
corrosion_datasets/datasets/combined_epit_soccol_mpea/processed/context_missingness_profile.json
--min_features 50 --max_features 50 --max_classes 0 --prior_n_jobs 1`.
The checkpoint then preserves NaNs in the scikit-learn preprocessing path and
maps them to a learned token. The flag also disables column feature grouping.
Fixed-schema evaluation must use
`remove_constant_features=False` and `feat_shuffle_method="none"`.

Incoming datasets do not need to supply columns in model order. Align names
before evaluation:

```python
from corrosion_datasets.datasets.combined_epit_soccol_mpea.schema_alignment import align_feature_frame

features, report = align_feature_frame(raw_table, unknown_columns="ignore")
```

The aligner applies only approved aliases, returns the canonical 50-column
order, fills absent features with `NaN`, and reports ignored columns. It does
not infer units or use fuzzy name matching. Missing-aware estimators also
reorder exact canonical DataFrame names automatically during prediction.

Missing-aware training uses `context_missingness_profile.json`. It samples
EPIT/Soccol source-shaped masks for both composition and environment columns;
the profile contains no MPEA query or scoring information. Informed synthetic
tables carry an internal source ID, so EPIT-informed tables receive only EPIT
masks and Soccol-informed tables receive only Soccol masks. Generic tables
sample either source mask using the empirical context row proportions.

The frozen prior mix is 25% generic, 25% EPIT-informed, and 50%
Soccol-informed. Generic tables retain the 0.7/0.3 MLP/tree mix and categorical
generation probability 0.2. Each informed regime retains its own selected
MLP/tree mix, formulas, coefficients, and coefficient variation; those values
are never averaged. MPEA rows and targets are not used in prior generation or
training.

The active v1 composition schema contains the 24 elements supported by EPIT or
Soccol. The registry reserves every other periodic-table element, but MPEA-only
elements remain inactive because the source priors do not teach their effects.

To train with heavier missingness in queries while preserving observed
composition, use:

```bash
sbatch scripts/combined_corrosion/train_final_v2.sbatch
```

This policy protects observed composition values in both context and queries;
existing NaNs remain NaNs. Non-composition features retain the usual
source-derived missingness in context and queries. In addition, each
non-composition column has an independent 80% probability per synthetic table
of being hidden across **all query rows**, without additional masking in
context. This includes environmental, preparation, and categorical features;
the selection does not depend on MPEA. Targets are generated before masking
and remain unchanged.

`QUERY_COLUMN_MASK_PROBABILITY` overrides the initial 0.8 setting. The launcher
uses a fresh `checkpoints/combined_epit_soccol_mpea/final_v2_seed42`
directory and refuses to overwrite an existing run unless `--resume` is explicit.
Resume the existing V2 run to the fixed 10,000-step endpoint with:

```bash
sbatch scripts/combined_corrosion/train_final_v2.sbatch --resume
```

To validate locally without submitting or training:

```bash
bash scripts/combined_corrosion/train_final_v2.sbatch --resume --dry-run
```

Resume requires an existing checkpoint and checks its prior hash, missingness
profile/policy, architecture, optimizer, and learning-rate schedule on CPU. It
pins the latest checkpoint explicitly and restores model, optimizer, scheduler,
and step counter. The shared launcher locks the checkpoint directory against
concurrent writers through that launcher. New checkpoints also save Python,
NumPy, Torch CPU/CUDA RNG states, private coefficient generators, consumed
data-loader progress and worker states, plus the AMP scaler. On restart,
unconsumed prefetched batches are regenerated from the saved worker states;
the worker count and prior settings must match. This covers combined and shared
specialist runs with CPU prior generation. Identical numerical training also
depends on the same software/hardware and deterministic operations.
Historical checkpoints lack this state: their old stream cannot be recovered,
but checkpoints saved after resuming them support future sampling restoration.
The fixed 10,000-step cosine schedule is preserved.

Checkpoints record both
new masking settings. The original `train_final.sbatch` defaults remain
unchanged. When evaluating this run, pass its path as `--candidate-run` and
choose a separate `--output-dir`; the frozen MPEA query partition stays fixed.

To retrain the individual source recipes using the same shared V2 setup:

```bash
sbatch scripts/combined_corrosion/train_epit_shared_v2.sbatch
sbatch scripts/combined_corrosion/train_soccol_shared_v2.sbatch
```

EPIT uses 50% generic and 50% EPIT-informed tables; Soccol uses 100%
Soccol-informed tables. Both reuse `train_final_v2.sbatch`, including the
50-column schema, missing-value embedding, composition protection, query
masking, and default 10,000-step schedule. Each informed generator retains its
frozen formulas, coefficients, feature profile, and perturbation settings.
Generic tables use the combined V2 generator with 50 features, rather than the
old EPIT run's 28-feature generic tables. These are controlled retrainings in
the shared setup, not exact reproductions of the old specialist runs.

To test sequential transfer from the existing EPIT shared V2 step-5,000 model:

```bash
sbatch --job-name=epit5k_control scripts/combined_corrosion/continue_epit_v2.sbatch control
sbatch --job-name=epit5k_soccol scripts/combined_corrosion/continue_epit_v2.sbatch soccol
```

Both start from `epit_shared_v2_seed42/step-5000.ckpt`, retain its model,
optimizer and scheduler state, and continue to step 10,000 on the original
10,000-step learning-rate schedule. The control keeps 50% EPIT / 50% generic;
the other branch uses 100% Soccol. Their directories are
`checkpoints/combined_epit_soccol_mpea/epit_v2_5000_control_seed42` and
`checkpoints/combined_epit_soccol_mpea/epit_v2_5000_to_soccol_seed42`.
The historical parent lacks sampling state; both branches restart sampling
using the same default seeds. Each branch records its parent checksum and
recipe in `continuation.json`. Fresh launches refuse existing checkpoints.

Use `bash scripts/combined_corrosion/continue_epit_v2.sbatch control --dry-run`
(or `soccol`) to validate without training or creating branch files. Append
`--resume` to a branch's submission command to resume that branch's latest
checkpoint, including its new sampling state. The original specialist
checkpoints are read only.

Compare the two branches on the frozen MPEA queries with the same settings as
the previous comparisons:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python -m scripts.combined_corrosion.evaluate_epit_continuations \
  --device cpu --checkpoint all \
  --output-dir corrosion_datasets/analysis/combined_epit_soccol_mpea/epit5k_continuation_comparison_cpu
```

For Slurm, use `sbatch scripts/combined_corrosion/evaluate_epit_continuations.sbatch`.
This reuses `evaluate_final.comparison_command`, including the 4,787 labeled
context rows, 267 frozen MPEA queries, model-specific preprocessing, inference
settings and CatBoost parameters. Output labels are `epit5k_control`,
`epit5k_to_soccol`, `generic_tabicl`, `pretrained_tabicl_v2` and `catboost`.
It uses steps shared by both branches and generic; the original specialists
and combined V2 do not restrict the available steps. Override checkpoint
directories with `--control-run`, `--soccol-run` and `--generic-run` if needed.
Results include the existing JSON/CSV tables and plots. Slurm output defaults
to `epit5k_continuation_comparison_<job-id>`. The parent step-5,000 result
remains in the original shared-specialist comparison.

Checkpoints go to `checkpoints/combined_epit_soccol_mpea/epit_shared_v2_seed42`
and `checkpoints/combined_epit_soccol_mpea/soccol_shared_v2_seed42`. Each run
saves its effective `prior_config.json` there, including the base configuration
hash; checkpoints record the effective configuration hash and regime weights.
Existing checkpoints or a conflicting saved configuration cause the launcher
to stop. The original launchers and frozen base configuration are unchanged.
The existing training environment overrides still apply; `PRIOR_CONFIG` selects
the base configuration and `CHECKPOINT_DIR` selects a fresh output directory.
Use `bash <launcher> --dry-run` to save and inspect the configuration without
starting training. These new 50-column models require the combined evaluator's
missing-aware preprocessing, not the old native-schema specialist evaluator;
the outer MPEA query partition remains fixed.

Compare both new specialists, combined V2, local generic TabICL, pretrained
TabICL V2, and CatBoost with:

```bash
sbatch scripts/combined_corrosion/evaluate_shared_specialists.sbatch
```

Or run locally on CPU:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python -m scripts.combined_corrosion.evaluate_shared_specialists \
  --device cpu --checkpoint all \
  --output-dir corrosion_datasets/analysis/combined_epit_soccol_mpea/shared_specialist_comparison_cpu
```

The new wrapper reuses the existing combined evaluator's settings and frozen
4,787 context / 267 query rows. The new specialists and combined V2 all retain
50 columns, fixed column order, and learned missing-value handling. Generic
and pretrained TabICL retain the same native preprocessing as earlier
comparisons; CatBoost settings are also unchanged. Output labels are
`epit_shared_v2`, `soccol_shared_v2`, `combined_corrpfn_v2`, `generic_tabicl`,
`pretrained_tabicl_v2`, and `catboost`.

By default, `--checkpoint all` evaluates only steps present in **all four local
runs**. Use `--skip-combined` to compare later specialist checkpoints with
baselines when combined V2 has no matching checkpoints. `--checkpoint <step>`
selects one explicit step. The `--epit-run`, `--soccol-run`, `--combined-run`,
and `--generic-run` options override the checkpoint directories. This wrapper
is for the new 50-column specialists; the old 28/37-column checkpoints still
use `evaluate_specialist_transfer.py`.

Results include `results.json`, `rows.csv`, `wide.csv`, `summary.csv`, and the
same plots as earlier comparisons. The Slurm launcher writes to
`shared_specialist_comparison_<job-id>` under this task's analysis directory;
the Python entry point defaults to `shared_specialist_comparison`. Earlier
evaluation outputs are separate.

Evaluate the currently available training checkpoints with:

```bash
sbatch scripts/combined_corrosion/evaluate_final.sbatch
```

The evaluator uses all 4,787 labeled context rows and only the 267 fixed MPEA
rows as queries. It compares the combined missing-aware checkpoint, the
unchanged local generic TabICL checkpoint, official pretrained TabICL V2, and
CatBoost. Generic and pretrained TabICL retain their original preprocessing;
only the combined checkpoint uses the learned missing-value embedding. The
ordinary prediction path retains columns that are entirely missing in MPEA:
generic/pretrained models impute or encode their missing values using fitted
context preprocessing. Only an explicit `feature_mask` requests feature
exclusion for explanations. Evaluation records the prediction-time feature
count. Comparisons saved before this correction require a fresh evaluation.
The combined script is a thin wrapper over the same shared corrosion evaluator used
by the existing EPIT and Soccol comparisons. By default it evaluates every
checkpoint step currently shared by the combined and local generic runs;
`--checkpoint <step>` restricts it to one explicit step.

### Separate MPEA-only five-fold comparison

The MPEA adaptation benchmark uses only the same 267 SCE MPEA rows. Five newly
frozen outer folds each supply queries once, with the other four folds as labeled
context. It is exploratory and does not replace the original EPIT/Soccol-context
final evaluation. No EPIT or Soccol rows enter its context or preprocessing.

Composition groups use connected components at L1 distance <= 1.0 wt.%, after
rounding the 24 shared elemental columns to 0.01 wt.%. Missing values are filled
with zero **for grouping only**; model inputs keep their original missingness.
Groups are assigned whole, balancing row counts without using target values.
Seed 42 gives 116 groups and query-fold sizes 53, 53, 54, 53, and 54.
This groups compositions, not publications.

```bash
sbatch scripts/combined_corrosion/evaluate_mpea_folds.sbatch
```

For a local CPU comparison at one step:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python -m scripts.combined_corrosion.evaluate_mpea_folds \
  --device cpu --checkpoint 1000 \
  --output-dir corrosion_datasets/analysis/combined_epit_soccol_mpea/mpea_only_fivefold_1000_cpu
```

The seven models are COMB V2, the shared 50-column EPIT and Soccol specialists,
COMB with 50% generic / 25% EPIT / 25% Soccol, generic TabICL, pretrained TabICL
V2, and CatBoost. Their inference settings match the previous comparisons.
Each fold fits preprocessing and CatBoost only on its own context. TabICL
weights remain frozen, with fresh estimator preprocessing for every fold.

`--checkpoint all` evaluates available checkpoints every 500 steps across the
five local runs by default, also including each run's final checkpoint when it
falls between those steps. Use `--checkpoint-step-interval 0` to include every
saved step. Each step uses only models with a saved checkpoint. Each model's results
continue through its own last checkpoint. Missing model/step combinations stay
blank in comparison tables. The pretrained and CatBoost results are computed
once per fold and reused across checkpoint comparisons. Use `--combined-run`,
`--epit-run`, `--soccol-run`, `--g50-run`, and
`--generic-run` to override directories. The existing checkpoint interval,
minimum step, estimator, random state, and CatBoost options remain available.

The first run creates `processed/mpea_only_folds_v1.json`, binding row IDs, groups,
folds, and input hashes. Later runs verify that exact manifest. `--prepare-only`
creates/verifies it without loading models; `--split-manifest` selects another
manifest path and `--split-seed` controls target-blind assignment ties.

Results go to the selected directory, with `results.json`, per-fold `rows.csv`,
`summary.csv`, averaged `wide.csv`, and checkpoint plots. `outer_fold_1` through
`outer_fold_5` contain individual fold tables. The top-level `summary.csv` reports
the **unweighted mean and sample standard deviation of five fold metrics**;
it does not average predictions or report a pooled correlation. Every model
scheduled at a step must complete all five folds before its checkpoint summary
is published.
Undefined metrics retain a missing aggregate and an explicit valid-fold count.
Completed checkpoints remain saved if a subsequent evaluation fails; details
are written to `failure.json`. Reruns overwrite result files in the selected
output directory; a successful run removes any previous `failure.json`.
