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
directory and refuses to overwrite an existing run. Checkpoints record both
new masking settings. The original `train_final.sbatch` defaults remain
unchanged. When evaluating this run, pass its path as `--candidate-run` and
choose a separate `--output-dir`; the frozen MPEA query partition stays fixed.

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
