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
