# Electrochemical Metrics For Corrosion Resistant Alloys

Source and raw-file provenance are in `manifest.yml`. The inspected workbook schema, feature groups, targets, leakage notes, and prior relevance are in `schema_summary.md`.

The latest standalone EPIT table, including corrected compositions, reviewed
environment/history features, preparation details and supported ion mappings,
was saved in
[`processed/epit_reference_v1/epit_processed.csv`](processed/epit_reference_v1/epit_processed.csv).
The feature-only view was saved in
[`epit_features.csv`](processed/epit_reference_v1/epit_features.csv), with **70
features** and names/units in
[`feature_registry.json`](processed/epit_reference_v1/feature_registry.json).
The complete processing rules and results were documented in
[`DATASET_PROCESSING.md`](../../DATASET_PROCESSING.md#epit-and-soccol-reference-type-feature).

Reference v1 added `reference_type = SCE` to all 760 rows after checking the
original workbook target header. All targets, existing features and frozen
splits were preserved. The evidence was saved in
[`reference_type_audit.csv`](processed/epit_reference_v1/reference_type_audit.csv).
It inherited composition v2, including the Ti correction below.

The standalone EPIT pipeline subsequently adopted the reviewed **composition
values only** in its loader and direct evaluator. Its default composition bank
became `epit_dataset_v3`, with 400 distinct accepted compositions, paired with
`epit_fe_nicrmo_features_v2`. Earlier banks were preserved. The pipeline still
used its historical 28-column layout and environment processing; this step did
not yet activate all 70 features. Details were recorded in the
[pipeline composition update](../../DATASET_PROCESSING.md#epit-pipeline-adoption-of-reviewed-compositions).

The next update restricted sampling to the frozen outer development partition.
At that stage, the defaults became `epit_dataset_v4` and
`epit_fe_nicrmo_features_v3`. The active Fe/Ni–Cr–Mo sampler drew from **411 Fe
and 41 Ni–Cr–Mo development rows**, using 255 distinct compositions and 452
environment tuples. Means and category lists were also fitted without final
queries. Families, target formulas, real evaluation rows and the 28-column
layout were retained. See the
[development-only sampling update](../../DATASET_PROCESSING.md#epit-pipeline-development-only-sampling).

The subsequent environment update activated **all 70 reviewed features** for
the default EPIT SCM-plus-formula workflow, using `epit_fe_nicrmo_features_v4`
with the same `epit_dataset_v4` composition bank. Each environment draw retained
all 46 environment/history fields and their missing values from one of the same
452 development rows. Training enabled the missing-value embedding, and the
generated evaluation commands selected the matching reviewed 70-column table.
The new default study name became
`epit_pipeline_optuna_empirical_features_scm_target_v9`. Previous profiles and
explicit legacy evaluation commands remained available. See the
[70-column sampling update](../../DATASET_PROCESSING.md#epit-pipeline-70-column-environment-sampling).

Formula fitting then adopted the reviewed composition and environment inputs.
The same nine formulas were refitted on the existing development folds, and
their updated coefficients and scores were saved in `target_rules_v4`.
Python and Slurm training defaults were connected to this refit. The frozen
final queries were excluded. See the
[reviewed formula-fitting update](../../DATASET_PROCESSING.md#epit-pipeline-formula-refit-on-reviewed-inputs).

The composition v2 review corrected `epit_r000359`: Ti changed from 1.88 to
1.86 wt.% and its Fe remainder from 77.592 to 77.612 wt.%. We retained all
12 Re-study rows and all 10 rows with combined Nb/Ta information as recorded,
with their uncertainty documented in
[`composition_retained_uncertainties.csv`](processed/epit_composition_v2/composition_retained_uncertainties.csv).
The user accepted these uncertainties for the common dataset comparison.
No rows were removed: all **608 development rows and 152 frozen final queries**
remained. The earlier environment v3 export was preserved.

The v3 review left one additional contradictory chloride value missing:
`epit_r000148` named 1 M HCl but reported zero chloride. Both original reports
were preserved. The [cell-level change log](processed/epit_environment_v3/review_changes.csv)
recorded the correction, and the earlier v2 export was preserved.

The v2 extension added `Prep_grinding_grit`, `surface_preparation_method`,
`preparation_solution_type`, `Prep_time`, `CP_time`, `CP_agitation`, `CP_Br`
and `CP_SO4`. Unknown values remained missing. Preparation voltage, preparation
pH and roughness remained deferred. The extension was generated with
[`prepare_epit_preparation.py`](prepare_epit_preparation.py), using the reviewed
[preparation rules](epit_preparation_rules_v1.json).

The earlier [environment v1 export](processed/epit_environment_v1/) and
[composition-only export](processed/epit_composition_v1/epit_processed.csv)
were both preserved.
All exports retained all 760 EPIT rows and the original split assignments.
Existing shared datasets, training and evaluation paths were not changed.

To reproduce the latest reference v1 export from composition v2:

```bash
.venv/bin/python -m corrosion_datasets.datasets.prepare_reference_types epit \
  --output-dir /tmp/epit_reference_check
```

To reproduce the earlier composition v2 export from the preserved environment v3 export:

```bash
.venv/bin/python -m corrosion_datasets.datasets.electrochemical_metrics_alloys.review_epit_composition \
  --output-dir /tmp/epit_composition_review_check
```

To reproduce the reviewed v3 export from the preserved v2 export:

```bash
.venv/bin/python -m corrosion_datasets.datasets.review_epit_soccol_environment epit \
  --output-dir /tmp/epit_environment_review_check
```
