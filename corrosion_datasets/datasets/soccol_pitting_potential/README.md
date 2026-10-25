# Soccol Pitting-Potential Database

This folder preserves the original workbook collection from Dimitri Soccol's
`Pitting_potential_database` repository. The 2024 workbook is the current
primary version; the two 2023 workbooks are retained as upstream historical
snapshots.

- Source provenance, versions, checksums, and licensing: `manifest.yml`
- Workbook fields and confirmed conventions: `schema_summary.md`
- Paper and source literature: `literature.md`
- Implemented decisions and remaining sensitivity analyses: `investigation_notes.md`
- Per-source material and handling registry:
  `../../analysis/soccol_source_conventions/source_conventions.csv`
- Reproducible preprocessing builder: `prepare_benchmark.py`
- Previous-EPIT-compatible split builder: `prepare_composition_split.py`
- Synthetic profile and calibrated-rule builder: `prepare_pipeline_assets.py`
- Optuna launcher: `../../../scripts/soccol_pipeline/run_optuna.py`
- Model-ready tables, frozen split, and feature roles: `processed/`

The files under `raw/` are preserved copies. No composition blank has been
replaced, Fe has not been reconstructed, and the confirmed `2000Russell.Ti`
error has not been altered in the raw workbook. All corrections and encodings are
confined to `processed/`.

## Reviewed composition and environment export

The latest standalone [reference table](processed/soccol_reference_v1/soccol_processed.csv)
combines 24 reviewed element columns in wt.% with the 33 environment/history
features from environment v2 and the categorical `reference_type` feature
(**58 model features**). It retains all 4,027 original regression rows,
their composition-review status and earlier values under `legacy_` names.

The reference column contains **565 SCE**, **112 AgAgCl**, and **3,350 missing**
values, matching the existing source-specific policy. Original targets were
unchanged and no electrode conversion was applied. The verification evidence
is in [reference_type_audit.csv](processed/soccol_reference_v1/reference_type_audit.csv)
and the [processing report](../../DATASET_PROCESSING.md#epit-and-soccol-reference-type-feature).
The preceding composition v1 export was preserved.

- [Accepted development features](processed/soccol_reference_v1/soccol_development_features.csv):
  2,950 rows; 272 development rows were withheld because their composition
  remained incomplete or ambiguous.
- [Frozen final-query features](processed/soccol_reference_v1/soccol_frozen_final_query_features.csv):
  all 805 original queries, including 10 unresolved `1994Malik` rows.
  **This export is not ready for final evaluation until those compositions are resolved.**
  The frozen queries were not dropped or replaced.
- [Rows requiring review](processed/soccol_composition_v1/composition_review_required.csv)
  records each unresolved composition and original partition.
- [Processing report](../../DATASET_PROCESSING.md#soccol-composition-processing),
  [builder](prepare_soccol_composition.py) and
  [source decisions](soccol_composition_rules_v1.json) document the corrections,
  evidence and remaining uncertainties.

Existing model and shared-schema paths remain unchanged. This export does not
automatically switch formula fitting, training or evaluation to the revised data.

```bash
.venv/bin/python -m corrosion_datasets.datasets.prepare_reference_types soccol \
  --output-dir /tmp/soccol_reference_check
```

The preceding composition export was built with:

```bash
.venv/bin/python -m corrosion_datasets.datasets.soccol_pitting_potential.prepare_soccol_composition \
  --output-dir /tmp/soccol_composition_check
```

## Reviewed environment export

The standalone [environment export](processed/soccol_environment_v2/) contains
the agreed preparation, solution and test-condition features. It excludes
`CP_ion_other` from extraction and model inputs, preserves unknown values as
missing, and retains the frozen regression rows and split assignments.

- [Environment features](processed/soccol_environment_v2/soccol_environment_features.csv)
  contains `row_id` and 33 environment/history columns, without composition or targets.
- [Reference table](processed/soccol_environment_v2/soccol_processed.csv)
  also retains the legacy composition and targets. Original environment values
  use `raw_` prefixes; this step did not correct or certify Soccol composition.
- [Processing report](../../DATASET_PROCESSING.md#epit-and-soccol-source-consistency-review)
  records the rules, coverage, exclusions and unresolved source conflicts.
- [Original builder](prepare_soccol_environment.py) and
  [reviewed text rules](soccol_environment_rules_v1.json) produced v1.
  The [source-consistency review](../review_epit_soccol_environment.py) produced v2.

The v2 review left 217 preparation times missing because only air exposure
was identified, without evidence linking the duration to a separate solution
treatment. It retained the 36 air-labelled rows with separate electrochemical
preparation information. It also sorted multiple preparation-operation names
as in EPIT for 32 rows. Original values and the v1 export were preserved;
[review_changes.csv](processed/soccol_environment_v2/review_changes.csv) recorded every change.

Existing training, evaluation and shared-schema paths still use their previous
exports. The environment export does not switch those paths automatically.

```bash
.venv/bin/python -m corrosion_datasets.datasets.review_epit_soccol_environment soccol \
  --output-dir /tmp/soccol_environment_review_check
```
