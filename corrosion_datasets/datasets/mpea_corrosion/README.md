# MPEA Corrosion Database

Prepared as an **evaluation candidate**, not a frozen benchmark. Raw files are unchanged.

- [Full candidate table](processed/mpea_evaluation_candidates.csv): 332 eligible rows; 17 composition features and 7 environmental feature slots.
- [Source-screened subset](processed/mpea_source_screened_candidates.csv): 267 rows, excluding shared and unresolved source references; not proven measurement-independent.
- [Preparation rules and issues](preprocessing_plan.md): conversions, missing values, exclusions, and source-overlap limitations.
- [Row audit](processed/mpea_row_audit.csv): all 619 workbook rows, stable Excel row IDs, and review flags.
- [Source audit](processed/mpea_source_audit.csv): DOI matching against EPIT and Soccol.
- [Feature manifest](processed/feature_manifest.json): exact predictor order and units. Only listed predictors belong in model inputs.
- [Preparation manifest](processed/preprocessing_manifest.json): counts, input/output hashes, and unresolved checks.

The 3 incomplete-composition rows were excluded from both exports, not deleted
from the raw workbook or audit. Of the 332 retained rows, 22 share source DOIs
with EPIT and 43 have unresolved source references. The paper reports 306
pitting-potential entries, whereas the workbook contains 335; the difference
remains unresolved. No train/test splits or model evaluations were created.

From the repository root:

```bash
.venv/bin/python corrosion_datasets/datasets/mpea_corrosion/prepare_evaluation.py --overwrite
.venv/bin/python -m pytest tests/test_mpea_preparation.py -q
```

`--overwrite` replaces only this processor's six named generated files. Omit
it when writing to a new directory with `--output-dir`.

Original provenance is in [manifest.yml](manifest.yml); the raw workbook
description is in [schema_summary.md](schema_summary.md). The source
[MPEA Corrosion Database, version 1](https://data.mendeley.com/datasets/nskb9khgsm/1)
is distributed under CC BY 4.0. These processed files are derived from it;
the changes are documented below.
