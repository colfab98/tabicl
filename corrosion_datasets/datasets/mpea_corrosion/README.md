# MPEA Corrosion Database

The original candidate exports are retained. A separate frozen split for developing
an MPEA specialist is now available below. Raw files are unchanged.

## Frozen MPEA specialist split (2026-10-22)

Use [processed/splits_v1/](processed/splits_v1/) for the MPEA specialist workflow:

- **214 development rows:** formula checks, coefficient fitting, prior construction, Optuna and model selection.
- **53 final-test rows:** evaluate the chosen frozen model with the 214 development rows as context.
- **Five development folds:** 43 / 43 / 43 / 43 / 42 rows, for development and Optuna only.

This uses the same 267 source-screened rows and the existing 50-column shared
schema. It reuses the EPIT grouping, split optimization and fold refinement code,
following the approach also used for Soccol: compositions rounded to 0.01 wt.%,
connected groups at an L1 distance of at most 1.0 wt.%, and an 80/20 outer split.
The 116 composition groups stay intact across both the outer split and the folds.
Missing values are treated as zero for grouping only; exported values remain unchanged.

As in EPIT/Soccol, assignment considers target deciles, composition isolation,
temperature, chloride, pH and source. Material-family and test-method blocks are
omitted because comparable recorded labels are unavailable here. Source is not
a grouping boundary. This does not rebalance the three datasets for later merging.

The folder contains a manifest, assignments, report and hash lock, plus
`development.csv`, target-free `final_test_queries.csv`, separate
`final_test_scoring_targets.csv`, and the shared feature registry. Only the
registry's feature columns belong in model inputs; IDs and source metadata do not.
Future fitting and evaluation should load and verify this frozen split through
`load_frozen_mpea_split()` in `prepare_composition_split.py`.

The original 267-row external-query experiment and exploratory MPEA fivefold
assignments remain separate. These rows have already been used in exploratory
evaluations; this split does not make them newly collected, untouched data.

Check the saved split without changing it:

```bash
.venv/bin/python -m corrosion_datasets.datasets.mpea_corrosion.prepare_composition_split --validate-only
```

The same command without `--validate-only` created the split. It refuses to
overwrite an existing output folder. Future MPEA steps must follow the existing
EPIT/Soccol procedures; any necessary dataset-specific differences must be documented.

## Development formula check (2026-10-22)

The [formula report](../../analysis/mpea_target_rules_v1/README.md) records the
completed development-only screen. It reuses EPIT's fold calibration and
coefficient fitter on all five frozen MPEA development folds. Eleven reduced
EPIT/Soccol candidates and eight composition-only controls were evaluated;
the existing production rule registries were not changed.

All original rules need unavailable MPEA inputs. The reduced candidates omit
those effects explicitly instead of inventing nitrogen, sulfur, pH or ion values.
The best fitted candidate scores 0.2334 mean fold Spearman (0.2424 before fitting).
A 13-row Ti/Nb-rich, Cr/Mo/W-free alloy group in nitric acid is a clear mismatch
for the inherited material formulas. The report separates this observation
from claims about its chemical cause or trained-model performance.

Coefficients, preprocessing states and predictions are saved in the report
folder. The 53 final-test rows were not fitted or scored. Run the same screen
into a new output directory with:

```bash
.venv/bin/python -m corrosion_datasets.datasets.mpea_corrosion.compare_target_rules \
  --output-dir /tmp/mpea_formula_check
```

## Original candidate preparation

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
remains unresolved. The original candidate processor creates no splits or evaluations;
the specialist split above is a separate operation.

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
