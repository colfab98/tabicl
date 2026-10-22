# Zhang 2023 duplex-steel external evaluation

Source: [Zhang et al., Nature Communications 14, 7869 (2023)](https://doi.org/10.1038/s41467-023-43752-8).
The archived source workbook contains individual pitting potentials in mV vs SCE.
Supplementary Tables 3 and 6 provide the experimental compositions in wt%.

The benchmark retains 56 measurements: seven experimental compositions, eight
replicates each, in one environment. The 16 industrial measurements in sheet 6d
are excluded because their separately measured compositions were not verified.
Theoretical compositions in Supplementary Tables 1 and 5 are not used.

`prepare_evaluation.py` exports the original 50-feature schema. Explicit zeros
remain zero; unreported elements and dashes remain NaN. Fe is the remainder of
the reported composition because the source explicitly specifies Fe balance.
This remainder is approximate where trace elements are unreported.

Conditions are 72 C, pH 8.2, 2000-grit final finish, 1 cm2, and 0.333 mV/s.
The seawater recipe follows reference 65, [Xin and Li (2014), Table 1](https://doi.org/10.1016/j.corsci.2013.12.004).
`raw/seawater_recipe.json` records the source and calculation basis. Chloride
includes NaCl, MgCl2, CaCl2, KCl and SrCl2; sulfate and bromide use their listed
salts. Bicarbonate is not assigned to carbonate, and hydroxide is not inferred
from pH. These are nominal preparation concentrations, not measured activities.
Unverified procedure units remain NaN; reported seconds and volts are retained
in `processed/row_audit.csv`. Aeration and agitation remain unknown.

All 56 rows are frozen external queries. Labels are stored separately in
`zhang2023_scoring_targets.csv`. The original 4,787 EPIT/Soccol context rows and
feature registry are copied byte-for-byte and verified by SHA-256. Neither
Zhang nor MPEA labels enter context, preprocessing fits or checkpoint selection.
Alloy IDs and the source's target-sorted replicate numbers are metadata only.

Prepare once:

```bash
.venv/bin/python corrosion_datasets/datasets/zhang2023_duplex_pitting/prepare_evaluation.py
```

Run or resume the same seven-model checkpoint comparison on CPU:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m scripts.combined_corrosion.evaluate_zhang2023 \
  --device cpu --workers 8 --torch-threads 8
```

This reuses the MPEA evaluator's model factories, inference settings, metrics
and plots. Static pretrained/CatBoost predictions are reused across checkpoint
steps. Available local checkpoints are scored every 500 steps plus the saved
endpoints included by the existing evaluator. Missing model checkpoints are
not substituted. The frozen protocol records every evaluated file and hash;
completed jobs are resumed only under the identical protocol.

Outputs are under `corrosion_datasets/analysis/zhang2023_duplex_pitting/comparison_cpu/`:
`protocol.json`, per-job predictions, `predictions.csv`, `rows.csv`, `wide.csv`,
`summary.csv`, `results.json`, and SVG metric plots. Seven distinct inputs and
Nb-dependent inclusion modification limit interpretation of the 56-row scores.
