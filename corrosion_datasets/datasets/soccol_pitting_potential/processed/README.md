# Processed Soccol benchmark v1

Generated from the unchanged 2024 workbook.

- `soccol_processed_all_rows.csv`: all 4,460 processed/audit rows.
- `soccol_regression_event1.csv`: 4,027 numeric actual-pitting rows for ordinary regression.
- `soccol_breakdown_survival.csv`: 4,384 numeric breakdown rows with defined censoring status.
- `feature_manifest.json`: predictor, target, and audit-only roles.
- `preprocessing_manifest.json`: transformations, counts, and output checksums.
- `splits_v1/`: frozen 80/20 composition-separated split plus five development folds.

Run `../prepare_benchmark.py` to reproduce preprocessing. `../prepare_composition_split.py` reproduces the established EPIT split method, but refuses to overwrite the frozen `splits_v1` directory.

The model uses 37 predictors: 13 composition columns, 20 other numeric columns,
and 4 categorical columns. Composition and ion blanks are encoded as zero.
Continuous means and categorical mappings are fitted on context rows only;
missing or unseen categories become `-1`. The retained composition missingness
columns and `source` are audit-only and never enter the model.

`Fe` is an approximate remainder only where the source and row chemistry support that calculation. Rows without adequate major composition use `Fe=0` together with `Fe_missing=1`. No raw file was changed.
