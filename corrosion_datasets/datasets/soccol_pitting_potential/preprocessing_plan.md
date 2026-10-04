# Model-Input Preprocessing

**Decision recorded:** 2026-09-28  
**Status:** Implemented.

This file records the implemented feature schema for rerunning the EPIT
prediction pipeline with the Soccol pitting-potential dataset.

## Prediction task

- Target: `E_pit` in mV versus Ag/AgCl (3 M KCl).
- Primary regression rows: numeric `E_pit` with `event == 1`.
- The already frozen composition-group split remains in use.
- Expected model input: 37 columns: 13 composition, 20 other numeric, and 4
  categorical predictors.

## Column treatment

| Raw or derived columns | Model role | Treatment |
|---|---|---|
| `C`, `N`, `Si`, `P`, `S`, `Ti`, `V`, `Cr`, `Mn`, `Ni`, `Nb`, `Mo` | Numeric predictors | Keep every element and replace blanks with `0`. Do not add missingness-indicator columns. |
| Derived `Fe` | Numeric predictor | For supported Fe-based rows, use the approximate balance `100 - sum(other recorded/zero-filled elements)`. Use `0` where Fe reconstruction is not supported. |
| `Prep_grinding_grit`, `Prep_Ra_micron`, `Prep_pH`, `Prep_redox`, `Prep_time` | Numeric predictors | Keep. Fit a separate mean for each column using only the current training/context rows, then use those means to replace missing values in both context and query rows. |
| `CP_time`, `CP_temp`, `CP_pH`, `Test_area_cm2`, `scan_rate` | Numeric predictors | Keep and apply the same training/context-only mean imputation. |
| `CP_Cl`, `CP_Br`, `CP_OH`, `CP_SO4`, `CP_CO3`, `CP_NO3`, `CP_PO4`, `CP_MoO4`, `CP_CrO4`, `CP_ion_other` | Numeric predictors | Keep and replace blanks with `0` under the benchmark convention that an unreported ion is encoded as absent. |
| `Prep_medium`, `CP_aeration`, `CP_agitation`, `CP_anions_info` | Categorical predictors | Keep. Fit integer category mappings using only the current training/context rows. Encode missing and previously unseen query categories as `-1`. |
| `label`, `source`, `ID` | Metadata | Keep for traceability; exclude from model inputs. |
| `alloy_designation` | Metadata | Keep for traceability; exclude from the primary model so the alloy name cannot circumvent the composition-group split. |
| `E_corr` | Excluded response | Exclude because it is another measured electrochemical response from the same experiment. |
| `E_pit` | Target | Never include among predictors. |
| `event` | Row-selection/status field | Select `event == 1` for primary regression, then exclude from predictors. |

## Special and global rules

- Replace the erroneous `Ti` values in all 24 `2000Russell` rows with `0`.
  Those cells contain a PRE expression rather than titanium composition.
- Keep rare columns and elements. Do not remove predictors based on coverage.
- Do not add `*_missing` columns or other missingness indicators.
- Do not apply logarithmic transformations or other new feature transforms.
- Do not use a universal numeric `-1` sentinel. Negative values are valid in
  columns such as `Prep_pH`, `Prep_redox`, and `CP_pH`.
- Store fold-dependent continuous values as missing in the canonical processed
  table. The evaluation pipeline must fit the numeric means from its own
  training/context partition and supply finite values to TabICL at inference.
- Preserve raw categorical strings in the canonical processed table. Apply the
  fitted category mapping when constructing each model input matrix.

## Implementation

- `prepare_benchmark.py` and `processed/feature_manifest.json` define the 37
  predictor roles while retaining missingness and material-family fields only
  for audit.
- `scripts/eval_corrosion_datasets.py` fits continuous means and categorical
  mappings separately on each context partition and verifies that all 37 model
  inputs are finite.
- `src/tabicl/prior/assets/soccol_pitting_features_v1.*` stores the target-free
  synthetic feature profile. `prepare_pipeline_assets.py` reproduces it and the
  frozen nine-rule calibration artifacts.
- `scripts/soccol_pipeline/run_optuna.py` reuses the previous search settings
  with the frozen Soccol composition split.
