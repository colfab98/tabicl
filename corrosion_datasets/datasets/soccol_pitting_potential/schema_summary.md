# Schema Summary

## Primary workbook

`raw/Pitting_potential_dataset_2024.xlsx` contains:

| Sheet | Observed data rows | Content |
|---|---:|---|
| `pitting_potentials` | 4,460 | 43 material, preparation, environment, procedure, and response fields |
| `references` | 154 | Source identifier, reference, and DOI |

A source can contribute many dependent measurements, alloys, or test
conditions. A row is not necessarily a unique material or independent study. The `label` field is not a unique row key: it has 4,448 unique values for 4,460 rows. The processed table adds `raw_workbook_row` as a stable row identifier.

## Columns and confirmed units

Identifiers and material labels:

- `label`, `source`, `ID`, `alloy_designation`

Composition, in **wt%**:

- `C`, `N`, `Si`, `P`, `S`, `Ti`, `V`, `Cr`, `Mn`, `Ni`, `Nb`, `Mo`

The collection paper describes composition as coming from a certificate or
standard. It limited its fitted alloy model to C, N, Si, P, S, Cr, Mn, Ni, and
Mo because the remaining elements had low reporting coverage.

Surface preparation:

- `Prep_grinding_grit`, `Prep_Ra_micron`, `Prep_medium`, `Prep_pH`,
  `Prep_redox`, `Prep_time`

Test environment and procedure:

- `CP_time`, `CP_aeration`, `CP_agitation`, `CP_temp`, `CP_pH`, `CP_Cl`,
  `CP_Br`, `CP_OH`, `CP_SO4`, `CP_CO3`, `CP_NO3`, `CP_PO4`, `CP_MoO4`,
  `CP_CrO4`, `CP_ion_other`, `CP_anions_info`, `Test_area_cm2`, `scan_rate`

The raw ion concentrations are molar. The paper transformed concentrations to
`log10` and assigned `10^-5 M` to anions not mentioned by a source for its
statistical model. Those are modeling steps, not raw-workbook units.

Electrochemical response/status:

- `E_corr`, `E_pit`, `event`

The paper standardizes potentials to mV versus Ag/AgCl (3 M KCl). The workbook
column `E_pit` is the observed breakdown potential `Ebr`: it may be a true
pitting potential or a competing non-pitting breakdown.

- `event=1`: actual pitting.
- `event=0`: competing non-pitting breakdown; right-censored for the latent
  pitting potential.
- `E_pit` is numeric in 4,408 rows.
- `event` has 4,029 ones, 398 zeros, and 33 blanks.

## Missing composition

A blank composition cell is missing/unreported information. It is not a
verified physical zero. The collection paper used multiple imputation for
missing covariates.

The source-pattern audit found:

| Structural pattern | Sources | Rows |
|---|---:|---:|
| Fully populated across all 12 recorded elements | 0 | 0 |
| Consistent source schema with one or more all-blank elements | 113 | 2,982 |
| Mixed blank/populated values within a source | 41 | 1,478 |

Explicit zeros and blanks coexist for the same element in two sources:

- `1994Carroll`: Mo has blanks, explicit zeros, and positive values.
- `1995Malik`: N has blanks, explicit zeros, and positive values.

Zero-filling is acceptable as a declared benchmark encoding for the current
model comparison. It should be accompanied by missingness indicators and must
not be described as recovered chemistry.

## Material scope and source review

The source-level registry accounts for all 154 sources and 4,460 rows:

| Review status | Sources | Rows |
|---|---:|---:|
| Usable Fe-based sources | 131 | 3,706 |
| Usable with a composition caveat | 14 | 265 |
| Usable with a material-taxonomy caveat | 3 | 204 |
| Mixed material families; split or filter | 4 | 169 |
| Separate non-Fe family | 1 | 92 |
| Repair required | 1 | 24 |

The four mixed-family sources are `1968Horvath`, `1988Roberge`, `1991Cortest`,
and `1994Carroll`. `2009Wong` is a Ni-based Ni-Cr-Mo dataset whose three
reported elements already sum to approximately 100 wt%; Fe must not be added.
`1997Stellwag` contains Alloy 800 and is flagged as a related Fe-Ni-Cr alloy
rather than conventional stainless steel.

`2000Russell` has a confirmed column error: its 24 `Ti` values equal
`Cr + 3.3*Mo + 20*N`, i.e. a PRE value. They are not titanium composition and
must be set to missing or removed before modeling.

Several sources vary an element that the workbook has no column for, including
Re (`1969Boehni`), W (`1983Bui`, `1998Ahn`), and Y (`2018Shi`). Other explicit
source-level omissions and row-handling rules are recorded in the registry.

Fe is the nominal balance for most Fe-based sources, but it is not always the
largest component. Any computed `100 - sum(recorded composition)` value is an
approximation because reported elements can be missing and the workbook omits
some alloying elements entirely.

## Modeling controls

- For the primary comparison, keep identical or closely linked compositions together using the frozen composition-group split. Source overlap is allowed, balanced, and reported. Use a source-held-out split only for a separate new-study sensitivity analysis.
- Treat `event=0` as censored for a censor-aware task. For plain regression,
  use `event=1` rows and state the selection rule.
- Do not use `event`, identifiers, references, or free text as ordinary
  predictors unless the intended prediction setting justifies them.
- Remove overlap with the older Nyby collection when this dataset serves as an
  independent evaluation set.

The complete source registry is in
`../../analysis/soccol_source_conventions/source_conventions.csv`.

## Processed benchmark and frozen split v1

`prepare_benchmark.py` creates the processed row tables under `processed/`:

| Table | Rows | Use |
|---|---:|---|
| `soccol_processed_all_rows.csv` | 4,460 | Complete processed/audit table |
| `soccol_regression_event1.csv` | 4,027 | Numeric actual-pitting rows for ordinary regression |
| `soccol_breakdown_survival.csv` | 4,384 | Numeric breakdown rows with defined censoring status |

`prepare_composition_split.py` applies the previous EPIT split method to the 4,027-row regression task. It rounds the 13 processed composition values to 0.01 wt.%, links pairs with L1 distance at most 1.0 wt.%, and keeps every connected component intact. The result contains 339 composition groups.

| Partition | Rows | Role |
|---|---:|---|
| Development | 3,222 | Model and checkpoint development |
| Final test | 805 | One final evaluation |
| Development folds | 645 / 645 / 644 / 644 / 644 | Five grouped validation rotations |

The optimizer balances EPIT deciles, material family, composition isolation, temperature, chloride, pH, and source. The prior test-method block is omitted because Soccol has no comparable field, as explicitly decided. Source does not define groups: 37 sources occur in both outer partitions, and the assignments record that overlap.

The frozen artifacts are in `processed/splits_v1/`. `split_assignments.csv` hides final-test EPIT values, while `split_lock.json` binds the input table, manifest, assignments, and report by hash.

The processed composition contains `Fe` plus the 12 workbook elements and one missingness indicator for every composition field. A computed `Fe` value is the remainder to 100 wt% after the reported/zero-filled fields; `Fe_is_approximate=1` records that assumption. It does not prove that Fe was measured or is the largest alloy component. Rows that do not support this calculation use `Fe=0, Fe_missing=1`.

`CP_aeration` and `CP_agitation` are categorical. The remaining recommended environmental/procedure predictors are numeric. Their blanks are preserved for context-only imputation. See `processed/feature_manifest.json` for the exact predictor and audit roles.
