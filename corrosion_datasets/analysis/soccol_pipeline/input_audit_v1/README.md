# Soccol input changes

Compared the original frozen CSV with the reviewed export. This audit did not fit coefficients, train models or calculate prediction scores. Targets, row order and frozen split assignments were unchanged.

The old schema had 37 features; the reviewed schema had 58: 34 retained, 24 added and 3 removed. Added columns were counted separately from changed common cells. Agitation yes/no to 1/0 and aeration renaming were recorded as representation changes, not changes in experimental conditions.

## Accepted development rows

Used 2,950 accepted development rows. Recorded 1015 changed common cells across 797 rows:

These comprised 71 corrected numeric amounts (Fe/Nb/Ti), 299 values deliberately left missing after review, and 645 category representation changes. The full audit also recorded withheld development rows and frozen queries separately; those rows did not enter the reviewed development inputs.

| Column | Changed cells |
|---|---:|
| CP_CO3 | 22 |
| CP_OH | 6 |
| CP_SO4 | 104 |
| CP_aeration | 252 |
| CP_agitation | 393 |
| Fe | 60 |
| Nb | 9 |
| Prep_time | 152 |
| Test_area_cm2 | 15 |
| Ti | 2 |

Added composition fields included positive W in 20 development rows and positive Cu in 38. The other nine added element columns were zero in all accepted development rows. These added fields were excluded from the common-cell counts above.

## Formula inputs

Among raw fields actually used by the nine formulas, only sulfate changed on accepted development rows: 104 disputed values became NaN. Cr, Mo, N, Mn, S, temperature, pH, chloride, bromide and nitrate raw values were unchanged on those rows. Corrected Fe/Nb/Ti and added W/Cu do not enter these formula definitions.

Internal formula inputs also changed through filling. The audit compared each accepted development validation row using its own fold's context only. It saved both the historical full context and a historical-data control using exactly the accepted context rows. This separated changes caused by row removal from changes caused by preprocessing; no final-query rows supplied filling statistics.

| Formula input | Changed effective validation cells |
|---|---:|
| CP_NO3 | 3 |
| CP_SO4 | 104 |
| CP_pH | 51 |
| CP_temp | 33 |

The 104 sulfate changes came from leaving disputed amounts missing and using context means internally. Three already-missing nitrate values changed from internal zero filling to context means. The 51 pH and 33 temperature changes came only from fitting means on the smaller accepted context. Model-facing unknowns remained NaN throughout. These are inputs before formula scaling: changes in scaling and coefficients can also affect other rows and were not measured here.

The formula-adapter CSV also includes fields loaded by the adapter but unused by these nine rules; `active_formula_families` identifies actual dependencies. Empty values in the CSVs mean missing, never zero.

## Checks and differences from EPIT

- Soccol's cr_mow_n_synergy name is misleading: its formula uses Mo, not W. EPIT's corresponding formula uses Mo + 0.5*W. No formula was changed by this audit.
- The reviewed Soccol bank contains 20 accepted development rows with positive W and 38 with positive Cu. All 58 features can feed the SCM, but the nine analytical formulas ignore W, Cu, Fe, Nb and Ti.
- Missingness and the shared 49 feature names/types/units agree with EPIT. Individual schemas remain 58 versus 70 columns; separate category codes must not be treated as universal merged-dataset codes.
- Ten frozen Soccol final queries retain unresolved composition NaNs. Accepted Soccol development compositions are complete. The final queries were retained unchanged; this audit did not evaluate predictions.
- Passed 19/19 checks: row/target/split identity, accepted compositions, saved bank and formula-adapter equality, missingness, shared units/types and saved formula filling states.

## Files and reproduction

- `input_changes_vs_legacy.csv`: every changed common raw feature cell across all 4,027 rows, with partition and eligibility.
- `formula_input_changes_vs_legacy.csv`: changed raw formula-adapter fields, with active-rule dependency labels.
- `effective_formula_input_changes.csv`: accepted development validation inputs after fold-context filling, with the same-row control.
- `fold_filling_values.csv`: old, same-row-control and reviewed filling values and context sizes.
- `schema_comparison.csv`: retained/added/removed columns and observed/missing counts by partition.
- `shared_epit_schema_check.csv`: all 49 shared names, types and units.
- `change_counts.csv` and `summary.json`: aggregate counts, input/output hashes, dependencies and checks.

```bash
.venv/bin/python -m scripts.soccol_pipeline.audit_input_changes --output-dir /tmp/soccol_input_audit
```

The command refused to overwrite an existing nonempty output directory. Numeric comparisons ignored serialization round-off only (absolute tolerance 1e-10, relative tolerance 1e-12).
