# Rebar evaluation inputs

## Reviewed candidate export, version 2 (19 November 2026)

Use [processed/rebar_reviewed_v2/](processed/rebar_reviewed_v2/) for the new
composition-screened candidate view and the combined-model evaluator's default
inputs. The existing version 1 files are preserved. **This remains provisional evaluation data; source-level
target verification and measurement-independence checks are incomplete.**

- **195 eligible candidates**, 44 completed composition vectors, 42 source identifiers.
- **152 eligible development-context rows**; original outer split retained.
- **All 56 original final-query rows retained**, including 13 with unresolved
  composition. The 43 eligible final rows are not a replacement final-query set.
  Native inference retains their missing values. The report's readiness flag
  records incomplete data verification; it does not prevent checkpoint comparison.
- **65 predictors**: 31 elemental wt.% columns, 33 environment/cement columns,
  and the workbook-supported `reference_type=SCE` category. Temperature, pH,
  chloride and equivalent ion fields use the established canonical names.

The review rereads composition cells from the raw workbook. It recovers 20
explicit Fe balances and one supported HRB400 Fe remainder. Six further explicit
balances remain unresolved because individual elements are unknown. Reported
numbers and zeros are preserved; accepted complete compositions receive logged
assumed zeros. No composition is rescaled. Ranges, unknown elemental amounts,
unexplained low totals and absent chemistry are withheld from the eligible pool.

The original paper for the Cr/RE source reports a combined **Ce/La** amount,
which the workbook assigned entirely to Ce. Those four rows now keep both
elements unknown. Its potential table also uses an Epass heading while the text
discusses Epit; all 12 source rows are withheld from the eligible pool pending
endpoint clarification. The unchanged source PDF and exact review decisions are
recorded in [review_rules_v2.json](review_rules_v2.json).

Environment values retain the v1 audited parsing and missingness. Ambiguous
`MO4` notation and the sodium-oleate hydrocarbon-fragment column are excluded
from predictors. Bicarbonate and hydrogen phosphate remain distinct from
carbonate and phosphate. No salt mass fraction is converted to molarity, and
unknown treatment/inhibitor details are not inferred. Targets remain the same
workbook values in mV/SCE; no model is fitted or evaluated.

Files include the full frozen-row feature view, eligible candidate features and
separate targets, development context features and targets, target-free final
queries and separate scoring targets, unchanged split assignments, composition
cell/row logs, all 357 originally audited target records, a source-overlap audit,
feature registry, report, and hash-locked provenance. Only registry-listed
features belong in model inputs. Source screening checks known/extracted DOIs
and normalized full-reference matches against both original EPIT/Soccol reference
sheets; an unmatched source is not certified independent.

```bash
PYTHONPATH=src:. .venv/bin/python -m scripts.rebar_pipeline.prepare_reviewed --validate-only
# Reproduce in a fresh directory; existing exports cannot be overwritten:
PYTHONPATH=src:. .venv/bin/python -m scripts.rebar_pipeline.prepare_reviewed --output-dir /tmp/rebar_reviewed_rebuild
```

## Preserved evaluation inputs, version 1

**Provisional external evaluation data, not a fully verified independent benchmark.**
278 workbook-reported pitting/breakdown-potential observations remain from 351 numeric candidates, across 52 publication identifiers. Original-paper verification and EPIT/Soccol source-overlap review are incomplete. These rows must not enter fitting, retention, synthetic priors or training.

Source: [Hou et al., Zenodo v1](https://zenodo.org/records/21987599), published 18 August 2026. The original workbook and Usage Notes are preserved in `raw/`. The Usage Notes specify potentials in V SCE. The export multiplies targets by 1000; no further reference-electrode correction is applied.

## Processing

- Preserve reported elemental weight percentages, temperature in Celsius, pH, solution concentrations in M and the cement weight-percent block. Duplicate Co/SiO2 and phase columns are reconciled only when their numeric values agree. Cb is the historical symbol for Nb. Conflicts become missing and are audited.
- Keep missing values as NaN, explicit zeros as zero. Do not infer unreported composition, close/rescale compositions, or fill unknown elements with zero. Bounds/ranges in predictors remain missing; temperature `value ± uncertainty` uses the central value. No global scaling or imputation is applied. Workbook author assumptions behind existing zeros are not recoverable.
- Preserve cached workbook values for calculated inputs such as pH; cell formulas and colors remain in the audit. Unobserved columns are omitted. Model preprocessing is fitted only on context, through the same native evaluator as EPIT/Soccol.
- Exclude Ecp/Ec, chloride thresholds and threshold ratios, testing times of ambiguous meaning, source IDs, tags, outcome-containing free text, duplicate cement molar columns, and author-default density/oxygen fields. Unreported inhibitor/treatment details can therefore remain unobserved; the 54-column export is not a complete experimental description.
- References inherit only within reference blocks; a uniquely identified reference may be recovered elsewhere in the same sheet. Physical inputs are never forward-filled.
- Source and row exclusions are recorded independently of model errors or performance. Equality between Epit and Ecp is a flag, not sufficient reason for exclusion.

Numeric exclusions: 35 applied-potential/threshold rows (Izquierdo), 11 threshold rows with placeholder stainless composition (Lollini), 8 rows requiring target verification (Liu/Chen), 4 chloride-free breakdown rows, 4 rows from a source also present in Soccol (Luo), 3 rows with mismatched source scope (Tian), 1 missing-reference row, 5 rows with evidently misplaced composition columns (17.31 wt% carbon for 316L), and 2 rows with an unresolved 76.24 wt% composition total. Six additional nonnumeric/censored target cells are also withheld. The DOI evidence links are in `source_audit.json`; a withheld designation means unresolved, not proof that every value is incorrect.

`row_audit.csv` identifies every candidate by sheet and Excel row. `cell_audit.json` records input parsing. `features.csv` and `targets.csv` use stable row IDs. `split_manifest.json` and `split_lock.json` lock the exports and raw workbook; evaluation rejects altered inputs. The preparer refuses to overwrite an existing export.

## Evaluation

The combined-model wrapper now loads reviewed v2 features through `reviewed_evaluation_data.py`. As in Soccol/MPEA, fold 1 keeps every frozen outer query, and folds 2–5 partition accepted development rows with seed 42. Query counts are **56/38/38/38/38**, with **152/170/170/170/170** context rows. The 70 excluded development rows never enter context or queries. All 278 original row indices remain in the backing table for provenance; 208 participate in evaluation. Context is always rebar only. The original v1 publication-grouped assignments remain archived unchanged; exploratory development folds are row folds and may share publications.

Fold 1 is the frozen outer final-query partition (**152 reviewed context / 56 original queries**). A final frozen model must use this partition only. The five-fold mean is an exploratory checkpoint comparison, not the final-test score. Native inference settings, three baselines, checkpoint selection, missing-checkpoint handling, metrics and plots use the existing shared engine. All models receive the same 65 registry-listed predictors, including the `reference_type` category. Missing inputs are handled by the existing context-fitted native preprocessing.

```bash
PYTHONPATH=src:. .venv/bin/python -m scripts.combined_corrosion.evaluate_selected_pairs \
  --dataset rebar \
  --checkpoint all --checkpoint-scope all --checkpoint-step-interval 0 \
  --max-checkpoint-step 3000 --exploratory-cv-folds 5 --device cpu \
  --output-dir corrosion_datasets/analysis/combined_corrosion/selected_pairs_through_3000_cv5
```

The output folder ends in `_rebar`. This command evaluates existing checkpoints; it does not train or modify them. A direct single final-partition evaluation uses `scripts/rebar_pipeline/evaluate_native.py` without either CV flag, with the same common native evaluator arguments. Final data status remains provisional even on that partition.

To reproduce in a new directory:

```bash
PYTHONPATH=src:. .venv/bin/python -m scripts.rebar_pipeline.prepare_evaluation --output-dir /tmp/rebar_rebuild
```

On another node, commit/push/pull the new `scripts/rebar_pipeline/` package,
the updated combined evaluation wrapper, `.gitignore`, and the rebar dataset
directory. Both processed versions' CSVs, JSON manifests and audits now travel
through Git. Raw files are ignored and must be copied separately at these same
repository-relative paths (existing identical copies can be reused):

- `corrosion_datasets/datasets/rebar_pitting/raw/rebar_electrochemical_parameters.xlsx`
- `corrosion_datasets/datasets/rebar_pitting/raw/rebar_usage.docx`
- `corrosion_datasets/datasets/rebar_pitting/raw/source_review_v2/zhu_2026_cr_re.pdf`
- `corrosion_datasets/datasets/electrochemical_metrics_alloys/raw/CRA_database_Scientific_Data_Publication_12102020.xlsx`
- `corrosion_datasets/datasets/soccol_pitting_potential/raw/Pitting_potential_dataset_2024.xlsx`

The evaluator verifies all five raw inputs against the reviewed manifest even
when using the prepared CSVs. After transfer, run the `prepare_reviewed
--validate-only` command above before evaluation. Model checkpoint files/cache
must also be present as for existing evaluations. No new package dependencies
are required beyond the existing evaluation environment.
