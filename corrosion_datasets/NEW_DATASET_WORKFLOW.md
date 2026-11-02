# Adding a corrosion dataset: complete chronological workflow

Audited against this project's working tree and the decisions in this session on **28 October 2026**, with the shared-stream training update on **30 October 2026**.

This is the procedure for taking a new literature dataset from its original files to a trained, selected specialist, and then, if required, adding its prior to a combined model. MPEA is the running example. It separates operations that change real measurements, operations that learn from development data, and operations that generate synthetic training tasks.

The detailed history of individual EPIT/Soccol corrections remains in [DATASET_PROCESSING.md](DATASET_PROCESSING.md). This document reorganizes that history into the order in which a new dataset should be handled. Earlier experimental reports, slides and old exports are historical evidence; they are not necessarily the current implementation.

**How to use this document:** follow the numbered steps in order. At each “Ready to continue” check, record the evidence or resolve the missing decision. A chemically ambiguous source cannot be resolved by blindly executing a command. The procedure tells you when to keep a value unknown, when to withhold a row, and when an explicit decision is necessary; it does not authorize inventing missing source information.

## Navigation

| Phase | Steps | Jump to |
|---|---|---|
| Define and inspect | 01–05 | [Experiment and source evidence](#01-define-the-experiment-before-editing-data) |
| Correct real data | 06–08 | [Composition](#06-process-the-complete-composition-before-projecting-it-into-any-model-schema), [environment and history](#07-process-environment-preparation-and-material-history-features) |
| Freeze and connect | 09–13 | [Splits](#09-freeze-the-outer-split-and-development-folds-or-reuse-the-existing-ones), [shared schema](#11-decide-the-common-schema-and-implement-explicit-mappings) |
| Establish target rules | 14–17 | [Material scope](#14-audit-material-families-and-formula-applicability), [formula fitting](#16-fit-formulas-with-the-established-development-fold-procedure) |
| Build the prior | 18–20 | [Profiles](#18-build-versioned-composition-and-environment-sampling-profiles), [sampling](#19-implement-and-audit-synthetic-sampling-in-the-correct-order), [NaNs/categories](#20-enforce-categories-and-missingness-consistently-through-inference) |
| Check before spending compute | 21–24 | [Pipeline integration](#21-connect-the-profile-formulas-and-schema-to-the-trainingevaluation-entry-points), [restart identity](#22-record-identities-and-decide-restart-behavior-before-expensive-training), [remote inputs](#23-make-the-complete-input-bundle-available-on-every-node) |
| Search and train | 25–29 | [Optuna](#25-set-and-freeze-the-optuna-search-configuration), [four final runs](#28-launch-final-training-as-four-separate-named-runs) |
| Combine and evaluate | 30–31 | [Three-source combination](#30-extend-the-combined-model-after-the-specialists-are-ready), [final evaluation](#31-evaluate-frozen-models-with-the-agreed-baselines-and-query-boundary) |
| Maintain and execute | 32 and appendices | [What must be repeated](#32-know-which-work-must-be-repeated-after-a-change), [commands](#command-cards-existing-commands-not-invented-mpea-entry-points), [remaining MPEA work](#the-mpea-work-that-remains-in-the-order-to-do-it), [completion checklist](#required-completion-record-for-each-new-dataset) |

## Current project state: read this before following the steps

| Item | Verified current state | Consequence for a new dataset |
|---|---|---|
| Reviewed EPIT export | `datasets/electrochemical_metrics_alloys/processed/epit_reference_v1/`; 760 rows, 70 predictors | Use this export and its registry, not an earlier workbook-derived feature table. |
| Reviewed Soccol export | `datasets/soccol_pitting_potential/processed/soccol_reference_v1/`; 4,027 reference rows, 58 predictors | Development eligibility is a separate filter; the full reference table is not the training bank. |
| EPIT development/final split | 608 / 152, with five frozen development folds | Keep existing row membership. |
| Soccol development/final split | 3,222 original development rows; 2,950 accepted development rows; all 805 frozen final queries retained | 272 development rows are withheld. Ten final-query compositions remain unresolved. |
| Reviewed EPIT–Soccol column alignment | 49 common predictors; 21 EPIT-only and 9 Soccol-only; proposed union **79** | Alignment is documented; it does not mean the combined runtime has been migrated. |
| Existing combined tables and prior | Historical **50-column** schema | They are not the reviewed 79-column union. Some historical fields are now excluded from the reviewed standalone schemas. |
| MPEA specialist split | `datasets/mpea_corrosion/processed/splits_v1/`: 214 development, 53 final, five development folds | Reuse this split. Do not replace it with the exploratory MPEA-only fivefold experiment. |
| MPEA schema in that frozen export | Historical 50-column shared representation | A revised MPEA feature view still needs preparation and alignment. Preserve row membership while versioning that view. |
| MPEA specialist implementation | Frozen split and development formula screen exist; `scripts/mpea_pipeline/` currently contains formula definitions, not a complete Optuna/final-training workflow | Do not assume an MPEA training command already exists. |
| EPIT synthetic inputs | `epit_dataset_v4` composition assets and `epit_fe_nicrmo_features_v4` feature assets | Earlier assets are retained for reproducibility, not interchangeable defaults. |
| Soccol synthetic inputs | `soccol_pitting_features_v3`, reviewed 58-column schema | This version samples distinct compositions uniformly. Older audit results describing observation-weighted compositions are historical. |
| Active EPIT search | Study `epit_pipeline_optuna_empirical_features_scm_target_v12`; formulas `target_rules_v4`; output root still named `optuna_v3` | Study versions and output-folder versions are different identifiers. Read the saved configuration. |
| Active Soccol search | Study `soccol_pipeline_optuna_empirical_features_scm_target_v6`; formulas `target_rules_v2`; output root `optuna_v4` | Do not select old trial 19 by default. |
| Sparse-column preprocessing | Applied to the main project | Columns with fewer than two known context values keep their values and NaNs without scaling/normalization/clipping. |
| Four-ratio EPIT comparison | Applied to the main project | Local evaluation no longer requires `/tmp/tabicl-standalone-fixes`. |
| Baseline inference across development/final wrappers | The final comparison uses native generic/pretrained settings; the current EPIT development-baseline helper, also reused by Soccol, explicitly passes `none` normalization/shuffling without the native-baseline flag | Resolve this baseline-protocol difference before using those development baseline scores as directly aligned evidence for a new experiment. Do not assume the final-comparison fix migrated every wrapper. |
| Shared streams and checkpoint identity | EPIT v12 / Soccol v6 use indexed streams, exact batch proportions, checkpoint profile identities and low-level worker-state recovery | Old checkpoints cannot provide this stream identity. The high-level final-training launcher still has no training-resume flag. See [shared streams](../scripts/epit_pipeline/PAIRED_STREAMS.md). |
| MPEA split-file transport | Manifest, lock, registry and report are tracked; the frozen split CSVs were not tracked at this audit | Add an explicit tracking/distribution solution before submitting an MPEA job on another node. |

“Same procedure as EPIT/Soccol” means the same safeguards and experiment structure. It does not mean forcing identical alloy families, identical physical formulas, or identical composition-noise ranges onto a different dataset.

## Terms used below

| Term | Meaning here |
|---|---|
| Raw data | Unchanged source workbook/table and supporting papers. |
| Reviewed export | Our documented, corrected real-data table. |
| Feature registry | The ordered list of model inputs, with meanings, units and types. |
| Development rows | Rows allowed for formula fitting, prior construction, Optuna and checkpoint selection. |
| Validation fold | Development rows temporarily held out as queries for a development score. |
| Final queries | The frozen outer final-test rows. They are never replaced by a development fold for final-model evaluation. |
| Context | Rows and their known targets supplied to TabICL during prediction. In final evaluation these come from development. |
| Sampling profile/bank | Versioned CSV/JSON assets from which synthetic composition and experimental conditions are sampled. |
| Informed prior | A generator of synthetic training tasks using reviewed feature distributions and analytical target rules. It is not a model checkpoint. |
| Formula coefficient center | Coefficients fitted on development data, around which synthetic-task coefficients may vary. |
| Missing-value embedding | The learned representation used by a compatible model when an input cell is NaN. |
| Frozen model | A checkpoint selected using development information and recorded in a final-model manifest/lock. |

## 01. Define the experiment before editing data

1. Give the dataset a stable ID and identify the exact scientific target: pitting potential, not corrosion potential, corrosion current, repassivation potential, or corrosion rate.
2. State whether the immediate task is external evaluation of existing models, development of a new specialist, or combined-prior training. These have different permitted uses of the new targets.
3. For the present MPEA plan, record: build an MPEA specialist; fit/check formulas on MPEA development; run Optuna; train the selected setup; eventually combine EPIT, Soccol and MPEA priors.
4. Record the later combined evaluation choice already agreed: concatenate the three development contexts and the three frozen final-query sets, without balancing their row counts for now.
5. Define the primary development selection metric. Current specialist searches use mean five-fold Spearman. Keep error metrics such as MAE as additional information.
6. Record what has already been inspected. MPEA rows have been used in earlier exploratory evaluations. The 214/53 split is frozen for the specialist workflow; it does not make those observations newly collected, historically untouched data.
7. Choose an output/version namespace before writing files. Keep old results reproducible.

**Save:** a short dataset decision record: purpose, target, source version, split version, intended feature version, primary metric, prior exposure and unresolved questions.

**Ready to continue:** nobody can confuse an exploratory external-query experiment with the new specialist's frozen final evaluation.

## 02. Acquire and preserve the original evidence

1. Save the original workbook/table, collection paper, supplement and data dictionary under the dataset's `raw/` directory or a documented external data location.
2. Record the repository DOI/URL, dataset version, download date and redistribution conditions. Follow the actual source terms when deciding which files to track in Git.
3. Calculate file hashes. Identify workbook sheet names, header rows and the physical Excel row corresponding to each record.
4. Compare the paper's advertised sample counts with the actual workbook. Explain differences where supported; otherwise record the discrepancy.
5. Keep raw files immutable. Corrections belong in a versioned processor or explicit correction table, never in an undocumented manual edit to the original workbook.
6. Keep copies of the source passages supporting non-obvious decisions, including alloy balance, concentration basis, electrode reference and specimen identity.

**MPEA example:** the workbook contains 619 rows and 335 usable numeric pitting-potential targets, while the collection paper reports 306. The discrepancy was not resolved by deleting an arbitrary 29 rows. The raw evidence and discrepancy were retained.

**Save:** `manifest.yml` or equivalent provenance file, raw-file hashes, source inventory, and a source-level audit table.

## 03. Give every row a permanent identity

1. Create an ID tied to the original file/sheet/row, not the row position after filtering or sorting.
2. Preserve source-paper identity, raw alloy name and any specimen/condition identifiers as metadata.
3. Keep a mapping from permanent ID to original source row and, where applicable, the frozen evaluator's `task_row_index`.
4. Require unique row IDs. An accidental join that duplicates rows must fail.
5. When filtering, carry IDs through every table. Never infer correspondence from two CSVs having the same number of rows.
6. Prefix IDs with the dataset name in combined data so two datasets' `row_001` cannot collide.
7. Keep original target text and original composition/environment strings available beside the processed values or in the audit.

**MPEA example:** its existing IDs encode the original Excel row. Reuse them, including the original header offset.

**Ready to continue:** any changed model-input cell can be traced to one source cell or one explicit extraction decision.

## 04. Inventory every column before deciding what to retain

Create one row per original column in a feature-review table. Record:

- Exact original name, description, unit and source of that interpretation.
- Whether it describes composition, test solution, test procedure, preparation, heat treatment, microstructure, target or provenance.
- Numeric/categorical/text type; unique text descriptions; known, blank, explicit-zero and inequality counts.
- Range, unusual values, repeated constants and suspected placeholders.
- Whether the value was known before the outcome was measured.
- Equivalent EPIT/Soccol/MPEA field, if any; whether equivalence requires conversion or splitting into several columns.
- Proposed action: retain, convert, split, audit only, exclude, or unresolved.
- Evidence and a concrete reason for the action.

Inspect unique descriptions, not just column headers. A column called “heat treatment” may also contain grinding or cleaning instructions; a “solution” column may contain temperature, gas condition and scan rate.

Use source-wide inspection to establish meanings and deterministic parsing rules. Once the split is frozen, use development data for learned statistics and performance-driven decisions. Do not use final-query scores to decide which features to keep.

**Exclude from predictors:** targets; other measured response variables such as corrosion current/potential when not valid pre-test inputs; outcome descriptions; row IDs; source IDs/DOIs; audit flags; acceptance flags; fold numbers; and raw comments that reveal results. Keep them for provenance where appropriate.

**Ready to continue:** each retained input has a physical meaning, a unit or category definition, and a reason it is available at prediction time.

## 05. Set the target and reference-electrode policy

1. Identify the exact target column and experiment. Do not silently merge distinct breakdown or corrosion quantities because they share millivolt units.
2. Keep genuine numeric zero targets. Missing targets are not zero and are never imputed.
3. Convert V to mV where justified and record the conversion. Unit conversion and electrode-reference conversion are separate operations.
4. Identify censoring or limit notation (`>800`, `<...`, “no pitting”). Retain the original notation and an audit flag.
5. Decide the modeling convention explicitly. Current EPIT retains its historical lower-bound approximations: `>800` becomes 800 and `>1256` becomes 1256. This is a dataset-specific accepted convention, not a general instruction to turn every inequality into an exact observation.
6. Soccol's current regression population uses `event=1`; `event=0` non-pitting observations were not silently treated as exact pitting targets. Check the equivalent distinction in the new source.
7. Preserve source-specific electrode evidence in `reference_type`. Current reviewed labels are `SCE` and `AgAgCl`; unverified references are NaN.
8. Do not label all rows from the reference stated in a collection paper if the underlying numeric targets retain different original references. Soccol showed this problem.
9. Preserve electrolyte details of the reference electrode in the audit. The current broad `AgAgCl` label does not make saturated KCl and 3 M KCl identical.
10. Follow the agreed current merge policy: retain the reported numerical potential in mV and its reference label, without an unsupported universal electrode conversion. The proposed merged target name is `reported_pitting_potential_mV`.
11. Do not describe such a merged target as uniformly SCE-converted. Missing reference labels and broad labels remain limitations of its interpretation.
12. A preparation potential such as `Prep_redox` needs its own reference evidence. The pitting target's `reference_type` cannot supply it automatically.

**Save:** target column mapping, units, numeric parsing rules, inequality/event policy and a row/source reference audit.

## 06. Process the complete composition before projecting it into any model schema

### 06a. Establish the composition basis

1. Read the source definition: wt.%, weight fraction, at.%, atomic fraction, or an alloy-name shorthand.
2. Keep all reported elements during conversion and completeness checks, including elements absent from current model schemas.
3. Use wt.% as the common model-facing composition unit.
4. Convert weight fractions by multiplying by 100.
5. Convert a complete atomic composition using frozen atomic weights:

   `wt%_i = 100 × atomic_amount_i × atomic_weight_i / sum(all atomic_amount_j × atomic_weight_j)`.

   The denominator includes **all reported elements**, not just those the old model happens to accept. Record the atomic-weight table/version.
6. Verify source-unit closure before conversion. An incomplete atomic composition must not be normalized into apparent completeness without evidence.
7. Distinguish an incomplete source composition from a complete composition projected onto an incomplete feature subset.

**MPEA example:** the historical export retained 17 elements supported by the old EPIT/Soccol models, but its conversion used all 24 elements in the workbook. Seven elements were omitted from the old predictor list: Be, Ga, Hf, La, Sn, Zn and Zr. Positive omitted amounts occurred in 67 of the 332 original candidates. Their retained 17-element subtotal was not renormalized to 100.

For the new MPEA specialist/shared schema, revisit those omissions. The 90% completeness rule applies to the **full source composition**, not an intentionally reduced feature subtotal. Expanding the element registry is preferable to pretending the omitted mass is Fe or redistributing it among retained elements.

### 06b. Resolve blanks, zeros and balance elements

1. Start with raw blanks as unknown.
2. Preserve reported numeric amounts and explicit zeros unless a documented source error is established.
3. If the alloy description and dataset context justify a balance element, calculate its remainder. It may be Fe or another base element; a low total by itself does not identify the base element.
4. Check for known omitted alloying elements **before** calculating the remainder. Otherwise Fe can absorb missing Cu, Nb, Re or another element and create a misleading total of 100.
5. Once the complete composition is justified, set other unreported elements to zero and record them as assumed zeros. This includes extra element columns introduced by a shared schema.
6. If completeness is not justified, retain unknowns in the audit and withhold the row from the complete-composition development population. Do not use mean composition or a guessed grade recipe.
7. Do not assign a combined `Nb + Ta` amount wholly to Nb or split it arbitrarily. Recover separate amounts from evidence, explicitly represent the unresolved quantity in a future schema, or keep the row unresolved.
8. If recovering amounts from another dataset, require the same original source and a matching composition with agreement on reported elements. Do not use a processed assumed zero as independent chemical evidence.
9. Keep composition recovery separate from duplicate-measurement removal. Recovering a source composition does not authorize a final query's measured target entering context.

### 06c. Apply acceptance checks without hiding uncertainty

1. After unit checks and justified balance completion, withhold/exclude a row with an unexplained full composition total below **90 wt.%**, the current general minimum.
2. A total above 90 is not automatic acceptance. A known important omission remains a problem even at 99.9 or after a balance made the total 100.
3. Require nonnegative finite accepted amounts, no amount over 100, and a plausible total near 100.
4. The chat proposed accepting very small deviations around ±0.3 percentage points without extensive investigation. The implemented reviewed EPIT export actually uses **99–101 wt.%** as its practical zero-fill acceptance interval; it retained totals 99.5, 99.8679 and 100.52 unchanged. Do not silently describe these as the same threshold.
5. For a new dataset, write the chosen tolerance and any source-specific exceptions explicitly. A known omission overrides a convenient rounding tolerance.
6. Preserve accepted real amounts rather than forcing every real row to exactly 100. Synthetic perturbation/closure is a later, separate operation.
7. Small upper-limit approximations must be explicit. EPIT's `<0.003` sulfur entries were accepted as 0.003 wt.%; this does not authorize replacing large limits, intervals or arbitrary text by a number.
8. Keep accepted-with-uncertainty decisions visible. EPIT's retained Re/Nb/Ta uncertainties were a specific user-approved decision, not proof that the uncertain amounts were recovered.

**Save:** full-basis composition, unit-conversion record, row-level totals, per-cell zero/balance/limit/correction evidence, acceptance status, exclusion reason and a before/after change table.

**Ready to continue:** every composition zero is either reported or justified by completeness; no balance element conceals a known omission; no incomplete projection has been normalized into a fake complete alloy.

## 07. Process environment, preparation and material-history features

Use the existing shared meanings wherever they are actually equivalent. Split multi-information text into fields with one meaning each.

| Information | Current compatible representation | Rule for a new dataset |
|---|---|---|
| Test temperature | `CP_temp`, °C | Convert explicit units. Do not assume “room temperature” means a particular number without a declared rule. |
| Test pH | `CP_pH`, dimensionless | Preserve reported values, including negative pH. Do not infer it from acid concentration. |
| Scan rate | `scan_rate`, mV/s | One numeric column. “Steady state” or a potential increment without a duration is not a numeric rate or zero. |
| Exposed area | `Test_area_cm2`, cm² | Convert area units, not lengths. Confirm which specimen is described. |
| Test medium identity | `test_solution_type`, categorical | Record an explicit whole medium/recipe. An additive note alone may not identify the whole bath. |
| Named chemicals | Separate fields such as `NaCl_mol_L`, `H2SO4_mol_L` | One column per named component and concentration basis; mixtures can contain several simultaneous amounts. |
| Normality/mass percentage | Fields such as `NaCl_eq_L`, `NaCl_wt_percent` | Keep distinct from mol/L unless a defensible conversion is available. |
| Reported ions | `CP_Cl`, `CP_Br`, `CP_OH`, `CP_SO4`, `CP_CO3`, `CP_NO3`, `CP_PO4`, `CP_MoO4`, `CP_CrO4`, mol/L | Do not reverse-infer salts or acids from ions. Preserve reported zeros and unknowns. |
| Gas condition | `CP_aeration`, categorical | Aerated/deaerated and explicitly named gas conditions; missing does not mean naturally aerated. |
| Gas pressure | For example `H2S_partial_pressure_atm` | Pressure is not dissolved concentration. |
| Hydrodynamic motion | `CP_agitation`, 1/0/NaN | Explicit yes/no/unknown. Bubbling or stationary current does not by itself establish agitation. |
| Abrasive grade | `Prep_grinding_grit` | Preserve a supported numeric designation and its convention in the audit. Do not equate grit, mesh and particle diameter. |
| Surface roughness | `Prep_Ra_micron`, µm | Measured/reported Ra; do not derive it from grit or polishing-particle size. |
| Preparation operation | `surface_preparation_method`, categorical | Explicit operations; standardized combination names do not imply a chronological sequence. |
| Preparation liquid | `preparation_solution_type`, categorical | Separate from the corrosion-test bath. |
| Preparation pH | `Prep_pH` | Separate from `CP_pH`. |
| Preparation potential | `Prep_redox`, V under the current registry | Preserve reference uncertainty; do not treat it as a common-reference voltage without evidence. |
| Preparation duration | `Prep_time`, hours | Chemical/electrochemical preparation; not annealing or an unexplained wait in air. |
| Test equilibration | `CP_time`, hours | Time in the test medium before measurement; not time to pitting or a purge before immersion. |
| Heat treatment | Four fields per reported stage | Type, temperature °C, duration h, cooling method. Current schema has two stages. |
| Phases | `phase_FCC`, `phase_BCC`, `phase_HCP`, `phase_IM` | Each is 1 present, 0 supported absent, NaN unknown. Several can be present. |
| Measurement reference | `reference_type` | Target-reference category established in step 05. |

### Solution rules that must not be simplified away

1. `test_solution_type` plus a single concentration cannot describe independently varying amounts in a mixture. Retain the agreed component-specific columns.
2. Example: an explicitly stated `1 M H2SO4 + 0.5 M NaCl, deaerated` supplies a mixture name, two chemical molarity columns and aeration.
3. Example: explicitly stated `10 wt.% HNO3` can justify `test_solution_type = HNO3` and a new `HNO3_wt_percent = 10`. This text is a source value, not the name of an existing column. Add the new column through the registry/version process.
4. A bare `%` or `ppm` with unknown basis does not automatically populate wt.% or mol/L. Preserve it in the audit.
5. Do not equate acid molarity with free sulfate/nitrate concentration. Do not infer pH or activities from formal concentration.
6. Only apply simple nominal ion mappings where the source and the adopted rule support them. EPIT retained explicit NaBr molarity and derived nominal bromide; it did not convert every normality or mixed acid/salt entry automatically.
7. Do not invent a complete seawater/PBS/Hanks recipe from a medium name.
8. Parse preparation solutions separately. A cleaning acid is not a corrosion-test acid.
9. Preserve the explicit decision to exclude `CP_ion_other` from the reviewed workflow. Do not reintroduce it merely because the historical combined schema contains it.

### Heat-treatment and phase rules

1. Keep stage 1 and stage 2 separately; do not average temperatures or add unrelated durations.
2. Attach cooling to the stage for which it was reported.
3. Alternative treatments for different specimens are not successive stages for one row.
4. An empty second stage is unknown/not recorded, not a known zero-hour treatment.
5. Naming FCC alone does not prove other phases absent. Use zero only for supported absence, such as an explicit single-phase description or an explicit exclusion.
6. General discussion of phases in a paper does not establish a sample's microstructure.
7. If future data require more stages/phases, extend the schema deliberately instead of overloading an existing column.

**Save:** explicit extraction mappings for unique descriptions; conversions; original text; matching specimen/stage; conflicts; per-feature coverage; and excluded-field reasons.

## 08. Review contradictions and finalize row eligibility

1. Compare explicit text with numeric cells. Check zeros as carefully as nonzero values.
2. If reports conflict and neither is established as correct, keep the affected processed cell NaN and retain both originals. An environment conflict usually does not require deleting the whole row.
3. Do not apply a global correction based on a subset of discrepancies. Soccol sulfate disagreements did not justify dividing all sulfate values by two.
4. Check that a preparation time describes the same operation in each dataset. Soccol's air-only durations were not all equivalent to EPIT's chemical/electrochemical `Prep_time`.
5. Review only the relevant specimen. Preparation of an exposure coupon is not necessarily preparation of the pitting electrode.
6. Verify reported element values have not come from a misplaced derived index. Soccol had a source-specific Ti-column error of this kind.
7. Check additions against old values and against source evidence. Record changed cells and affected row IDs, not just the new total row count.
8. Separate three statuses: accepted; accepted with a documented qualification; withheld pending information. Do not label every reviewed table “perfect”.
9. If a split already exists, apply new development eligibility without silently changing final-query membership or fold numbers.

**Existing Soccol exception:** its accepted development file excludes 272 unresolved rows, while all 805 frozen queries remain, including ten unresolved compositions. The model can represent NaNs; that does not make those compositions verified. Preserve the boundary and disclose this exception. Do not generalize it into a rule to accept incomplete new development compositions.

**Ready to continue:** exclusions and qualifications are explicit and source-based, not chosen because they improve model scores.

## 09. Freeze the outer split and development folds, or reuse the existing ones

For an entirely new dataset, complete deterministic target/composition eligibility sufficiently to define the population before freezing a split. Do this before fitted preprocessing, formula selection, profiles or Optuna. For MPEA, reuse the already-frozen split and apply corrections as versioned feature views of its IDs.

1. Use approximately 80% development / 20% final, following the established grouped procedure.
2. Group repeated and very similar compositions, not arbitrary individual rows. The established machinery rounds compositions to 0.01 wt.% and uses connected groups with L1 distance at most 1.0 wt.%.
3. Record the exact composition columns used for grouping. Connected grouping is transitive: document any large groups and do not assume every pair in a group is within the threshold.
4. Current MPEA grouping used missing-as-zero only in the grouping representation; this did not authorize changing the exported composition values to zero.
5. Reuse the EPIT split optimization/refinement code and document meaningful dataset-specific omissions. Target-decile and feature/source balancing help construct the split; they are not permission for later score-driven resplitting.
6. Source-paper identity is not automatically a hard grouping boundary in the current procedure. Audit it separately.
7. Freeze five grouped development folds. For MPEA they contain 43 / 43 / 43 / 43 / 42 rows; the final 53 remain separate.
8. Save manifest, assignments, grouping definitions, report and hash lock. Save target-free final-query features separately from scoring targets where the workflow supports this.
9. Verify ID uniqueness, no overlap, correct union, intact composition groups and the expected fold populations.
10. When adding features later, retain the existing assignments. EPIT deliberately retained a split made with its legacy 17-element grouping after expanding its model features.
11. If new evidence reveals a duplicate across a frozen boundary, record it and resolve the protocol explicitly; do not quietly reshuffle until scores look better.

**MPEA paths:** [frozen specialist split](datasets/mpea_corrosion/processed/splits_v1/) and [split builder/validator](datasets/mpea_corrosion/prepare_composition_split.py). Its `load_frozen_mpea_split()` is the intended validated entry point.

**Ready to continue:** you can name the exact development IDs and final-query IDs, and all later code must use them.

## 10. Publish a versioned reviewed dataset and its feature registry

Write a reproducible processor rather than a one-off notebook edit. It should read the source and decision tables, validate them, and write a new version without overwriting previous evidence.

Recommended outputs, following the existing exports:

| Output | Required content |
|---|---|
| `<dataset>_processed.csv` | Stable IDs, corrected values, target and required provenance/source fields. |
| `<dataset>_features.csv` | Stable ID plus precisely the registered inputs; no target or unregistered text. |
| Accepted development view | The accepted development IDs, their exact inputs and corresponding labels. |
| Frozen final-query view | Every frozen final-query ID in the required order; target-free when exported for inference. |
| Separate scoring targets | Final targets joined by stable ID, never positional guesswork. |
| `feature_registry.json` | Ordered inputs, types, units, categorical fields, composition fields, exclusions and qualifications. |
| `split_assignments.csv` | Original frozen membership and fold assignment, not a new split. |
| Composition/environment audits | Original value, output value, action, evidence and row eligibility. |
| `processing_manifest.json` | Source hashes, processor/rule versions, output hashes, row/feature counts and unresolved issues. |
| Before/after table | Added, removed and changed cells/columns relative to the previous version. |

1. Read written CSVs back and check IDs, types, zeros, NaNs and target alignment.
2. Check that serialization has not turned an unknown category into the literal observed category `"nan"`, `"None"` or `"unknown"`.
3. Compare every declared model column against the feature table. Do not infer features as “everything except the last column”.
4. Keep metadata and raw text outside the feature list even if they share the processed CSV.
5. Record the processed dataset version separately from the split version; they are different objects.
6. Keep the old exports. An existing checkpoint's provenance must remain interpretable after a new export is published.

**Ready to continue:** the reviewed export is reproducible and inspectable, and its model inputs are unambiguous. Saving this CSV alone has not yet migrated any loader, sampler or model.

## 11. Decide the common schema and implement explicit mappings

Do this before committing to an expensive model that will later require a different fixed schema.

1. Compare registered fields by **meaning + unit + type**, not name alone.
2. Build a mapping table: source dataset, source field, shared field, conversion, type, unit, missing rule and evidence.
3. Use the union of justified features when preparing a combined schema. Do not discard a useful field solely because one other dataset lacks it.
4. Keep names stable where meanings already match. EPIT/Soccol's reviewed 49 common names already match in declared type/unit.
5. Separate genuinely different quantities: chemical versus ion amount, preparation versus test pH, grit versus Ra, target reference versus preparation-potential reference.
6. Extend the composition registry for real elements missing from the old model, where this is part of the new experiment. EPIT/Soccol's 24 symbols are not a permanent restriction on future datasets.
7. Add new chemical-concentration columns with explicit units as needed. Do not reuse `CP_NO3` to store HNO3 wt.% merely to avoid a new column.
8. Choose one explicit shared column order. Save it in a versioned registry and validate it everywhere.
9. Map absent environment/history fields to NaN. Map newly introduced elements to zero only when that source composition was accepted as complete under step 06.
10. Keep known zero and missing distinct after concatenation. A blanket `fillna(0)` is not a schema-alignment operation.
11. Resolve categorical spelling and combination-order differences before assigning integer codes. Preserve physically meaningful distinctions, such as natural and synthetic seawater.
12. Check for collisions between old and new meanings. For example, the historical combined schema treated some fields categorically that the reviewed schema now represents numerically.

**Current gap:** `datasets/combined_epit_soccol_mpea/processed/epit_soccol_context.csv` belongs to the historical 50-column workflow. It does not automatically acquire the reviewed corrections because EPIT/Soccol standalone exports changed. The recorded 79-column alignment is a plan for a union, not proof that this CSV, its registry, the combined sampler or old checkpoints use that union.

**Architecture decision:** either train a specialist in its reviewed native schema and later implement/test a named mapping for its prior, or train a new specialist with the agreed common schema. A native fixed-schema checkpoint cannot be made equivalent to a common-schema-trained model by blindly appending or reordering columns at inference. Make this choice explicit; a different schema can require new training.

**Save:** a versioned shared registry, mapping table, mapping implementation and field-by-field comparison. Build merged diagnostic tables now if useful, but do not mix final-query rows into development banks.

## 12. Define one authoritative accepted-development selector

1. Start from the frozen original row membership.
2. Apply reviewed development eligibility using stable IDs.
3. Preserve each accepted row's original development fold and original task-row index.
4. Expose a single selector that loaders, formula fitting, profile builders, Optuna and final context all reuse.
5. Return its ID list, fold assignments, counts and a hash/identity. Do not replicate the selection in several slightly different filters.
6. Separately expose the complete frozen final-query membership. Its selection must not be accidentally inherited from the development acceptance mask.
7. Check exact membership, not just totals. Two different 2,950-row subsets are not the same dataset.

**Current Soccol reference:** [data.py](../scripts/soccol_pipeline/data.py), especially `select_accepted_development()`. It selects 2,950 accepted development rows and preserves fold sizes **622, 615, 638, 445, 630**, with corresponding context sizes **2,328, 2,335, 2,312, 2,505, 2,320**.

For EPIT, distinguish the full 608-row development population from the 452-row Fe/Ni sampling/formula population. For MPEA, 214 is the frozen development count before any future explicitly documented eligibility changes; do not silently assume every later feature review preserves acceptance.

**Ready to continue:** all development consumers agree on the same IDs, with any additional material-family applicability filter stated separately.

## 13. Connect every real-data loader to the reviewed export

1. Audit the actual call paths for splitting validation, formulas, profile building, Optuna evaluation, final context and final queries.
2. Replace legacy source tables in active model inputs with the reviewed export. Retain an explicit legacy path only for reproducing old experiments.
3. Validate saved hashes, IDs, schema order, types and eligibility before returning model inputs.
4. Preserve composition values exactly: no new Fe remainder, blanket blank-to-zero conversion, mean composition or renormalization of real rows.
5. Preserve environment NaNs and explicit zeros.
6. Map old formula input names to reviewed column names without changing definitions accidentally. An adapter is often safer than rewriting every formula.
7. Keep targets aligned by ID. A changed feature CSV must not shift target order.
8. Ensure an evaluator cannot accidentally load the old source simply because a default flag, path, feature count or task ID was not updated.
9. Inspect underlying dependencies. EPIT still uses original workbook information in some loaders; having the reviewed CSV on a remote node does not prove every raw dependency is present.

**References:** [EPIT reviewed composition loader](../scripts/epit_pipeline/composition_data.py), [EPIT formula adapter](../scripts/epit_pipeline/formula_data.py), [Soccol reviewed loader](../scripts/soccol_pipeline/data.py), and [Soccol formula adapter](../scripts/soccol_pipeline/formula_data.py).

**Check:** loading the same accepted ID through the real evaluator, formula adapter and profile builder yields the intended same physical values. Differences must be an explicit projection or internal preprocessing step, not an unexplained old-data path.

## 14. Audit material families and formula applicability

Before fitting coefficients, determine what population the existing formulas actually describe.

1. Count accepted development observations, distinct compositions and sources by recorded material family.
2. Compare the labels with the actual composition. A family label is not necessarily a largest-element classifier.
3. Count the rows to which each existing formula is applicable and why. List required elements, environmental inputs, units and any auxiliary method information.
4. Distinguish unavailable measurements from justified zero amounts. A formula cannot treat an unknown pH as a measured neutral solution.
5. Identify inputs that are always zero, always missing or constant in development. The corresponding formula term may contain no information for this dataset.
6. Check whether alloy families or test solutions absent from the original rule's support are common in the new dataset.
7. Define the material scope deliberately. Do not copy EPIT's Fe/Ni restriction merely because its implementation is convenient.

| Existing pipeline | Current relevant scope |
|---|---|
| EPIT | 608 development rows overall. The active informed feature bank/formula population uses **411 Fe + 41 Ni–Cr–Mo observations**, corresponding to **241 + 14 = 255** eligible distinct composition templates. Other development families remain available for real-model evaluation. |
| EPIT composition asset | Contains 323 distinct development templates across its recorded families, but the current informed Fe/Ni path uses the eligible 255. Asset inventory is not identical to runtime sampling scope. |
| Soccol | All **2,950 accepted development rows**, **349 distinct compositions**, without an additional alloy-family filter. Historical family “include/review/exclude” labels do not override the current accepted-row policy. |
| MPEA | Must be reviewed for its own support. Existing screening found a 13-row Ti/Nb-rich, Cr/Mo/W-free group in nitric acid that was poorly represented by inherited rules. This observation does not establish a complete chemical explanation or trained-model outcome. |

Soccol's recorded development-family counts were 2,745 stainless-steel rows, 65 Fe-labelled model-alloy rows, 64 Alloy-800 rows, 39 Ni–Cr–Mo rows, 35 broadly labelled high-alloy rows and two Alloy-600 rows. Some stored labels differed from the largest element. These are descriptive row counts, not current synthetic family probabilities.

**Save:** material-scope report with observation/composition/source counts, applicability masks and caveats. Use [Soccol's material-scope audit](analysis/soccol_pipeline/material_scope_audit_v1/README.md) as an example, while remembering that its old composition-draw shares preceded the later uniform-template change.

## 15. Map and screen candidate target formulas

1. Start from named existing definitions, not just their fitted numerical coefficients.
2. Document each term, sign, unit, coefficient anchor/bounds, material applicability and required input.
3. Map reviewed inputs explicitly. Keep a test showing that the old and new names produce the same formula value when given the same physical inputs.
4. Expanded model features do not automatically become formula terms. The SCM can use more fields while an analytical rule retains a smaller named projection.
5. If an input is unavailable, choose explicitly among: retaining the established internal filling convention where justified; evaluating only the applicable rows; defining a clearly named reduced rule; or rejecting that candidate for the new dataset.
6. Removing nitrogen, sulfur, pH or ion terms creates a modified rule. Label it as such; do not present it as the original formula with equivalent evidence.
7. Evaluate simple composition-only controls where useful, to distinguish weak environmental support from weak composition ranking.
8. Inspect term variation. A coefficient optimizer cannot rescue a formula whose informative terms are all constant or absent.
9. Keep the source interpretation fixed while comparing rules. Do not switch chemical units to make a formula score better.
10. PySR/symbolic discovery is **not required now**. Add it only as a separate, explicitly chosen extension after the reviewed-input workflow is correct.

**MPEA starting evidence:** [existing development formula screen](analysis/mpea_target_rules_v1/README.md) evaluated eleven reduced candidates and eight composition-only controls. Its best fitted candidate achieved about **0.2334** mean fold Spearman, versus **0.2424** before fitting. It did not promote a production MPEA prior or use the 53 final targets. A revised feature schema can invalidate these input assumptions and requires another versioned check.

## 16. Fit formulas with the established development-fold procedure

For each candidate and each frozen development fold:

1. Select the other four folds as context, applying the same accepted-row and formula-applicability rules.
2. Fit missing-value filling values on those context rows only. Keep these internal to formula calculation.
3. Fit term/input scaling and any method offsets on those same context rows only.
4. Fit coefficients against the context targets. Preserve the existing definitions, coefficient term order, constraints and bounds when the intent is a refit rather than a new formula.
5. The established coefficient fitter uses nonnegative coefficients summing to one, rule-specific upper bounds, standardized target ranks and SLSQP, with anchor regularization strength 0.1. Reuse that fitter when applicable; do not substitute an unconstrained regression without documenting the change.
6. Transform and predict the held-out development rows with the saved context state. Do not refit means, scales or coefficients using them.
7. Save fold coefficients, filling/scaling state, IDs, predictions and metrics.
8. Report unweighted mean fold Spearman, the individual fold scores and population sizes. Keep pooled out-of-fold scores distinct from mean fold scores.
9. Mark correlations undefined when there are too few rows or no variation. Do not turn them into zero silently.
10. After the development comparison, fit one coefficient center per retained rule on all applicable accepted development rows for synthetic training.

**Important distinction in the current implementation:** a formula's cross-validation check fits each fold on its own context. However, the production synthetic prior used in Optuna contains the coefficient centers/rule weights fitted using the full outer development partition, and its feature bank also uses outer development. The same synthetic checkpoint is then evaluated across development folds. Those Optuna scores are model-selection evidence; they are not a fully nested estimate with each fold excluded from every upstream prior-fitting step.

For alignment, retain that current outer-development-only workflow and describe it accurately. A stricter nested experiment would require fold-specific banks/coefficient fits and usually separately trained models, substantially increasing cost. It is a different protocol to decide before running, not a hidden requirement to introduce afterward. **Neither version may use outer final queries to fit the prior.**

**Save:** versioned coefficient files, calibration summary, fold metrics, out-of-fold predictions, applicability counts, preprocessing states, source/split/selection hashes and a report. Current references are [EPIT calibration](../scripts/epit_pipeline/calibrate_target_rules.py) and [Soccol calibration](../scripts/soccol_pipeline/calibrate_target_rules.py).

## 17. Review formula scores and decide what becomes a production rule

1. Compare fixed/default coefficients with the fitted version on the same held-out rows.
2. If old results exist, preserve them and report whether old/new populations match. A change from 3,222 to 2,950 rows confounds a simple “formula improved” claim.
3. Where feasible, compare old/new rules on matched IDs as well as on their full eligible populations.
4. Break down scores by sufficiently populated material family, solution/source and missingness conditions. State observation and source counts; a single-source family with three rows in a fold cannot support a broad generalization.
5. Inspect whether stronger aggregate ranking merely separates groups while within-family ranking remains weak.
6. Keep, modify, remove or add formulas based on development evidence and physical applicability. Save the decision and new registry version.
7. Make the runtime registry agree exactly with the calibration artifacts: rule names, order, terms, bounds and coefficients.
8. Current rule sampling normalizes finite nonnegative rule scores into probabilities. Negative scores cannot be fed to the runtime unchanged, and their absolute values must not be used as evidence of a good rule. Decide explicitly how unusable/negative rules are excluded or treated, then version that policy.
9. Validate that probabilities are nonnegative and sum to one, and at least one usable rule remains.

**Do not start Optuna merely because a coefficient JSON exists.** First establish that its inputs are correct, its scope is suitable, and the generator implements the same calculations. Weak formulas may remain a deliberate experimental choice, but record that choice rather than assuming fitting solved transfer.

## 18. Build versioned composition and environment sampling profiles

A profile is not just a CSV copy. It defines what synthetic data can be generated and with what probabilities.

1. Build only from accepted **outer development** IDs. Exclude final-query features as well as labels from the empirical bank and its fitted statistics.
2. Apply the declared material scope. Record the selected source IDs, task indices and family/applicability filter.
3. Build a distinct-composition inventory. Use the declared exact/rounded deduplication rule and record it; do not confuse split grouping with profile-template deduplication.
4. Keep the full accepted environment/history observation bank separately. Repeated measurements can remain repeated environment rows.
5. Store the exact ordered feature schema and feature types.
6. Fit category mappings and finite internal filling values on the documented development scope, never on final queries.
7. Current EPIT fits its category vocabulary on all 608 development rows while sampling environment observations from its 452 eligible Fe/Ni rows. Soccol uses its 2,950 accepted development rows. Preserve or explicitly adapt this distinction.
8. Store original NaNs in the model-visible bank. Store finite internal numeric means/category modes separately; a wholly unobserved field may use a neutral internal placeholder as documented, without becoming an observed value.
9. Record composition sampling, environment sampling, category coding, missingness, internal filling, perturbation and normalization policies in JSON.
10. Store CSV and metadata hashes, source export identity, frozen split identity, accepted selection identity and builder/software versions.
11. Add a deterministic rebuild/check mode. Rebuilding from the same inputs must reproduce the frozen assets, or explain an intentional version change.
12. Prefer loaders that reject unknown/stale profile names. A permissive fallback to an old profile can waste the entire search.

**Current policy:** sample distinct accepted compositions uniformly; sample complete environment/history observations uniformly and independently of composition. EPIT's current Fe/Ni family probabilities and within-family template draws together implement uniform selection of its eligible templates. Soccol v3 implements uniform selection of its 349 distinct templates. This is not equal weighting of material families or source papers.

**Existing builders:** [EPIT development profiles](../scripts/build_epit_development_profiles.py), [EPIT expanded feature profile](../scripts/build_epit_expanded_feature_profile.py), and [Soccol pipeline assets](datasets/soccol_pitting_potential/prepare_pipeline_assets.py).

**Ready to continue:** profile rows and statistics are traceable to the intended development population, and no query-derived information is present.

## 19. Implement and audit synthetic sampling in the correct order

First adopt the [shared-stream contract](../scripts/epit_pipeline/PAIRED_STREAMS.md): independent indexed generic/informed streams, exact per-batch source counts, a common generic recipe, per-table random seeds and checkpointed consumed positions. Keep compatible groups together for micro-batching. An MPEA adaptation must extend the profile identity and informed schema support.

For each informed synthetic task, follow the existing generator contract:

1. Allocate source groups using the exact requested proportions, advance each source's own table counter, and select the SCM type using that group's indexed random draws.
2. Draw composition templates using the declared composition distribution.
3. Perturb the positive element amounts using the existing composition-noise mechanism. Preserve zeros. Close **synthetic** compositions to 100 as specified; leave saved real compositions unchanged.
4. Draw an environment row independently from the accepted environment bank.
5. Copy all that row's solution, concentrations, preparation, heat treatment, phases, reference information and NaNs together. Do not draw pH, acid amount, heat temperature and treatment duration independently.
6. Keep composition and environment independent as in the current EPIT/Soccol workflow. This can create combinations not observed together; it is a deliberate shared assumption, not an EPIT/Soccol difference by itself.
7. Create the separate finite representation needed by the SCM/formula calculation, using the same sampled row and saved filling state.
8. Draw the analytical rule and, when enabled, perturbed coefficients at the established task scope. Do not independently change the rule per row if the current prior defines one task relationship.
9. Give formulas their named raw-physical inputs; give the SCM its standardized finite physical inputs. Do not accidentally feed a standardized chloride value into a formula expecting mol/L.
10. Combine the standardized SCM target and formula contribution using the configured formula mix, then apply the established final target standardization. Preserve the existing treatment of constant signals.
11. Apply the generator's model-input transformations and restore the original sampled missingness before the tensor reaches the model.
12. Verify that added model columns do not shift formula terms or auxiliary fields. EPIT's private formula test-method input is not an extra public model predictor.

### Sampling checks before a long run

- Count distinct templates and actual draw shares; inspect whether repeated source observations accidentally overweight a composition.
- Verify environment tuples and their missing masks come from one allowed source row.
- Verify composition and environment draws are independent under the current policy.
- Check nonnegative synthetic compositions, zeros staying zero, finite closure and realistic perturbation size.
- Measure perturbation in wt.% points as well as the sampler parameter. Strength 0.30 is a log-normal-noise setting, not a hard ±0.30 wt.% or ±30% physical bound.
- Check whether perturbation crosses the original family/domain support. Existing samplers do not automatically enforce every source range after perturbation; a new restriction would be a documented change.
- Count missing and constant columns in both the source bank and generated data.
- Verify both the MLP and tree SCM routes, and formula contribution endpoints 0 and 1.
- Compare synthetic formula outputs with the standalone calibration implementation on identical finite inputs.
- Reproduce draws with a fixed seed and check the intended coefficient RNG independently.

Use [the Soccol sampling audit](../scripts/soccol_pipeline/audit_synthetic_sampling.py) as a structural example, but note that the historical audit/report described v2 weighting before v3 changed composition draws. Recompute the diagnostic for the actual selected profile/version.

**Ready to continue:** an observed issue is traced to a specific difference or shared assumption. Do not explain weaker Soccol performance merely by something also present at similar magnitude in EPIT, such as some missing/constant columns.

## 20. Enforce categories and missingness consistently through inference

| Case | Saved real data | Synthetic/internal behavior | Model-facing behavior |
|---|---|---|---|
| Complete composition, unreported extra element | Justified assumed zero, audited | Zero remains zero under perturbation | Numeric zero |
| Important unresolved composition amount | NaN plus qualification; normally withheld from accepted development | Do not invent a complete template | Existing frozen-query exceptions retain NaN and their qualification |
| Unknown environment value | NaN | Internal filling only for target calculations | Restore NaN for a missing-aware checkpoint |
| Explicit reported environmental zero | 0 unless contradicted by evidence | Preserve it | Numeric zero |
| Known category with code 0 | Known label | Saved mapping gives 0 | Valid observed category, not missing |
| Missing or unseen category | Missing/unsupported | No vocabulary learned from queries | NaN under the reviewed mapping |
| Field absent from a dataset | NaN for environment/history | Same absence represented in its synthetic component | NaN in the fixed shared position |
| Zero/one known numeric context values | Values retained | No invented evidence | Skip scaling, normalization and outlier clipping for that column |

1. Reuse the stored development category mapping for real and synthetic reviewed inputs. Do not independently fit a new encoder in the evaluator or each dataset before merging.
2. For a combined schema, establish common semantic category mappings. Native code `1` in EPIT need not mean native code `1` in Soccol. Translate through names or an explicit mapping.
3. Preserve feature count and order, including constant and entirely missing fields where the fixed informed schema requires them.
4. Enable the learned missing-value embedding for newly trained reviewed models and verify it is saved in model configuration and used on reload.
5. The missing-aware path must not silently mean-impute every NaN before the model sees it.
6. Generic/pretrained baselines retain their native missing-value and preprocessing behavior. The comparison uses the same underlying real measurements and split, while retaining the inference conventions appropriate to each checkpoint.
7. Current standalone reviewed sampling adds no independent extra missingness mask. Do not inherit historical combined masking/augmentation by accident; inspect and document the combined policy separately.
8. Test missing categorical values, unknown labels, entirely missing numeric columns, and single-observation columns, including known query values first appearing in those columns.
9. Also test the outlier stage when its preliminary rejection leaves fewer than two context observations. Known values must not become NaN because a spread cannot be computed.
10. Check prediction in one batch versus separate batches. Query composition must not change fitted context statistics or feature selection.

**Failure found in this session:** the old scaler/outlier path erased two known final-query values each in `WO4_mol_L` and `CP_SO4`, plus one development `NaClO4_mol_L` value. Finite predictions alone did not catch the lost information. The shared preprocessing fix now preserves all five; regression checks must verify values as well as output finiteness.

**References:** [shared preprocessing](../src/tabicl/sklearn/preprocessing.py), [sparse-column tests](../tests/test_sparse_preprocessing.py), [prediction/missing-column tests](../tests/test_prediction_missing_columns.py), and [missing-value embedding tests](../tests/test_missing_value_embedding.py).

## 21. Connect the profile, formulas and schema to the training/evaluation entry points

Update and verify every link; changing a processor or profile file alone is insufficient.

| Location | What must agree |
|---|---|
| Schema module | Ordered feature names, numeric/categorical types, count and block boundaries. |
| Profile loader | Exact supported profile version, source hashes and accepted development selection. |
| Synthetic generator | Composition/environment columns, formula projection, category codes and missingness. |
| Target-rule loader | Exact rule inventory, term names/order, coefficients, bounds, scores and input identity. |
| Training command builder | Profile name, composition mode, fixed feature count, block allocation, missing-value embedding and optional descriptors. |
| Optuna trial parser/fingerprint | Search ranges and fixed settings, including the fixed informed ratio. |
| Fold evaluator | Reviewed data flags, correct schema, accepted context and intended development queries. |
| Final trainer | Same selected prior/formula configuration plus explicit permitted overrides. |
| Final evaluator/comparison | Frozen final queries, correct checkpoint identity and native/fixed-schema model settings. |
| Slurm scripts | Same defaults as Python, correct project/environment paths and data availability. |
| Documentation/tests | Current names and behavior, not just historical examples. |

1. Make the reviewed version the active default. Keep legacy behavior explicitly opt-in.
2. Set minimum/maximum feature counts to the intended schema width on the fixed informed path; inspect all block-allocation arguments.
3. Keep optional derived descriptors versioned and ordered. EPIT's optional ten Magpie descriptors currently use its historical 17-element descriptor subset even though base composition expanded. Do not silently claim those descriptors include newly added elements.
4. If descriptors are enabled, verify the same calculations and appended positions in synthetic training and real evaluation, and record the resulting feature count.
5. Retain raw physical formula inputs even when the model receives transformed or expanded inputs.
6. Inspect the exact emitted training and evaluation commands before starting a GPU job. A module with the correct name can still forward old profile constants.
7. Use a new study/version when the data, schema, profile, formulas, fixed settings or search space change incompatibly.
8. Make incompatible artifact combinations fail early. Do not bypass mismatched hashes to make a launcher run.

**Known baseline-alignment item:** the current EPIT `evaluate_baseline_folds.py` explicitly supplies `none` normalization/shuffling and does not request `--native-tabicl-baselines`; Soccol reuses that helper. The updated final comparison does request native baseline settings. Before producing a new dataset's baseline-fold comparison, decide and implement the intended matching baseline convention, version its outputs, and check the emitted flags. This guide records the difference; it did not change those settings or rerun historical baseline results.

**MPEA implementation work:** create/adapt its reviewed loader, authoritative accepted-development selector, schema, profile builder/loader, generator integration, formula-calibration artifact loader, Optuna wrapper, final trainer and evaluators. Its existing `scripts/mpea_pipeline/target_rules.py` and frozen split are starting components, not the complete pipeline.

## 22. Record identities and decide restart behavior before expensive training

The following objects need separate identities because changing any one can change the experiment:

- Raw source files and correction/extraction rules.
- Reviewed export and exact accepted development IDs.
- Frozen split manifest/lock and original row identities.
- Ordered model feature registry and category mappings.
- Composition and environment profiles, their sampling weights and internal filling state.
- Formula definitions, coefficient centers/bounds, scores and selection probabilities.
- Optional descriptor definitions.
- Search space, fixed settings and trial-selection rule.
- Training/model configuration, software revision and relevant dependency versions.
- Selected trial, selected checkpoint and its hash.

Record these in processing/profile/calibration manifests and in the study fingerprint. For a new production training implementation, also save the relevant profile/training identity in the checkpoint so an old weight file cannot silently be paired with the newest CSV having the same filename.

**Current implementation:** new EPIT v12/Soccol v6 paired checkpoints contain stream/profile identities and consumed worker state. The low-level trainer supports restoration of those checkpoints and rejects incompatible or legacy stream metadata. The final-training launcher still has no training-resume option; `--resume-checkpoint-selection` is evaluation recovery only. The separate deferred launcher/writer-lock work was not applied. See [shared streams](../scripts/epit_pipeline/PAIRED_STREAMS.md) before planning recovery.

Distinguish these three operations:

| Operation | Meaning | What it does not guarantee |
|---|---|---|
| Join/resume an Optuna journal | Continue the study's trial records and search | Recovery of a killed trial's partially trained model. |
| Resume checkpoint selection | Reuse completed training and continue development checkpoint evaluation | Restart of interrupted model training. |
| Exact training resume | Restore model, optimizer, scheduler, scaler and the consumed synthetic sample/RNG state | Available from an older checkpoint that never saved the required worker state. |

For exact continuation, preserve Python/NumPy/Torch and coefficient RNGs, dataloader worker state, consumed-versus-prefetched batch boundary, seeds, worker count, world size, scheduler horizon and model/prior configuration. Saving a seed alone is insufficient.

Checkpoint writes are now atomic. The separate deferred implementation provides a per-run writer lock; that launcher safeguard remains unapplied. Do not run concurrent writers in the same checkpoint directory. A new missingness/schema setting is not a valid exact-resume continuation of a checkpoint trained under another setting.

**Ready to continue:** the run can explain exactly what it was trained with, and its interruption/restart policy is known in advance.

## 23. Make the complete input bundle available on every node

1. List every file the actual loader opens, including referenced manifests, registries, assignments, profile CSV/JSON files, formula JSONs and any still-required raw workbooks.
2. Check whether each required artifact is tracked, copied explicitly, or stored on a genuinely shared filesystem.
3. Add small reproducibility-critical reviewed CSVs and metadata to Git where their source terms allow it. Update `.gitignore` deliberately; adding a JSON manifest while its referenced CSV remains ignored is not sufficient.
4. Keep large checkpoints, raw payloads and bulky audit dumps out of ordinary Git unless there is a separately chosen storage policy. Keep their locations/hashes recorded.
5. Verify `git ls-files` and `git check-ignore` for the exact paths. Do not assume `git add` collected an ignored dataset.
6. Check the state on the execution node, not just the submission machine. The same absolute pathname on sv1224 and sv1225 does not prove the files are shared.
7. Confirm the node uses the intended code revision, virtual environment and source import path. Current final-training launchers still have their existing main-project path convention; the newly applied comparison launcher can select a project directory. Do not assume all launchers inherited that change.
8. Verify numerical/ML dependencies and checkpoint availability before occupying a GPU. A pretrained baseline may need an existing local cache or a permitted initial download.
9. Test opening the reviewed dataset and validating its hashes on the actual node.
10. Save the command and input identity with the job log.

**Failure already encountered:** EPIT training failed on sv1224 because `epit_reference_v1/epit_processed.csv` was absent. The reviewed manifests/code existed, but the required CSV had not traveled to the node.

**MPEA-specific check still needed:** the frozen split CSV payloads were not tracked at this audit even though the split manifest/lock/registry were. Resolve this before MPEA training; do not regenerate the split independently on each node to compensate.

**Ready to continue:** a clean, intended checkout on the execution node can load all required data and validate its identities without relying on accidental local leftovers.

## 24. Run cheap, meaningful verification before Optuna

This is the last stage before committing substantial training time. Run these against the new dataset implementation, not only the older EPIT/Soccol examples.

### Real-data checks

- Source-to-processed conversions and a sample of every correction/extraction rule agree with evidence.
- Accepted composition zeros, balances and units are correct; unresolved cases are not silently completed.
- Feature names/order/types match the registry, and targets/metadata are excluded from predictors.
- Unknown categories, environment NaNs and sparse columns survive serialization and model preprocessing correctly.
- Loaders return the exact expected development/final memberships and aligned targets.
- Shared-schema mappings preserve values and distinguish absent environment information from absent elements in complete alloys.

### Fitting and sampling checks

- Formula folds fit their context only, and saved states reproduce held-out predictions.
- Runtime formula outputs agree with calibration formulas on identical inputs.
- Profile rebuilds are deterministic, with correct development-only IDs and sampling probabilities.
- Altering final-query features/targets does not change a sampling bank or fitted formula coefficients. Perform such checks on in-memory/test copies, not frozen files.
- Altering a formula validation fold does not change that fold's fitted context state/coefficients; distinguish this from the deliberately full-development production coefficient center.
- Category codes match in real and synthetic paths.
- Joint environment tuples and their NaNs remain together after both SCM routes.
- Formula weight 0/1 and informed ratios 0.25/0.50/0.75/1.00 generate valid tasks.

### End-to-end checks

- Generate a small synthetic batch: correct width/order, finite internal calculations/targets, intended NaN mask and valid compositions.
- Run a tiny CPU training/inference check to catch shape errors, missing-embedding configuration errors and nonfinite gradients/losses.
- Reload that tiny checkpoint and verify the schema/missingness contract.
- If exact resume has been integrated, compare resumed and uninterrupted tiny runs, including subsequent sample/RNG draws.
- Exercise the new fold-evaluation wrapper using development rows; verify exact context/query IDs and all required model outputs.
- Check the final evaluator's membership logic with mock scores or a tiny test. Do not use final model performance as a preflight tuning signal.
- Validate all generated commands and Slurm shell syntax.

Do not rerun a full search merely to test a path or schema. Do not stop at “predictions are finite”: this session's sparse-column bug produced finite predictions while erasing known input values.

**Save:** concise preflight results and exact tested versions. Advance only after failures that affect data/selection/model meaning are resolved.

## 25. Set and freeze the Optuna search configuration

The current specialist design is:

| Parameter | EPIT | Soccol | New-dataset decision |
|---|---|---|---|
| Informed-task ratio during search | Fixed 1.00 | Fixed 1.00 | Use the same agreed informed-only search design unless explicitly changed. |
| MLP SCM share | Continuous 0–1 | Continuous 0–1 | Same starting convention; tree share is the remainder. |
| Formula contribution | Continuous 0–1 | Continuous 0–1 | Same starting convention with the new dataset's validated formulas. |
| Composition perturbation | 0–0.30 | 0–0.15 | Review the new support; do not claim one interval is universally correct. |
| Coefficient variation | Fixed 0.8, seed 42 | Fixed 0.8, seed 42 | Current a priori shared choice, not fitted to final queries. |
| Training per trial | 1,000 steps | 1,000 steps | Keep aligned for the first comparable search. |
| Scheduler horizon | 10,000 steps | 10,000 steps | Preserve this; a 1,000-step proxy is not a separately compressed 1,000-step schedule. |
| Trial budget | 50 requested per worker | 50 requested per worker | State whether the intended budget is per worker or global. |
| Startup trials | 10 | 10 | Keep explicit. |
| Development evaluation | All five frozen folds; eight estimators | Same | Use the new frozen development folds only. |
| Objective | Mean fold Spearman | Same | Keep aggregation explicit. |

1. Review old completed trials if they exist and are cheaply accessible. Exclude incomplete/stale records from that analysis.
2. Use their development results to justify broad ranges, not to claim isolated causal effects from a joint adaptive search.
3. EPIT widened composition noise because its prior optimum was near the old upper boundary. Soccol's stronger old trials favored much smaller noise, so its approved range remained 0–0.15. MPEA has no equivalent specialist search history yet.
4. Freeze all other architecture/training settings using the established baseline configuration. Document any intentional change.
5. Remove the fixed informed ratio from searched parameters but store its value in the fingerprint. Final training must reconstruct fixed parameters, not look only at Optuna's `trial.params`.
6. Store precise numerical coefficients/scores rather than rounded display values in emitted commands/artifacts.
7. Use a new study name for a different data/schema/profile/formula/search contract. Reject a mismatched existing fingerprint.
8. Keep a record of source version and software revision for every worker.

**Coefficient variation means:** multiply each nonzero fitted coefficient by an independent factor in 0.2–1.8, renormalize, enforce existing bounds and redraw if necessary. Zero coefficients stay zero. The existing sampler falls back to the center after its retry limit. “0.8” is not a statistical variance or a guaranteed final ±80% coefficient change after normalization.

## 26. Run Optuna and verify its artifacts, not just the dashboard score

1. Place the journal on the established shared filesystem if workers on different nodes must join the same study.
2. Distinguish that shared journal from node-local checkpoints, evaluation CSVs and logs. Sharing one does not share the others.
3. Start with the validated command/configuration from steps 21–25.
4. Check an early trial's generated command, dataset/profile/formula identities, saved checkpoint and five development-fold populations.
5. Confirm final-query rows never appear in the trial's evaluation queries or context.
6. Confirm the objective is the intended unweighted mean over the five folds rather than a different pooled statistic.
7. Monitor failed trials for nonfinite values, missing files, shape/category errors and changed configuration. Do not interpret such failures as scientific evidence that the prior is poor.
8. Record job/host/trial IDs and the locations of its local artifacts.
9. Understand the budget: two workers each requesting 50 trials can request roughly 100 additional trials, not a guaranteed global total of 50.
10. Treat `RUNNING`/`WAITING` entries as state to inspect. They are not by themselves evidence that a final-training launch failed. Do not mark a genuinely active worker's trial failed to tidy the dashboard.
11. Preserve completed trial parameters, fixed settings, objective/fold scores and artifact identities in durable records.

The current final-selection code can use a validated shared-journal record when another node's local proxy artifacts are absent. It still checks files that exist locally and rejects corrupt hashes or incomplete journal evidence. Missing artifacts are not permission to disable validation.

**Ready to continue:** there are valid completed trials with reproducible configurations and development evaluation evidence.

## 27. Select and pin the configuration for final training

1. Prefer completing the planned search before declaring its final best configuration. If selecting while workers remain active, explicitly record that it is the best completed trial at that time.
2. Select by the agreed development objective, not by any final-query plot.
3. Validate the trial against the current fingerprint, reviewed inputs, profiles, formula artifacts and saved evaluation evidence.
4. Inspect actual input counts, missingness mode, model settings and all searched/fixed parameters.
5. For the requested 25/50/75/100 comparison, **pin the same selected trial for all four jobs**. Otherwise automatic selection can change between submissions if Optuna continues.
6. Do not blindly reuse old trial numbers such as Soccol 19 or EPIT 2 for a new study. A trial number has meaning only within its study.
7. Freeze a copy of the chosen configuration, including formula-noise and composition-noise settings, before launching long training.
8. Decide whether optional descriptors are part of that chosen configuration. Do not add them independently to only the final run.

**Current selector constraint:** for a fresh final run, the pinned trial must still be the best completed trial when the job starts. If Optuna continues and a new winner appears, a later pinned job will refuse the old trial. Finish the intended study, or establish an explicitly supported frozen study/selection snapshot, before relying on four submissions to use one best configuration. Do not bypass validation or silently let each job choose a different winner.

**Ready to continue:** the four runs will differ in the intended informed ratio, not in an accidentally changed best trial, feature set or coefficient file.

## 28. Launch final training as four separate, named runs

The requested procedure is manual individual submissions, not a hidden loop or array.

1. Reconstruct the best completed trial, including its fixed informed ratio and all validated artifacts.
2. Train with the established full **10,000-step** budget and matching scheduler horizon.
3. Submit four runs with explicit informed-ratio overrides: **0.25, 0.50, 0.75, 1.00**. Hold the selected formula/SCM/composition settings constant unless another explicit experiment is intended.
4. Give every run a distinct checkpoint directory and analysis/output directory. Record the ratio in its metadata, not just an abbreviated folder suffix.
5. Use an explicit `1.0` override for the 100% run too. In the existing launcher, the implicit default can target the study root; other ratio subdirectories can make that root nonempty and trigger `FileExistsError` before training.
6. Do not delete existing output folders to work around that error. Choose the correct dedicated run directory, or use a supported continuation mode only for that exact run.
7. Inspect the saved/generated command and early log: correct trial, ratio, profile, input width, missingness mode, coefficients, seeds and scheduler.
8. Verify periodic checkpoint saving. Current defaults save temporary checkpoints every 100 steps and permanent checkpoints every 500; temporary retention does not imply every intermediate step will remain available forever.
9. Keep the four jobs' metadata/logs separate even if they run on different nodes.
10. Check that the saved checkpoint's configuration matches the intended model. Before depending on exact continuation or full profile identity, verify the paired-stream metadata and the actual recovery entry point described in step 22.

“100% informed” here means all synthetic training tasks use the informed path. It does not mean a 100% analytical formula target: the selected formula/SCM mix still applies. “75% informed” means the remaining task share is generic under the existing mixture design; it is not an ensemble of separately trained final model weights.

**Manual command pattern for existing EPIT/Soccol implementations:** see the command cards below. There is intentionally no asserted ready-to-run MPEA final-training command until its implementation exists and passes the earlier steps.

## 29. Select the final checkpoint using development data and freeze it

1. After training, evaluate the intended saved checkpoints using the five development folds, with the same reviewed inputs and settings used for model selection.
2. Retain the accepted development population and fold IDs; exclude every outer final query.
3. Use the declared checkpoint schedule and selection metric. Current defaults start at step 500 and inspect 500-step checkpoints; verify any included endpoint behavior in the actual command.
4. Select the checkpoint from development scores, record the decision and save its path/hash.
5. Write the final-model manifest and lock: dataset/profile/formula identities, split identity, feature count/order, inference settings, selected trial, overrides and selected checkpoint.
6. Verify that the manifest points to the correct ratio run. A top-level default manifest is not automatically the 25%, 50% or 75% result.
7. Preserve the selection metrics and logs so the chosen file is reproducible after node-local cleanup.
8. If checkpoint selection was interrupted after training completed, use its supported evaluation-resume mode. Do not confuse that with retraining or exact training-state recovery.

**Ready to continue:** you have a frozen model chosen without final-query performance and a manifest that identifies precisely what it is.

## 30. Extend the combined model after the specialists are ready

This is a separate training experiment. Merging real CSVs, mixing synthetic priors and averaging model predictions are different operations.

### 30a. Build the reviewed shared real-data views

1. Use the common registry and name/unit mappings from step 11, now including the revised MPEA fields.
2. Map each dataset's accepted development rows into the same ordered schema.
3. Map each dataset's frozen final-query rows into that same schema separately.
4. Use one shared categorical meaning/mapping; do not concatenate independently encoded integers.
5. Check source/measurement overlaps across datasets. A held-out measurement from one dataset must not reappear with its target in another dataset's labeled context.
6. Sharing a paper or a composition is not automatically the same as sharing a measurement. Inspect enough source/specimen/condition evidence to distinguish them; keep unresolved overlap qualifications explicit.
7. If an exact query/context duplicate is established, preserve frozen queries and consistently remove the conflicting context observation for all compared models. Record IDs and changed context counts.
8. Do not assume equal leakage makes every comparison unbiased: models can exploit duplicated context differently. Also do not label the entire dataset invalid merely because source overlap is possible.
9. Concatenate the three contexts and the three query sets, with no row balancing for the current agreed experiment.
10. Retain dataset IDs for bookkeeping and per-dataset metrics; do not silently introduce dataset ID as a predictor.

If all current accepted/frozen counts remain and no conflicting context rows are removed, the merged context would have **608 + 2,950 + 214 = 3,772 rows**, and the merged final queries **152 + 805 + 53 = 1,010 rows**. These are expected counts under stated conditions, not a substitute for checking actual IDs. Soccol will dominate the pooled query population; report per-dataset metrics alongside the pooled score without changing the agreed row balance.

### 30b. Build the combined synthetic prior

1. Map each dataset's generator into the same shared feature schema, including its real missingness pattern and meaning of zeros.
2. Use each dataset's own accepted profile, fitted rules and validated generation assumptions.
3. Add missing feature positions deliberately; never reuse the historical 50-column positional mapping as though it were the new reviewed mapping.
4. Define top-level generic/EPIT/Soccol/MPEA synthetic-task probabilities explicitly and record them.
5. If retaining the earlier 50%-generic design with three equally weighted informed sources, the task shares would be **50% generic and 16⅔% each informed source**. That is a choice to confirm/version, not an already implemented three-source default.
6. Avoid counting generic tasks twice. If an informed component itself retains a generic share, the effective generic proportion becomes the top-level generic share plus the generic fractions hidden inside the components. Current informed-only searches make a clean component definition easier, but verify the actual generator configuration.
7. State whether composition/environment independence remains within each source and whether any source-missingness augmentation is applied. Match training and evaluation deliberately.
8. Re-run schema/value/formula/missingness audits on mapped samples from **each** component and on the generic component.
9. Save the combined prior configuration with all component artifact identities and schema/category mappings.
10. Train and select a new combined checkpoint. New merged features do not retroactively change an old COMB checkpoint's learned input meaning.

The current combined implementation and its resumable loader provide useful infrastructure, but its historical feature mapping/prior definitions are not a ready-made reviewed three-source model. Review [combined_corrosion.py](../src/tabicl/prior/combined_corrosion.py) and the [combined scripts](../scripts/combined_corrosion/) before adapting them.

## 31. Evaluate frozen models with the agreed baselines and query boundary

1. For a specialist's final evaluation, use its accepted development context and its frozen outer final queries. For the agreed combined experiment, use the merged development context and merged frozen queries from step 30.
2. Keep exactly the same query IDs across compared models. Context-size diagnostics may reduce development context; they must not substitute a development fold for the outer final queries.
3. Compare the intended informed runs with local generic TabICL, official pretrained TabICL and CatBoost. Distinguish these baselines from each other.
4. Preserve fixed schema/order and missing-aware mode for reviewed informed checkpoints. Use native normalization/shuffling/constant-feature handling for generic/pretrained baselines as in the current combined comparison protocol.
5. Current shared comparison settings include eight estimators, seed 42, median point prediction, pretrained `tabicl-regressor-v2-20260212.ckpt`, and CatBoost with 1,000 iterations, depth 6, learning rate 0.03 and L2 regularization 3. Record actual versions/settings in each result.
6. Save row-level or task-level results, summary tables, model errors, context/query membership and the requested metrics/plots. Do not silently drop failed models.
7. Report Spearman with MAE and the other recorded metrics; ranks and errors answer different questions. Report target units/reference limitations accurately.
8. Keep frozen-selected-model reporting distinct from all-checkpoint diagnostic curves. Do not select a better-looking final-query checkpoint after viewing the curve and still call that selection independent.
9. Do not interpret `npinball_iqr=NaN` or `cov80=NaN` as failure when uncertainty evaluation is explicitly disabled; these are not the missing-value warnings fixed earlier.
10. Record that MPEA's historical exploratory exposure remains part of the experiment history.

### Checkpoint transfer and plotting checks

- Copy the correct ratio directories from the actual training host, including needed metadata; verify files exist, sizes are plausible and transfers are complete.
- Prefer stable completed/permanent files when training is still writing new checkpoints. A directory name in an SCP command does not prove a successful job created it.
- The current evaluator discovers available checkpoints when constructing its evaluation schedule. A file arriving afterward may require another evaluation invocation.
- It evaluates available steps per model; a shorter informed run does not truncate a longer generic run.
- Pretrained and CatBoost results may be reused across steps because their fitted comparison is unchanged.
- Confirm that the desired checkpoint has a result row before diagnosing a plotting problem.
- Open the plot from the actual output directory used by that invocation. This session's old folder had only the 100% step-500 point; the newer `_sparse_fixed` folder also had its step-1,000 point.
- Use a fresh output directory after changing preprocessing or comparison semantics. Do not mix corrected and old scores in one apparently homogeneous run.

## 32. Know which work must be repeated after a change

| Change | Minimum downstream action |
|---|---|
| Documentation-only clarification, no values/semantics changed | Update the record; no model rerun solely for prose. |
| Corrected real composition/environment value | Version export/audit; revalidate loader and accepted IDs; rebuild affected profiles/statistics; refit affected formulas; invalidate incompatible studies; determine whether models trained with old inputs need a new version. |
| New accepted-development selection | Keep frozen membership/fold IDs; regenerate selector identity, affected banks, fitting artifacts and evaluations; do not compare changed-population scores as a pure model change. |
| New/renamed/reordered feature or category meaning | Update registry, loaders, mappings, profiles, generator, model configuration, evaluators and tests; fixed-schema models generally require a new training version. |
| Different atomic-weight/unit/reference conversion policy | Recompute affected values and identities; review grouping compatibility and formula units; propagate through fitting/training/evaluation as needed. |
| New formula/term/bounds/coefficient center/rule weighting | New calibration artifacts and prior identity; new study/training experiment unless strictly reproducing a saved old configuration. |
| Different composition weighting, perturbation or missingness augmentation | New profile/prior configuration and study fingerprint; assess with new training, not by relabeling old checkpoints. |
| Different Optuna range/fixed parameter | New compatible study contract; do not append incompatible trials to a journal. |
| Sparse-column inference bug fixed | Re-evaluate existing checkpoints with the corrected code in a new output directory. Weights are not retroactively altered; earlier development selection scores may also need recomputing if affected. |
| Different checkpoint selected on development | Freeze a new final manifest, preserving old selection evidence. |
| New dataset added to the shared schema/prior | Rebuild shared mappings and combined training configuration; do not assume standalone changes migrated COMB. |
| New outer split | New experiment/split version with an explicit reason; never silently overwrite the existing frozen boundary. |

Do not recompute every expensive stage reflexively. Trace which values/semantics changed and which saved artifacts depend on them. Conversely, do not reuse a stale artifact merely because its filename still exists. Keep the dependency record explicit enough to make that decision.

## Command cards: existing commands, not invented MPEA entry points

Run commands from the main project unless an explicitly selected isolated experiment requires another checkout:

```bash
cd /home/fcolanto/projects/tabicl
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

These commands are references for the corresponding stages. This report did not launch a search, final training or trained-model evaluation.

### Validate existing frozen inputs/profiles without rebuilding them

```bash
.venv/bin/python -m corrosion_datasets.datasets.mpea_corrosion.prepare_composition_split --validate-only

.venv/bin/python -m scripts.build_epit_development_profiles --check

.venv/bin/python -m scripts.build_epit_expanded_feature_profile --check

.venv/bin/python -m corrosion_datasets.datasets.soccol_pitting_potential.prepare_pipeline_assets --check
```

Check required CSV transport explicitly:

```bash
git ls-files -- corrosion_datasets/datasets/mpea_corrosion/processed/splits_v1/

git check-ignore -v corrosion_datasets/datasets/mpea_corrosion/processed/splits_v1/development.csv
```

If a required file is ignored, fix the intended tracking/distribution policy. A nonzero exit status from `git check-ignore` can simply mean the path is not ignored; inspect its output rather than treating every status as a dataset failure.

### Existing MPEA formula screen

This reproduces the existing reduced-candidate screen, not a completed new-schema MPEA production prior. Use a fresh output directory:

```bash
.venv/bin/python -m corrosion_datasets.datasets.mpea_corrosion.compare_target_rules \
  --output-dir /tmp/mpea_formula_check_new
```

Revise its reviewed-input adapter and candidate decisions before treating such a rerun as the new schema's result. Do not overwrite the historical report to make two versions look identical.

### Existing EPIT/Soccol search launches

Only after the preflight and study-identity checks:

```bash
sbatch scripts/epit_pipeline/run_optuna.sbatch
```

```bash
sbatch scripts/soccol_pipeline/run_optuna.sbatch
```

These select their documented current studies/defaults. A changed dataset/profile/search requires the appropriate new study configuration. There is **no ready-to-run `scripts/mpea_pipeline/run_optuna.sbatch` at this audit**.

### Four manual final-training submissions

First determine the selected trial from the correct completed study in step 27 and set `SELECTED_TRIAL_NUMBER` to that integer. The guard below intentionally stops if it has not been set. Do not use a historical trial number from this chat merely because it is familiar.

For EPIT, submit individually:

```bash
: "${SELECTED_TRIAL_NUMBER:?Set the selected EPIT study trial number first}"
sbatch --export=ALL,SELECTED_TRIAL_NUMBER="$SELECTED_TRIAL_NUMBER",OVERRIDE_INFORMED_PRIOR_RATIO=0.25 scripts/epit_pipeline/train_final.sbatch
```

```bash
: "${SELECTED_TRIAL_NUMBER:?Set the selected EPIT study trial number first}"
sbatch --export=ALL,SELECTED_TRIAL_NUMBER="$SELECTED_TRIAL_NUMBER",OVERRIDE_INFORMED_PRIOR_RATIO=0.50 scripts/epit_pipeline/train_final.sbatch
```

```bash
: "${SELECTED_TRIAL_NUMBER:?Set the selected EPIT study trial number first}"
sbatch --export=ALL,SELECTED_TRIAL_NUMBER="$SELECTED_TRIAL_NUMBER",OVERRIDE_INFORMED_PRIOR_RATIO=0.75 scripts/epit_pipeline/train_final.sbatch
```

```bash
: "${SELECTED_TRIAL_NUMBER:?Set the selected EPIT study trial number first}"
sbatch --export=ALL,SELECTED_TRIAL_NUMBER="$SELECTED_TRIAL_NUMBER",OVERRIDE_INFORMED_PRIOR_RATIO=1.00 scripts/epit_pipeline/train_final.sbatch
```

For Soccol, repeat this pattern with the **Soccol study's** selected trial and `scripts/soccol_pipeline/train_final.sbatch`. Clear or review inherited `STUDY_NAME`, `STORAGE`, `FINAL_RUN_NAME`, `OUTPUT_DIR` and unrelated override variables before submission; an old exported value can override the expected default or make four jobs share a directory.

The generated ratio suffix depends on the complete override configuration. Do not predict future directory names from the old EPIT trial-2 hashes. Record the directory printed/written for each actual run. The EPIT comparison now includes the optional 0% control and trusts the supplied 0/25/50/75/100 path order; old checkpoints do not necessarily contain enough provenance to detect a mislabeled ratio automatically.

### Preview an existing EPIT comparison without running models

```bash
.venv/bin/python -m scripts.epit_pipeline.evaluate_checkpoint_comparison \
  --candidate-run /path/to/the/actual/reviewed/epit/run \
  --device cpu \
  --dry-run
```

The `/path/to/...` value here is an explicitly marked placeholder. A final evaluation requires the real checkpoint path. For all five EPIT ratios, use `--ratio-runs RUN0 RUN25 RUN50 RUN75 RUN100` and a fresh `--output-dir`. Add `--max-checkpoint-step 5000` to cap evaluated checkpoints at step 5,000, including the generic baseline and latest endpoints. The implemented wrapper adds generic, pretrained and CatBoost baselines and keeps frozen final queries.

Do not assume every script supports `--dry-run`, `--resume-training` or the same options. Check that script's `--help`; in particular, the deferred exact-training-resume option must not be presented as available in the current main final-training CLI.

## The MPEA work that remains, in the order to do it

The following is an implementation checklist, not a claim that these items have already been completed:

1. Reuse the 214/53 frozen IDs and existing five folds; validate them before editing a feature view.
2. Review the complete original MPEA compositions and the seven elements omitted by the old supported-feature projection. Decide the revised composition registry without inventing omitted-element mass.
3. Apply the agreed complete-composition zero/balance policy using source evidence. Preserve the three originally excluded incomplete-composition rows in the audit; do not silently add them to the frozen specialist population.
4. Parse the original solution descriptions into the expanded chemical/unit fields. Keep conflicting or unsupported concentrations unknown; preserve the original concentration values for audit.
5. Review available temperature, scan rate, heat treatment, phases, preparation and reference information against the corrected EPIT/Soccol definitions. Missing pH is not a reason to invent pH.
6. Produce a versioned reviewed MPEA export, feature registry, audits and before/after comparison, keeping the frozen IDs and memberships.
7. Decide native-specialist versus common-schema training and create the correct mapping/registry. Do not reuse the historical 50-column table as the final answer by default.
8. Implement its authoritative development selector and reviewed loaders. Resolve remote availability of the frozen CSV payloads.
9. Re-evaluate formula applicability on the revised inputs. Refit reduced/retained/new candidate definitions using the frozen development folds; inspect family and solution-specific results.
10. Explicitly choose the production rule set and save coefficient centers, bounds, scores and runtime-equivalence checks. The existing weak formula screen does not make this decision automatically.
11. Build development-only composition/environment/category profiles with documented material scope, uniform distinct-composition weighting and coherent environment rows.
12. Integrate the generator, formula projection, missing-value embedding, schema width and configuration identity into the shared training machinery.
13. Extend the paired-stream profile identity to MPEA, verify its low-level resume behavior, and distinguish that from the still-pending high-level resume launcher.
14. Implement aligned MPEA Optuna/final-training/evaluation wrappers, including fixed 100% informed search and explicit later ratio overrides. Choose its composition-noise range based on development support, not its final scores.
15. Run Optuna, pin the selected completed trial, launch the four manual final runs and select their checkpoints using development folds only.
16. Freeze the model manifests and perform the stated specialist comparisons on the 53 frozen queries. Retain the historical-exposure qualification.
17. Only then build/test the reviewed three-source combined-prior configuration and merged contexts/queries, and train a new combined model.

## Required completion record for each new dataset

Copy this checklist into the dataset's versioned decision record. Mark an item complete only when its named evidence exists.

- [ ] Source version, raw hashes, license/distribution decision and permanent IDs recorded.
- [ ] Every original column assigned a meaning, unit/type and retention/exclusion decision.
- [ ] Target, censoring/event and electrode-reference policies recorded.
- [ ] Full composition conversion/completeness reviewed before any model projection.
- [ ] Every zero/balance/limit approximation justified; known omissions and exclusions retained in the audit.
- [ ] Environment/preparation/history extraction checked against the correct specimen and stage.
- [ ] Contradictory cells resolved or retained as NaN with evidence; unresolved composition rows handled explicitly.
- [ ] Frozen outer/fold identities validated and unchanged where already established.
- [ ] Reviewed feature/target tables and registry versioned, reproducible and read-back checked.
- [ ] Shared name/unit/type mappings and category semantics verified; old combined schema not mistaken for the new union.
- [ ] One accepted-development selector reused everywhere; additional family scope stated separately.
- [ ] All active loaders point to the reviewed inputs without repeating old composition preprocessing.
- [ ] Material-family, distinct-composition and source coverage documented.
- [ ] Formula definitions/inputs/units/applicability verified; fold filling/scaling/coefficients fitted on context only.
- [ ] Fold and per-family scores reviewed; production rule registry and all-development centers explicitly selected.
- [ ] Profile rows/statistics/categories come only from allowed development data.
- [ ] Distinct-composition weighting, coherent environment sampling, perturbation and coefficient RNG behavior audited.
- [ ] Real/synthetic category codes and missingness match; sparse known values survive preprocessing.
- [ ] Training/evaluation command builders, feature counts, descriptors, missing-value mode and Python/Slurm defaults agree.
- [ ] Artifact identities, checkpoint provenance and actual resume capabilities established.
- [ ] Required files and dependencies verified on the execution node, not only locally.
- [ ] Small end-to-end checks passed; no full search was used as a substitute for debugging.
- [ ] Optuna ranges, fixed settings, budget, scheduler horizon and study fingerprint frozen.
- [ ] Completed trial selected and pinned consistently for every final ratio run.
- [ ] Four independent output/checkpoint directories confirmed, including explicit 1.00 override.
- [ ] Final checkpoint selection used development data; manifests/locks saved.
- [ ] Final evaluation used the frozen outer queries, correct baselines and a new output directory for changed semantics.
- [ ] Combined-prior migration, if required, separately implemented and checked rather than inferred from merged CSVs.

## Experiment record template

Fill this with actual paths/hashes rather than relying on version names alone:

```text
dataset_id:
purpose: specialist / combined / external evaluation
raw_source_version_and_hashes:
reviewed_export_directory_and_hashes:
feature_registry_version_order_types_units:
target_column_unit_reference_policy:
composition_basis_and_acceptance_rules:
zero_balance_limit_and_conflict_decisions:
accepted_with_qualifications:
withheld_row_ids_and_reasons:
frozen_split_manifest_and_lock:
accepted_development_selection_identity:
final_query_ids_and_prior_exposure:
material_scope_and_formula_applicability:
composition_profile_and_sampling_weights:
environment_profile_and_joint_sampling_policy:
category_mappings_and_internal_fill_state:
model_missingness_and_sparse_column_policy:
formula_definitions_centers_bounds_scores_probabilities:
coefficient_variation_and_rng_seed:
composition_perturbation_policy_and_range:
optional_descriptors_and_element_scope:
code_revision_and_uncommitted_patch_if_any:
software_environment:
execution_host_and_required_input_locations:
study_name_storage_fingerprint_and_budget:
selected_completed_trial_and_evidence:
final_ratio_override:
checkpoint_directory_and_analysis_directory:
training_command_seeds_workers_and_scheduler:
resume_capability_verified:
selected_checkpoint_path_hash_and_development_scores:
final_model_manifest_and_lock:
final_context_and_query_membership:
comparison_baselines_and_inference_settings:
results_directory:
unresolved_limitations_and_followup_decisions:
```

## Implementation map and historical pitfalls

| Need | Current reference implementation or evidence |
|---|---|
| Dataset-specific processing decisions | [Processing history](DATASET_PROCESSING.md) |
| EPIT composition processing | [prepare_epit_composition.py](datasets/electrochemical_metrics_alloys/prepare_epit_composition.py) |
| EPIT environment/preparation extraction | [prepare_epit_environment.py](datasets/electrochemical_metrics_alloys/prepare_epit_environment.py), [prepare_epit_preparation.py](datasets/electrochemical_metrics_alloys/prepare_epit_preparation.py) |
| Soccol environment/composition processing | [prepare_soccol_environment.py](datasets/soccol_pitting_potential/prepare_soccol_environment.py), [prepare_soccol_composition.py](datasets/soccol_pitting_potential/prepare_soccol_composition.py) |
| Reviewed reference labels | [prepare_reference_types.py](datasets/prepare_reference_types.py), [reference rules](datasets/reference_type_rules_v1.json) |
| Reviewed EPIT–Soccol mapping | [Column-alignment report](analysis/epit_soccol_column_alignment_v1/column_mapping.md) |
| Reviewed schema constants | [EPIT schema](../src/tabicl/prior/epit_expanded_schema.py), [Soccol schema](../src/tabicl/prior/soccol_reviewed_schema.py) |
| Frozen grouped splitting | [EPIT split machinery](../scripts/epit_pipeline/split_data.py), [MPEA split implementation](datasets/mpea_corrosion/prepare_composition_split.py) |
| MPEA original conversion/source limitations | [Preparation rules](datasets/mpea_corrosion/preprocessing_plan.md), [current dataset README](datasets/mpea_corrosion/README.md) |
| Accepted-row reviewed loading | [Soccol data/selection](../scripts/soccol_pipeline/data.py), [EPIT composition adapter](../scripts/epit_pipeline/composition_data.py) |
| Formula calibration | [EPIT](../scripts/epit_pipeline/calibrate_target_rules.py), [Soccol](../scripts/soccol_pipeline/calibrate_target_rules.py) |
| MPEA development screen | [compare_target_rules.py](datasets/mpea_corrosion/compare_target_rules.py), [report](analysis/mpea_target_rules_v1/README.md) |
| Profile construction | [EPIT development](../scripts/build_epit_development_profiles.py), [EPIT expanded](../scripts/build_epit_expanded_feature_profile.py), [Soccol](datasets/soccol_pitting_potential/prepare_pipeline_assets.py) |
| Feature sampling | [EPIT generator](../src/tabicl/prior/epit_feature_generator.py), [Soccol generator](../src/tabicl/prior/soccol_feature_generator.py) |
| SCM/formula target integration | [PriorDataset](../src/tabicl/prior/dataset.py) |
| Sampling and family audits | [Soccol audit scripts](../scripts/soccol_pipeline/), [historical sampling audit](analysis/soccol_pipeline/synthetic_sampling_audit_v1/README.md) |
| Search/defaults/fingerprints | [EPIT run_optuna.py](../scripts/epit_pipeline/run_optuna.py), [Soccol run_optuna.py](../scripts/soccol_pipeline/run_optuna.py) |
| Final training and trial verification | [EPIT train_final.py](../scripts/epit_pipeline/train_final.py), [Soccol train_final.py](../scripts/soccol_pipeline/train_final.py) |
| Core real-data evaluation | [eval_corrosion_datasets.py](../scripts/eval_corrosion_datasets.py) |
| Shared comparison command structure | [Combined evaluator](../scripts/combined_corrosion/evaluate_final.py), [EPIT four-ratio evaluator](../scripts/epit_pipeline/evaluate_checkpoint_comparison.py) |
| Sparse-context fix | [preprocessing.py](../src/tabicl/sklearn/preprocessing.py), [regression tests](../tests/test_sparse_preprocessing.py) |
| Current specialist usage documentation | [EPIT README](../scripts/epit_pipeline/README.md), [Soccol README](../scripts/soccol_pipeline/README.md) |

The expensive mistakes this workflow is designed to prevent are: fixing an export while training still reads an older one; rebuilding profiles with final-query features; accepting fake composition closure; discarding real omitted-element mass; mixing incompatible units or category codes; erasing known sparse values; fitting formula folds on their own validation rows; claiming fully nested Optuna scores when the prior was fitted on all development; forgetting remote CSV dependencies; joining an incompatible study; changing selected trials between ratio runs; letting the 100% run collide with a nonempty study root; assuming saved seeds imply exact resume; and treating old combined mappings or plots as automatically updated by standalone changes.



