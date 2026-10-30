# Dataset processing guidelines

For the complete chronological procedure from source inspection through final
training and combined-model integration, use
[Adding a new dataset](NEW_DATASET_WORKFLOW.md). It includes the current
implementation status, remaining MPEA work, commands and completion checks.

The general rules below define how datasets will be prepared for the shared
schema. The dataset-specific sections record completed processing in past
tense. Each section identifies its processed export; it does not imply that
older exports or model pipelines already use that version.

The latest standalone exports after the composition and source-consistency reviews were
[EPIT reference v1](datasets/electrochemical_metrics_alloys/processed/epit_reference_v1/epit_processed.csv)
and [Soccol reference v1](datasets/soccol_pitting_potential/processed/soccol_reference_v1/soccol_processed.csv).
Both added the [reviewed reference-type feature](#epit-and-soccol-reference-type-feature)
to their latest composition exports without changing targets.
The Soccol export inherited environment v2 and retained unresolved compositions
for review. Its accepted development subset contained 2,950 rows; all 805 frozen
queries were preserved, including 10 that still needed composition evidence.
The [environment review](#epit-and-soccol-source-consistency-review) and
[Soccol composition review](#soccol-composition-processing) recorded the additional
corrections. Earlier sections described the preserved earlier versions.
The subsequent [EPIT composition source investigation](#epit-composition-source-investigation)
supported one Ti correction and left Re and Nb/Ta amounts unverified in 22 rows.
The [EPIT review decision](#epit-composition-review-decision) then applied the
Ti correction and retained all 22 uncertain rows as recorded, with explicit
user acceptance. No EPIT rows or frozen split assignments were removed.

## Composition features

- **One column per element:** use the same element names across every dataset,
  such as `Fe`, `Cr`, and `Ni`.

- **One unit: weight percent (wt.%).** Check the source units first. Convert
  atomic percentages or atomic fractions using atomic weights. Convert weight
  fractions to percentages. Do not assume an incomplete list of elements is a
  complete composition when converting units.

- **Keep reported amounts**, including explicitly reported zeros.

- **Treat blanks and NaNs as missing initially.** Decide how to fill them using
  the alloy description and dataset context.

- **Fill a missing base element when justified:** for example, for a clearly
  Fe-based alloy whose other reported amounts reasonably describe its alloying
  additions, calculate `Fe = 100 - sum of the other element amounts`. Apply the
  same principle to another base element when appropriate. The source does not
  have to explicitly say "balance Fe" if the alloy description and dataset
  context support that interpretation. A low total alone is not enough reason
  to assume Fe.

- **Set other unreported elements to `0` when we have enough confidence that
  the reported amounts plus any justified balance describe the alloy.** This
  includes missing entries in original element columns and additional element
  columns introduced by the shared schema. These are assumed zeros; preserve
  the distinction from explicitly reported zeros in the processing record.

- **Exclude rows with an unexplained composition total below 90 wt.%.** Use
  **90 wt.% as the current default minimum**. Apply this check after verifying
  units and applying any justified balance calculation. If the total remains
  below 90 wt.% and the dataset provides no explanation or basis for completing
  it, exclude the row from the model-ready data. Keep the original row and
  record the exclusion reason. An explanation requires review; it is not
  automatic acceptance of an incomplete composition.

- **Flag other unclear compositions for review:** if an important amount is
  unknown and the context does not support filling it, do not invent a value.
  A total of 90 wt.% or more does not by itself establish completeness.

- **Check the completed composition:** amounts must be between 0 and 100, and
  the total should be approximately 100%. Allow small differences consistent
  with the source's rounding precision; do not automatically rescale incomplete
  compositions. A total of 100% produced by our balance calculation is not
  independent evidence that the original composition was complete.

- **Record our changes:** preserve the original data and document unit
  conversions, assumed zeros, calculated balance amounts and their justification,
  unresolved rows, and exclusions.

## Examples

- Reported alloying additions total 25 wt.%, and the alloy context supports an
  Fe balance: set `Fe = 75`, then set other unreported elements to `0` if the
  reconstructed composition is accepted as complete.
- Reported elements total 85 wt.%, with no explanation or justified balance
  element: exclude the row and record the unexplained shortfall.
- Reported elements total 96 wt.%, but an important element amount is unclear:
  flag for review; passing the 90 wt.% threshold is not enough to fill the gap.
- A complete composition is mapped to a shared schema containing additional
  elements: set those additional element columns to `0`.

## EPIT: composition processing

- **Source and rows:** we used the `Pitting Potential` sheet of
  `CRA_database_Scientific_Data_Publication_12102020.xlsx`. We retained the
  existing 760 samples with a usable average pitting-potential target. The
  source contained 810 rows; the existing target-availability rule had already
  excluded 50 rows before this composition-processing step.

- **Columns and units:** we retained all 24 composition columns in **weight
  percent (wt.%)**. We named them using their element symbols, such as `Fe`,
  `Cr`, and `Ni`. No unit conversion was required.

- **Reported amounts:** we preserved all reported numeric amounts, including
  explicit zeros. We did not rescale compositions to exactly 100%.

- **Fe:** we retained the Fe amount already provided in the source dataset.
  We did not perform an additional Fe-balance calculation. A later source
  investigation established that all 23 rows reviewed for Ti, Re and Nb/Ta
  contained an upstream Excel Fe-balance formula. Their cached Fe amounts
  were therefore calculated remainders, not independent measurements.

- **Reported limits:** we represented the 16 sulfur entries written as
  `<0.003` by **0.003 wt.%**. We recorded these as upper-limit approximations
  and preserved their original notation in the processing log. This matched
  the existing standalone EPIT parser and corrected the information lost in
  the previous shared CSV.

- **Missing values:** we treated unreported elements as **0** when the
  composition total, including the numeric upper-limit approximations, lay
  between **99 and 101 wt.%**. We recorded these as assumed zeros, separately
  from explicitly reported zeros. No known-omission exceptions were configured
  for this EPIT export; the processing rule did not permit zero-filling an
  element identified as a known omission.

- **Tolerance adjustment:** for this export, we widened the earlier EPIT
  zero-filling interval from **99.9–100.1 wt.%** to **99–101 wt.%**. We used
  this as a practical acceptance tolerance, rather than assuming every
  difference was caused by rounding. The three additional accepted rows had
  totals of **99.5%, 99.8679%, and 100.52%**; we left those totals unchanged.

- **Corrections:** compared with the previous shared CSV, we restored the
  sulfur amounts in **16 rows** and filled **17 missing entries across three
  additional rows**. In total, we corrected **33 cells across 19 rows**.
  Across the original source data, we recorded 3,685 assumed zeros, including
  the 3,668 already supplied by the earlier EPIT zero-filling rule.

- **Exclusions and validation:** we excluded no additional samples on
  composition grounds. All 760 processed rows contained 24 finite element
  amounts between 0 and 100 wt.%, and every composition total lay within the
  accepted interval. No EPIT row fell below the general 90 wt.% cutoff.

- **Traceability and split:** we preserved the original workbook and existing
  row IDs, and copied the frozen assignments without regrouping or resplitting:
  **608 development rows and 152 final-test rows**, with the same five
  development folds. We recorded each assumed zero and upper-limit conversion,
  the changes from the previous shared CSV, and checksums of the inputs and
  outputs.

- **Scope:** we saved a standalone processed EPIT reference table. We copied
  non-composition source fields without changing their values or processing
  the environment features. We did not update the shared EPIT–Soccol CSV,
  existing training profiles, model loaders, or evaluation paths in this step.

We saved the results under
[`datasets/electrochemical_metrics_alloys/processed/epit_composition_v1/`](datasets/electrochemical_metrics_alloys/processed/epit_composition_v1/):

- [`epit_processed.csv`](datasets/electrochemical_metrics_alloys/processed/epit_composition_v1/epit_processed.csv)
  contained the corrected composition and the original non-composition fields.
- [`composition_processing_log.csv`](datasets/electrochemical_metrics_alloys/processed/epit_composition_v1/composition_processing_log.csv)
  recorded the original notation and treatment of each assumed zero or upper limit.
- [`changes_vs_previous_shared.csv`](datasets/electrochemical_metrics_alloys/processed/epit_composition_v1/changes_vs_previous_shared.csv)
  listed the 33 corrected cells.
- [`composition_row_audit.csv`](datasets/electrochemical_metrics_alloys/processed/epit_composition_v1/composition_row_audit.csv)
  recorded each row's composition total and processing counts.
- [`split_assignments.csv`](datasets/electrochemical_metrics_alloys/processed/epit_composition_v1/split_assignments.csv)
  retained the original outer split and development-fold membership.
- [`processing_manifest.json`](datasets/electrochemical_metrics_alloys/processed/epit_composition_v1/processing_manifest.json)
  recorded the rules, column meanings, counts, and file checksums.

We generated this export with
[`prepare_epit_composition.py`](datasets/electrochemical_metrics_alloys/prepare_epit_composition.py).
To reproduce it in a new directory:

```bash
.venv/bin/python -m corrosion_datasets.datasets.electrochemical_metrics_alloys.prepare_epit_composition \
  --output-dir /tmp/epit_composition_check
```

## EPIT: retained target-value convention

- **Target:** we retained the existing handling of `Epit, mV (SCE) Avg.`.
  The source contained 13 entries written as `>800` and one as `>1256`.
  We kept the existing numeric modelling convention of using **800** and
  **1256 mV**, respectively. These were approximations using the reported
  lower bounds, rather than known exact pitting potentials.
- **Original notation:** we preserved the `>` notation in both standalone
  processed reference tables. We did not change targets, exclude these rows,
  or alter frozen split assignments in this review.
- **Comparison with the other datasets:** we checked the original Soccol and
  MPEA target columns. Soccol marked non-pitting breakdown cases separately
  with `event=0`; its current regression export had already excluded those
  cases and contained **4,027 rows, all with `event=1`**. MPEA's workbook
  contained numeric or missing pitting potentials, without explicit inequality
  target entries or a separate limit-status column. This check did not establish
  whether individual original papers had unrecorded limits.
- **Decision:** we kept EPIT's existing numeric-bound approximation for the
  current workflow and documented it; we did not leave a new target-handling
  change pending.

## EPIT: environment, heat treatment and microstructure processing

- **Input and scope:** we extended the corrected `epit_composition_v1` export
  and saved the result in `epit_environment_v1`. We retained all **760 rows**,
  their corrected compositions, original source columns and target notation.
  We copied the frozen **608 development / 152 final-test** assignments and
  five development folds unchanged. We did not fit parameters, use target
  values to choose extraction rules, or exclude additional rows.

- **Shared names:** we used the same names and units as the existing shared
  schema where an equivalent column existed. We added named chemical amounts,
  treatment stages and phase flags for later use by other datasets.

- **Missing information:** we left unknown values missing. We did not fill
  missing environment values with zero. We also left unused later treatment
  stages missing; this did not establish that no further treatment occurred.
  We treated source markers `NA`, blank and `None` as missing information,
  rather than proof of no treatment or no phase.

### Temperature, pH, chloride and scan rate

| Original EPIT column | Processed column | Unit | Populated rows |
|---|---|---|---:|
| `Test Temp. oC` | `CP_temp` | °C | 677 |
| `[Cl-] pH` | `CP_pH` | dimensionless | 712 |
| `[Cl-] M` | `CP_Cl` | mol/L | 759 |
| `Scan Rate mV/s` | `scan_rate` | mV/s | 604 |

- **Numeric values:** we retained the reported numeric temperature, pH and
  scan-rate values, including negative pH. We did not estimate missing values
  from other features.
- **Scan-rate text:** we left the numeric scan rate missing for 18 entries
  stating `steady state` and 10 stating `20mV steps by manual increament`.
  The original text remained available. Another 128 entries were missing in
  the source; neither procedural description was interpreted as a zero rate.
- **Chloride conflict:** we retained all original chloride values in the
  source column. For `epit_r000756`, the solution text `0.1 M NaCIO4`
  identified perchlorate after notation correction, while `[Cl-] M` reported
  0.1. We left the new `CP_Cl` value missing and logged the conflict for source
  verification. We did not assume zero chloride or treat perchlorate as
  chloride. The other 759 chloride values were copied unchanged.

### Test solution

- **Source:** all 760 rows contained `Test Solution`, with 111 distinct written
  descriptions. We converted these to **21 standardized solution names** in
  `test_solution_type`. Mixture names listed the reported components in a
  consistent alphabetical order. Natural seawater, synthetic seawater and
  unspecified seawater retained distinct names.
- **Separate amounts:** we recorded each reported component separately.
  For example, `1M H2SO4 + 0.5M NaCl, H2 deaerated` produced
  `test_solution_type = H2SO4 + NaCl`, `H2SO4_mol_L = 1`,
  `NaCl_mol_L = 0.5`, and `CP_aeration = deaerated`.
- **Units:** we retained molarity, normality and weight percent separately.
  We did not assume that N meant M, or convert weight percent to mol/L without
  sufficient information. We did not infer an unreported counterion or
  calculate free-ion concentrations from acid amounts.

| Column group created | Unit and interpretation |
|---|---|
| `H2SO4_mol_L`, `HCl_mol_L`, `KCl_mol_L`, `MgCl2_mol_L`, `Na2S2O3_mol_L`, `Na2SO4_mol_L`, `NaBr_mol_L`, `NaClO4_mol_L`, `NaCl_mol_L`, `NaHCO3_mol_L`, `WO4_mol_L` | Reported component amount in mol/L. `WO4` represented a reported ion with an unidentified counterion. |
| `H2SO4_eq_L`, `HCl_eq_L`, `KCl_eq_L`, `NaBr_eq_L`, `NaCl_eq_L` | Reported normality in equivalents/L. |
| `NaCl_wt_percent` | Explicitly reported weight percent. |
| `H2S_partial_pressure_atm` | Reported gas partial pressure in atm; not dissolved concentration. |
| `NaCl_percent_unspecified`, `NaCl_ppm_unspecified` | Original amounts whose concentration basis was not specified; retained for audit, excluded from the feature list. |

- **Unknown units and amounts:** we preserved 25 rows reporting unspecified
  `%` and 59 reporting unspecified `ppm` in the two audit-only columns. We
  left corresponding model concentration fields missing. We retained unnamed
  amounts, seawater dilution conventions and unquantified acid additions in
  the source text. A chemical absent from a description did not automatically
  receive a zero concentration.
- **Chloride-only descriptions:** we used `chloride_unspecified` when the
  solution text identified chloride without naming its source chemical.
  We retained the existing chloride measurement without inventing NaCl.
- **Notation corrections:** we recorded explicit corrections such as `NaCI`
  to `NaCl`, `HCI` to `HCl`, `NaCIO4` to `NaClO4`, and `W04` to `WO4`.
  The original strings remained unchanged in the processed reference table.
- **Aeration:** we populated `CP_aeration` for **291 rows** from the solution
  description or an explicit statement in the test method. We preserved named
  gas conditions such as nitrogen saturation, oxygen sparging and nitrogen
  atmosphere rather than inventing dissolved gas concentrations. Missing
  aeration did not mean naturally aerated.

### Heat treatment

- **Source:** `Heat treatment` contained 421 nonmissing entries and 79 distinct
  descriptions. We extracted at least one thermal-treatment detail for
  **397 rows**. We kept unsupported or ambiguous details missing.
- **Columns:** we created the following four columns for the first described
  treatment, plus the same four with the `heat_treatment_2_` prefix for a
  second explicitly described stage. **95 rows** contained extracted details
  of a second stage. These stage numbers followed the reported sequence; they
  did not establish a complete manufacturing history.

| First-stage column | Meaning and unit |
|---|---|
| `heat_treatment_type` | Reported treatment, such as annealing, aging or solution treatment. |
| `heat_treatment_temperature_C` | Treatment temperature in °C. |
| `heat_treatment_duration_h` | Duration in hours. |
| `heat_treatment_cooling` | Reported cooling method, such as water quench, brine quench or still-air cooling. |

- **Conversions:** we converted minutes and days to hours, and the two
  Fahrenheit descriptions to Celsius using `(F - 32) × 5/9`. We recorded
  source notation corrections such as `ll00` to `1100` and `I250` to `1250`.
  We retained unusual reported values, including 1000 hours, without guessing
  corrections.
- **Several stages:** we retained solution treatment followed by another
  heating or aging stage separately. We attached cooling only to the stage
  for which it was reported. We did not average temperatures or add durations
  across stages.
- **Alternative conditions:** descriptions listing different temperatures
  for different alloys, or aging durations of 30 minutes, 1 hour and 10 hours,
  did not establish a sequence. We extracted only common, clearly applicable
  details and left unresolved temperatures or durations missing. Ten rows
  mixed as-received and quenched conditions without identifying the applicable
  condition; we left their treatment fields missing.
- **Other processing:** we retained the explicitly reported thermal
  spark-plasma-sintering stage. We did not assign melting, hot rolling,
  pickling, electropolishing, ion implantation or rapid solidification to an
  annealing stage. Their descriptions stayed in the source text. In particular,
  preparation solutions and preparation temperatures were not test-bath
  solution or temperature features. Fourteen rows contained only manufacturing
  or as-received/as-cast/as-rolled information without extractable heat-treatment
  details.

### Microstructure

- **Source:** `Microstructures` contained 217 nonmissing entries and 39 distinct
  descriptions. We extracted phase information for **154 rows**.
- **Columns:** we created `phase_FCC`, `phase_BCC`, `phase_HCP` and `phase_IM`.
  Each used **1 for presence, 0 for supported absence, and missing for unknown**.
  Several phase-presence columns could equal 1 for the same sample.
- **Names:** we mapped austenite to FCC and ferrite to BCC using the standard
  structures described by the [University of Cambridge Materials Algorithms
  Project](https://www.phase-trans.msm.cam.ac.uk/map/steel/programs/dilat-b.html).
  We mapped explicit austenite–ferrite and duplex stainless-steel descriptions
  to both flags. We kept ambiguous alpha-phase labels unresolved.
- **Absence:** we set absent-phase flags only when the text explicitly excluded
  them or clearly identified a single phase. Simply naming FCC did not imply
  that BCC, HCP or intermetallic phases were absent. Unspecified secondary
  phases and nonmetallic inclusions did not automatically imply `phase_IM = 1`.
- **Other text:** 30 rows contained a general discussion of nitrogen and phase
  boundaries, rather than a sample observation. We did not assign phases from
  it. Another 33 rows provided other information, including grain size,
  amorphous/crystalline descriptions, an ordered DO3 structure or unspecified
  secondary phases; we preserved that information in the source text without
  forcing it into the four flags.

| Phase column | Present (1) | Supported absent (0) | Unknown |
|---|---:|---:|---:|
| `phase_FCC` | 125 | 0 | 635 |
| `phase_BCC` | 73 | 9 | 678 |
| `phase_HCP` | 0 | 9 | 751 |
| `phase_IM` | 1 | 31 | 728 |

### Test-method text and feature boundaries

- **Exposed area:** we extracted `Test_area_cm2` for **144 rows** from explicit
  exposed-area statements in `[Cl-] Test Method`. Where the exposed face was
  explicitly 0.5 by 0.6 cm, we calculated 0.3 cm². Approximate reported areas
  remained nominal values and were identified in the log. We did not confuse
  current-density units or dimensions of other specimens with exposed area.
- **Full descriptions:** we retained test-method text, comments, references,
  identifiers and original solution/history descriptions in the reference
  table, but excluded them from the feature list. We did not extract features
  from target values or comments about test outcomes.
- **Feature list:** we registered **24 composition features and 37 environment,
  procedure and history features**. The reference table also contained the two
  audit-only concentration columns with unspecified bases. No targets, source
  IDs other than the joining `row_id`, or original free-text fields appeared in
  `epit_features.csv`.
- **Validation:** the focused composition and environment checks passed. They
  covered unit separation, treatment stages versus alternatives, phase presence
  versus absence, nonnumeric scan descriptions, preserved original values and
  frozen splits, target exclusion, checksums and refusal to overwrite an
  existing export.
- **Pipeline scope:** we did not modify Soccol, MPEA, the existing shared CSV,
  trained checkpoints, synthetic generators, model loaders or evaluation paths.
  This export completed the EPIT data-processing step; existing models did not
  automatically start using these features.

We saved the results under
[`datasets/electrochemical_metrics_alloys/processed/epit_environment_v1/`](datasets/electrochemical_metrics_alloys/processed/epit_environment_v1/):

- [`epit_processed.csv`](datasets/electrochemical_metrics_alloys/processed/epit_environment_v1/epit_processed.csv)
  contained the preserved original columns and all extracted fields.
- [`epit_features.csv`](datasets/electrochemical_metrics_alloys/processed/epit_environment_v1/epit_features.csv)
  contained `row_id` and the 61 registered features, without targets.
- [`feature_registry.json`](datasets/electrochemical_metrics_alloys/processed/epit_environment_v1/feature_registry.json)
  listed feature names, meanings, units and excluded fields.
- [`environment_processing_log.csv`](datasets/electrochemical_metrics_alloys/processed/epit_environment_v1/environment_processing_log.csv)
  recorded source text, extracted values and decisions for every row and source field.
- [`environment_review_notes.csv`](datasets/electrochemical_metrics_alloys/processed/epit_environment_v1/environment_review_notes.csv)
  collected partial interpretations, unresolved descriptions and the chloride conflict.
- [`environment_column_summary.csv`](datasets/electrochemical_metrics_alloys/processed/epit_environment_v1/environment_column_summary.csv)
  recorded coverage and units for every derived column.
- The composition audits, split assignments and composition manifest were copied
  into this export. `processing_manifest.json` recorded counts and checksums.

We stored the exact-text extraction decisions in
[`epit_environment_rules_v1.json`](datasets/electrochemical_metrics_alloys/epit_environment_rules_v1.json).
Unseen descriptions required review rather than a silent approximate match.
We generated the export with
[`prepare_epit_environment.py`](datasets/electrochemical_metrics_alloys/prepare_epit_environment.py).
To reproduce it in a new directory:

```bash
.venv/bin/python -m corrosion_datasets.datasets.electrochemical_metrics_alloys.prepare_epit_environment \
  --output-dir /tmp/epit_environment_check
```

## EPIT: preparation and shared-ion additions

- **Input and output:** we extended `epit_environment_v1` and saved the result
  in **`epit_environment_v2`**. We preserved every existing source and processed
  column, including corrected composition and target notation. We retained all
  **760 rows**, the **608 development / 152 final-test** assignments and the
  same five development folds. The previous export remained unchanged.
- **Added columns:** we added six preparation/procedure fields and two
  supported ion fields. The feature table therefore contained **69 features**:
  24 composition features and 45 environment, procedure and history features.
  The new column names, numeric/categorical types and units matched Soccol.

| Added column | Meaning and unit | Populated rows |
|---|---|---:|
| `Prep_grinding_grit` | Reported numeric abrasive grade | 126 |
| `surface_preparation_method` | Explicit preparation operation or combination | 291 |
| `preparation_solution_type` | Named preparation or cleaning liquid | 101 |
| `Prep_time` | Chemical/electrochemical preparation duration, hours | 87 |
| `CP_time` | Waiting in the test solution before measurement, hours | 177 |
| `CP_agitation` | Explicit hydrodynamic motion: 1; explicit absence: 0 | 6 |
| `CP_Br` | Nominal bromide amount from reported NaBr molarity, mol/L | 25 |
| `CP_SO4` | Supported nominal sulfate amount from reported Na2SO4 molarity, mol/L | 2 |

### Preparation descriptions and abrasive grades

- **Sources:** we reviewed the 79 distinct nonmissing `Heat treatment`
  descriptions, 51 `[Cl-] Test Method` descriptions and 80 `Comments`
  descriptions. We recorded the specific preparation clauses supporting each
  extraction. Comments about composition, test outcomes or parallel experiments
  did not supply added feature values. Original full descriptions remained
  available for checking, and were excluded from model inputs.
- **Numeric grades:** we retained the explicitly reported numeric grade for
  **126 rows**: 38 at grade 120, 70 at grade 600, 17 at grade 1000 and one at
  grade 2000. These included 17 entries written simply as `Ground to 600`;
  the context identified a grinding grade, but did not establish its standard.
  We did not claim that all reported grades used one grading convention.
- **Unconverted grades:** 16 rows reported `600 mesh SiC`, and 44 reported
  slash or ambiguous emery notation such as `3/0`, `4/0` or `2.0 emery`.
  We preserved these descriptions and left their numeric grit missing. We
  did not interpret `3/0` as a division, `2.0` as grit 2, or mesh as an
  automatically equivalent grit standard.
- **Correct specimen:** three method descriptions gave a 600-grade finish
  for separate exposure-test coupons, while the pitting-test electrode had
  been polished with an alumina slurry. We extracted only the pitting-electrode
  preparation and did not copy the other coupons' grade or solvent treatment.
- **Particle size:** alumina or diamond particle sizes, and vague micrometre
  surface-finish descriptions, did not populate a grit or roughness column.
  The ambiguous `1 mm` diamond-paste description remained uncorrected in the
  source text; it did not affect the extracted numeric grade.
- **Methods:** we retained explicit operations such as grinding, polishing,
  pickling, electropolishing, cathodic reduction and ion implantation. When
  several operations were reported, we listed their distinct names together.
  This represented reported operations, including earlier surface processing
  where described, rather than an inferred complete sequence or a declaration
  of the final surface condition. Repeated descriptions did not create
  duplicated operations.

### Preparation liquids, durations and agitation

- **Separate liquids:** we populated `preparation_solution_type` with
  `HF + HNO3` for 48 rows, `H2SO4 + NaCl` for 22, `ethanol` for 18, `NaOH`
  for 12 and `HCl` for one. These described preparation baths or explicitly
  named cleaning liquids, not the corrosion-test solution. Water used for
  rinsing or as an abrasive carrier was not combined with an active bath
  into an invented mixture. Preparation concentrations remained in the source
  text; no additional preparation-concentration fields were created.
- **Preparation duration:** we converted reported minutes and seconds to
  hours. For example, five minutes and 300 seconds both became `1/12` hour.
  A specified-potential conditioning hold populated `Prep_time`; annealing
  time, epoxy-curing time and time spent waiting in air did not.
- **Different stages and bounds:** 18 rows described conditioning for
  **at least 15 minutes**, alongside a usual five-minute pickling step.
  We left `Prep_time` missing rather than substituting the pickling duration,
  treating the lower bound as an exact value, or adding the two durations.
  The separate operations and both original descriptions remained recorded.
- **Test-bath waiting:** we extracted explicit pre-scan waiting or open-circuit
  intervals into `CP_time`. These ranged from five minutes to two hours.
  Seventy-three rows used the source's approximate wording: 65 reported
  about one hour and eight reported about two hours. We retained these as
  nominal durations and noted the qualification in the extraction log.
- **Excluded intervals:** a gas purge before specimen immersion was not
  specimen waiting time. Nor were waits between potential steps, hours spent
  observing pitting, or holds for separate surface-analysis specimens. In the
  30-row description that specified a nitrogen purge beginning 30 minutes
  before immersion and polarization 50 minutes after immersion, `CP_time`
  became **50/60 hour**.
- **Agitation:** six rows explicitly used rotating disk electrodes with a
  nonzero rotation rate; we assigned `CP_agitation = 1`. The other 754 rows
  stayed missing. We found no explicit quiescent condition to encode as zero.
  Stationary electrical current, fixed electrode mounting and gas bubbling
  did not establish whether the test solution was agitated.

### Supported ion mappings and deferred fields

- **Bromide:** 25 rows explicitly reported NaBr in mol/L. We mapped the
  nominal bromide amount one-to-one into `CP_Br` and retained the original
  NaBr concentration as well. The 12 entries reported in normality remained
  in their existing `NaBr_eq_L` field; we did not introduce an automatic
  normality-to-molarity conversion.
- **Sulfate:** two rows reported `0.1M Na2SO4 + 0.6M NaCl`, without naming an
  acid in the recipe. We populated `CP_SO4 = 0.1 mol/L` as a nominal amount
  supplied by the reported sulfate salt. This was not a calculation of free-ion
  activity or pH-dependent speciation. The six recipes combining Na2SO4 with
  H2SO4 remained missing in `CP_SO4`; we did not add acid molarity to the salt
  amount or equate H2SO4 concentration with free sulfate. The original chemical
  concentration fields remained unchanged for every row.
- **Other ions:** we did not infer ion concentrations from preparation liquids,
  unreported counterions, or pH. Other absent Soccol ion columns were not added
  as entirely empty fields in this standalone export.
- **Deferred fields:** we did not add `Prep_redox`, `Prep_pH` or
  `Prep_Ra_micron`. Preparation potentials required alignment of voltage
  references with Soccol, and the reviewed EPIT descriptions did not establish
  preparation-solution pH or measured surface Ra. We preserved potentially
  useful original information without inventing these values.
- **Missing values and validation:** all unknown added values stayed `NaN`.
  Focused checks covered grade conventions, different specimens, preparation
  versus measurement timing, bounds, chemical versus test liquids, rotating
  electrodes versus stationary current, supported ion mappings, outcome
  exclusion, compatibility with Soccol units, and preservation of all earlier
  columns and frozen assignments. No additional rows were excluded and no
  model parameters were fitted.
- **Scope:** this completed the EPIT preparation extension as a standalone
  export. We preserved Soccol, MPEA, the shared tables and all existing model,
  synthetic-generator, training and evaluation paths.

We saved the results under
[`datasets/electrochemical_metrics_alloys/processed/epit_environment_v2/`](datasets/electrochemical_metrics_alloys/processed/epit_environment_v2/):

- [`epit_processed.csv`](datasets/electrochemical_metrics_alloys/processed/epit_environment_v2/epit_processed.csv)
  contained every previous reference column plus the eight additions.
- [`epit_features.csv`](datasets/electrochemical_metrics_alloys/processed/epit_environment_v2/epit_features.csv)
  contained `row_id` and the 69 registered features, without targets or free text.
- [`preparation_processing_log.csv`](datasets/electrochemical_metrics_alloys/processed/epit_environment_v2/preparation_processing_log.csv)
  recorded the added extraction decisions. The combined environment log and
  review notes also retained all earlier processing records.
- `feature_registry.json`, `environment_column_summary.csv` and
  `processing_manifest.json` recorded definitions, coverage and checksums.
  The composition audits and split assignments were copied unchanged.

We stored the reviewed clauses in
[`epit_preparation_rules_v1.json`](datasets/electrochemical_metrics_alloys/epit_preparation_rules_v1.json)
and generated the export with
[`prepare_epit_preparation.py`](datasets/electrochemical_metrics_alloys/prepare_epit_preparation.py).
To reproduce the extension in a new directory:

```bash
.venv/bin/python -m corrosion_datasets.datasets.electrochemical_metrics_alloys.prepare_epit_preparation \
  --output-dir /tmp/epit_preparation_check
```

## Soccol: environment and preparation processing

- **Source and rows:** we read environment values from the original
  `Pitting_potential_dataset_2024.xlsx` workbook, using exactly the existing
  **4,027 regression rows**. We preserved the frozen **3,222 development /
  805 final-test** split and all five development folds. We retained the shared
  `soccol_rXXXXXX` identifiers based on original workbook row numbers.
- **Scope:** we created a standalone environment export. We copied the legacy
  composition, targets and other audit fields into its reference table without
  changing them. We did **not** apply the new composition policy to Soccol in
  this step or certify its old zero-filling and Fe reconstruction. The separate
  environment feature table contained no composition or target columns.
- **Unknown values:** we left blanks and whitespace-only cells as `NaN`, and
  retained numeric zeros already recorded in the workbook. A recorded zero was
  not evidence that the original study measured absence. We did not replace
  missing ion concentrations with zero or with the collection paper's modelling
  floor of `10^-5 M`. We fitted no imputation, scaling or other parameters.
- **Original values:** we retained all selected original workbook cells in
  `source_rows.csv`. In `soccol_processed.csv`, original environment fields
  received a `raw_` prefix, and the processed fields used the shared names.
  Original text therefore remained recoverable when a processed value was
  missing or disagreed with another source field.
- **Feature list:** we registered **33 environment, procedure and history
  columns**. Sixteen names and units matched columns already present in the
  updated EPIT export, including the heat-treatment stage fields. We left
  unsupported stage fields missing. We did not infer microstructure from an
  alloy name or from heat treatment.

### Test conditions and named ions

| Processed column | Meaning and unit | Populated rows |
|---|---|---:|
| `CP_temp` | Test temperature, °C | 3,977 |
| `CP_pH` | Test-solution pH | 3,963 |
| `scan_rate` | Scan rate, mV/s | 3,767 |
| `Test_area_cm2` | Exposed test area, cm² | 2,000 |
| `CP_time` | Time in the test medium before the scan, hours | 1,982 |
| `CP_Cl` | Reported chloride concentration, mol/L | 4,027 |
| `CP_Br` | Reported bromide concentration, mol/L | 4,027 |
| `CP_OH` | Reported hydroxide concentration, mol/L | 4,005 |
| `CP_SO4` | Reported sulfate concentration, mol/L | 3,895 |
| `CP_CO3` | Reported carbonate concentration, mol/L | 3,985 |
| `CP_NO3`, `CP_PO4`, `CP_MoO4`, `CP_CrO4` | Reported nitrate, phosphate, molybdate and chromate concentrations, mol/L | 4,024 each |

- **Numeric values:** we retained the original units and negative pH values.
  We did not infer chemical salts from ions: chloride did not establish NaCl,
  and sulfate did not establish H2SO4. The term "reported concentration"
  referred to the workbook entry, not necessarily a measured free-ion amount.
- **Timing:** the collection paper identified `CP_time` as equilibration in
  the test medium before measurement, rather than time until pitting. We kept
  the raw durations in hours; we did not apply the paper's log transformation.
- **Aeration:** we retained **2,764** reported entries in `CP_aeration`:
  2,231 `deaerated`, 360 `naturally_aerated`, 171 `aerated`, and 2 `ozonated`.
  Missing aeration did not imply natural aeration.
- **Agitation:** we converted explicit `yes` to **1** for 105 rows and `no` to
  **0** for 413 rows. The other 3,509 rows stayed missing, including 12 cells
  containing only whitespace. We kept agitation separate from aeration.

### Additional test-solution descriptions

- **Text extraction:** we reviewed the 38 distinct nonblank `CP_anions_info`
  descriptions. We used explicit amounts to fill missing matching fields or
  check existing values. We converted mM to mol/L by dividing by 1,000.
  We did not feed the original free-text column to the model.
- **Named acid:** eleven notes explicitly reported **0.1 M H2SO4**. We
  populated the existing EPIT column `H2SO4_mol_L` for these rows. We did not
  equate this chemical amount to a free sulfate-ion concentration.
- **Solution type:** we populated `test_solution_type` only for the explicitly
  named whole media: 18 `seawater` and 5 `synthetic_seawater` entries, using
  EPIT's category names. Other notes generally described additives rather
  than complete recipes. We therefore did not label an entire mixed bath as
  H2SO4 merely because an additive note mentioned it.
- **Unspecified concentrations:** salt percentages without a stated basis,
  such as `0.5% MgSO4`, remained in the original text. We did not assume wt.%
  or convert them to mol/L. `none` meant no additional note, rather than pure
  water or absence of every ion. Dichromate was not treated as chromate.
- **Excluded other ions:** following the explicit decision for this export,
  we excluded `CP_ion_other` from both extraction and model features. We also
  deferred the proposed borate/citrate concentration columns. The 52 positive
  original values and their accompanying notes remained available for audit;
  they did not enter a replacement generic ion field.

### Conflicting numeric values and descriptions

- **Handling:** we found **192 rows** with an explicit disagreement between
  a numeric field and its accompanying description. We left only the affected
  processed cell missing, retained the row and both original reports, and
  recorded the disagreement in `environment_review_notes.csv`. We did not
  silently choose one source field over the other.
- **Sulfate:** 132 rows disagreed. Many numeric values were twice the
  concentration written in the note, for example **0.12 versus 0.06 M**.
  We did not assume that every sulfate value in the workbook was expressed
  in equivalents/L or apply a global division by two. The convention remained
  unresolved for these cells; unchanged entries were not individually
  revalidated against every original paper.
- **Carbonate:** 39 rows disagreed. These included 18 notes saying 0.015 M
  against numeric 0.02, and 21 notes reporting **25 mM each of bicarbonate
  and carbonate** against numeric carbonate 0.05. We did not treat the sum
  of two different ions as carbonate alone.
- **Hydroxide:** six notes stated 0.05 M while the numeric field contained
  approximately 0.03162 M. We left the processed hydroxide value missing.
- **Exposed area:** 15 notes stated **0.436 mm²**, equivalent to
  **0.00436 cm²**, while the numeric field contained **0.000436 cm²**.
  We logged the tenfold disagreement and left the processed area missing.
  The other 2,000 reported numeric areas were retained.

### Surface preparation and preparation conditions

| Processed column | Meaning and unit | Populated rows |
|---|---|---:|
| `Prep_grinding_grit` | Reported abrasive grade | 2,990 |
| `Prep_Ra_micron` | Reported average surface roughness Ra, µm | 367 |
| `Prep_pH` | Preparation-solution pH | 875 |
| `Prep_redox` | Reported preparation potential, V; reference unresolved | 1,241 |
| `Prep_time` | Reported preparation-treatment duration, hours | 1,006 |
| `surface_preparation_method` | Explicit preparation method or combination | 1,600 |
| `preparation_solution_type` | Named preparation chemical or mixture | 283 |

- **Grit:** we retained the workbook's reported numeric grades. The 31 rows
  marked `(1200 = 600 US grit)` already contained **600**; we did not convert
  them again. We did not assume that every other entry used the same grading
  standard, and did not undertake a full per-paper grit-standard conversion.
  We did not convert abrasive particle size into roughness or grit.
- **Roughness:** we kept `Prep_Ra_micron` separate from abrasive grade. We
  copied the reported Ra values without estimating them from grit or treating
  an abrasive particle diameter as a measured surface roughness.
- **Preparation pH:** we kept `Prep_pH` separate from `CP_pH`. Preparation
  liquids and corrosion-test liquids could differ; one did not fill the other.
- **Preparation potential:** the collection paper confirmed volts for
  `Prep_redox`, but its statement about the Ag/AgCl reference explicitly
  concerned `E_corr` and `E_pit`. We retained the reported preparation voltages
  and marked their reference as unresolved in the registry. We did not convert
  them to SCE or claim that they were already comparable to future preparation
  potentials from other datasets.
- **Preparation duration:** we retained hours in `Prep_time`, separate from
  `CP_time` and heat-treatment duration. Notes such as `0.5 V - 30'` supported
  the existing 0.5-hour preparation value. Air-exposure durations stayed in
  the original notes when assigning them to `Prep_time` could confuse two
  different preparation steps. We did not call 24 hours in air 24 hours in
  the corrosion-test bath.
- **Mixed preparation text:** we reviewed the 73 distinct nonblank
  `Prep_medium` descriptions. We replaced this model category with the two
  new fields for method and preparation solution, plus the applicable existing
  heat-treatment and numeric fields. For example, `pickling HNO3 + HF`
  produced method `pickling` and preparation solution `HF + HNO3`.
- **Preparation chemicals:** nitric acid, sulfuric acid, citric acid and
  named mixtures populated preparation fields only. They did not populate
  test-solution concentrations. Unspecified preparation percentages stayed
  in the notes, and a reported preparation temperature was not copied into
  `CP_temp` or a heat-treatment temperature.
- **Partial descriptions:** we retained explicit combinations such as
  `wet_grinding + acid_treatment`, and distinguished edge electropolishing
  from whole-surface electropolishing. Unclear abbreviations, as-delivered
  descriptions and `none` did not establish an absent treatment. Free-text
  details without a supported mapping remained in the audit.

### Heat treatment and feature boundaries

- **Shared heat columns:** we used the same two sets of four heat-treatment
  fields as EPIT: type, temperature in °C, duration in hours and cooling.
  **89 rows** contained a first-stage treatment type: 41 sensitization
  descriptions, 24 anneal-and-quench descriptions, and 24 source-specific
  annealing descriptions. Quenching was cooling of the annealing stage,
  rather than an invented second heating stage.
- **Verified source units:** the [2019Ha Materials section](https://www.sciencedirect.com/science/article/pii/S0010938X17313847)
  confirmed that the six temperatures written as `annealed 1040` through
  `annealed 1200` were in °C, with one hour of treatment and water quenching.
  We populated all four fields for its 24 rows. We restricted this additional
  source evidence to `2019Ha`; we did not apply it to similarly worded notes
  from other papers.
- **Unknown heat details:** for the other 24 anneal-and-quench rows, we kept
  the temperature, duration and quenching medium unknown. The sensitized rows
  also lacked a temperature or duration. No description established a second
  thermal stage, so all four second-stage columns stayed missing.
- **Excluded predictors:** source identifiers, raw free text, `E_corr`,
  `E_pit`, `event`, legacy composition and `CP_ion_other` did not enter the
  environment feature table. Extraction did not use outcomes. The frozen
  regression membership supplied the row list; we performed no new selection.
- **Validation:** focused checks verified preparation versus test conditions,
  explicit yes/no versus missing values, unit conversions, contradictions,
  source-specific heat evidence, exclusion of other-ion and target values,
  original-value preservation, unchanged frozen assignments and checksums,
  and refusal to overwrite an existing export.
- **Pipeline scope:** we left EPIT, MPEA, the shared datasets, synthetic
  generators, trained models and evaluation paths unchanged. Adding grit
  and other preparation information to EPIT was completed separately in the
  EPIT preparation extension documented above.

We saved the results under
[`datasets/soccol_pitting_potential/processed/soccol_environment_v1/`](datasets/soccol_pitting_potential/processed/soccol_environment_v1/):

- [`soccol_environment_features.csv`](datasets/soccol_pitting_potential/processed/soccol_environment_v1/soccol_environment_features.csv)
  contained `row_id` and the 33 registered environment/history features.
- [`soccol_processed.csv`](datasets/soccol_pitting_potential/processed/soccol_environment_v1/soccol_processed.csv)
  included the processed environment fields, original environment values under
  `raw_` names, and unchanged legacy composition, target and audit fields.
- [`source_rows.csv`](datasets/soccol_pitting_potential/processed/soccol_environment_v1/source_rows.csv)
  preserved the selected original workbook rows with their shared row IDs.
- [`feature_registry.json`](datasets/soccol_pitting_potential/processed/soccol_environment_v1/feature_registry.json)
  documented names, units, exclusions and qualifications.
- [`environment_processing_log.csv`](datasets/soccol_pitting_potential/processed/soccol_environment_v1/environment_processing_log.csv)
  recorded numeric copying, missing values, extraction and conflicts.
- [`environment_review_notes.csv`](datasets/soccol_pitting_potential/processed/soccol_environment_v1/environment_review_notes.csv)
  collected partial interpretations and unresolved conflicts.
- `environment_column_summary.csv`, `split_assignments.csv` and
  `processing_manifest.json` recorded coverage, frozen assignments and checksums.

We stored the extraction decisions in
[`soccol_environment_rules_v1.json`](datasets/soccol_pitting_potential/soccol_environment_rules_v1.json)
and generated the export with
[`prepare_soccol_environment.py`](datasets/soccol_pitting_potential/prepare_soccol_environment.py).
Unreviewed descriptions failed rather than receiving an approximate match.
To reproduce the export in a new directory:

```bash
.venv/bin/python -m corrosion_datasets.datasets.soccol_pitting_potential.prepare_soccol_environment \
  --output-dir /tmp/soccol_environment_check
```

## EPIT and Soccol: source-consistency review

- **Scope:** we rechecked the modified EPIT composition and environment
  exports and the Soccol environment export against their source cells,
  reviewed text rules and frozen row assignments. We reviewed solution
  amounts, preparation versus test conditions, time and temperature
  conversions, heat stages, phase indicators, grade descriptions, missing
  values and feature exclusions. This did not independently verify every
  measurement in the original literature.
- **EPIT composition:** the 33 earlier corrections across 19 rows passed
  the checks. The 24 composition columns remained finite, with no additional
  balance calculation, rescaling or row removal.
- **EPIT chloride contradiction:** `epit_r000148` named
  `deareated (by H2) 1 M HCl` but recorded `[Cl-] M = 0`. We left the derived
  `CP_Cl` missing instead of treating zero as reliable or silently choosing
  the solution description. We preserved both source reports and the
  explicitly reported HCl amount. The other explicit molar chloride recipes
  agreed with the reported chloride within source precision. The earlier
  perchlorate/chloride conflict also remained missing. Consequently, `CP_Cl`
  contained 758 populated values rather than 759.
- **Soccol preparation-time mismatch:** 217 rows contained a numeric
  preparation duration but named only `air`, without a preparation potential
  or pH identifying a separate solution treatment. EPIT's shared `Prep_time`
  represented chemical/electrochemical treatment, excluding waiting in air.
  We therefore left these 217 values missing in the processed column, while
  preserving their original times under `raw_Prep_time`. We did not assume
  which step they timed or move them into `CP_time`.
- **Separate treatment was retained:** another 36 air-labelled Soccol rows
  also reported a preparation potential and pH. We retained their preparation
  durations; the word `air` alone did not justify removing a separately
  described treatment. The revised `Prep_time` contained **789** populated
  values instead of 1,006. The collection paper's description distinguished
  passivation in solution from air exposure
  ([author manuscript, Section 2.1](datasets/soccol_pitting_potential/raw/Soccol_updated_PREN_author_manuscript.pdf)).
- **Consistent operation names:** 32 Soccol rows used
  `wet_grinding + acid_treatment`. We reordered the names to
  `acid_treatment + wet_grinding`, matching EPIT's alphabetical convention.
  The operations were unchanged and their order did not imply chronology.
- **Existing uncertainties:** the 192 previously identified Soccol
  sulfate, carbonate, hydroxide and exposed-area conflicts remained missing.
  The reference for Soccol's preparation voltage remained unverified, and
  reported grit grades did not establish a common grading standard. We did
  not certify every source zero as a measured absence. Soccol's legacy
  composition remained outside this environment review and still required
  its separate review under the new composition policy.
- **Validation:** 22 existing focused checks and four additional review
  checks passed. The new checks confirmed that the chloride contradiction
  remained missing, air-only timing was separated from solution treatment,
  separately documented conditioning survived, and the exports changed
  exactly the logged cells. All row IDs, targets, compositions, original
  source fields and frozen splits were preserved. The feature counts stayed
  at **69 for EPIT** and **33 environment/history features for Soccol**.
  An independent numeric cross-check verified 332 populated EPIT chloride
  values against explicit molar recipes, accounted for all 409 Soccol numeric
  differences as logged missing values, and checked the types and units of
  all 24 shared columns. We saved the checksum and count results in
  [the review summary](analysis/processed_epit_soccol_review_v1.json).
- **Saved results:** we preserved the earlier exports and saved the reviewed
  versions below. Each included `review_changes.csv`, updated coverage and
  extraction logs, feature definitions and file checksums. Existing shared
  CSVs and model paths were not changed.

| Dataset | Reviewed reference table | Cell-level corrections |
|---|---|---|
| EPIT | [epit_environment_v3/epit_processed.csv](datasets/electrochemical_metrics_alloys/processed/epit_environment_v3/epit_processed.csv) | [One chloride cell](datasets/electrochemical_metrics_alloys/processed/epit_environment_v3/review_changes.csv) |
| Soccol | [soccol_environment_v2/soccol_processed.csv](datasets/soccol_pitting_potential/processed/soccol_environment_v2/soccol_processed.csv) | [217 timing cells and 32 operation names](datasets/soccol_pitting_potential/processed/soccol_environment_v2/review_changes.csv) |

We generated both reviewed exports with
[`review_epit_soccol_environment.py`](datasets/review_epit_soccol_environment.py).
The following commands reproduced them in new directories:

```bash
.venv/bin/python -m corrosion_datasets.datasets.review_epit_soccol_environment epit \
  --output-dir /tmp/epit_environment_review_check
.venv/bin/python -m corrosion_datasets.datasets.review_epit_soccol_environment soccol \
  --output-dir /tmp/soccol_environment_review_check
```

## Soccol: composition processing

- **Source and scope:** we reviewed the 4,027 existing regression rows from
  the `pitting_potentials` sheet of `Pitting_potential_dataset_2024.xlsx`.
  We read the original composition cells again, before the earlier blank-to-zero
  conversion. We inherited all 33 reviewed environment/history features from
  `soccol_environment_v2` without changing them.
- **Columns and units:** we used the same 24 element columns as EPIT, in
  **wt.%**: `Fe`, `Cr`, `Ni`, `Mo`, `W`, `N`, `Nb`, `C`, `Si`, `Mn`, `Cu`,
  `P`, `S`, `Al`, `V`, `Ta`, `Re`, `Ce`, `Ti`, `Co`, `B`, `Mg`, `Y`, `Gd`.
  No unit conversion was needed. Together with the environment columns,
  the feature table contained **57 features**.
- **Original amounts:** we retained every numeric element amount from Soccol,
  except the already-confirmed `2000Russell.Ti` error: those cells contained
  PRE values, not Ti amounts. The 11 affected regression rows used assumed
  zero Ti after removal of the misfiled values. We retained explicit source
  zeros and distinguished them from assumed zeros in the cell log.
- **Four explicit alloy descriptions:** we recovered nominal amounts that
  Soccol had written in the alloy name but omitted from its element columns:

  | Row ID | Recovered amounts, wt.% | Recalculated Fe, wt.% |
  |---|---|---|
  | `soccol_r000112` | Ti = 1.86 | 77.612 |
  | `soccol_r000117` | Ti = 1.75 | 75.47 |
  | `soccol_r000149` | Nb = 1 | 67.93 |
  | `soccol_r000152` | Nb = 1; Cu = 3.5 | 43.44 |

- **Additional source evidence:** we recovered missing W, Cu and Nb amounts
  from the original EPIT workbook where it described the **same paper and
  matching composition**. We verified the paper DOI and required agreement
  on every numeric element originally reported by Soccol. We accepted a
  recovered amount only when the matching donors' reported amounts agreed.
  We did not use target values, environmental conditions, row order or a
  generic grade recipe to choose amounts. Raw EPIT blanks did not supply
  values, and its processed zero fills were not used as evidence.

  | Soccol source | EPIT reference | Recovered fields | Recovered cells, including explicit zeros |
  |---|---|---|---:|
  | `1969Lizlovs` | 11 | Cu | 4 |
  | `1974Machin` | 84 | Cu, in addition to the alloy-name recovery | 5 |
  | `1983Bui` | 73 | W | 17 |
  | `1998Ahn` | 74 | W and Nb | 16 |
  | `1995Malik` | 49 | Cu and separately reported Nb | 115 |

- **Recovery totals:** the five nominal cells from the four alloy descriptions
  and 157 same-paper cells supplied **162 element values across 103 rows**.
  These comprised **105 positive amounts across 91 rows**, plus **57 explicit
  zeros**. Some rows with recovered Cu still lacked separate Nb/Ta amounts
  and therefore remained unresolved.
- **Fe and missing elements:** after the recoveries, we calculated
  `Fe = 100 - sum(other element amounts)` where the existing alloy/source
  review supported an Fe balance and no identified omission remained.
  We kept Fe at zero for the 61 explicitly complete non-Fe alloys. We set
  other unreported elements to zero only in accepted compositions and logged
  them as assumptions. We did not rescale reported amounts. Accepted totals
  lay between 99 and 101 wt.%; a calculated total of 100 was not itself
  evidence that the original data were complete.
- **Unresolved compositions:** we left unknown amounts missing and withheld
  the Fe balance where it could absorb missing alloying elements. We did not
  replace these unknowns with zero, a mean, or a generic grade composition.
  We retained the original rows and recorded the following unresolved cases:

  | Reason | Development rows withheld | Frozen query rows preserved for review |
  |---|---:|---:|
  | No reported element amounts | 95 | 0 |
  | Only sulfur reported (`2014Zhou`) | 87 | 0 |
  | Incomplete Cr/Ni or G3 composition, without a supported balance | 7 | 0 |
  | Missing Re (`1969Boehni`) | 12 | 0 |
  | Missing row-to-Y-amount link (`2018Shi`) | 4 | 0 |
  | Unrecovered source-specific Cu; also Nb for relevant grades (`1994Malik`) | 56 | 10 |
  | Combined Nb + Ta without separate amounts (`1995Malik`) | 10 | 0 |
  | Nb named but not quantified in `430TiNb` (`2000DiSchino`) | 1 | 0 |
  | **Total** | **272** | **10** |

- **Interpretation of exclusions:** the first three groups accounted for the
  original **189 incomplete development compositions**. Their unsupported
  totals remained below 90 wt.%. Additional known omissions required review
  even where the earlier Fe calculation had made the total appear complete.
  For `1994Malik`, we conservatively left the whole source pending because
  it included copper-containing grades but lacked source-specific Cu amounts.
  This did not mean that every row necessarily contained positive Cu. We did
  not transfer chemistry from the different 1995 paper by grade name alone.
- **Frozen split:** we copied the original assignments without resplitting.
  Of 3,222 development rows, **2,950** were accepted and **272** were withheld
  from the new development feature file. All **805 final-test/query rows**
  remained in the query export. Ten remained unresolved: five `316L` rows
  (`soccol_r001215`–`soccol_r001219`) and five `17-4PH` rows
  (`soccol_r001235`–`soccol_r001239`), all from `1994Malik`. We marked this
  export **not ready for final evaluation** while those queries lacked
  verified composition. We did not substitute development rows or produce
  an accepted-only final-query subset.
- **Limits of the EPIT evidence:** Soccol's first Ti description said 1.86,
  while EPIT's numeric Ti said 1.88 and its comment also said 1.86. We used
  Soccol's explicit nominal 1.86 and recorded the disagreement. EPIT also
  recorded zero Re throughout the Re study and split combined Nb + Ta equally
  without independently verified justification. We did not use those entries
  to resolve the Soccol omissions. These findings remained source-review
  caveats for EPIT; this step did not change or certify its entire workbook.
- **Traceability and validation:** we saved each element's original, previous
  and processed amounts, treatment and evidence. We moved superseded balance
  and missingness fields under `legacy_` names. Four focused checks passed:
  reported values and recoveries were correct; unknown amounts were not
  invented; all frozen queries and development fold assignments survived;
  and environment values, targets, source files and input checksums remained
  unchanged. Recovery was unchanged by donor order or altered outcomes, and
  conflicting donor amounts remained unresolved.
- **Pipeline scope:** we saved a new standalone export and preserved earlier
  files. Existing shared CSVs, formula fitting, synthetic generation, training
  and evaluation paths remained unchanged. No models were retrained or evaluated.

We saved the results under
[`datasets/soccol_pitting_potential/processed/soccol_composition_v1/`](datasets/soccol_pitting_potential/processed/soccol_composition_v1/):

- [`soccol_processed.csv`](datasets/soccol_pitting_potential/processed/soccol_composition_v1/soccol_processed.csv)
  retained all 4,027 reference rows, with accepted/unresolved status and legacy values.
- [`soccol_development_features.csv`](datasets/soccol_pitting_potential/processed/soccol_composition_v1/soccol_development_features.csv)
  contained the 2,950 accepted development rows, with 24 populated composition columns.
- [`soccol_frozen_final_query_features.csv`](datasets/soccol_pitting_potential/processed/soccol_composition_v1/soccol_frozen_final_query_features.csv)
  retained all 805 frozen queries, including unresolved compositions.
- [`composition_review_required.csv`](datasets/soccol_pitting_potential/processed/soccol_composition_v1/composition_review_required.csv)
  listed the 282 unresolved rows, reasons and original partitions.
- `composition_processing_log.csv`, `composition_changes.csv` and
  `composition_row_audit.csv` recorded cell decisions and row status.
- `feature_registry.json`, `split_assignments.csv` and `processing_manifest.json`
  recorded units, feature roles, unchanged assignments, readiness and checksums.

We stored source decisions and literature-search results in
[`soccol_composition_rules_v1.json`](datasets/soccol_pitting_potential/soccol_composition_rules_v1.json)
and built the export with
[`prepare_soccol_composition.py`](datasets/soccol_pitting_potential/prepare_soccol_composition.py).
The following command reproduced the export in a new directory:

```bash
.venv/bin/python -m corrosion_datasets.datasets.soccol_pitting_potential.prepare_soccol_composition \
  --output-dir /tmp/soccol_composition_check
```

## EPIT: composition source investigation

- **Scope:** on 2026-10-24, we investigated the three composition concerns
  identified during the Soccol review. We compared the original EPIT workbook,
  including its Excel formulas, with `epit_environment_v3`, the frozen split
  assignments and available primary-source evidence. This was a review of
  these specific cases, not an independent verification of every EPIT row.

  | Case | EPIT row IDs | Development rows | Frozen final-query rows | Finding |
  |---|---|---:|---:|---|
  | Ti disagreement | `epit_r000359` | 1 | 0 | Source table text supported Ti = 1.86 rather than 1.88 wt.% |
  | Re recorded as zero | `epit_r000211`–`epit_r000222` | 0 | 12 | Exact Re amounts and their row mapping remained unresolved |
  | Combined Nb + Ta | `epit_r000051`–`epit_r000055`; `epit_r000061`–`epit_r000065` | 5 | 5 | Separate amounts and combined totals remained unverified |
  | **Total** | **23 rows** | **6** | **17** | |

- **Ti:** EPIT reference 11 contained numeric Ti = 1.88, whereas its alloy
  description said 1.86. The indexed text of Table I on page 575 of
  [Lizlovs and Bond's original paper](https://citeseerx.ist.psu.edu/document?doi=5984cfb24bae9d77f62f3bb3a07ec357368db56a&repid=rep1&type=pdf)
  also reported Ti = 1.86 for the matching titanium-series alloy, with
  C = 0.035 and N = 0.023. This supported a transcription-error interpretation.
  We recorded a proposed correction of **Ti 1.88 → 1.86 wt.%** and the
  corresponding Fe remainder of **77.592 → 77.612 wt.%**. We read the
  search-indexed table text; direct PDF access did not permit visual
  verification of the scan. We did not apply the proposed correction in
  this investigation.
- **Re:** all 12 rows from EPIT reference 81 contained literal numeric
  Re = 0 in the original workbook, rather than blanks converted by our
  preprocessing. The workbook's heat-treatment description mentioned Re,
  and [Böhni and Uhlig's original abstract](https://www.sciencedirect.com/science/article/abs/pii/0010938X69800077)
  described experiments with varying Re additions. We did not recover the
  original composition table or a verified row-to-Re mapping. A zero-Re
  control could have been present, so the evidence did not establish that
  every zero was wrong. All 12 rows belonged to the frozen final-query set.
- **Nb + Ta:** ten rows from EPIT reference 49 already contained equal
  numeric Nb and Ta amounts in the original workbook. The five N08020
  rows had Nb = 2.5 and Ta = 2.5 alongside a comment giving a combined
  5 wt.%; these were frozen queries. The five S31803 rows had Nb = 0.3
  and Ta = 0.3 alongside a combined 0.6 wt.%; these were development rows.
  We found no independent support for dividing either total equally.
  We located [Malik et al.'s 1995 paper](https://www.sciencedirect.com/science/article/pii/0010938X9500043J)
  but could not recover its composition table. A related 1992 author report
  also lacked readable composition-table content in the accessible text.
  We therefore left both the separate amounts and the combined totals
  unverified; we did not substitute a generic composition for either grade.
- **Fe and provenance:** every reviewed row contained the original Excel
  formula `Fe = 100 - sum(other element columns)`. Consequently, a missing
  or incorrect alloying amount would have changed the Fe remainder while
  leaving the total at 100%. The suspect Ti, Re and Nb/Ta entries existed
  upstream of our processing; our parser had not introduced them.
- **Result and limits:** source text supported one proposed correction;
  **22 rows remained unresolved**, including **17 frozen final queries**.
  We recorded this limitation without dropping or replacing any queries.
  We did not change dataset values, targets, frozen assignments or model
  paths. Checksums confirmed that the workbook and all files in the reviewed
  EPIT export remained unchanged.
- **Saved evidence:** we recorded each affected row, its partition, original
  values, Fe formula, evidence status and any proposed values in
  [row_findings.csv](analysis/epit_composition_source_review_v1/row_findings.csv).
  We saved source links, access limitations, counts and input checksums in
  [summary.json](analysis/epit_composition_source_review_v1/summary.json).

## EPIT: composition review decision

- **Agreed treatment:** we corrected the supported Ti discrepancy and kept
  the recorded Re and Nb/Ta amounts. The user accepted the remaining source
  uncertainties so that every compared model could use the same rows and
  values. This decision retained uncertain information; it did not establish
  that those amounts were correct or that their errors affected all models
  equally.
- **Correction:** for `epit_r000359`, we changed **Ti from 1.88 to 1.86 wt.%**
  and its **Fe remainder from 77.592 to 77.612 wt.%**. The total remained
  100 wt.%. We applied these two corrections to both the reference table and
  the feature-only CSV.
- **Re and Nb/Ta:** we retained all 12 Re-study rows with Re = 0 and all
  10 Nb/Ta rows with their existing equal allocations. Their amounts remained
  unverified. We recorded the affected rows in a separate audit file; the
  review status was not added as a model feature. We did not exclude the
  five development rows or the 17 frozen queries involved in these cases.
- **Rows and splits:** we retained **all 760 rows: 608 development rows and
  152 frozen final queries**. We preserved their existing development folds.
  No EPIT query rows had been removed, so this correction required no split
  rebalancing. The decision applied to EPIT; it did not change Soccol's
  separately documented exclusions or unresolved compositions.
- **Export and validation:** we saved the new version in
  [`epit_composition_v2/`](datasets/electrochemical_metrics_alloys/processed/epit_composition_v2/).
  The export inherited all reviewed environment/history features from
  `epit_environment_v3` and retained the same 69 model features. We verified
  that only the two agreed cells changed in each data table. All other values,
  row ordering and frozen assignments were preserved. Earlier exports and
  source files remained unchanged. Existing shared datasets, formula fitting,
  training and evaluation paths were not switched to this export.
- **Records:** we saved the two changes in
  [`composition_review_changes.csv`](datasets/electrochemical_metrics_alloys/processed/epit_composition_v2/composition_review_changes.csv)
  and the 22 retained uncertain rows in
  [`composition_retained_uncertainties.csv`](datasets/electrochemical_metrics_alloys/processed/epit_composition_v2/composition_retained_uncertainties.csv).
  The processing manifest recorded this acceptance policy and file checksums.
  Inherited logs described earlier stages; the new change log recorded this
  additional correction.

We generated the export with
[`review_epit_composition.py`](datasets/electrochemical_metrics_alloys/review_epit_composition.py).
The following command reproduced it in a new directory:

```bash
.venv/bin/python -m corrosion_datasets.datasets.electrochemical_metrics_alloys.review_epit_composition \
  --output-dir /tmp/epit_composition_review_check
```

## EPIT and Soccol: reference-type feature

- **Added column:** we added the categorical model input `reference_type`
  to both standalone reference tables and their full model feature views.
  Its populated values were **`SCE`** and **`AgAgCl`**. Unverified references
  remained empty CSV fields, read as `NaN`; we did not introduce an
  `unknown` category, a numeric code or an assumed reference.
- **Target handling:** we kept every target value and its original column
  unchanged. We applied no electrode conversion. The new column recorded
  the reference category adopted for the reported pitting-potential target,
  following the existing combined-dataset policy. It did not establish a
  reference for preparation voltages such as `Prep_redox`.

  | Table | SCE | AgAgCl | Missing reference | Total rows |
  |---|---:|---:|---:|---:|
  | EPIT, all rows | 760 | 0 | 0 | 760 |
  | Soccol, all reference rows | 565 | 112 | 3,350 | 4,027 |
  | Soccol, accepted development subset | 348 | 112 | 2,490 | 2,950 |
  | Soccol, frozen final queries | 110 | 0 | 695 | 805 |

- **EPIT evidence:** we checked cell `Z1` of the original `Pitting Potential`
  sheet, which explicitly declared `Epit, mV (SCE)`. Its processed average
  target retained this convention, so all 760 rows received `SCE`.
- **Soccol evidence:** we retained the previous 13 source assignments and
  checked their evidence again. Twelve source groups were supported by a
  freshly read original paper or publisher abstract. For `1999Wegrelius`,
  the one Soccol row matched the same paper's EPIT entry in composition,
  test conditions and the 549.6 mV target explicitly labelled SCE. This was
  a dataset cross-check; we did not claim to have independently re-read that
  paper's electrode statement. Eight `1970Wilde` values also matched the
  corresponding SCE-labelled EPIT values, and its original methods confirmed
  SCE. We recorded source URLs, locations and evidence strength separately
  in [source_evidence.csv](analysis/epit_soccol_reference_review_v1/source_evidence.csv).
- **Check against the collection-wide description:** the preserved Soccol
  manuscript described Ag/AgCl (3 M KCl), but that statement did not establish
  a verified conversion of every workbook target. In
  [Alar et al., Table 2](https://www.electrochemsci.org/papers/vol11/110907674.pdf),
  the source targets were explicitly reported in mV/SCE. **64 of the 66**
  corresponding retained Soccol values matched those numbers exactly. The
  two already-known discrepancies remained unchanged: `2016Alar06` contained
  777 versus the paper's 771, and `2016Alar12` contained 911 versus 977.
  This supported retaining the source-specific policy rather than assigning
  all Soccol rows the collection's stated reference. We saved the numerical
  comparison in [alar_numeric_check.csv](analysis/epit_soccol_reference_review_v1/alar_numeric_check.csv).
- **Verification limits:** checking an original paper's electrode did not
  independently verify the transcription or conversion history of every
  workbook target. We kept 3,350 Soccol reference categories missing where
  the existing source audit had not established a label. For the 112
  `2020Melia` rows, the [original methods](https://doi.org/10.1038/s41529-020-00126-5)
  specified Ag/AgCl with **saturated KCl**. We recorded this detail in the
  audit, while using the agreed broad `AgAgCl` feature category. That category
  did not imply that saturated-KCl and 3-M-KCl electrodes were identical.
- **Rows, values and feature counts:** all original predictors, target strings,
  row ordering and frozen assignments were preserved. The existing Soccol
  accepted-development subset and frozen-query subset were unchanged.
  EPIT now contained **70 model features** and Soccol **58**. The added
  measurement feature was separate from their 45 and 33 environment/history
  features. Environment-only historical views retained their original scope.
- **Audit and validation:** each new export included `reference_type_audit.csv`,
  with a row's assigned value, evidence, source URL and verification status.
  Evidence fields were not model features. We checked every populated label
  against the frozen source rules, compared all Soccol labels with the prior
  combined table using workbook row IDs, and verified that unknown labels
  read back as missing. Removing the new column reproduced every prior data
  table exactly. File checksums and split-file equality confirmed that earlier
  exports, source files and memberships were unchanged.
- **Saved versions:** we saved
  [`epit_reference_v1/`](datasets/electrochemical_metrics_alloys/processed/epit_reference_v1/)
  and [`soccol_reference_v1/`](datasets/soccol_pitting_potential/processed/soccol_reference_v1/).
  Existing shared datasets and model paths were not switched automatically.
  Earlier composition caveats were inherited; adding reference labels did
  not resolve or change composition decisions.

We stored the decisions in
[`reference_type_rules_v1.json`](datasets/reference_type_rules_v1.json)
and generated the exports with
[`prepare_reference_types.py`](datasets/prepare_reference_types.py):

```bash
.venv/bin/python -m corrosion_datasets.datasets.prepare_reference_types epit \
  --output-dir /tmp/epit_reference_check
.venv/bin/python -m corrosion_datasets.datasets.prepare_reference_types soccol \
  --output-dir /tmp/soccol_reference_check
```

## EPIT and Soccol: column-name alignment

- **Inputs:** we compared the latest standalone `epit_reference_v1` and
  `soccol_reference_v1` exports. We recorded the mapping before merging rows.
- **Predictors:** EPIT contained 70 predictors and Soccol contained 58.
  **49 names were shared**, including all 24 composition columns. Their
  declared data types and units matched. **21 predictors were EPIT-only and
  9 were Soccol-only**, giving a proposed union of **79 predictors**. We
  retained the existing predictor names; no predictor rename was needed.
  Shared names described schema availability, not complete data coverage:
  Soccol's four second heat-treatment columns existed but were entirely empty.
- **Target name:** we mapped EPIT's `Epit, mV (SCE) Avg.` and Soccol's `E_pit`
  to the proposed shared name `reported_pitting_potential_mV`. We kept this
  target separate from the 79 predictors and retained `reference_type` as
  a predictor. We applied no electrode conversion. EPIT's existing numeric
  modeling convention used 800 for `>800` and 1256 for `>1256`; the original
  notation remained in the standalone reference table. This mapping did
  not change or parse any source target values.
- **Separate meanings:** we kept chemical concentrations separate from ion
  concentrations, including `NaCl_mol_L` versus `CP_Cl` and `H2SO4_mol_L`
  versus `CP_SO4`. We kept molarity, normality, mass percentage and gas
  partial pressure separate. Preparation pH remained separate from test pH;
  roughness remained separate from grit. `Prep_redox` retained its reported
  voltage and its unresolved reference qualification; the target's
  `reference_type` did not supply that reference.
- **Missing data and categories:** the mapping specified missing values
  for environment/history/reference columns absent from a dataset. Existing
  composition decisions and categorical labels were preserved. We introduced
  no new zero filling, phase inference or category collapsing.
- **Exclusions:** `CP_ion_other` remained excluded. Source descriptions, row
  identifiers, literature references and audit fields were not predictors.
- **Outputs and checks:** we saved the full
  [column mapping CSV](analysis/epit_soccol_column_alignment_v1/column_mapping.csv),
  a [readable mapping table](analysis/epit_soccol_column_alignment_v1/column_mapping.md)
  and a [summary with input checksums](analysis/epit_soccol_column_alignment_v1/summary.json).
  We checked that every registered predictor appeared exactly once and that
  all 49 common columns had matching declared types and units. Populated-row
  counts described the full 760-row EPIT and 4,027-row Soccol reference
  tables; they did not change row eligibility. Checksums confirmed that
  source exports and frozen assignments were unchanged. We did not merge
  datasets or switch existing model paths at this step.

## EPIT pipeline: adoption of reviewed compositions

- **Real-data loading:** we connected the standalone EPIT loader and direct
  evaluator to the 24 composition columns in `epit_reference_v1`. We matched
  the source identifiers to the reviewed export and verified the export's
  checksums, row IDs and frozen assignments. We preserved the saved numeric
  values, including the sulfur upper-limit approximations, assumed zeros and
  the corrected Ti/Fe row. Loading performed no new filling, Fe balancing or
  normalization. All 760 compositions were finite and accepted.
- **Old tolerance:** the current path stopped applying the old
  99.9–100.1 wt.% rule. It validated the reviewed 99–101 wt.% acceptance
  interval instead. The compositions totaling 100.52%, 99.8679% and 99.5%
  remained unchanged and became eligible as composition templates. The
  historical raw-workbook path retained its previous rule for reproducibility.
- **Sampling bank:** we created
  [`epit_dataset_v3.csv`](../src/tabicl/prior/assets/epit_dataset_v3.csv) and its
  [metadata](../src/tabicl/prior/assets/epit_dataset_v3.json), containing
  **400 distinct compositions** from the 760 reviewed rows. Explicit zeros
  replaced the former distinction between missing and zero entries, so some
  previously separate templates became identical. The family counts were
  296 Fe, 56 Al, 18 HEA, 17 Ni–Cr–Mo and 13 Other; all were composition-eligible.
  We recorded the source row IDs for every template. We did not normalize
  these stored real compositions. Generated synthetic compositions retained
  the existing perturbation and closure to 100 wt.%.
- **Feature sampler:** we created `epit_fe_nicrmo_features_v2`, retaining the
  earlier environment tuples, category codes and environment imputation
  values while referencing the corrected composition bank. Its existing
  Fe/Ni–Cr–Mo restriction remained: it used 313 eligible distinct compositions
  from those two families. We made the new versions the standalone EPIT
  defaults and included both profile names explicitly in new training commands.
- **Versioning:** old composition and feature assets were preserved. Old
  combined configurations and training commands that explicitly selected the
  historical feature profile retained their implicit `epit_dataset_v2` bank.
  New EPIT study identities included the reviewed export's checksums, so an
  existing study could not silently continue with the changed composition data.
  Future calibration exports recorded the reviewed input identity separately
  from the original workbook and split identity. Existing formula coefficients,
  saved predictions and checkpoints were not refitted or overwritten.
- **Scope:** this was a composition-only pipeline update. We retained the
  historical 28-column layout through a column-name adapter, including the old
  environment fields and method feature. The 70-feature environment schema,
  missing-value treatment and development-only sampling migration remained
  separate work. The new bank retained the historical all-760-row, feature-only
  sampling scope; it did not store or use target values. Formula fitting and
  model selection retained their development-only boundaries. Final-model
  evaluation retained the frozen outer queries.
- **Checks:** we compared every loaded composition cell with the saved export,
  checked sampling-template values against their source rows, checked direct
  evaluation inputs, and exercised the synthetic sampler and training command
  builders. We verified that the same 608 development rows, 152 final queries
  and five development folds were retained. We also checked historical profile
  loading and combined-prior compatibility. We ran no model training or new
  formula calibration.

We generated and checked the new assets with
[`build_epit_reviewed_composition_profile.py`](../scripts/build_epit_reviewed_composition_profile.py):

```bash
.venv/bin/python -m scripts.build_epit_reviewed_composition_profile --check
```

## EPIT pipeline: development-only sampling

- **Partition:** we restricted the new sampling assets to the existing outer
  development partition. We retained the same 608 development rows, 152 frozen
  final queries and five development folds; we created no new split.
- **Composition bank:** we created `epit_dataset_v4`, containing 323 distinct
  reviewed compositions from the 608 development rows. The general composition
  bank retained its five family labels: 241 Fe, 43 Al, 13 HEA, 14 Ni–Cr–Mo and
  12 Other templates. Every template recorded only development source row IDs.
  No final-query composition contributed a template or family probability.
- **Active informed sampling:** we retained the Fe/Ni–Cr–Mo family restriction.
  The selected rows were **411 Fe and 41 Ni–Cr–Mo**, represented by **255 distinct
  compositions**. We created `epit_fe_nicrmo_features_v3` with **452 environment
  tuples** from those same eligible development rows. We preserved each tuple's
  temperature, chloride, pH and test method, and retained the existing independent
  draws of composition and environment. Other development families remained
  outside this active informed sampling path.
- **Sampling statistics:** we fitted numeric imputation means and the method
  vocabulary on the 608 development rows only, preserving the earlier role of
  these statistics across the full real development context. The vocabulary
  contained 50 categories, of which 39 occurred in the eligible Fe/Ni–Cr–Mo
  environment bank. Its missing-value counts were 37 temperatures, zero chloride
  concentrations and 38 pH values. Final-query conditions and categories did not
  contribute to these statistics. Neither development nor final-query target
  values were used to build the sampling assets.
- **Defaults and compatibility:** we switched the standalone EPIT defaults and
  generated training commands to the new profiles. Training commands obtained
  the method-category count from the selected profile. Old profiles and their
  composition-bank pairings remained available for historical runs. Study
  fingerprints continued to record the selected profile identities, preventing
  reuse of an existing study under silently changed sampling inputs.
- **Scope:** we changed only the sampling partition and the statistics derived
  from it. We retained the formulas, their coefficients, the family-selection
  rule and the 28-column layout. We did not activate the expanded environment
  schema or change model missing-value handling. This was outer final-query
  exclusion; the sampling bank used the full development partition rather than
  separate banks for each inner development fold. Frozen final-model evaluation
  still used the frozen outer queries.
- **Validation:** we checked that all composition sources, environment sources
  and preprocessing sources belonged to development. A regression test changed
  every final-query composition, environment value, method and family label,
  plus all target values, and confirmed that the generated assets were identical.
  We also checked reproducible generation, sampler output, new training commands
  and historical profile compatibility. The relevant suite passed 111 tests.
  Checksums confirmed that the previous assets, processed datasets, frozen split
  and saved formula artifacts were unchanged. We ran no model training.

We built the new assets with
[`build_epit_development_profiles.py`](../scripts/build_epit_development_profiles.py)
and checked them with:

```bash
.venv/bin/python -m scripts.build_epit_development_profiles --check
```

## EPIT pipeline: 70-column environment sampling

- **Inputs:** we expanded the default EPIT SCM-plus-formula generator from 28
  inputs to the exact **70-column order** in `epit_reference_v1/feature_registry.json`:
  24 composition features followed by 46 environment/history/measurement features.
  We retained the saved names and units, including separate molar, normal and
  weight-percent solution concentrations. We kept sparse and constant columns.
- **Sampling bank:** we created `epit_fe_nicrmo_features_v4.csv/json`, paired with
  the existing `epit_dataset_v4` composition bank. We used the same **411 Fe and
  41 Ni–Cr–Mo development rows** and the same 255 eligible composition templates.
  We retained the existing composition perturbation and family-selection rules.
- **Related fields:** we sampled all 46 non-composition fields together from
  one development row. This kept solution identity, reported concentrations,
  temperature, pH, preparation, heat treatment, phase flags and missingness
  together. We retained the earlier independent draws of composition and
  environment; we did not introduce a new composition/environment matching rule.
- **Corrected environment:** we read the reviewed columns rather than the old
  workbook environment fields. In particular, the contradictory chloride value
  for `epit_r000148` remained missing. The eligible sampling bank contained
  37 missing temperatures, one missing chloride value and 38 missing pH values.
- **Categories:** we fitted the nine categorical vocabularies on the 608 outer
  development rows only. Synthetic sampling and expanded real-data evaluation
  used the same codes. Unknown or missing categories remained missing; final
  queries did not contribute categories or statistics.
- **Missing values:** we preserved unknown environment values as `NaN` in the
  synthetic model inputs and enabled the model's learned missing-value embedding.
  We retained the exact missingness pattern of the sampled environment row.
  We added no independent missingness augmentation. Composition inputs remained
  complete, including their accepted zeros. Entirely unknown environment columns,
  including `WO4_mol_L` and `CP_SO4` in development, remained missing model inputs.
- **Internal target calculation:** the SCM and formulas required finite values.
  For those calculations only, we filled missing numeric fields with development
  means and categorical fields with development modes. An entirely unobserved
  field used a zero placeholder internally. We restored the observed missingness
  before passing inputs to the model; these placeholders did not become reported
  measurements or visible replacements for missing cells.
- **Formulas:** we retained the existing nine formula definitions, coefficients,
  Fe/Ni–Cr–Mo handling and SCM/formula mixing procedure. The SCM received all 70
  fields. The formulas received a separate projection containing composition,
  temperature, chloride, pH and the original auxiliary test-method code.
  The raw test-method field was **not** included among the 70 model inputs.
  We fitted no new formula coefficients during this update.
- **Training and evaluation:** we connected the new schema to the default EPIT
  Optuna and final-training commands. They selected 70 inputs and enabled the
  missing-value embedding. Optional Magpie descriptors increased the input count
  to 80. Generated evaluation commands used `--epit-expanded-features` and the
  matching missing-aware model setting. Final-model evaluation retained all
  **152 frozen outer queries**; development folds remained for model selection.
- **Versioning:** we used the new default study name
  `epit_pipeline_optuna_empirical_features_scm_target_v9`. Study fingerprints
  recorded the expanded schema and profile identity, and evaluation checks
  rejected a mismatched schema. We preserved earlier assets and explicit
  legacy 28-column paths, including the previous EPIT-only empirical mode.
  The existing combined and Soccol workflows were not migrated by this update.
- **Validation:** 262 relevant regression tests passed, including all nine
  formula projections, both SCM generators, optional Magpie descriptors,
  frozen-query selection and historical Soccol/combined compatibility. One
  small CPU training step verified that the sampled missingness reached the
  model and produced finite training results. A mutation test changed all
  final-query features and all target values without changing the sampling
  assets. We verified reproducible asset generation. No full model training or
  Optuna search was run.

We built the new profile with
[`build_epit_expanded_feature_profile.py`](../scripts/build_epit_expanded_feature_profile.py)
and checked it with:

```bash
.venv/bin/python -m scripts.build_epit_expanded_feature_profile --check
```

## EPIT pipeline: formula refit on reviewed inputs

- **Input source:** we connected formula fitting to `epit_reference_v1` through
  `scripts/epit_pipeline/formula_data.py`. We used the saved composition values
  and mapped `CP_temp`, `CP_Cl` and `CP_pH` to the formulas' existing temperature,
  chloride and pH inputs. The auxiliary test-method value also came from the
  reviewed export. The historical loader and existing sampling assets were retained.
- **Missing values:** we preserved missing environment values during loading.
  Within each development fold, each formula fitted its filling values and
  scaling from its applicable context rows only. Validation rows and final
  queries did not contribute these statistics. We retained complete composition
  values, including zeros, without normalization.
- **Corrections adopted:** relative to the historical loader, the input audit
  recorded 21 changed numeric cells across six rows. The environment changes
  were two chloride values that the reviewed dataset had left missing:
  `epit_r000148` (Fe) and `epit_r000756` (Al). Only the Fe row belonged to the
  nine formulas' fitting population. Temperature and pH values were unchanged.
  The audit also recorded the existing reviewed composition corrections.
- **Rules and fitting:** we retained the same nine formula definitions, term
  order, coefficient anchors, bounds, regularization and alloy applicability.
  We refitted each formula on the context portion of each of the same five
  development folds, and scored its held-out development rows. We then fitted
  its final coefficient center on all **452 eligible development rows: 411 Fe
  and 41 Ni–Cr–Mo**. We masked all **152 frozen final-query targets** before
  calibration. No final-query values were used for fitting or scoring.
- **Outputs:** we saved the new coefficient centers, fold-specific coefficients,
  preprocessing states, scores, predictions and HTML report in
  [`target_rules_v4/`](analysis/epit_pipeline/target_rules_v4/).
  We retained `target_rules_v2` and `target_rules_v3` unchanged. Each new artifact
  recorded the reviewed input identity and frozen split hashes.
- **Pipeline defaults:** Python and Slurm EPIT launchers were connected to
  `target_rules_v4` and the v9 study. The generated training command passed the
  refitted coefficients and new formula scores to the prior. Existing checks
  continued to reject mixing incompatible studies or altered artifacts.
- **Results:** the average Spearman scores across the five development folds
  changed only slightly. The largest absolute change was **0.0034**.
  `method_aware_pren_n` remained the strongest of these nine formulas, at
  **0.5908**. These were formula scores on development folds, not final-model
  evaluation scores.

| Formula | Previous mean fold Spearman | Reviewed-input refit |
| --- | ---: | ---: |
| `pren_n_linear` | 0.5061 | 0.5060 |
| `cr_mow_n_synergy` | 0.5335 | 0.5339 |
| `threshold_saturation` | 0.5122 | 0.5101 |
| `pren_n_improved_environment` | 0.5517 | 0.5505 |
| `mo_n_acid_repassivation` | 0.5514 | 0.5494 |
| `mns_inclusion_penalty` | 0.5695 | 0.5708 |
| `coupled_breakdown` | 0.5313 | 0.5340 |
| `fe_ni_cr_threshold` | 0.5246 | 0.5212 |
| `method_aware_pren_n` | 0.5904 | 0.5908 |

- **Validation:** 164 relevant tests passed. For each of the nine formulas,
  changing every final-query feature and removing its target left fitted
  coefficients and development predictions unchanged. We checked context-only
  filling values, unchanged formula definitions, artifact loading and prior
  integration. The three updated Slurm launchers passed shell syntax checks.
  No Optuna search or full model training was run.

We ran the refit with:

```bash
.venv/bin/python -m scripts.epit_pipeline.calibrate_target_rules
```

The command refused to overwrite existing outputs without an explicit `--force`.
The before/after scores were saved in
[`comparison_vs_v3.csv`](analysis/epit_pipeline/target_rules_v4/comparison_vs_v3.csv),
and the changed input cells in
[`formula_input_changes_vs_legacy.csv`](analysis/epit_pipeline/target_rules_v4/formula_input_changes_vs_legacy.csv).

## EPIT pipeline: fixed informed share and broader Optuna ranges

- **Informed share:** we fixed the current EPIT Optuna search at **100% informed**
  and removed `informed_prior_ratio` from its sampled parameters. We recorded
  the fixed value in the study fingerprint and generated training commands.
- **Search ranges:** we retained the full **0–1** formula contribution range,
  changed the MLP share from discrete choices to continuous **0–1**, and widened
  composition perturbation strength from **0–0.15 to 0–0.30**. The tree share
  remained one minus the MLP share. Perturbation strength remained the existing
  sampler parameter, rather than a percentage-point change in composition.
- **Versioning:** we changed the Python and Slurm defaults to
  `epit_pipeline_optuna_empirical_features_scm_target_v10`. The fingerprint
  recorded the revised ranges and rejected incompatible study reuse. Earlier
  studies, Soccol settings and explicit historical EPIT modes were retained.
- **Final training:** we reconstructed the selected configuration using the
  study's saved fixed ratio. We preserved the final-training override for the
  planned **25%, 50%, 75% and 100%** informed comparison, without changing the
  other selected settings. Overrides were recorded separately from the selected
  trial and received distinct default output paths.
- **Data and evaluation:** we retained the reviewed 70-column inputs, nine
  formulas in `target_rules_v4`, 50-trial default and 1,000-step trial budget.
  Selection continued to use the same five development folds. Final-model
  evaluation continued to use the **152 frozen outer queries**.
- **Validation:** 106 relevant tests passed, including actual Optuna parameter
  sampling, saved-trial reconstruction, the four final-training ratio overrides,
  incompatible-study rejection and Soccol compatibility. Two preview trials
  generated commands at 100% informed, including a perturbation strength above
  the previous 0.15 limit. The three launchers passed shell syntax checks. No
  Optuna training or full model training was run.

## Soccol pipeline: reviewed-data loader

- **Input source:** we added `scripts/soccol_pipeline/data.py` to read
  `soccol_reference_v1` directly. It loaded all 58 registered features,
  including the shared 24 composition columns, and retained the reported
  `E_pit` target without an electrode conversion. The reviewed route no
  longer read model inputs from `soccol_regression_event1.csv`.
- **Values and missingness:** we preserved saved numeric values, accepted
  zeros, recovered element amounts and unresolved `NaN` values. Loading did
  not calculate a new Fe balance, normalize a real composition, replace
  missing elements or ions with zero, or fit imputation values. Category
  labels remained as saved for the later encoding step.
- **Identity and checks:** we verified file checksums, the registered feature
  order, wt.% units, agreement between feature and reference tables, and
  composition acceptance against the saved audit. We matched rows by their
  original workbook-derived IDs and returned them in frozen task order.
  Duplicate or missing IDs, changed partitions and altered files caused errors.
- **Rows:** we retained all 4,027 rows and exposed the saved acceptance and
  partition flags. The loader identified 2,950 accepted development rows,
  272 unresolved development rows and all 805 frozen queries, including ten
  unresolved compositions. We did not filter rows or revise the frozen split
  during this step.
- **Integration and scope:** we connected an explicit
  `--soccol-reviewed-data` route in the general evaluator for task inspection.
  It rejected model evaluation until the accepted-row filter, training schema
  and missing-value handling were connected, preventing the old 37-column
  imputer from filling or discarding reviewed inputs. Historical Optuna and
  checkpoint evaluation retained their original route. We did not refit
  formulas, rebuild synthetic assets, start training or change dataset files.
- **Validation:** 47 relevant tests passed. They checked exact saved values,
  known composition recoveries, retained zeros and NaNs, row reordering by
  frozen IDs, rejection of altered files or partitions, all 805 final queries,
  and historical Soccol/EPIT compatibility. The reviewed loader and frozen
  query selection succeeded without reading the old source CSV. The documented
  task-inspection command also ran successfully without model evaluation.

## Soccol pipeline: shared accepted-development selection

- **One selection:** we added `select_accepted_development()` to the reviewed
  loader. It selected rows whose saved composition status was accepted and
  whose frozen outer partition was development. It verified the saved
  eligibility flags and preserved original row IDs, source task indices and
  fold labels. It returned **2,950 rows**, withholding the same **272**
  unresolved development rows. No new split or source-data edit was made.
- **Formula inputs:** `formula_data.py` and the comparison driver's explicit
  `--reviewed-data` path used this selection before fitting any statistics or
  coefficients. No final-query row reached formula fitting. Saved composition
  values remained complete and unchanged. Missing numeric/ion values received
  context-only means for internal formula calculations; entirely unobserved
  fields used an internal zero placeholder. Source inputs retained their NaNs,
  and `CP_ion_other` remained excluded. Formula definitions were unchanged;
  we did not run coefficient calibration during this step.
- **Sampling preparation:** `load_reviewed_feature_source()` returned the same
  accepted rows as target-free source data, retaining all 58 registered fields,
  missing values and original IDs. It did not use any target values. This
  selection preceded the later sampling-bank rebuild and any fitted sampling
  statistics. The existing historical 37-column bank was retained.
- **Optuna folds:** the reviewed evaluator used only accepted rows for both
  labeled context and validation. Validation counts were **622, 615, 638,
  445 and 630**; corresponding context counts were **2,328, 2,335, 2,312,
  2,505 and 2,320**. The fold wrapper forwarded the reviewed-data flag and
  recorded the selection identity in its reviewed output metadata.
- **Final-query boundary:** reviewed final-test selection used the 2,950
  accepted development rows as context and preserved all **805 frozen queries**,
  including the ten unresolved compositions. Context-size diagnostics could
  reduce the accepted context to a saved fold while retaining all 805 queries.
  Preprocessing-row selection also excluded withheld development rows.
- **Versioning and scope:** formula and sampling input records shared a
  checksum over selected row IDs, original indices and fold labels. The
  reviewed formula comparison used a separate output default. Historical
  pipelines retained their original inputs and row counts. Reviewed model
  execution remained blocked pending the training-schema and missing-value
  migration; no Optuna run or full model training was started.
- **Validation:** 108 relevant tests passed. They checked exact saved inputs,
  identical selected rows across formula and sampling sources, original fold
  memberships, exclusion of withheld and final-query rows from development,
  retention of all 805 final queries, and historical Soccol/EPIT compatibility.
  Formula, sampling and evaluator inspection commands also ran successfully
  without coefficient fitting or model evaluation.

## Soccol pipeline: development-only synthetic sampling bank

- **Bank:** we built `src/tabicl/prior/assets/soccol_pitting_features_v2.csv`
  and its JSON metadata from the same **2,950 accepted development rows**.
  The composition bank, joint environment/history bank, category mappings and
  internal filling statistics excluded all **805 final queries** and **272
  withheld development rows**. No target value was used.
- **Material scope and weights:** we retained every accepted Soccol material
  without applying EPIT's Fe/Ni–Cr–Mo restriction. Each accepted observation
  remained one equally weighted bank entry; repeated compositions retained
  their source frequency. These entries contained **349 distinct compositions**.
  Composition and environment/history entries were sampled independently,
  with replacement, as in the original Soccol sampler.
- **Saved values:** we preserved the 24 corrected composition columns exactly,
  including reported zeros and accepted totals slightly different from 100%.
  Only generated compositions received the existing positive-entry
  multiplicative perturbation and closure to 100 wt.%; their zeros stayed zero.
- **Environment and missingness:** the bank retained all 34 remaining reviewed
  fields. We sampled their values and missingness together from one development
  observation. We encoded categories using sorted accepted-development labels
  and retained unknown entries as NaN. Means for numeric fields and modes for
  categories were fitted only on accepted development observations and stored
  separately for internal calculations. Entirely missing fields used zero only
  in that internal representation; model-facing bank values remained missing.
- **Identity and validation:** metadata recorded original row IDs, task indices,
  unchanged fold labels, the shared selection checksum, input hashes, missing
  counts and fitted statistics. The loader checked the bank against these
  records. Rebuilding with `--reviewed-data --check` required identical bytes.
- **Scope:** the loader and sampler supported the new 58-column bank explicitly.
  We retained the historical `v1` assets and existing 37-column wrappers for
  earlier runs. Switching training and evaluation to the reviewed schema
  remained a separate migration step. We did not refit target formulas or run
  Optuna or full model training.
- **Checks:** 122 relevant tests passed. Altering final-query features,
  withheld-development features or any targets left the generated bank and
  its statistics unchanged. Tests also checked exact source compositions,
  unchanged observation weights, joint environment/missingness sampling,
  preserved zeros under perturbation, rejection of inconsistent metadata,
  and historical Soccol/EPIT compatibility. The CLI rebuild check passed.

## Soccol pipeline: reviewed 58-column training and evaluation schema

- **Shared schema:** we connected the exact saved names, order and types to
  the reviewed loader, synthetic sampler, training runtime and real-data
  evaluator. The inputs contained **24 composition columns and 34 other
  fields**, with **49 numeric fields and nine categories** overall. The old
  `Prep_medium`, `CP_anions_info` and `CP_ion_other` predictors were excluded;
  their accepted structured replacements remained in the reviewed schema.
- **Training:** the reviewed path used `soccol_pitting_features_v2`, exactly
  58 features and a 24-column material/34-column context allocation. Category
  roles followed the saved column positions. Both MLP and tree SCM targets
  received all 58 fields. The nine existing Soccol formulas read their
  original named inputs from the new order; their calculations and
  coefficients were not refitted during this step.
- **Missing values and categories:** real inputs used the sampling bank's
  accepted-development category mappings. Query-only categories stayed
  missing. Saved numeric values, zeros and NaNs were preserved. Synthetic
  target calculations used the bank's finite internal representation, then
  restored the sampled row's missingness before model input. Reviewed
  training required the missing-value embedding and did not require a
  separate masking profile. All-empty columns retained their positions.
- **Evaluation:** we enabled model evaluation through `--soccol-reviewed-data`
  and connected the development-fold wrapper to matching fixed-column and
  missing-aware settings. CatBoost's default reviewed route identified the
  nine categorical columns explicitly. The same 2,950 accepted development
  rows and all 805 frozen final queries remained in their original partitions.
  Ten unresolved final-query compositions remained missing and were retained.
- **Versioning:** the explicit reviewed search route generated 58-column
  commands, recorded the schema and sampling-bank identity, and used separate
  `optuna_v2` defaults. It required a matching reviewed `target_rules_v2`
  calibration and rejected the historical fit. Existing historical sbatch and
  frozen-model wrappers remained reproducible. The formula refit and Optuna
  search-policy update remained pending; no study or full model training ran.
- **Validation:** 139 tests passed. They checked all nine formulas against
  the same values in their historical order, both SCM types, real/synthetic
  encoding equality, retained missingness, frozen query membership and
  historical EPIT/Soccol compatibility. A tiny CPU training step and local
  checkpoint prediction completed with all 58 columns and finite outputs.
  This smoke test used development data only and did not produce final-model
  evaluation scores. NumPy emitted warnings for all-empty columns; predictions
  remained finite and the tests verified that the columns and NaNs survived.

## Soccol pipeline: joint environment sampling verification

- We confirmed that the reviewed bank and runtime already sampled all **34
  non-composition fields** from one accepted development observation. Solution
  information, concentrations, preparation, heat treatment, reference type and
  their missing values stayed together. Fields were not drawn separately.
- We continued drawing composition independently from the same 2,950-row
  accepted development pool, as in EPIT. Independent draws could select the
  same observation by chance; they were not forced to match or differ.
- The finite internal representation used the same selected environment row.
  After target calculations and standardization, training restored that row's
  original missing-value pattern. No additional environment draw was made.
- Seven existing focused checks passed for paired fields, unchanged zeros,
  missingness through both SCM types and EPIT alignment. We clarified the
  sampler documentation; its sampling behavior and bank files were unchanged.

## Soccol pipeline: category codes and missing-value handling verification

- We confirmed that all **nine categorical mappings** were fitted on the
  **2,950 accepted development rows** and stored in the reviewed sampling bank.
  Synthetic samples and real-data evaluation reused those exact mappings;
  evaluation did not refit them on its context or query rows.
- Missing and unseen category labels remained NaN. A known category assigned
  code `0` remained a valid observed value. The entirely unreported
  `heat_treatment_2_type` and `heat_treatment_2_cooling` fields retained empty
  mappings and missing values; no category was invented for them.
- Environment measurements, including ions, retained their saved zeros and
  NaNs. Finite development means/category modes were used only in the internal
  representation for synthetic calculations. The original joint missingness
  was restored before model input, using the same procedure as reviewed EPIT.
- Reviewed training enabled the learned missing-value embedding and rejected
  configurations that disabled it. Checkpoint loading retained this mode for
  inference, so reviewed-model inputs were not mean-imputed. The historical
  37-column preprocessing remained separate for reproducing earlier runs.
- We expanded the evaluation check to cover known, unseen and missing values
  for every categorical field. **16 focused tests passed**, including
  development-only mapping/statistic checks, learned-vector gradient checks,
  both SCM types and a tiny CPU training/checkpoint prediction check. The
  existing all-empty-column NumPy warnings recurred; outputs remained finite
  and missing columns retained their positions. No sampling-bank or dataset
  values were changed during this verification.

## Soccol pipeline: reviewed nine-formula refit

- **Inputs and scope:** we refitted the same nine Soccol formulas using all
  **2,950 accepted development rows**. We excluded the **272 withheld
  development rows** and all **805 frozen final queries** before calibration.
  The original row IDs and five fold assignments were preserved.
- **Fitting:** each fold fitted numeric/ion filling values, all scaling values
  and formula coefficients on its context rows only. Validation sizes remained
  **622, 615, 638, 445 and 630**; their context sizes remained **2,328, 2,335,
  2,312, 2,505 and 2,320**. Held-out rows were transformed using the saved
  context state. No filling values or coefficients were fitted on validation
  rows for that fold.
- **Unchanged definitions:** we retained all nine formulas, their named inputs,
  coefficient anchors and upper bounds. We kept the nonnegative, sum-to-one
  SLSQP fit against standardized target ranks and anchor strength **0.1**.
  The 58-column training implementation read each formula's existing named
  inputs correctly. No extra terms, family restrictions or target conversions
  were introduced.
- **Final coefficient centers:** after the five-fold evaluations, we fitted
  each formula once on all accepted development rows for synthetic training.
  We saved these coefficients separately from each fold's fitted coefficients,
  filling values and scaling state. Real inputs retained their saved NaNs;
  filling remained internal to formula calculations.
- **Saved artifacts:** we wrote the new version to
  `analysis/soccol_pipeline/target_rules_v2/`, including nine rule JSON files,
  `calibration_summary.json`, `rule_scores.csv`, `fold_metrics.csv`, and
  `oof_predictions.csv`. These recorded input hashes, accepted row IDs, frozen
  folds, per-fold preprocessing and coefficients, held-out predictions, scores
  and score-based rule probabilities. The historical `target_rules_v1` files
  were retained unchanged.
- **Results:** mean development-fold Spearman ranged from **0.431203** to
  **0.571158**. The highest score belonged to
  `pren_n_coupled_mns_weak_anions`; `pren_n_coupled_mns` scored **0.560894**.
  We retained all nine formulas. These were development formula scores, not
  final-model scores; the old and new calibrations used different row sets.
- **Training connection:** the reviewed Optuna loader accepted the new artifact
  version and checked the corrected input identity, 2,950-row count, unchanged
  folds, complete nine-rule inventory, coefficient terms/bounds and checksums.
  It rejected historical or mismatched calibrations. We corrected only
  floating-point excess beyond a coefficient bound after normalization
  (`0.15000000000000002` became `0.15`); the bound was not relaxed.
- **Validation:** 103 relevant tests passed. We reproduced all 45 held-out
  fold predictions and scores from saved states and coefficients, verified
  context-only preprocessing, and confirmed that changing a validation fold's
  features/targets did not change that fold's fitted state or coefficients.
  All nine formulas agreed with the named training implementation. The saved
  coefficients also passed a tiny CPU training step and local checkpoint
  prediction check, with historical EPIT/Soccol compatibility retained. No
  Optuna study or full model training was started.

## Soccol pipeline: Optuna settings aligned with reviewed EPIT

- **Search settings:** we fixed the informed ratio at **100%** and removed it
  from Optuna's searched parameters. We used continuous ranges of **0–1** for
  the MLP share, **0–1** for formula contribution and **0–0.30** for composition
  perturbation strength. The tree SCM received the remaining share after the
  MLP. The reviewed EPIT and Soccol searches used the same range definitions.
- **Retained settings:** we kept **50 trials per worker**, **10 startup
  trials**, **1,000 training steps**, a **10,000-step scheduler**, all **five
  frozen development folds** and **eight evaluation estimators**. Mean
  development-fold Spearman remained the objective. Coefficient variation
  remained **0.8**, with seed **42**.
- **Reviewed inputs:** we made the reviewed route the default for the Soccol
  search command and selected it explicitly in the sbatch launcher. It used
  the **58-column** `soccol_pitting_features_v2` bank, the same **2,950 accepted
  development rows** and the refitted nine-rule `target_rules_v2` calibration.
  We kept Soccol's own column order, joint environment sampling, category
  mappings and missing-value embedding. Sharing EPIT's search policy did not
  replace Soccol's schema with EPIT's 70-column schema.
- **Study separation:** we selected
  `soccol_pipeline_optuna_empirical_features_scm_target_v3` and separate
  `analysis/soccol_pipeline/optuna_v3/` outputs. The study fingerprint recorded
  the fixed ratio, broader ranges, data/schema and formula identities and
  rejected incompatible study reuse. The journal retained its shared
  cross-node storage location, under the new study directory.
- **Historical compatibility:** we retained the old search through
  `--legacy-data`. Existing final-training and frozen-model wrappers continued
  to select the historical v1 route explicitly; migration of those wrappers
  to the new selected trial remained a separate step. No final-test rows were
  used as Optuna evaluation queries.
- **Validation:** **79 focused tests passed**, including actual Optuna
  parameter distributions, range endpoints, fixed-ratio reconstruction,
  58-column training/evaluation commands, incompatible-study rejection and
  existing EPIT/Soccol behavior. Two command-only previews used the reviewed
  data and calibration; one sampled perturbation **0.203**, above the former
  **0.15** limit. The launcher passed shell syntax checking. No Optuna training
  job was launched and no final-test scores were produced.

## Soccol pipeline: reviewed final training and frozen-query evaluation

- **Trial selection:** we removed the launcher's default selection of old
  trial **19**. Final training selected the best completed trial from
  `soccol_pipeline_optuna_empirical_features_scm_target_v3` automatically.
  It checked the selected trial's saved results, study fingerprint, reviewed
  sampling bank and `target_rules_v2` calibration before training.
- **Training and checkpoint selection:** we retained **10,000 training
  steps** and selected the final checkpoint using only the five accepted
  development folds. The final workflow used the reviewed **58-column**
  schema, shared category mappings and learned missing-value embedding.
  No frozen final-query observations were used to choose the checkpoint.
- **Informed-ratio overrides:** we recovered the selected trial's fixed
  **100%** ratio from the study fingerprint. We preserved later **25%, 50%,
  75% and 100%** final-training overrides. Other selected parameters remained
  unchanged, and override runs retained separate names and output locations
  by default.
- **Final evaluation:** we supplied all **2,950 accepted development rows**
  as labeled context and retained the exact **805 frozen final queries**,
  including the ten queries with unresolved compositions. We recorded the
  original **3,222-row** development count separately from the accepted
  context count. The frozen model manifest stored accepted context IDs,
  original final-query IDs and reviewed file identities; result validation
  checked actual row membership as well as row counts.
- **Optional context diagnostics:** we retained the ability to use one
  accepted development fold as context: **622, 615, 638, 445 or 630** rows.
  Each evaluation still used the same **805 outer final queries**. Development
  folds were not substituted for final-model queries.
- **Fold-result correction:** we corrected reviewed Soccol fold summaries to
  record the processed CSV's hash, matching the refitted formula artifacts.
  They had still recorded the original CSV's hash, which would have caused
  valid reviewed results to fail later identity checks. We also required the
  accepted-development selection in Optuna and checkpoint-selection results.
  Historical EPIT and Soccol file identities remained unchanged.
- **Outputs and compatibility:** we moved reviewed final outputs to
  `analysis/soccol_pipeline/final_v2/` and
  `checkpoints/soccol_pipeline_final_v2/`. The final-training and evaluation
  launchers selected the reviewed route explicitly. Historical final workflows
  remained available through `--legacy-data`; the old checkpoint-comparison
  script retained its historical v1 behavior.
- **Validation:** **96 relevant tests passed**. A complete simulated workflow
  exercised study selection, real artifact/hash checks, development-only
  checkpoint selection, manifest freezing, full-context evaluation and all
  five reduced-context evaluations. It retained every original query and
  rejected changed membership even when row counts stayed equal. All four
  informed-ratio overrides passed. Training jobs, checkpoint bytes and
  prediction scores in that integration test were artificial fixtures; no
  real training or final-test prediction job was launched. Both launchers
  passed shell syntax checks.

## Soccol pipeline: reviewed profile fixed as the default

- We changed the profile loader and feature sampler to default explicitly to
  **`soccol_pitting_features_v2`**, containing **2,950 accepted development
  rows and 58 features**. Missing v2 files caused an error; the loader did not
  fall back to historical v1 data.
- We changed the sampling-asset builder's default route to the reviewed v2
  bank. `--check` and `--inspect-inputs` worked without an extra data-selection
  flag. `--reviewed-data` remained an explicit equivalent, and `--legacy-data`
  was required to run historical v1 generation. Historical builder functions
  received explicit `legacy` names.
- We retained explicit v1 loading for old 37-column models and combined-prior
  configurations. The historical schema's identifiers continued to describe
  v1; current loading and sampling defaults came from the reviewed profile.
  Current Optuna and final-training launchers already selected v2 explicitly.
- We preserved all frozen dataset, split, bank and formula files. Rebuilding
  the reviewed bank in a temporary directory produced identical bytes, and
  the builder continued to refuse overwriting existing versioned files.
- **Validation:** **70 relevant tests passed**, covering default 58-column
  sampling, retained missingness, no fallback when v2 files were missing,
  default asset generation/checking and explicit historical compatibility.
  The default builder check passed, and all 18 files staged during the preceding
  Git step remained unchanged.

## Soccol pipeline: before/after input audit

- **Scope:** we compared `soccol_regression_event1.csv` with the reviewed
  `soccol_reference_v1` export in frozen row order. We saved every changed
  common feature cell, its original and reviewed values, row ID, partition,
  development eligibility and formula dependencies in
  [`input_audit_v1/`](analysis/soccol_pipeline/input_audit_v1/). We compared
  preprocessing inputs only; we did not refit coefficients or calculate
  prediction scores.
- **Rows and targets:** all **4,027 rows**, their target values and frozen
  assignments were unchanged. The reviewed workflow used **2,950 accepted
  development rows**, withheld **272 development rows**, and retained all
  **805 final queries**. Ten final queries still had unresolved composition
  cells. Accepted development compositions were complete, including their
  saved zeros.
- **Schema changes:** the model schema changed from **37 to 58 columns**:
  **34 retained, 24 added and 3 removed**. The removed model fields were
  `CP_ion_other`, `Prep_medium` and `CP_anions_info`; supported information
  from the latter two had been extracted into separate fields. Added columns
  were counted separately from corrections to existing columns.
- **Changed common cells:** across all rows, we recorded **4,554 changes in
  1,231 rows**. Within accepted development only, we recorded **1,015 changes
  in 797 rows**. These comprised **71 corrected numeric composition amounts**,
  **299 values deliberately left missing after review**, and **645 category
  representation changes**. Aeration renaming and agitation `yes/no` to
  `1/0` did not change the reported experimental conditions.

| Common feature | Changed accepted-development cells | Change |
| --- | ---: | --- |
| `Fe` | 60 | Corrected balance after recovering other elements |
| `Nb` | 9 | Recovered reported amount |
| `Ti` | 2 | Recovered reported amount |
| `Prep_time` | 152 | Removed air-exposure durations from the preparation-treatment field |
| `Test_area_cm2` | 15 | Left conflicting reports missing |
| `CP_SO4` | 104 | Left conflicting reports missing |
| `CP_CO3` | 22 | Left conflicting reports missing |
| `CP_OH` | 6 | Left conflicting reports missing |
| `CP_aeration` | 252 | Renamed `natural` to `naturally_aerated` |
| `CP_agitation` | 393 | Encoded explicit `yes/no` as `1/0` |

- **Added composition fields:** **20 accepted development rows** contained
  positive `W`, and **38** contained positive `Cu`. The other nine newly
  introduced element columns were zero throughout accepted development.
  These additions were outside the common-cell correction counts above.
- **Raw formula inputs:** among fields actually used by the existing nine
  formulas, only **104 sulfate cells** changed in accepted development.
  `Cr`, `Mo`, `N`, `Mn`, `S`, temperature, pH, chloride, bromide and nitrate
  raw values were unchanged on those rows. The formula-adapter audit also
  recorded corrections to loaded but unused fields, identifying actual
  formula dependencies separately.
- **Internal formula filling:** we compared each accepted validation row
  using only its fold's context. We also evaluated the old filling procedure
  on the same accepted context rows, separating the effect of row removal
  from preprocessing changes. The **104 sulfate values** used context means
  internally after their disputed values were removed. **Three already-missing
  nitrate values** changed from internal zero filling to context means.
  **51 missing pH values and 33 missing temperature values** received different
  internal means solely because the accepted context was smaller. Model-facing
  unknowns remained NaN. These counts described inputs before formula scaling;
  they did not measure all effects of changed scaling or coefficients.
- **Consistency within Soccol:** all **19 audit checks passed**. The reviewed
  loader, formula adapter and saved sampling bank agreed on accepted rows,
  feature values and missingness. All nine saved formula artifacts used the
  same accepted selection and context-specific filling values.
- **Comparison with EPIT:** all **49 shared feature names** had matching
  types and units. Both reviewed workflows preserved missing model inputs
  and used context means internally for missing formula inputs. Their full
  schemas remained **58 Soccol versus 70 EPIT columns**; category codes were
  local to each specialist, rather than universal codes for a future merge.
- **Formula difference identified:** Soccol's formula named
  `cr_mow_n_synergy` used `Mo` without `W`, whereas EPIT's corresponding
  formula used `Mo + 0.5*W`. Soccol's nine analytical formulas did not use
  `W`, `Cu`, `Fe`, `Nb` or `Ti`. Consequently, the composition corrections
  reached the model and SCM but did not directly change these formula terms.
  We documented this existing difference and preserved the approved formula
  definitions; the audit did not establish its effect on prediction scores.
- **Validation:** **44 tests passed**, including checks that the reported
  formula dependencies matched actual calculations, missing values were
  distinguished from zeros, and final-query or validation-row changes did
  not alter that fold's context filling statistics. No dataset, bank,
  formula coefficient or frozen split file was changed by this audit.

We generated the audit with:

```bash
.venv/bin/python -m scripts.soccol_pipeline.audit_input_changes
```

The command refused to overwrite a nonempty output directory. A repeat audit
could use a new `--output-dir`. The saved
[`README.md`](analysis/soccol_pipeline/input_audit_v1/README.md) described each
CSV; `summary.json` recorded the input/output hashes and consistency checks.

## Soccol pipeline: previous versus reviewed formula scores

- **Saved results:** we found the previous nine promoted formula artifacts
  in `analysis/soccol_pipeline/target_rules_v1/` and their development
  predictions in `analysis/soccol_target_rules_v1/oof_predictions.csv`.
  We compared them with `soccol_pipeline/target_rules_v2/`, without running
  training or refitting coefficients. We saved the comparison in
  [`target_rule_comparison_v1/`](analysis/soccol_pipeline/target_rule_comparison_v1/).
- **Matching rows:** the previous results covered **3,222 development rows**,
  whereas the reviewed results covered **2,950 accepted development rows**.
  We therefore also re-scored the saved old predictions on exactly the same
  **2,950 validation rows**, retaining their frozen fold assignments. Every
  score below was the unweighted mean of five development-fold Spearman
  correlations. All **805 frozen final queries** were excluded.

| Formula | Old: 3,222 rows | Old: same 2,950 rows | Reviewed: 2,950 rows | Change on same rows |
| --- | ---: | ---: | ---: | ---: |
| `pren_n_linear` | 0.463315 | 0.441652 | 0.431203 | −0.010448 |
| `cr_mow_n_synergy` | 0.494717 | 0.472136 | 0.479619 | +0.007483 |
| `threshold_saturation` | 0.459631 | 0.434369 | 0.439298 | +0.004930 |
| `pren_n_improved_environment` | 0.459677 | 0.436797 | 0.438341 | +0.001544 |
| `mo_n_acid_repassivation` | 0.459924 | 0.436328 | 0.433876 | −0.002452 |
| `mns_inclusion_penalty` | 0.498030 | 0.477475 | 0.467508 | −0.009967 |
| `coupled_breakdown` | 0.502793 | 0.482026 | 0.475886 | −0.006141 |
| `pren_n_coupled_mns` | 0.554193 | 0.564396 | 0.560894 | −0.003502 |
| `pren_n_coupled_mns_weak_anions` | 0.586618 | 0.579633 | 0.571158 | −0.008475 |

- **Results:** three formulas improved and six declined on matched validation
  rows. The mean across the nine formulas changed from **0.480534 to
  0.477531**, a decrease of **0.003003**. This descriptive mean was not an
  ensemble score. `pren_n_coupled_mns_weak_anions` remained strongest, and
  `pren_n_coupled_mns` remained second.
- **Interpretation:** comparing the original populations directly gave a
  larger mean decrease, from **0.497655 to 0.477531**. Much of that apparent
  decrease came from changing which validation rows were scored. The matched
  comparison controlled validation membership, but retained the old
  predictions' original larger context fits. Context membership,
  preprocessing, scaling and fitted coefficients still differed. The
  comparison did not isolate the effect of cleaning alone or establish an
  improvement or decline in trained TabICL performance.
- **Validation:** we reproduced all **45 previous and 45 reviewed fold
  scores**, plus both sets of nine saved mean scores. We checked frozen split
  identities, formula artifact hashes, unchanged formula definitions, row
  membership and fold assignments. CSV target round-off was at most
  **1.14 × 10⁻¹³**; target ranks matched exactly in every paired fold.
  Existing datasets, split files, formula artifacts and sampling banks were
  unchanged.

We generated the comparison with:

```bash
.venv/bin/python -m scripts.soccol_pipeline.compare_saved_rule_results
```

The command refused to overwrite a nonempty output directory. It saved
[`comparison_vs_v1.csv`](analysis/soccol_pipeline/target_rule_comparison_v1/comparison_vs_v1.csv),
per-fold comparisons, a readable summary and source/output hashes. A repeat
comparison could use a fresh `--output-dir`.

## Soccol pipeline: material-family coverage

- **Active scope:** we verified that formula fitting and synthetic sampling
  used all **2,950 accepted development rows**, covering **349 distinct
  compositions**. No alloy-family filter was applied. Composition and joint
  environment rows were sampled independently and uniformly by observation.
  The material-family identifiers passed by the Soccol sampler were a single
  placeholder category; they did not select different Fe/Ni rules.

| Stored material group | Accepted development rows | Share of composition draws |
| --- | ---: | ---: |
| Stainless steels | 2,745 | 93.05% |
| Fe-labelled model alloys | 65 | 2.20% |
| Fe–Ni–Cr Alloy 800 | 64 | 2.17% |
| Ni–Cr–Mo model alloys | 39 | 1.32% |
| High-alloy group with broad historical labels | 35 | 1.19% |
| Nickel Alloy 600 | 2 | 0.07% |

- **Historical labels:** `material_scope_primary` still contained **2,810
  include, 99 review and 41 exclude** entries within accepted development.
  Those entries were historical notes, not current filters. The reviewed
  composition status governed acceptance. The 272 withheld development rows
  had unresolved compositions; they were not excluded by material family.
- **Composition cross-check:** Fe was the largest element in **2,901**
  accepted rows, Ni in **47**, and Cr in **two**. Eight rows with broad
  `fe_based...` labels had another largest element: six Ni and two Cr.
  We recorded their IDs separately and retained the frozen labels. This
  largest-element diagnostic was not a replacement alloy taxonomy.
- **Formula coverage:** all nine Soccol formulas used the same definitions
  and globally fitted coefficient distributions across the accepted material
  population. Formula selection was not conditioned on alloy family. We
  calculated each formula's within-family development-fold scores from saved
  held-out predictions, without refitting coefficients. The complete table
  was saved in the [material-scope analysis](analysis/soccol_pipeline/material_scope_audit_v1/README.md).
- **Main result:** `pren_n_coupled_mns_weak_anions`, the strongest formula
  overall, scored **0.5592** on the 2,745-row stainless group, **0.2958** on
  the 65-row Fe-labelled model-alloy group, and **0.0330** on the 39-row
  Ni–Cr–Mo group. Within Ni–Cr–Mo, `threshold_saturation` was highest at
  **0.2734**. These were means of within-family development-fold Spearman
  correlations, rather than separately fitted family models or final scores.
- **Small-group limits:** all 39 Ni–Cr–Mo rows came from `2009Wong`, with
  **8, 14, 8, 6 and 3** observations across the five validation folds. The
  64 Alloy 800 rows belonged to one composition and fold 5; their high
  correlations described environmental variation for that composition. We
  did not report a correlation for the two Alloy 600 rows. Folds with fewer
  than three rows or constant targets/predictions were left unscored.
- **Signal available in nickel rows:** all 39 Ni–Cr–Mo rows had zero Mn, S,
  N, sulfate and nitrate; pH and chloride were constant. Consequently, the
  extra Mn/S and weak-anion terms supplied no varying signal within that
  subgroup. Temperature and composition varied. This observation did not
  establish why a trained Soccol model performed differently from EPIT.
- **Difference from EPIT sampling:** EPIT used **411 Fe and 41 Ni–Cr–Mo
  development observations**, represented by **241 Fe and 14 Ni–Cr–Mo
  distinct composition templates**. Its current composition sampling gave
  nickel **14/255 = 5.49%**, whereas Soccol's observation sampling gave
  nickel-labelled rows **41/2,950 = 1.39%**. EPIT's environment-row share,
  41/452, was separate from this composition probability. We documented the
  template-versus-observation difference without changing either sampler.
- **Difference from EPIT formulas:** EPIT included `fe_ni_cr_threshold`,
  with Cr thresholds of **12 wt.% for Fe and 15 wt.% for Ni–Cr–Mo**. Soccol
  had no promoted rule with a separate Fe/Ni threshold. EPIT nevertheless
  also scored lower on nickel: its strongest formula overall scored
  **0.5782 on Fe versus 0.3594 on Ni–Cr–Mo**; its best nickel formula was
  `fe_ni_cr_threshold` at **0.4644**. Weak nickel performance alone was
  therefore not a unique explanation for Soccol. The datasets' subgroup
  rows and contexts differed, so the scores did not isolate the effect of
  family-specific rules.
- **Assessment:** we recommended keeping the current material inclusion for
  this Optuna run. The small nickel groups did not justify automatic removal
  or use of the old exclusion flags. Observation-based composition weighting
  and the absence of a family-specific formula remained separate findings
  for the sampling/formula review. No dataset values, family labels, rules,
  weights or frozen partitions were changed.
- **Validation and outputs:** we checked bank membership and prediction
  membership against accepted development rows and frozen folds, and
  reproduced the overall mean scores of all nine Soccol and nine EPIT
  formulas. We saved family counts, partition counts, label caveats and all
  subgroup/fold scores in
  [`material_scope_audit_v1/`](analysis/soccol_pipeline/material_scope_audit_v1/).
  No final-query predictions were used.

We generated the analysis with:

```bash
.venv/bin/python -m scripts.soccol_pipeline.analyze_material_scope
```

The command refused to overwrite an existing nonempty output directory.
`summary.json` recorded the source/output hashes and the sampling comparison.

## Soccol pipeline: synthetic sampling audit

- **Scope:** we traced the reviewed source banks through composition and
  environment sampling, internal filling, formula/SCM target generation,
  standardization and missing-value restoration. We sampled **20,000 rows
  per dataset** at each perturbation strength **0, 0.05 and 0.30**, using
  seed **606**, and compared Soccol with the current EPIT path. We saved the
  numerical results in
  [`synthetic_sampling_audit_v1/`](analysis/soccol_pipeline/synthetic_sampling_audit_v1/).
- **Main difference:** Soccol sampled composition observations uniformly,
  whereas EPIT sampled distinct composition templates. Soccol's 2,950
  accepted observations contained 349 distinct compositions; repeated
  measurements therefore increased a composition's chance of being drawn.

| Composition-weight statistic | Soccol | EPIT |
| --- | ---: | ---: |
| Distinct active source compositions | 349 | 255 |
| Largest single-composition probability | 6.64% | 0.39% |
| Share of draws assigned to top 10 compositions | 34.31% | 3.92% |
| Share of draws assigned to top 20 compositions | 52.71% | 7.84% |
| Effective template count from source weights | 52.26 | 255.00 |

- **Weighting example:** the most repeated Soccol composition occurred
  **196 times**: Fe 69, Cr 17.5, Ni 11 and Mo 2.5 wt.%. It received 196 times
  the probability of a composition recorded once. The actual sampled top-20
  share was **52.52%**, consistent with the expected **52.71%**. All 349
  distinct source compositions appeared in the sample. The effective count
  above was `1 / sum(probability²)`, describing weight concentration rather
  than the number of possible perturbed alloys.
- **Correct source handling:** the Soccol bank matched the real-data encoder
  and accepted development row IDs. Saved compositions remained unchanged.
  Both samplers perturbed positive element amounts, preserved zeros and
  normalized only synthetic compositions to 100 wt.%. Sampled values were
  finite and composition totals agreed with 100 within numerical precision.
- **Correct context handling:** all environment, preparation, heat-treatment
  and measurement fields were drawn together from one observed row, including
  their joint missingness. Composition was drawn independently of that row
  in both pipelines. The audit found no accidental independent resampling of
  individual context fields. Composition/environment independence was a
  shared design, not a unique Soccol difference.
- **Missing values and categories:** the model received the original sampled
  NaN pattern and recorded category codes. Internal numeric means and category
  modes supplied finite values for SCM/formula calculations only. Soccol used
  its 2,950 accepted development rows for these statistics; EPIT retained
  statistics from all 608 development rows while sampling 452 Fe/Ni environment
  rows. This scope difference was documented; no final-query rows contributed
  filling statistics. EPIT's formula and SCM temperature/chloride/pH inputs
  agreed exactly after internal filling.
- **Comparable missingness:** across the **25 shared context columns**, missing
  entries were **62.69% for Soccol versus 59.91% for EPIT**. Soccol had five
  entirely missing context fields in its accepted bank, and EPIT had four
  in its active bank. After internal filling, **15/58 Soccol columns and
  16/70 EPIT columns** were constant over their source supports. Constant
  internal features standardized to zero, while missing model inputs remained
  NaN. Missing or constant columns alone did not uniquely explain Soccol's
  weaker performance.
- **Noise magnitude:** at strength **0.05**, the 95th percentile of the largest
  per-element change in a generated row was **2.54 wt.% points for Soccol**
  and **2.67 for EPIT**. At **0.30**, these values were **15.63 and 16.08**.
  The parameter controlled log-normal noise; it was not a hard ±30% limit.
  Both samplers allowed values outside the original observed ranges. Neither
  reclassified generated alloys or applied a family-specific acceptance filter
  after perturbation. This behavior was shared; we did not change the search
  range in this audit.
- **Training connection:** the checked Soccol path passed all 58 columns to
  the SCM, read the intended named formula inputs, retained fixed column
  positions and restored missingness after finite target calculations.
  The corresponding EPIT path used 70 model columns and its private formula
  projection. Both MLP and tree SCM paths and the learned missing-value
  handling passed the existing focused checks.
- **Recommendation:** we recommended drawing Soccol's **349 distinct reviewed
  compositions equally**, while keeping observation-weighted joint environment
  sampling. This would align composition weighting with EPIT without removing
  material families or balancing families equally. Under that proposal any
  20 source compositions would receive **5.73%** of draws. We documented the
  proposal but left the sampler, bank, dataset and training settings unchanged.
  A trained-model benefit had not been demonstrated.
- **Validation:** all **51 sampling assertions** and **80 existing focused
  tests** passed. The tests included small CPU training/inference checks.
  Five existing empty-column warnings occurred in the Soccol inference check;
  predictions remained finite and missingness/schema assertions passed.
  No Optuna run, full training or frozen final-query scoring was performed.

We generated the numerical audit with:

```bash
.venv/bin/python -m scripts.soccol_pipeline.audit_synthetic_sampling
```

The command refused to overwrite a nonempty output directory. It saved source
composition weights, feature summaries, sampled coverage, perturbation ranges
and source/output hashes. Repeated source entries and distinct compositions
were counted separately.

## Soccol pipeline: uniform composition sampling and fixed coefficient variation

- **Approved sampling change:** we gave each of the **349 distinct accepted
  development compositions** equal probability, matching EPIT's treatment of
  distinct compositions. Repeated measurements no longer increased a
  composition's sampling probability. The 20 most repeated compositions
  therefore received **5.73%**, instead of **52.71%**, of the expected draws.
  We did not balance the material families equally.
- **Source handling:** we retained the first accepted source row as the ID
  for each distinct composition. We kept all **2,950 observations** in the
  bank and drew environment, preparation, history and missingness together
  from one independently selected observation. These context observations
  retained equal probability. Source values, category codes, internal filling
  statistics, formula coefficients, development folds and all **805 frozen
  final queries** remained unchanged.
- **Versioning:** we created `soccol_pitting_features_v3.csv` and `.json`.
  The CSV bytes matched v2; the metadata recorded the new sampling policy
  and the 349 representative source indices. We preserved explicitly selected
  v1/v2 sampling behavior and kept the earlier audit commands pinned to v2
  so their reported weighting remained reproducible. Both reviewed versions
  used the same 58-column real-input schema.
- **Training connection:** we set the current Optuna and final-training
  defaults to profile v3 and the new
  `soccol_pipeline_optuna_empirical_features_scm_target_v4` study, with search
  results under `analysis/soccol_pipeline/optuna_v4/`. The study fingerprint
  included the sampling policy and template count and rejected reuse of
  studies with the previous weighting. Existing `target_rules_v2` fits
  remained valid because their accepted real-data inputs had not changed.
- **Coefficient variation (item 7):** we verified that Soccol already used
  the same fixed **0.8** variation and seed **42** as EPIT. We documented
  this as an **a priori design choice**, not a value fitted to either dataset
  or an Optuna search parameter. The setting applied both during Optuna and
  during final training at **25%, 50%, 75% and 100%** informed ratios.
- **Meaning of 0.8:** both pipelines used the same coefficient sampler.
  It multiplied fitted coefficients by independent uniform factors from
  **0.2 to 1.8**, normalized their sum to one, and rejected values exceeding
  the existing upper bounds. Zero coefficients remained zero. If 100 attempts
  failed, it returned the original coefficients. The parameter was not a
  variance or a guaranteed ±80% final coefficient change after normalization.
  We found no mismatch between Soccol and EPIT in this setting or mechanism.
- **Verification:** all **146 focused tests** passed, including 50,000 draws
  from each Soccol profile, exact source-data equality, context/missingness
  preservation, version checks and final-training command checks. We also
  checked that the frozen v3 assets rebuilt exactly from accepted development
  inputs. Five existing empty-column warnings occurred in the small CPU
  inference check; predictions remained finite. We made no claim of improved
  trained-model performance and launched no Optuna or full-training job.

## Soccol pipeline: historical Optuna ranges (item 8)

- **Evidence:** we read temporary copies of the old Soccol v1 and EPIT v8
  journals. Soccol had **50 completed trials** and five entries still marked
  `RUNNING`; we excluded the latter. All completed trials shared one pipeline
  fingerprint. We checked the 24 available local result files against their
  journal scores, parameters and hashes. Only development-fold scores entered
  this review.
- **Best old configuration:** Soccol trial **19** scored **0.604646** mean
  development-fold Spearman with **100% informed**, MLP share **1.0**, formula
  contribution **0.662955**, and composition perturbation **0.035766**.
- **Broader pattern:** all top ten trials used 100% informed. Eight used MLP
  share 0.75 or 1.0; the others used 0.25 and 0.50. Their formula contribution
  ranged from **0.581274 to 0.836408**, and composition perturbation from
  **0.018839 to 0.045396**. These were joint adaptive-search results, not
  isolated causal effects of individual parameters.
- **Difference from EPIT:** EPIT's best old trial used composition
  perturbation **0.149675**, near its previous 0.15 upper limit. Its top ten
  ranged from **0.095352 to 0.149675**. Soccol did not show this pressure
  towards the upper limit, so extending its range to 0.30 was not supported
  by the same historical pattern.
- **Recommendation:** we proposed continuous **0–1 MLP share**, **0–1 formula
  contribution**, and **0–0.15 composition perturbation**, with informed share
  fixed at **100%** and coefficient variation fixed at **0.8**, seed **42**.
  We retained broad ranges because the source data, features, formula fits
  and composition weighting had changed. We did not restrict the search to
  the old top-ten interval or claim an optimum for the corrected workflow.
- **Initial review status and outputs:** at this stage the proposed noise-range
  change had not been implemented; the scripts still used **0–0.30**. We saved the completed
  trials, parameter summaries, source hashes and recommendation in
  [`optuna_range_review_v1/`](analysis/soccol_pipeline/optuna_range_review_v1/README.md).
  We launched no training and left the original journals untouched.

### Approved range implementation

- We changed Soccol's composition-perturbation search range to **0–0.15**.
  EPIT retained **0–0.30**. Both retained continuous **0–1** MLP share and
  formula contribution, fixed **100% informed**, and coefficient variation
  **0.8** with seed **42**.
- We used the same dataset-specific range in Optuna sampling, parameter
  validation, study fingerprints and saved trial metadata. A study previously
  bound to **0–0.30** could not silently resume with the reduced range.
  The current Soccol study name remained **v4**.
- Final training retained the selected trial's composition-noise value for
  each **25%, 50%, 75% and 100%** informed-ratio override. Historical search
  modes retained their previous settings.
- All **26 focused checks** passed, covering the Soccol and EPIT search
  distributions, old-range fingerprint rejection and final-training commands.
  We updated both pipeline READMEs and the range-review implementation status.
  We launched no Optuna or full-training job.

## Soccol pipeline: reviewed baseline and checkpoint comparisons (item 9)

- **Checkpoint comparison:** we replaced the default old trial-19/37-column
  route with the reviewed **58-column** inputs. Soccol, generic TabICL,
  pretrained TabICL v2 and fixed CatBoost received the same **2,950 accepted
  development context rows** and all **805 frozen final queries**, including
  the ten unresolved compositions. We used the reviewed category mappings
  and preserved source NaNs. Soccol used fixed columns and learned missing-value
  handling; generic and pretrained models retained their existing handling.
- **Category handling:** we replaced the historical four-category list with
  the **nine reviewed categories**. We did not treat numeric `CP_agitation`
  as categorical or request excluded `Prep_medium`/`CP_anions_info` columns.
- **Model selection:** the comparison default read the frozen reviewed v4
  final-model manifest, checked its locked checkpoint and reviewed input/split
  identity, and used that checkpoint's run directory. Other informed-ratio
  runs could be selected through their own manifest or an explicit run path.
  A missing new model produced an error instead of selecting old trial 19.
  The comparison curves did not change the development-selected checkpoint.
- **Development baselines before training:** we added
  `scripts/soccol_pipeline/evaluate_baseline_folds.py`. It compared the generic
  **step-1000** checkpoint, pretrained TabICL v2 and fixed CatBoost on the same
  five accepted development folds as Optuna. Query counts were **622, 615,
  638, 445 and 630**, with the other accepted development rows as context.
  All 805 final queries were excluded. The script saved individual scores,
  their five-fold means, reviewed source hashes and original row membership.
  Its defaults matched the EPIT baseline protocol: eight estimators, seed 42,
  no feature shuffling or power transform, and CatBoost with 1,000 iterations,
  depth 6, learning rate 0.03 and L2 penalty 3.0.
- **Membership checks:** we checked original source-row indices in the saved
  results, not only their counts. For development evaluation, we translated
  the evaluator's local 2,950-row positions back to frozen source indices.
  Context-size diagnostics used each development fold as context while keeping
  all 805 final queries. They did not substitute development rows as final
  model queries.
- **Checkpoint coverage and outputs:** the comparison retained available
  checkpoints from either run, including each run's latest endpoint after
  interval filtering. It required all four models, no recorded model errors,
  and all six metric SVGs. Nonempty result directories were rejected. Reviewed
  outputs were separate from historical results, and `--legacy-data` remained
  available for the old input/model combination. The sbatch launcher called
  the same reviewed Python entry point as local execution.
- **Preparation and verification:** both commands supported `--dry-run`.
  We verified the real split/command/result paths using mock model scores:
  **91 regression checks** passed, and all **seven comparison checks** passed
  after the final label-handling adjustment. We also parsed the launcher
  preview and all five baseline commands through the real evaluator. We ran
  no actual baseline scoring, checkpoint evaluation, training or Optuna job.
- **Remaining EPIT difference:** EPIT's older checkpoint-comparison launchers
  still referenced historical v7/v8 runs. This Soccol update did not migrate
  those EPIT launcher defaults.

We prepared the development-baseline command as:

```bash
.venv/bin/python -m scripts.soccol_pipeline.evaluate_baseline_folds --device cpu
```

The reviewed checkpoint comparison, after final training, used:

```bash
sbatch scripts/soccol_pipeline/evaluate_checkpoint_comparison.sbatch
```

## EPIT and Soccol: automatic final-trial selection across nodes

- **Cause:** automatic selection correctly found the best completed trial in
  the shared Optuna journal. Final training then required that trial's local
  evaluation CSV/JSON files and proxy checkpoint. The journal stored their
  paths and hashes, but those worker-local files were not necessarily present
  on another node. The previous journal fallback was enabled only when an
  explicit trial number was supplied through the launcher.
- **Fix:** we enabled validated journal fallback by default for both automatic
  and explicitly pinned selection in the shared EPIT/Soccol final-training
  implementation. Neither launcher required `SELECTED_TRIAL_NUMBER` for this
  behavior. Final training still selected the current best completed trial.
- **Verification:** available local evidence remained preferred. When files
  were missing, we checked the recorded study and data identities, feature
  profile, folds, settings, metrics and artifact records. We also checked the
  hashes of any files that were present, so one missing file could not hide
  another changed file. Incomplete journal records and mismatched inputs
  remained errors. `--no-allow-missing-selected-trial-artifacts` retained an
  explicit strict mode requiring all proxy files.
- **Saved evidence:** the final manifest recorded journal-based verification,
  missing files, verified local files, source/split hashes and, for reviewed
  Soccol, the accepted-development selection. The 25/50/75/100% training
  overrides and development-only checkpoint selection remained unchanged.
- **Checks:** all **65 focused tests** passed. These covered automatic
  selection with missing files, partial local evidence, corrupted files,
  incomplete journal records, strict mode and the full mocked Soccol final
  workflow. Using a temporary copy of the real EPIT v10 journal, we also
  automatically selected and verified trial **2** without a trial-number
  override. We left the shared journal unchanged and launched no training.

## EPIT: reviewed comparison inputs and frozen queries

- We updated EPIT's baseline-fold and final-comparison scripts to use the
  reviewed 70 features and their category columns. Baseline folds used only
  development rows; final comparisons retained all 152 frozen outer queries.
- We obtained the default final candidate from its frozen model manifest,
  accepted explicit checkpoint directories, enabled the informed model's
  learned missing-value handling and checked actual context/query membership.
- We retained historical inputs only through an explicit legacy option and
  used a separate output location for reviewed development baselines.

## EPIT: four informed ratios in the combined-evaluator structure

- We added `--ratio-runs` for the 25%, 50%, 75% and 100% checkpoint directories.
  We evaluated them in one invocation with generic TabICL, pretrained TabICL
  and CatBoost, producing seven curves in the same tables and plot directory.
- We reused the combined evaluator's command builder and common settings
  instead of maintaining a separate copy. We retained its 500-step schedule,
  latest endpoints, static baseline reuse, metrics and output filenames.
- We corrected an alignment gap: EPIT's prior wrapper did not request native
  generic/pretrained inference, and its final split implicitly forced constant
  columns to remain. The new comparison used native baseline normalization,
  shuffling and constant-feature handling, while the informed models retained
  the fixed reviewed schema and learned missing-value handling.
- We kept all 608 development context rows and all 152 frozen final queries by
  default. Optional context-fold diagnostics still kept the same final queries.
- We verified seven-model orchestration, native baseline settings, unequal
  checkpoint endpoints and full/reduced context membership using mock scores.
  We did not run the user's models or change the active training checkout.

## Shared evaluation preprocessing: columns with very few known values

- We corrected the shared TabICL preprocessing used by EPIT, Soccol, MPEA and
  combined evaluations, including development-fold evaluation. Columns with
  fewer than two known context values bypassed scaling, normalization and
  outlier clipping. We preserved their known values and genuine NaNs without
  changing column order or fitting any statistics to query rows.
- We also disabled outlier clipping for columns whose preliminary outlier
  removal left fewer than two known context values. Columns with sufficient
  observations retained the existing processing rules.
- We confirmed the original failure on reviewed EPIT: `WO4_mol_L` and `CP_SO4`
  each had zero known development values and two known final-query values;
  `NaClO4_mol_L` had one known development value. Previously, preprocessing
  turned all five known cells into NaN. The fix preserved them.
- We made the change in the isolated checkout and left running training,
  processed CSVs, frozen partitions and saved model weights unchanged.
  Evaluations started before the fix required a restart with the corrected
  code and a fresh output directory to produce corrected scores.
- We passed 100 focused checks covering sparse values, all five normalization
  methods, missing-value embeddings and prediction caches, plus EPIT, Soccol,
  MPEA and combined evaluation wrappers. We checked the five affected EPIT
  cells directly without rerunning trained-model evaluations.

## Applying the evaluation fixes to the main project

- We applied the reviewed evaluator, four-ratio comparison and shared sparse-
  column preprocessing fixes to `/home/fcolanto/projects/tabicl`. Evaluation
  no longer required `/tmp/tabicl-standalone-fixes`.
- We kept training/resume changes separate. We did not modify the active
  training/prior code, Optuna or final-training launchers, datasets, frozen
  partitions, checkpoints or existing evaluation output directories.
- We passed 138 checks from the main checkout and verified that the local CPU
  comparison loaded the main-project code. We ran no trained-model evaluation.


## Shared synthetic streams for comparable training (30 October 2026)

- We introduced indexed generic and informed streams for EPIT v11 and Soccol v5. All ratios consumed the same ordered source tables at different rates, with exact counts per batch.
- We separated table seeds from mixture allocation, trial numbers, worker assignment and maximum training steps. We separated the SCM target RNG from physical draws and formula draws, and separated EPIT environment draws from composition perturbation.
- We restored the standard generic recipe inside mixed runs: 2–100 initial features, 70/30 MLP/tree, category generation and feature permutation. This recipe no longer inherited specialist-only restrictions.
- We fixed Python, NumPy, Torch and data-stream seed defaults at 42, and bound final training to the selected study's stream policy, seeds, schedule and process settings. Optuna's parameter-suggestion seed remained separate.
- We saved profile and stream identities plus the consumed worker position in new checkpoints, and wrote checkpoints atomically. We did not retrofit historical checkpoints or apply the separate high-level training-resume launcher.
- We retained all reviewed real values, frozen partitions, formula definitions, coefficient variation and search ranges. Existing models/results were retained.
- We documented the implementation and limitations in [Shared synthetic streams](../scripts/epit_pipeline/PAIRED_STREAMS.md).
