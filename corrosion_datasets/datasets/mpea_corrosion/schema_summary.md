# Schema Summary

## Inspected Files

- `raw/Mendeley_MPEA_corrosion_database.xlsx`
- `raw/s41529-025-00700-9.pdf`

## Workbook Structure

Observed workbook sheets:

| Sheet | Observed Data Rows | Max Columns Seen |
|---|---:|---:|
| `Sheet1` | 619 | 39 |

Primary columns:

- Alloy and phase: `Alloy name`, `Phases present`, `FCC`, `BCC`, `HCP`, `IM`.
- Environment: `Test environment`, `Concentration in M`, `Electrolyte`.
- Electrochemical response: `Corrosion potential (mV vs SCE)`, `Pitting potential (mV vs SCE)`, `Corrosion current density (microA/cm2)`, `Calculated passive window`.
- Processing: `Processing`.
- Composition: element fraction columns from `Al` through `Zr`.
- Provenance: `Reference`.

## Feature Groups

- Material/composition: alloy name and atomic/element fractions.
- Microstructure/phase: phase-present text and one-hot phase flags.
- Environment: electrolyte and concentration.
- Processing/history: processing route.
- Electrochemical response: Ecorr, Epit, icorr, passive window.
- Provenance: reference DOI/source.

## Targets

Likely target candidates include Ecorr, Epit, icorr, and passive window. `Calculated passive window` is derived from electrochemical potentials and should be treated as target-derived.

## Leakage Risks

- Passive window is calculated from response variables and can leak outcome information.
- Ecorr/Epit/icorr are measured corrosion responses.
- Reference can identify source paper and should stay as metadata.
- Phase labels may be partly characterized after processing; use as a structural group carefully depending on the prediction timing.

## Relevance To Informed Prior

High relevance. This dataset supports composition, phase, processing, environment, and electrochemical response blocks for MPEA/HEA corrosion.

Safe prior implications:

- Composition and environment should interact.
- Phase/microstructure is a distinct material-state group.
- Processing route supports a history/process block.
- Electrochemical variables should be grouped as responses, not used blindly as pre-outcome features.

## Evaluation-candidate export (2026-10-12)

The new [preparation rules](preprocessing_plan.md) narrow the general raw-schema
description above to the proposed EPIT/Soccol transfer task. The
[candidate table](processed/mpea_evaluation_candidates.csv) contains 332
eligible target-bearing rows, 17 composition predictors in wt%, and 7 environmental
slots. Phase, processing, identifiers, and other electrochemical responses
are not predictors. Unknown values remain `NaN`.

Three incomplete-composition rows are excluded from this export but preserved
in the raw workbook and audit. A separate 267-row source-screened subset omits
shared and unresolved sources; it is not proven measurement-independent.
This is not a frozen evaluation set. The paper's 306-versus-workbook-335 count
discrepancy remains documented. See [README.md](README.md) for reproduction.
