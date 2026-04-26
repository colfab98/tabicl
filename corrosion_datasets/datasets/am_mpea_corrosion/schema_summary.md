# Schema Summary

## Inspected Files

- `raw/Additively Manufactured Multi Principal Element Alloy Corrosion Database.zip`
- `raw/s41529-025-00701-8.pdf`

## Archive Structure

Internal workbooks:

| Internal Workbook | Sheet | Observed Data Rows | Max Columns Seen | Notes |
|---|---|---:|---:|---|
| `Mendeley_AM_MPEA_corrosion_database.xlsx` | `Sheet1` | 78 | 27 | Earlier/smaller workbook. |
| `AM_MPEA_corrosion_database_V3.xlsx` | `Sheet1` | 97 | 25 | Primary/current workbook for analysis. |

Primary V3 columns:

- Provenance/material: `DOI`, `Alloy name`, `Alloy formula`.
- Processing: `AM process`.
- Environment: `Test electrolyte`, `Electrolyte concentration (M)`.
- Microstructure/phase: `Phases present`.
- Electrochemical response: `Corrosion potential (mV vs. SCE)`, `Pitting potential (mV vs. SCE)`, `Corrosion current density (microA/cm2)`.
- Composition/descriptors: `Al`, `Co`, `Cr`, `Cu`, `Fe`, `Mn`, `Mo`, `Nb`, `Ni`, `Si`, `Ti`, `Entropy of mixing (J/K.mol)`, `Average VEC`, `St. Deviation of VEC`.

## Feature Groups

- Material/composition: alloy formula and element fractions.
- Derived material descriptors: entropy of mixing and VEC-derived descriptors.
- Processing/history: AM process and post-processing labels such as HIP or heat treatment where present.
- Environment: electrolyte and concentration.
- Microstructure/phase: phase-present label.
- Electrochemical response: Ecorr, Epit, icorr.

## Targets

Likely target candidates are corrosion potential, pitting potential, and corrosion current density.

## Leakage Risks

- Electrochemical columns are measured responses.
- Process labels are safe only if they are known before the predicted corrosion outcome.
- Phase labels and derived descriptors are generally material-state features, but their availability should match the intended prediction setting.
- DOI/source fields are metadata only.

## Relevance To Informed Prior

High relevance for process/history structure. This is the clearest downloaded dataset showing AM process and post-processing labels alongside corrosion response.

Safe prior implications:

- A processing/history block is defensible.
- Processing can interact with material composition and environment.
- Exact intervention/process effect sizes are not calibrated by this small dataset.
