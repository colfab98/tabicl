# Schema Summary

## Inspected Files

- `raw/Data for The effect of environmental variables on corrosion of high–strength low–alloy mooring steel immersed in seawater.zip`
- `raw/1-s2.0-S0951833917300655-main.pdf`

## Archive Structure

Internal files:

- `OCP R4 S31 Omax.xlsx`
- `Optical microscopy analysis of the R4 samples after immersion corrosion tests .docx`

Observed workbook:

| Internal Workbook | Sheet | Observed Nonempty Rows | Max Columns Seen |
|---|---|---:|---:|
| `OCP R4 S31 Omax.xlsx` | `Hoja1` | 29 | 10 |
| `OCP R4 S31 Omax.xlsx` | `Hoja2` | 0 | 0 |

The nonempty sheet describes open circuit potential vs Ag/AgCl for steel grade R4 in synthetic seawater under salinity/temperature/oxygen conditions. The visible condition blocks include `S=31 T=2 Omax`, `S=31 T=32 Omax`, and `S=31 T=17 Omax`.

## Feature Groups

- Material: fixed R4 high-strength low-alloy mooring steel.
- Environment: salinity, temperature, oxygen concentration condition.
- History/time: exposure days.
- Electrochemical response: open circuit potential vs Ag/AgCl.
- Post-exposure morphology: optical microscopy document.

## Targets

Open circuit potential is the main structured electrochemical response in the workbook. Optical microscopy is a post-exposure observation source, not a pre-outcome covariate.

## Leakage Risks

- OCP is a measured response and should not be used as a generic input feature when predicting corrosion status.
- Optical microscopy is post-exposure and likely post-outcome.
- Material variation is absent in the structured workbook, so do not use this dataset to calibrate composition-block strength.

## Relevance To Informed Prior

Moderate relevance. This dataset is small but useful for environment/history logic because it explicitly varies temperature and exposure time under seawater conditions.

Safe prior implications:

- Environment variables can alter electrochemical response over time.
- Exposure/time can be represented as a history component.
- The dataset supports environment/history structure, not broad alloy-composition structure.
