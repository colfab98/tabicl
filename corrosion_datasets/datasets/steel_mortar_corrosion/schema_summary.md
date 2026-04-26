# Schema Summary

## Inspected Files

- `raw/Dataset of Steel Corrosion in Cementitious Mortar Due to Carbonation and Chlorides.xlsx`
- `raw/1-s2.0-S2352340924005626-main.pdf`

## Workbook Structure

Observed workbook sheets:

| Sheet | Observed Data Rows | Max Columns Seen | Scope |
|---|---:|---:|---|
| `01_Carbonation` | 180 | 17 | Carbonation exposure data. |
| `02_Chloride` | 95 | 16 | Chloride exposure data. |

The workbook has explicit merged/header groups:

- Mixture Parameters
- Materials Properties
- Environmental Parameters
- Electrochemical Parameters

## Feature Groups

- Mixture: cement, GGBS, fly ash, silica fume proportions; water-to-binder ratio; incorporated chloride concentration.
- Material/pore solution: free chloride concentration, pore solution pH, porosity.
- Environment: relative humidity, degree of saturation, water content.
- Electrochemical response: corrosion potential, electrical resistivity, chloride-to-hydroxide concentration ratio, corrosion rate.

## Targets

`Corrosion Rate of Steel` is the clearest target. Corrosion potential, resistivity, and chloride-to-hydroxide ratio are measured response/diagnostic variables and should be treated carefully depending on the prediction task.

## Leakage Risks

- Electrochemical parameters may be simultaneous or downstream measurements relative to corrosion rate.
- Chloride-to-hydroxide ratio may be a mechanistic predictor in some settings, but can also be tightly tied to observed corrosion state.
- If predicting final corrosion rate/status, avoid using post-outcome diagnostics unless the analysis goal explicitly permits them.

## Relevance To Informed Prior

Very high relevance for feature-block structure. This workbook is unusually clean because it explicitly separates mixture, material, environment, and electrochemical columns.

Safe prior implications:

- Distinct feature blocks are scientifically defensible.
- Environment and material/pore chemistry interact strongly.
- Electrochemical response variables should be grouped separately and treated as leakage-prone for prediction.
