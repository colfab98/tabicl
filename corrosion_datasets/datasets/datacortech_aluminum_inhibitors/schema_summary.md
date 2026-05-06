# Schema Summary

## Inspected Files

- `raw/Datacortech_initial_data.xlsx`
- `raw/Datacortech_final_data.xlsx`

## Workbook Structure

Observed workbook sheets:

| File | Sheet | Observed Data Rows | Max Columns Seen |
|---|---|---:|---:|
| `Datacortech_initial_data.xlsx` | `Efficiencies` | 2011 | 19 |
| `Datacortech_initial_data.xlsx` | `Sheet1` | 7 | 5 |
| `Datacortech_final_data.xlsx` | `Sheet 1` | 1966 | 229 |

The evaluator loader exposes:

| Loader Table | Rows | Columns | Numeric Target |
|---|---:|---:|---|
| `Efficiencies_with_descriptors` | 2011 | 229 | `Efficiency` |

## Feature Groups

- Material: `Metal`, `Alloy`.
- Environment: temperature, pH, and salt concentration.
- History: exposure time.
- Intervention: inhibitor concentration, synergistic-inhibitor fields, encapsulation, and molecular descriptors joined from the final workbook.
- Metadata: inhibitor name, SMILES, reference, link, contributor, methodology, and descriptor source.

## Targets

The target is numeric `Efficiency` from the initial workbook. Values span from strongly negative apparent efficiencies to 100.

The final workbook's binary `Efficiency` label is not used as a target. It is only used as a source of molecular descriptors.

## Leakage Risks

- Do not use the final workbook's binary `Efficiency` label for regression-binned evaluation.
- Some numeric efficiencies are extreme negative outliers. They are preserved as raw experimental/literature values; median and quantile binning are less sensitive than direct squared-error regression, but task interpretation should note this.
- Inhibitor identity and literature reference can be shortcut features and are kept as metadata.

## Relevance To Informed Prior

High relevance for environment-intervention structure. It is a broad aluminum-inhibitor literature dataset with pH, concentration, alloy, and molecular-descriptor variation.
