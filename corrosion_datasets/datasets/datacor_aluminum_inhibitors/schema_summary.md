# Schema Summary

## Inspected Files

- `raw/DataCor.xlsx`

## Workbook Structure

Observed workbook sheets:

| Sheet | Observed Data Rows | Max Columns Seen |
|---|---:|---:|
| `Main` | 102 | 20 |
| `Non-covalent` | 102 | 10 |

The evaluator loader uses the `Main` sheet and reshapes the four fixed response columns into a long table:

| Loader Table | Rows | Columns | Numeric Target |
|---|---:|---:|---|
| `DataCor_long_efficiency` | 408 | 23 | `efficiency_score` |

## Feature Groups

- Material: derived `alloy` from the target column name, either AA2024 or AA7075.
- Environment: derived `pH`, either 4 or 10.
- Intervention/descriptors: inhibitor molecular descriptors, molecular weight, logP, surface area, donor/acceptor counts, rotatable bonds, and dimerization descriptors.
- Metadata: inhibitor name, numeric identifier, and source response column.

## Targets

The target is `efficiency_score`, loaded from `Efficiency_AA2024_pH4`, `Efficiency_AA2024_pH10`, `Efficiency_AA7075_pH4`, and `Efficiency_AA7075_pH10`.

These are numeric inhibitor-efficiency scores on a 0-10 scale. They are suitable for the project setup where regression-like responses are converted to ordered bins for classifier evaluation.

## Leakage Risks

- The original wide response columns encode alloy and pH in the column name. The loader moves those into feature columns and keeps only one response column.
- The inhibitor identity should not be used as a categorical shortcut for strict extrapolation tests; it is left as metadata.
- This dataset is small and chemically descriptor-heavy, so it should not dominate aggregate evaluation.

## Relevance To Informed Prior

High relevance for inhibitor/intervention structure. It supports pH/alloy-specific response variation and molecule-environment interactions.
