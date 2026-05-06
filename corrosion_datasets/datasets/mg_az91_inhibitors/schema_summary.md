# Schema Summary

## Inspected Files

- `raw/Features.csv`

## Table Structure

Observed CSV structure:

| Table | Rows | Columns | Numeric Target |
|---|---:|---:|---|
| `Features` | 68 | 877 | `IE` |

## Feature Groups

- Material: added constant `substrate_alloy = AZ91`.
- Intervention/descriptors: molecular descriptors for candidate organic inhibitors.
- Metadata: row number and compound/file name.

## Targets

The target is `IE`, interpreted as inhibition efficiency for Mg AZ91. It is numeric and therefore compatible with the project's binning-based regression evaluation.

Observed finite range during inspection: -563.3441057 to 83.0.

## Leakage Risks

- The table is very small and descriptor-heavy.
- Negative efficiencies are preserved; they likely represent corrosion acceleration or derived efficiency artifacts and should be interpreted carefully.
- Compound/file names are metadata, not generalizable features.

## Relevance To Informed Prior

Moderate to high relevance for inhibitor/intervention descriptors and magnesium-alloy corrosion response. It should be useful for informed settings, but not as a high-weight benchmark task.
