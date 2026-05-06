# Schema Summary

## Inspected Files

- `raw/ze41_mol_desc_db_red.csv`

## Table Structure

Observed CSV structure:

| Table | Rows | Columns | Numeric Target |
|---|---:|---:|---|
| `ze41_mol_desc_db_red` | 60 | 1264 | `inhibition efficiency ZE41 / %` |

## Feature Groups

- Material: added constant `substrate_alloy = ZE41`.
- Intervention/descriptors: molecular descriptors for candidate dissolution modulators.
- Metadata: compound name.
- Excluded: `LinIE ZE41`, because it is a transformed target-like quantity.

## Targets

The target is `inhibition efficiency ZE41 / %`, a numeric inhibition-efficiency response for Mg ZE41. It is suitable for binned regression-style evaluation.

Observed finite range during inspection: -270 to 75.

## Leakage Risks

- The table is very small and descriptor-heavy.
- `LinIE ZE41` is derived from the target and must not be used as a feature.
- Negative efficiencies are preserved and should be interpreted carefully.

## Relevance To Informed Prior

Moderate to high relevance for inhibitor/intervention descriptors and magnesium-alloy corrosion response. It complements the Mg AZ91 inhibitor table.
