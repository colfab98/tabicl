# Schema Summary

## Inspected Files

- `raw/Research_data_Saenzetal.docx`

## Document Structure

Observed DOCX table:

| Table | Rows | Columns | Numeric Target |
|---|---:|---:|---|
| `Research_data_Saenzetal_table1` | 613 | 10 | `ER.CREV. VECS` |

Primary columns:

- Test ID: `Test ID#`.
- Repassivation response: `ER.CREV. VECS`.
- Material: `Alloy`.
- Environment: `T. deg C`, chloride, sulfate, nitrate, and molybdate concentrations.
- Electrochemical/test setting: `iGS. uA/cm2`.
- Outcome label: `CC Attack?`.

## Feature Groups

- Material: alloy label.
- Environment: temperature and ion concentrations.
- Electrochem: galvanostatic current density `iGS. uA/cm2`.
- Target: `ER.CREV. VECS`.
- Excluded: `CC Attack?`, because it is an observed attack outcome and could leak corrosion response information.

## Targets

The target is `ER.CREV. VECS`, the crevice corrosion repassivation potential in V vs ECS. It is continuous and suitable for binned regression-style evaluation.

Observed finite range during inspection: -0.329 to 0.89.

## Leakage Risks

- `CC Attack?` is an outcome label and should not be used as a feature for repassivation-potential prediction.
- Alloy is categorical rather than full composition; this is useful but less physically detailed than composition-resolved alloy tables.
- DOCX extraction depends on the published table layout.

## Relevance To Informed Prior

High relevance for material-environment-intervention corrosion behavior. The dataset has alloy, temperature, chloride, and inhibitor-anion concentrations with a direct electrochemical potential response.
