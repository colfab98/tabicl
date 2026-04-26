# Schema Summary

## Inspected Files

- `raw/CORR-DATA_Database.zip`

## Archive Structure

Internal files:

- `CORR-DATA_Database.csv`
- `about-CORR-DATA_Database.txt`
- `columnDefinitions.txt`

Observed CSV schema:

| File | Estimated Data Rows | Columns |
|---|---:|---:|
| `CORR-DATA_Database.csv` | 24721 | 15 |

Columns:

```text
Environment
Material Group
Material Family
Material
Rate (mm/yr) or Rating
Rate (mils/yr) or Rating
Localized Attack
UNS
Condition/Comment
Concentration (Vol %)
Temperature (deg C)
Temperature (deg F)
Duration
Reference #
Reference
```

## Feature Groups

- Material: material group, family, material name, UNS.
- Environment: environment, concentration, temperature.
- History/exposure: duration, condition/comment.
- Corrosion outcome: rate/rating and localized attack.
- Provenance: reference number and reference text.

## Targets

Potential targets are rate/rating and localized attack. The rate/rating fields mix numeric rates and categorical ratings, so any modeling task needs parsing rules.

## Leakage Risks

- `Localized Attack` can be a target label rather than a feature.
- `Rate (mm/yr) or Rating` and `Rate (mils/yr) or Rating` are direct outcomes.
- `Condition/Comment` may include experimental context but can also contain outcome-like notes.
- Reference fields can leak source identity.
- Mixed numeric/categorical rate fields require careful parsing before quantitative analysis.

## Relevance To Informed Prior

High relevance for broad material-environment structure because the dataset is large and heterogeneous.

Safe prior implications:

- Material and environment coupling is a core corrosion pattern across many source documents.
- Temperature, concentration, and duration are recurring exposure context variables.
- Because the schema is heterogeneous and text-heavy, it is better for high-level structural support than for directly estimating precise prior strengths.
