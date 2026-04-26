# Schema Summary

## Inspected Files

- `raw/Epit and Epass descriptors of 316L stainless steel estimated by Machine Learning.zip`
- `raw/Micro-scale potentiodynamic polarisation (log(j)) curves of 316L stainless steel.zip`

## Descriptor Archive

All descriptor CSVs have columns:

```text
Maps, Epit_x, Epit_y, Epass_x, Epass_y
```

Observed descriptor files:

| Internal CSV | Data Rows | Condition Encoded In Filename |
|---|---:|---|
| `Epit_Epass_005_50.csv` | 47 | 0.05 M NaCl, 50 mV/s |
| `Epit_Epass_001_50.csv` | 119 | 0.01 M NaCl, 50 mV/s |
| `Epit_Epass_005_100.csv` | 125 | 0.05 M NaCl, 100 mV/s |
| `Epit_Epass_001_100.csv` | 377 | 0.01 M NaCl, 100 mV/s |
| `Epit_Epass_0005_100.csv` | 287 | 0.005 M NaCl, 100 mV/s |

## Curve Archive

The curve archive contains paired `logj_uA-cm2_*` CSVs and `E(V)_*` CSVs. The `E(V)` files provide the potential axis, while `logj_uA-cm2` files contain multiple measured polarisation traces as columns.

Observed curve files:

| Internal CSV | Rows | Approx. Trace Columns |
|---|---:|---:|
| `logj_uA-cm2_0.05_M_NaCl_50_mV_per_s.csv` | 2217 | 47 |
| `logj_uA-cm2_0.05_M_NaCl_100_mV_per_s.csv` | 1107 | 125 |
| `logj_uA-cm2_0.01_M_NaCl_50_mV_per_s.csv` | 2218 | 119 |
| `logj_uA-cm2_0.01_M_NaCl_100_mV_per_s.csv` | 1108 | 377 |
| `logj_uA-cm2_0.005_M_NaCl_100_mV_per_s.csv` | 1108 | 287 |

## Feature Groups

- Material: fixed 316L stainless steel.
- Environment/test condition: NaCl concentration and scan rate encoded in filenames.
- Electrochemical curve response: potential and log-current trajectories.
- Derived electrochemical descriptors: Epit/Epass coordinate descriptors.

## Targets

The descriptor archive is essentially target/response data extracted from polarisation curves. It is best treated as electrochemical response evidence rather than generic feature data.

## Leakage Risks

- `Epit_*` and `Epass_*` are derived from measured curves and are outcome-like.
- The raw curves are experimental response trajectories.
- Because material is fixed, this dataset should not be used to infer alloy-composition prior strength.

## Relevance To Informed Prior

Moderate relevance. This dataset strongly supports the idea that electrochemical response variables are internally structured and correlated, but it contributes less to material-environment interaction because the material is fixed.

Safe prior implications:

- Electrochemical descriptors should form a coherent response block.
- Scan rate and electrolyte concentration are useful examples of test/environment context.
- Do not use this dataset to set composition-block strength.
