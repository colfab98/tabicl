# Informed versus generic: paired sample diagnostic

Positive penalty means the informed model is worse. Rows below average checkpoints within each sample before aggregating samples.
Outer fold 1 is reported separately from exploratory development folds. Category confidence intervals are descriptive, not evidence of causation.

## epit_100pct: exploratory_development_folds

| Category | n | Sources | MAE penalty mV | Samples worse |
|---|---:|---:|---:|---:|
| family_and_heat: Other / missing | 37 | 4 | +9.12 | 48.6% |
| CP_pH_bin: missing | 38 | 5 | +8.37 | 47.4% |
| Material class: Other | 40 | 6 | +7.57 | 47.5% |
| test_solution_type: NaBr | 26 | 4 | +7.29 | 53.8% |
| family_and_heat: Fe Alloy / missing | 124 | 13 | +4.70 | 52.4% |
| CP_Cl_bin: Cl >1 | 20 | 5 | +3.21 | 50.0% |
| family_and_heat: HEA / missing | 15 | 7 | +2.86 | 33.3% |
| CP_temp_bin: T 30–60 | 117 | 9 | +1.47 | 50.4% |
| CP_temp_bin: missing | 66 | 7 | +1.24 | 47.0% |
| N_bin: N >0.1 | 118 | 18 | +0.44 | 44.9% |
| Mo_bin: Mo >3 | 186 | 22 | +0.11 | 44.6% |
| heat_treatment_type: homogenization | 27 | 3 | -1.95 | 37.0% |

## epit_100pct: outer_fold_1

| Category | n | Sources | MAE penalty mV | Samples worse |
|---|---:|---:|---:|---:|
| family_and_heat: Al Alloy / missing | 24 | 6 | +40.12 | 70.8% |
| Material class: Al Alloy | 24 | 6 | +40.12 | 70.8% |
| Mo_bin: Mo >3 | 30 | 10 | +30.41 | 66.7% |
| phase_BCC_known: present | 10 | 4 | +24.53 | 60.0% |
| family_and_heat: Fe Alloy / missing | 26 | 7 | +15.80 | 50.0% |
| CP_pH_bin: pH <=3 | 27 | 9 | +15.03 | 51.9% |
| heat_treatment_type: nan | 73 | 18 | +13.48 | 52.1% |
| CP_temp_bin: T <=30 | 80 | 23 | +10.19 | 51.2% |
| N_bin: N 0–0.1 | 25 | 11 | +9.60 | 52.0% |
| test_solution_type: NaCl | 84 | 22 | +9.13 | 51.2% |
| surface_preparation_method: nan | 90 | 22 | +8.64 | 48.9% |
| heat_treatment_cooling: nan | 102 | 25 | +6.70 | 49.0% |

Inspect `persistently_worse_samples.csv` for exact sample IDs, source references, compositions and conditions.
Inspect `category_errors_by_step.csv` before attributing an average loss to a persistent category problem.
Groups overlap and small/source-specific groups are not independent discoveries.
