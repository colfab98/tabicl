# EPIT and Soccol column mapping

We matched the latest `epit_reference_v1` and `soccol_reference_v1` schemas.

- **49 predictor names were already shared**, with matching declared types and units.
- **21 predictors existed only in EPIT; 9 existed only in Soccol.**
- The proposed union contained **79 predictors and one separate target**.
- We retained the existing predictor names. We mapped the two target names to `reported_pitting_potential_mV`.
- We saved this mapping without merging rows or changing any dataset, value or frozen split.

A dash means the column was absent from that dataset. A column present in both schemas could still contain missing values. In particular, all four second heat-treatment columns were empty in Soccol.

The [CSV](column_mapping.csv) also recorded meanings, qualifications and populated-row counts. Counts used all 760 EPIT and 4,027 Soccol reference rows; they did not select rows for training or evaluation. The accepted Soccol development subset and frozen queries remained unchanged.

## Composition

| EPIT column | Soccol column | Shared column | Unit |
|---|---|---|---|
| Fe | Fe | Fe | wt.% |
| Cr | Cr | Cr | wt.% |
| Ni | Ni | Ni | wt.% |
| Mo | Mo | Mo | wt.% |
| W | W | W | wt.% |
| N | N | N | wt.% |
| Nb | Nb | Nb | wt.% |
| C | C | C | wt.% |
| Si | Si | Si | wt.% |
| Mn | Mn | Mn | wt.% |
| Cu | Cu | Cu | wt.% |
| P | P | P | wt.% |
| S | S | S | wt.% |
| Al | Al | Al | wt.% |
| V | V | V | wt.% |
| Ta | Ta | Ta | wt.% |
| Re | Re | Re | wt.% |
| Ce | Ce | Ce | wt.% |
| Ti | Ti | Ti | wt.% |
| Co | Co | Co | wt.% |
| B | B | B | wt.% |
| Mg | Mg | Mg | wt.% |
| Y | Y | Y | wt.% |
| Gd | Gd | Gd | wt.% |

## Test conditions

| EPIT column | Soccol column | Shared column | Unit |
|---|---|---|---|
| CP_temp | CP_temp | CP_temp | degC |
| CP_pH | CP_pH | CP_pH | dimensionless |
| scan_rate | scan_rate | scan_rate | mV/s |
| Test_area_cm2 | Test_area_cm2 | Test_area_cm2 | cm2 |
| CP_time | CP_time | CP_time | h |
| CP_agitation | CP_agitation | CP_agitation | binary |

## Test solution

| EPIT column | Soccol column | Shared column | Unit |
|---|---|---|---|
| CP_Cl | CP_Cl | CP_Cl | mol/L |
| test_solution_type | test_solution_type | test_solution_type | category |
| CP_aeration | CP_aeration | CP_aeration | category |
| H2SO4_eq_L | — | H2SO4_eq_L | eq/L |
| H2SO4_mol_L | H2SO4_mol_L | H2SO4_mol_L | mol/L |
| H2S_partial_pressure_atm | — | H2S_partial_pressure_atm | atm |
| HCl_eq_L | — | HCl_eq_L | eq/L |
| HCl_mol_L | — | HCl_mol_L | mol/L |
| KCl_eq_L | — | KCl_eq_L | eq/L |
| KCl_mol_L | — | KCl_mol_L | mol/L |
| MgCl2_mol_L | — | MgCl2_mol_L | mol/L |
| Na2S2O3_mol_L | — | Na2S2O3_mol_L | mol/L |
| Na2SO4_mol_L | — | Na2SO4_mol_L | mol/L |
| NaBr_eq_L | — | NaBr_eq_L | eq/L |
| NaBr_mol_L | — | NaBr_mol_L | mol/L |
| NaClO4_mol_L | — | NaClO4_mol_L | mol/L |
| NaCl_eq_L | — | NaCl_eq_L | eq/L |
| NaCl_mol_L | — | NaCl_mol_L | mol/L |
| NaCl_wt_percent | — | NaCl_wt_percent | wt.% |
| NaHCO3_mol_L | — | NaHCO3_mol_L | mol/L |
| WO4_mol_L | — | WO4_mol_L | mol/L |
| CP_Br | CP_Br | CP_Br | mol/L |
| CP_SO4 | CP_SO4 | CP_SO4 | mol/L |
| — | CP_OH | CP_OH | mol/L |
| — | CP_CO3 | CP_CO3 | mol/L |
| — | CP_NO3 | CP_NO3 | mol/L |
| — | CP_PO4 | CP_PO4 | mol/L |
| — | CP_MoO4 | CP_MoO4 | mol/L |
| — | CP_CrO4 | CP_CrO4 | mol/L |

## Surface preparation

| EPIT column | Soccol column | Shared column | Unit |
|---|---|---|---|
| Prep_grinding_grit | Prep_grinding_grit | Prep_grinding_grit | reported grit grade |
| surface_preparation_method | surface_preparation_method | surface_preparation_method | category |
| preparation_solution_type | preparation_solution_type | preparation_solution_type | category |
| Prep_time | Prep_time | Prep_time | h |
| — | Prep_Ra_micron | Prep_Ra_micron | um |
| — | Prep_pH | Prep_pH | dimensionless |
| — | Prep_redox | Prep_redox | V; reference not established for this column |

## Heat treatment

| EPIT column | Soccol column | Shared column | Unit |
|---|---|---|---|
| heat_treatment_type | heat_treatment_type | heat_treatment_type | category |
| heat_treatment_temperature_C | heat_treatment_temperature_C | heat_treatment_temperature_C | degC |
| heat_treatment_duration_h | heat_treatment_duration_h | heat_treatment_duration_h | h |
| heat_treatment_cooling | heat_treatment_cooling | heat_treatment_cooling | category |
| heat_treatment_2_type | heat_treatment_2_type | heat_treatment_2_type | category |
| heat_treatment_2_temperature_C | heat_treatment_2_temperature_C | heat_treatment_2_temperature_C | degC |
| heat_treatment_2_duration_h | heat_treatment_2_duration_h | heat_treatment_2_duration_h | h |
| heat_treatment_2_cooling | heat_treatment_2_cooling | heat_treatment_2_cooling | category |

## Microstructure

| EPIT column | Soccol column | Shared column | Unit |
|---|---|---|---|
| phase_FCC | — | phase_FCC | binary |
| phase_BCC | — | phase_BCC | binary |
| phase_HCP | — | phase_HCP | binary |
| phase_IM | — | phase_IM | binary |

## Measurement reference

| EPIT column | Soccol column | Shared column | Unit |
|---|---|---|---|
| reference_type | reference_type | reference_type | category |

## Target

| EPIT column | Soccol column | Shared column | Unit |
|---|---|---|---|
| Epit, mV (SCE) Avg. | E_pit | reported_pitting_potential_mV | mV |

## Matching rules

- Chemical amounts and ion concentrations remained separate: for example, `NaCl_mol_L` and `CP_Cl`, or `H2SO4_mol_L` and `CP_SO4`. We did not infer the whole solution from an ion.
- Molarity, normality, mass percentage and gas partial pressure retained their separate columns and units.
- Test-solution pH and preparation-solution pH remained separate, as did grit and surface roughness.
- `Prep_redox` retained its reported voltage with an unspecified reference. The target reference column did not establish its reference.
- Existing categorical labels were preserved, including more specific gas conditions and preparation operations.
- Missing environment, heat-treatment, phase and reference information remained missing. Composition values followed the previous dataset-specific decisions; we introduced no new zero filling.
- `CP_ion_other`, raw descriptions, source identifiers, audit flags and row identifiers were excluded from the predictor list. Identifiers and original target notation remained metadata.
- The target retained reported mV values without electrode conversion. The existing EPIT modeling approximation used 800 for `>800` and 1256 for `>1256`; the source notation remained available. This mapping itself changed no values.
