# Structural Analysis Results

This is a first quantitative pass over the downloaded external corrosion datasets. It estimates structural evidence for informed-SCM settings without using DatacorTech performance.

## Dataset Coverage

| Dataset | Table | Rows | Columns | Numeric material | Numeric environment | Numeric process/history | Numeric molecular descriptors | Numeric targets |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `electrochemical_metrics_alloys` | `Pitting Potential` | 810 | 41 | 24 | 3 | 0 | 0 | 3 |
| `electrochemical_metrics_alloys` | `Repassivation Potential` | 189 | 38 | 19 | 3 | 0 | 0 | 1 |
| `electrochemical_metrics_alloys` | `Crevice Corrosion Potential` | 31 | 30 | 5 | 2 | 0 | 0 | 1 |
| `electrochemical_metrics_alloys` | `Pitting Temp` | 117 | 28 | 12 | 0 | 0 | 0 | 3 |
| `electrochemical_metrics_alloys` | `Crevice Corrosion Temp` | 70 | 34 | 17 | 0 | 0 | 0 | 2 |
| `electrochemical_metrics_alloys` | `HEAs_Ecorr, icorr, ipass, Rate` | 51 | 115 | 9 | 2 | 0 | 0 | 2 |
| `mpea_corrosion` | `Sheet1` | 619 | 39 | 28 | 1 | 0 | 0 | 4 |
| `am_mpea_corrosion` | `AM_MPEA_corrosion_database_V3` | 97 | 25 | 14 | 1 | 0 | 0 | 3 |
| `steel_mortar_corrosion` | `01_Carbonation` | 180 | 17 | 6 | 5 | 0 | 0 | 4 |
| `steel_mortar_corrosion` | `02_Chloride` | 95 | 16 | 5 | 5 | 0 | 0 | 4 |
| `316l_pitting_passivity` | `Epit_Epass_descriptors` | 955 | 8 | 0 | 1 | 0 | 0 | 4 |
| `nace_nist_corr_data` | `CORR-DATA_Database` | 24721 | 15 | 0 | 3 | 1 | 0 | 2 |
| `mooring_steel_seawater` | `OCP_R4_S31_Omax` | 62 | 6 | 0 | 1 | 1 | 0 | 1 |
| `datacor_aluminum_inhibitors` | `DataCor_long_efficiency` | 408 | 23 | 0 | 1 | 0 | 0 | 1 |
| `datacortech_aluminum_inhibitors` | `Efficiencies_with_descriptors` | 2011 | 229 | 0 | 3 | 1 | 0 | 1 |
| `mg_az91_inhibitors` | `Features` | 68 | 877 | 0 | 0 | 0 | 0 | 1 |
| `mg_ze41_inhibitors` | `ze41_mol_desc_db_red` | 60 | 1264 | 0 | 0 | 0 | 0 | 1 |
| `ni_crevice_repassivation` | `Research_data_Saenzetal_table1` | 613 | 10 | 1 | 5 | 0 | 0 | 1 |

## Within-Block Correlation Evidence

| Dataset | Table | Group | Numeric Cols | Pairs | Mean | Median | Q75 |
|---|---|---|---:|---:|---:|---:|---:|
| `316l_pitting_passivity` | `Epit_Epass_descriptors` | target | 4 | 6 | 0.281 | 0.116 | 0.411 |
| `am_mpea_corrosion` | `AM_MPEA_corrosion_database_V3` | material | 14 | 91 | 0.275 | 0.211 | 0.400 |
| `am_mpea_corrosion` | `AM_MPEA_corrosion_database_V3` | target | 3 | 3 | 0.220 | 0.218 | 0.283 |
| `datacortech_aluminum_inhibitors` | `Efficiencies_with_descriptors` | direct_intervention | 2 | 1 | 0.018 | 0.018 | 0.018 |
| `datacortech_aluminum_inhibitors` | `Efficiencies_with_descriptors` | environment | 3 | 3 | 0.360 | 0.222 | 0.452 |
| `electrochemical_metrics_alloys` | `Crevice Corrosion Potential` | environment | 2 | 1 | 0.966 | 0.966 | 0.966 |
| `electrochemical_metrics_alloys` | `Crevice Corrosion Potential` | material | 5 | 10 | 0.398 | 0.358 | 0.596 |
| `electrochemical_metrics_alloys` | `Crevice Corrosion Temp` | material | 17 | 120 | 0.394 | 0.363 | 0.583 |
| `electrochemical_metrics_alloys` | `Crevice Corrosion Temp` | target | 2 | 1 | 0.989 | 0.989 | 0.989 |
| `electrochemical_metrics_alloys` | `HEAs_Ecorr, icorr, ipass, Rate` | environment | 2 | 1 | 0.510 | 0.510 | 0.510 |
| `electrochemical_metrics_alloys` | `HEAs_Ecorr, icorr, ipass, Rate` | material | 9 | 36 | 0.294 | 0.299 | 0.434 |
| `electrochemical_metrics_alloys` | `HEAs_Ecorr, icorr, ipass, Rate` | target | 2 | 1 | 0.106 | 0.106 | 0.106 |
| `electrochemical_metrics_alloys` | `Pitting Potential` | environment | 3 | 3 | 0.145 | 0.123 | 0.215 |
| `electrochemical_metrics_alloys` | `Pitting Potential` | material | 24 | 234 | 0.116 | 0.068 | 0.176 |
| `electrochemical_metrics_alloys` | `Pitting Potential` | target | 3 | 3 | 0.989 | 0.993 | 0.993 |
| `electrochemical_metrics_alloys` | `Pitting Temp` | electrochem_control | 2 | 1 | 0.459 | 0.459 | 0.459 |
| `electrochemical_metrics_alloys` | `Pitting Temp` | material | 12 | 61 | 0.281 | 0.231 | 0.423 |
| `electrochemical_metrics_alloys` | `Pitting Temp` | target | 3 | 3 | 0.995 | 0.996 | 0.997 |
| `electrochemical_metrics_alloys` | `Repassivation Potential` | environment | 3 | 3 | 0.340 | 0.356 | 0.449 |
| `electrochemical_metrics_alloys` | `Repassivation Potential` | material | 19 | 128 | 0.310 | 0.280 | 0.488 |
| `mpea_corrosion` | `Sheet1` | material | 28 | 378 | 0.124 | 0.067 | 0.171 |
| `mpea_corrosion` | `Sheet1` | target | 4 | 6 | 0.276 | 0.144 | 0.364 |
| `nace_nist_corr_data` | `CORR-DATA_Database` | environment | 3 | 3 | 0.434 | 0.154 | 0.575 |
| `nace_nist_corr_data` | `CORR-DATA_Database` | target | 2 | 1 | 0.905 | 0.905 | 0.905 |
| `ni_crevice_repassivation` | `Research_data_Saenzetal_table1` | environment | 5 | 10 | 0.201 | 0.148 | 0.340 |
| `steel_mortar_corrosion` | `01_Carbonation` | environment | 5 | 10 | 0.278 | 0.161 | 0.371 |
| `steel_mortar_corrosion` | `01_Carbonation` | material | 6 | 15 | 0.291 | 0.246 | 0.405 |
| `steel_mortar_corrosion` | `01_Carbonation` | target | 4 | 6 | 0.622 | 0.698 | 0.813 |
| `steel_mortar_corrosion` | `02_Chloride` | environment | 5 | 10 | 0.238 | 0.125 | 0.217 |
| `steel_mortar_corrosion` | `02_Chloride` | material | 5 | 10 | 0.319 | 0.330 | 0.490 |
| `steel_mortar_corrosion` | `02_Chloride` | target | 4 | 6 | 0.341 | 0.310 | 0.505 |

## Material-Environment Interaction Probes

The probe compares cross-validated ridge R2 from material+environment numeric features versus the same features plus material x environment products. Positive delta supports interaction structure, but this is a rough diagnostic rather than a final model.

| Dataset | Table | Target | N | Base R2 | Interaction R2 | Delta |
|---|---|---|---:|---:|---:|---:|
| `am_mpea_corrosion` | `AM_MPEA_corrosion_database_V3` | `Corrosion current density (µA/cm2)` | 58 | 0.797 | 0.796 | -0.001 |
| `am_mpea_corrosion` | `AM_MPEA_corrosion_database_V3` | `Corrosion potential (mV vs. SCE)` | 77 | 0.207 | 0.205 | -0.003 |
| `am_mpea_corrosion` | `AM_MPEA_corrosion_database_V3` | `Pitting potential (mV vs. SCE)` | 58 | 0.803 | 0.803 | 0.000 |
| `mpea_corrosion` | `Sheet1` | `Calculated passive window` | 336 | 0.315 | 0.418 | 0.103 |
| `mpea_corrosion` | `Sheet1` | `Corrosion current density (microA/cm2)` | 555 | 0.048 | 0.046 | -0.002 |
| `mpea_corrosion` | `Sheet1` | `Corrosion potential (mV vs SCE)` | 576 | 0.057 | -0.633 | -0.690 |
| `mpea_corrosion` | `Sheet1` | `Pitting potential (mV vs SCE)` | 316 | 0.516 | 0.647 | 0.132 |
| `ni_crevice_repassivation` | `Research_data_Saenzetal_table1` | `ER.CREV. VECS` | 407 | 0.217 | 0.324 | 0.107 |
| `steel_mortar_corrosion` | `01_Carbonation` | `Chloride-to-hydroxide concentration ratio` | 180 | -0.625 | -0.837 | -0.211 |
| `steel_mortar_corrosion` | `01_Carbonation` | `Corrosion Potential of Steel vs Cu/CuSO4` | 180 | 0.626 | 0.644 | 0.018 |
| `steel_mortar_corrosion` | `01_Carbonation` | `Corrosion Rate of Steel` | 180 | 0.688 | 0.716 | 0.028 |
| `steel_mortar_corrosion` | `02_Chloride` | `Chloride-to-hydroxide concentration ratio` | 95 | 0.958 | 0.967 | 0.010 |
| `steel_mortar_corrosion` | `02_Chloride` | `Corrosion Potential of Steel vs Cu/CuSO4` | 95 | 0.284 | 0.211 | -0.074 |
| `steel_mortar_corrosion` | `02_Chloride` | `Corrosion Rate of Steel` | 95 | -0.032 | -0.033 | -0.001 |
| `steel_mortar_corrosion` | `02_Chloride` | `Electrical Resistivity of mortar` | 95 | 0.830 | 0.883 | 0.053 |

## History Evidence

| Dataset | Condition | N | Lag-1 Pearson | Lag-1 Spearman |
|---|---|---:|---:|---:|
| `mooring_steel_seawater` | `S=31 T=2 Omax` | 21 | 0.735 | 0.495 |
| `mooring_steel_seawater` | `S=31 T=32 Omax` | 21 | 0.247 | 0.018 |
| `mooring_steel_seawater` | `S=31 T=17 Omax` | 20 | 0.257 | -0.068 |

## Parameter Implications

| Parameter | Suggested Default | Suggested Range | Evidence Summary | Interpretation |
|---|---:|---:|---|---|
| `informed_feature_block_strength` | 0.25 | 0.20-0.35 | mean=0.337; median=0.302; n=20 | Observed within-block numeric dependence is moderate but not a calibrated prior strength; use a soft block signal rather than increasing it aggressively. |
| `informed_interaction_strength` | 0.25 | 0.10-0.35 | mat-env corr median=0.148; interaction delta median=0.000 | Material-environment coupling is structurally sensible, but simple interaction gains are small and mixed; keep this moderate. |
| `informed_history_strength` | 0.25 | 0.00-0.50 | mean=0.194; median=0.068; n=3 | Time-series evidence is narrow and mixed; path dependence is a corrosion motif, but a universal strong autoregressive component is not supported. |
| `informed_intervention_strength` | 0.10 | 0.05-0.20 | mean=0.382; median=0.377; n=10 | Processing/intervention labels are sometimes informative, but the evidence is sparse and categorical, so keep this weak and conditional. |
| `informed_prior_ratio` | 0.50 | 0.25-0.75 | mean=NA; median=NA; n=0 | External datasets do not directly estimate this training-mixture parameter; choose via pre-specified ablation, not DatacorTech test selection. |

## Caveats

- This is structural evidence, not DatacorTech model selection.
- Numeric correlations use pairwise complete Spearman correlations over column-gated scalar measurement fields.
- Text identifiers, formulas, process labels, prose methods, and categorical environment labels are excluded from numeric evidence; bounded categorical target association is still estimated with eta squared.
- The physical-range-prior question is still separate: this pass mostly addresses block and interaction settings.
- Recommended values should become a small pre-specified ablation grid, not a final claim that one value is optimal.
