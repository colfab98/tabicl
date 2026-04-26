# Structural Analysis Results

This is a first quantitative pass over the downloaded external corrosion datasets. It estimates structural evidence for informed-SCM settings without using DatacorTech performance.

## Dataset Coverage

| Dataset | Table | Rows | Columns | Numeric material | Numeric environment | Numeric history/intervention | Numeric targets |
|---|---|---:|---:|---:|---:|---:|---:|
| `electrochemical_metrics_alloys` | `Pitting Potential` | 810 | 41 | 24 | 4 | 4 | 3 |
| `electrochemical_metrics_alloys` | `Repassivation Potential` | 189 | 38 | 19 | 4 | 2 | 1 |
| `electrochemical_metrics_alloys` | `Crevice Corrosion Potential` | 31 | 30 | 5 | 2 | 1 | 1 |
| `electrochemical_metrics_alloys` | `Pitting Temp` | 117 | 28 | 12 | 1 | 2 | 3 |
| `electrochemical_metrics_alloys` | `Crevice Corrosion Temp` | 70 | 34 | 17 | 0 | 1 | 2 |
| `electrochemical_metrics_alloys` | `HEAs_Ecorr, icorr, ipass, Rate` | 51 | 115 | 9 | 3 | 1 | 2 |
| `mpea_corrosion` | `Sheet1` | 619 | 39 | 29 | 2 | 1 | 4 |
| `am_mpea_corrosion` | `AM_MPEA_corrosion_database_V3` | 97 | 25 | 15 | 1 | 0 | 3 |
| `steel_mortar_corrosion` | `01_Carbonation` | 180 | 17 | 7 | 5 | 0 | 4 |
| `steel_mortar_corrosion` | `02_Chloride` | 95 | 16 | 6 | 5 | 0 | 4 |
| `316l_pitting_passivity` | `Epit_Epass_descriptors` | 955 | 8 | 0 | 1 | 1 | 4 |
| `nace_nist_corr_data` | `CORR-DATA_Database` | 24721 | 15 | 2 | 3 | 2 | 2 |
| `mooring_steel_seawater` | `OCP_R4_S31_Omax` | 62 | 6 | 0 | 1 | 1 | 1 |

## Within-Block Correlation Evidence

| Dataset | Table | Group | Numeric Cols | Pairs | Mean | Median | Q75 |
|---|---|---|---:|---:|---:|---:|---:|
| `316l_pitting_passivity` | `Epit_Epass_descriptors` | target | 4 | 6 | 0.281 | 0.116 | 0.411 |
| `am_mpea_corrosion` | `AM_MPEA_corrosion_database_V3` | material | 15 | 104 | 0.296 | 0.221 | 0.496 |
| `am_mpea_corrosion` | `AM_MPEA_corrosion_database_V3` | target | 3 | 3 | 0.220 | 0.218 | 0.283 |
| `electrochemical_metrics_alloys` | `Crevice Corrosion Potential` | environment | 2 | 1 | 0.966 | 0.966 | 0.966 |
| `electrochemical_metrics_alloys` | `Crevice Corrosion Potential` | material | 5 | 10 | 0.398 | 0.358 | 0.596 |
| `electrochemical_metrics_alloys` | `Crevice Corrosion Temp` | material | 17 | 120 | 0.394 | 0.363 | 0.583 |
| `electrochemical_metrics_alloys` | `Crevice Corrosion Temp` | target | 2 | 1 | 0.989 | 0.989 | 0.989 |
| `electrochemical_metrics_alloys` | `HEAs_Ecorr, icorr, ipass, Rate` | environment | 3 | 3 | 0.347 | 0.510 | 0.519 |
| `electrochemical_metrics_alloys` | `HEAs_Ecorr, icorr, ipass, Rate` | material | 9 | 36 | 0.294 | 0.299 | 0.434 |
| `electrochemical_metrics_alloys` | `HEAs_Ecorr, icorr, ipass, Rate` | target | 2 | 1 | 0.106 | 0.106 | 0.106 |
| `electrochemical_metrics_alloys` | `Pitting Potential` | environment | 4 | 6 | 0.264 | 0.173 | 0.285 |
| `electrochemical_metrics_alloys` | `Pitting Potential` | history | 4 | 2 | 0.450 | 0.450 | 0.664 |
| `electrochemical_metrics_alloys` | `Pitting Potential` | material | 24 | 234 | 0.116 | 0.068 | 0.176 |
| `electrochemical_metrics_alloys` | `Pitting Potential` | target | 3 | 3 | 0.989 | 0.993 | 0.993 |
| `electrochemical_metrics_alloys` | `Pitting Temp` | history | 2 | 1 | 0.459 | 0.459 | 0.459 |
| `electrochemical_metrics_alloys` | `Pitting Temp` | material | 12 | 61 | 0.281 | 0.231 | 0.423 |
| `electrochemical_metrics_alloys` | `Pitting Temp` | target | 3 | 3 | 0.995 | 0.996 | 0.997 |
| `electrochemical_metrics_alloys` | `Repassivation Potential` | environment | 4 | 5 | 0.463 | 0.356 | 0.541 |
| `electrochemical_metrics_alloys` | `Repassivation Potential` | history | 2 | 1 | 0.587 | 0.587 | 0.587 |
| `electrochemical_metrics_alloys` | `Repassivation Potential` | material | 19 | 128 | 0.310 | 0.280 | 0.488 |
| `mpea_corrosion` | `Sheet1` | environment | 2 | 1 | 0.720 | 0.720 | 0.720 |
| `mpea_corrosion` | `Sheet1` | material | 29 | 393 | 0.124 | 0.067 | 0.168 |
| `mpea_corrosion` | `Sheet1` | target | 4 | 6 | 0.276 | 0.144 | 0.364 |
| `nace_nist_corr_data` | `CORR-DATA_Database` | environment | 3 | 3 | 0.434 | 0.156 | 0.575 |
| `nace_nist_corr_data` | `CORR-DATA_Database` | history | 2 | 1 | 0.114 | 0.114 | 0.114 |
| `nace_nist_corr_data` | `CORR-DATA_Database` | material | 2 | 1 | 0.470 | 0.470 | 0.470 |
| `nace_nist_corr_data` | `CORR-DATA_Database` | target | 2 | 1 | 0.906 | 0.906 | 0.906 |
| `steel_mortar_corrosion` | `01_Carbonation` | environment | 5 | 10 | 0.368 | 0.140 | 0.778 |
| `steel_mortar_corrosion` | `01_Carbonation` | material | 7 | 21 | 0.354 | 0.347 | 0.555 |
| `steel_mortar_corrosion` | `01_Carbonation` | target | 4 | 6 | 0.622 | 0.698 | 0.813 |
| `steel_mortar_corrosion` | `02_Chloride` | environment | 5 | 10 | 0.352 | 0.125 | 0.739 |
| `steel_mortar_corrosion` | `02_Chloride` | material | 6 | 15 | 0.290 | 0.267 | 0.407 |
| `steel_mortar_corrosion` | `02_Chloride` | target | 4 | 6 | 0.341 | 0.310 | 0.505 |

## Material-Environment Interaction Probes

The probe compares cross-validated ridge R2 from material+environment numeric features versus the same features plus material x environment products. Positive delta supports interaction structure, but this is a rough diagnostic rather than a final model.

| Dataset | Table | Target | N | Base R2 | Interaction R2 | Delta |
|---|---|---|---:|---:|---:|---:|
| `am_mpea_corrosion` | `AM_MPEA_corrosion_database_V3` | `Corrosion current density (µA/cm2)` | 58 | 0.797 | 0.796 | -0.001 |
| `am_mpea_corrosion` | `AM_MPEA_corrosion_database_V3` | `Pitting potential (mV vs. SCE)` | 58 | 0.803 | 0.803 | 0.000 |
| `mpea_corrosion` | `Sheet1` | `Corrosion potential (mV vs SCE)` | 425 | -0.037 | -0.645 | -0.608 |
| `mpea_corrosion` | `Sheet1` | `Pitting potential (mV vs SCE)` | 257 | 0.417 | 0.515 | 0.098 |
| `steel_mortar_corrosion` | `01_Carbonation` | `Chloride-to-hydroxide concentration ratio` | 180 | -0.614 | -0.631 | -0.017 |
| `steel_mortar_corrosion` | `01_Carbonation` | `Corrosion Potential of Steel vs Cu/CuSO4` | 180 | 0.782 | 0.788 | 0.006 |
| `steel_mortar_corrosion` | `01_Carbonation` | `Corrosion Rate of Steel` | 180 | 0.707 | 0.752 | 0.045 |
| `steel_mortar_corrosion` | `01_Carbonation` | `Electrical Resistivity of mortar` | 180 | -0.842 | -0.519 | 0.322 |
| `steel_mortar_corrosion` | `02_Chloride` | `Chloride-to-hydroxide concentration ratio` | 95 | 0.957 | 0.967 | 0.010 |
| `steel_mortar_corrosion` | `02_Chloride` | `Corrosion Potential of Steel vs Cu/CuSO4` | 95 | 0.413 | 0.409 | -0.004 |
| `steel_mortar_corrosion` | `02_Chloride` | `Corrosion Rate of Steel` | 95 | 0.101 | 0.162 | 0.061 |
| `steel_mortar_corrosion` | `02_Chloride` | `Electrical Resistivity of mortar` | 95 | 0.843 | 0.925 | 0.082 |

## History Evidence

| Dataset | Condition | N | Lag-1 Pearson | Lag-1 Spearman |
|---|---|---:|---:|---:|
| `mooring_steel_seawater` | `S=31 T=2 Omax` | 21 | 0.735 | 0.495 |
| `mooring_steel_seawater` | `S=31 T=32 Omax` | 21 | 0.247 | 0.018 |
| `mooring_steel_seawater` | `S=31 T=17 Omax` | 20 | 0.257 | -0.068 |

## Parameter Implications

| Parameter | Suggested Default | Suggested Range | Evidence Summary | Interpretation |
|---|---:|---:|---|---|
| `informed_feature_block_strength` | 0.30 | 0.20-0.45 | mean=0.385; median=0.354; n=23 | Observed within-block numeric dependence is usually moderate, supporting the current conservative default rather than a large increase. |
| `informed_interaction_strength` | 0.35 | 0.25-0.55 | mat-env corr median=0.129; interaction delta median=0.008 | Material-environment coupling is structurally present; interaction probes should be reviewed per target before increasing above the current default. |
| `informed_history_strength` | 0.70 | 0.50-0.80 | mean=0.194; median=0.068; n=3 | Time-series evidence is narrow but strongly path-dependent where available, so the current high default is plausible but not broadly calibrated. |
| `informed_intervention_strength` | 0.20 | 0.10-0.30 | mean=0.177; median=0.180; n=7 | Processing/intervention labels are informative in MPEA/AM-MPEA, but mostly categorical and dataset-specific, so keep this conservative. |
| `informed_prior_ratio` | 0.50 | 0.25-0.75 | mean=NA; median=NA; n=0 | External datasets do not directly estimate this training-mixture parameter; choose via pre-specified ablation, not DatacorTech test selection. |

## Caveats

- This is structural evidence, not DatacorTech model selection.
- Numeric correlations use pairwise complete Spearman correlations and ignore many text/categorical fields except for target association via eta squared.
- The physical-range-prior question is still separate: this pass mostly addresses block and interaction settings.
- Recommended values should become a small pre-specified ablation grid, not a final claim that one value is optimal.
