# Corrosion Dataset Evidence Workspace

This folder keeps external corrosion datasets separate from TabICL training code. The purpose is to document source provenance, inspect feature/target schemas, and extract only high-level corrosion structure that can safely inform synthetic prior configuration.

Raw files should stay unchanged. Any cleaning, derived tables, or scripts should go under a separate analysis or processed-data folder.

## Current Dataset Status

| Dataset | Status | Primary Use For Informed Prior Work |
|---|---|---|
| `electrochemical_metrics_alloys` | downloaded and inspected | Broad alloy composition, environment, test-method, and electrochemical target structure. |
| `mpea_corrosion` | downloaded and inspected | MPEA composition, phases, processing, electrolyte, and electrochemical targets. |
| `am_mpea_corrosion` | downloaded and inspected | Additive-manufacturing/processing signal plus MPEA corrosion targets. |
| `316l_pitting_passivity` | downloaded and inspected | Electrochemical curve/descriptor structure for fixed 316L under controlled NaCl/scan-rate settings. |
| `steel_mortar_corrosion` | downloaded and inspected | Clean block structure: mixture, material, environment, and electrochemical response. |
| `nace_nist_corr_data` | downloaded and inspected | Large broad material-environment-rate corpus; useful but heterogeneous. |
| `mooring_steel_seawater` | downloaded and inspected | Small controlled seawater time-series data; useful for environment/history logic. |
| `pipeline_steel_pitting` | pending | Supplementary dataset was not found/downloaded yet; ignored for now. |

## Documentation Files

Each inspected dataset has:

- `manifest.yml`: source links, local raw files, checksums, and provenance notes.
- `schema_summary.md`: rows/sheets/columns, feature groups, targets, leakage risks, and relevance to the informed prior.
- `README.md`: short local entry point.

The cross-dataset analysis notes are in `analysis/`.

## Leakage Boundary

Use these datasets to justify structural prior assumptions such as feature grouping, material-environment interaction, processing/history relevance, and electrochemical response grouping.

Do not tune informed-prior numeric values against DatacorTech test performance, copy target thresholds, or encode dataset-specific outcome distributions into the TabICL prior.
