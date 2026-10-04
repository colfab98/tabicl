# Soccol Pitting-Potential Database

This folder preserves the original workbook collection from Dimitri Soccol's
`Pitting_potential_database` repository. The 2024 workbook is the current
primary version; the two 2023 workbooks are retained as upstream historical
snapshots.

- Source provenance, versions, checksums, and licensing: `manifest.yml`
- Workbook fields and confirmed conventions: `schema_summary.md`
- Paper and source literature: `literature.md`
- Implemented decisions and remaining sensitivity analyses: `investigation_notes.md`
- Per-source material and handling registry:
  `../../analysis/soccol_source_conventions/source_conventions.csv`
- Reproducible preprocessing builder: `prepare_benchmark.py`
- Previous-EPIT-compatible split builder: `prepare_composition_split.py`
- Synthetic profile and calibrated-rule builder: `prepare_pipeline_assets.py`
- Optuna launcher: `../../../scripts/soccol_pipeline/run_optuna.py`
- Model-ready tables, frozen split, and feature roles: `processed/`

The files under `raw/` are preserved copies. No composition blank has been
replaced, Fe has not been reconstructed, and the confirmed `2000Russell.Ti`
error has not been altered in the raw workbook. All corrections and encodings are
confined to `processed/`.
