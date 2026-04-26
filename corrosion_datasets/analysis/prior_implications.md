# Preliminary Prior Implications

These are first-pass implications from the downloaded external datasets. They should be treated as design evidence, not tuned values.

## Strongly Supported

- Keep material/composition and environment as distinct feature blocks.
- Include explicit material-environment interaction structure.
- Keep processing/history as a separate block where such variables exist.
- Treat electrochemical measurements as a tightly related response block.
- Use references/provenance for documentation only, not model features.

## Moderately Supported

- A moderate within-block strength is defensible for composition and environment groups because real datasets repeatedly organize columns this way.
- A moderate material-environment interaction strength is defensible because nearly every dataset conditions corrosion outcome on electrolyte, concentration, pH, temperature, humidity, salinity, or oxygen.
- A stronger history/process component is defensible for datasets with heat treatment, AM process, immersion duration, or exposure time, but this signal is not equally rich in every dataset.

## Keep Conservative

- Do not increase intervention/treatment strength aggressively from these files alone. Treatment/process variables are present, but often as coarse labels.
- Do not treat electrochemical targets as safe covariates. They are useful for response-block structure, but leakage-prone as features.
- Do not calibrate numerical strengths directly from row counts or target correlations until the analysis protocol is fixed.

## Candidate Safe Ablation Ranges

These ranges are a starting point for pre-specified experiments, not fitted estimates:

| Config | Conservative Range | Rationale |
|---|---:|---|
| `informed_feature_block_strength` | 0.15 to 0.45 | Recurring grouped schemas, but heterogeneous datasets. |
| `informed_interaction_strength` | 0.20 to 0.50 | Material-environment coupling is pervasive. |
| `informed_history_strength` | 0.50 to 0.80 | History/process is important but unevenly observed. |
| `informed_intervention_strength` | 0.10 to 0.30 | Treatment/process variables exist, but causal effects are not calibrated here. |
| `informed_prior_ratio` | 0.25 to 0.75 | Controls how much domain structure enters synthetic training; should be selected before final testing. |

The current v11/v12-style settings remain scientifically plausible as conservative heuristic values. The external datasets mainly justify the direction of the structure, not exact constants.

## Quantitative First Pass

A first structural analysis has now been added:

- `scripts/analyze_structure.py`
- `column_group_map.md`
- `structural_analysis_results.md`
- `structural_analysis_results.json`
- `parameter_decision.md`

The main conclusion is not to make the informed prior uniformly stronger. The external datasets support moderate block structure and moderate material-environment coupling, but the simple interaction probes are mixed and the history evidence is narrow. The most useful next experiment is therefore a small pre-specified ablation around the current v11/v12 settings, especially `informed_prior_ratio` and weaker history/structure settings.
