# Preliminary Prior Implications

These are first-pass implications from the downloaded external datasets. They should be treated as design evidence, not tuned values.

## Strongly Supported

- Keep material/composition and environment as distinct feature blocks.
- Include material-environment interaction structure, but keep the strength moderate.
- Keep processing/history as separate concepts where such variables exist.
- Treat electrochemical measurements as a tightly related response block.
- Use references/provenance for documentation only, not model features.

## Moderately Supported

- A moderate within-block strength is defensible for composition and environment groups because real datasets repeatedly organize columns this way.
- A moderate material-environment interaction strength is defensible because corrosion outcomes are conditioned by electrolyte, concentration, pH, temperature, humidity, salinity, or oxygen, but the simple corrected interaction probes are small and mixed.
- A weak history/process component is defensible as a corrosion motif, but the downloaded time-series evidence is narrow and mixed.

## Keep Conservative

- Do not increase intervention/treatment strength aggressively from these files alone. Treatment/process variables are present, but often as coarse labels.
- Do not treat electrochemical targets as safe covariates. They are useful for response-block structure, but leakage-prone as features.
- Do not calibrate numerical strengths directly from row counts or target correlations until the analysis protocol is fixed.
- Do not let identifiers, alloy formulas, process labels, method prose, or categorical environment labels enter numeric correlation evidence.

## Candidate Safe Ablation Ranges

These ranges are a starting point for pre-specified experiments, not fitted estimates:

| Config | Conservative Range | Rationale |
|---|---:|---|
| `informed_feature_block_strength` | 0.20 to 0.35 | Corrected within-block dependence is moderate, not strong. |
| `informed_interaction_strength` | 0.10 to 0.35 | Coupling is scientifically sensible, but corrected interaction deltas are small and mixed. |
| `informed_history_strength` | 0.00 to 0.50 | Path dependence exists in corrosion, but this dataset collection does not justify a strong universal autoregressive term. |
| `informed_intervention_strength` | 0.05 to 0.20 | Treatment/process variables exist, but the usable evidence is sparse and mostly categorical. |
| `informed_prior_ratio` | 0.25 to 0.75 | Controls how much domain structure enters synthetic training; should be selected before final testing. |

The current v11/v12-style settings remain scientifically motivated as a
historical control, but the stricter analysis no longer supports them as
preferred defaults. The external datasets justify the direction of the
structure, not exact constants.

## Quantitative First Pass

A first structural analysis has now been added:

- `scripts/analyze_structure.py`
- `column_group_map.md`
- `structural_analysis_results.md`
- `structural_analysis_results.json`
- `parameter_decision.md`

The main conclusion is not to make the informed prior uniformly stronger. The
external datasets support corrosion-like feature families and moderate block
structure, but the corrected interaction probes are mixed and the history
evidence is narrow. The most useful next experiment is therefore a pre-specified
comparison against softer structural settings, especially weaker history,
weaker intervention, and possibly a material-heavy block allocation.
