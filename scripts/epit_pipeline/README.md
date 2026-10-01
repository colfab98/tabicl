# CorrPFN EPIT pipeline

This directory now contains two deliberately separated workflows.

| Workflow | Status | Composition schema | Base/with-Magpie width | Rule artifacts |
| --- | --- | --- | --- | --- |
| v7 (`epit_pipeline_optuna_empirical_features_scm_target_v7`) | Frozen historical model and results | `epit_dataset_v1`, 17 elements | 21/31 | `target_rules_v2` |
| v8 (`epit_pipeline_optuna_empirical_features_scm_target_v8`) | Current Optuna study; no selected final model yet | `epit_dataset_v2`, 24 elements | 28/38 | promoted subset of `target_rules_v3` |

The v7 split, model, and result artifacts are not overwritten. Current launchers
use new `optuna_v3`, `final_v2`, and `epit_pipeline_final_v2` paths. The legacy
coefficient-variation diagnostic explicitly pins v7 and `epit_dataset_v1`.

## Current 24-element policy

The fixed material order is:

`Fe, Cr, Ni, Mo, W, N, Nb, C, Si, Mn, Cu, P, S, Al, V, Ta, Re, Ce, Ti, Co, B, Mg, Y, Gd`.

A column's coverage is merely its populated-cell fraction; it cannot distinguish
a truly absent element from incomplete reporting. Missing elemental values are
therefore set to zero only when all reported components sum to 100 ± 0.1 wt.%.
All 760 real rows remain available for evaluation. Three non-closing composition
templates (source Nos. 581, 601, and 773) remain in evaluation but are excluded
from empirical pretraining sampling. The current template bank has 400 eligible
distinct compositions.

The direct evaluator exposes the 24 composition columns first, followed by
temperature, chloride, pH, and test method. Optional Magpie descriptors continue
to use the historical 17-element descriptor subset and are appended after all 28
base features.

## Frozen split policy

The existing 608-development/152-final split and its five development folds are
reused exactly. They were constructed with the legacy 17-element grouping. Adding
columns does not add rows and is not a reason to spend a new outer split. The v3
calibrator accepts this exact frozen legacy schema only when the current dataset
is the exact 24-element schema and records that compatibility in every artifact.
Do not regenerate `splits_v2` for a like-for-like comparison.

## Target-rule evaluation

`target_rules_v2` remains the frozen rule set used by v7. `target_rules_v3`
contains the 24-element direct comparison and the coefficient centers used by
v8. It contains all historical baselines plus:

- `pren_n_linear`: `Cr + 3.3*(Mo + 0.5*W) + 16*N`.
- `cr_mow_n_synergy`: PREN-N plus `sqrt(Cr*(Mo + 0.5*W))`.
- `pren_n_improved_environment`: PREN-N with separate chloride, temperature,
  temperature–chloride, and acidic-pH terms.
- `mo_n_acid_repassivation`: an exploratory positive Mo/W–N–acid interaction.
- `mns_inclusion_penalty`: an exploratory negative `sqrt(Mn*S)` bulk-composition
  proxy; it is not a measured inclusion descriptor.
- `method_aware_pren_n`: the N-aware synergy rule plus context-fitted,
  shrinkage-regularized method offsets.

The v8 runtime uses these nine Fe/Ni rules, in this fixed order:

1. `pren_n_linear` (replaces `pren_linear`)
2. `cr_mow_n_synergy` (replaces `cr_mow_synergy`)
3. `threshold_saturation` (retained)
4. `pren_n_improved_environment` (replaces `improved_environment`)
5. `mo_n_acid_repassivation` (new)
6. `mns_inclusion_penalty` (new)
7. `coupled_breakdown` (retained)
8. `fe_ni_cr_threshold` (retained)
9. `method_aware_pren_n` (replaces `method_aware`)

The loader filters `target_rules_v3` to exactly this set and fails if a rule is
missing or reordered. The three Al rules remain evaluation-only because the
synthetic generator has no Al-family routing.

V8 also enables coefficient variation with strength 0.8. For every synthetic
task, each nonzero calibrated coefficient is multiplied independently by
`U(0.2, 1.8)`. The resulting vector is renormalized to sum to one and redrawn if
it violates a rule-specific upper bound. A zero coefficient stays zero. Thus,
0.8 is a relative multiplier range, not a statistical variance and not an
80-percentage-point change. The dedicated coefficient RNG is seeded with 42;
the run fingerprint records the strength, seed, coefficient centers, and bounds.

The v3 comparison reused the saved folds, masked all 152 final-test targets, and
kept applicability matched. Leading Fe/Ni results were:

| Rule | Mean fold Spearman | Pooled OOF Spearman |
| --- | ---: | ---: |
| `method_aware_pren_n` | 0.5904 | 0.5842 |
| `method_aware` | 0.5833 | 0.5791 |
| `mns_inclusion_penalty` | 0.5695 | 0.5516 |
| `improved_environment` | 0.5675 | 0.5540 |
| `pren_n_improved_environment` | 0.5517 | 0.5433 |
| `mo_n_acid_repassivation` | 0.5514 | 0.5433 |

The small improvements are exploratory development evidence, not a final-test
claim. See `target_rules_v3/calibration_summary.json` and
`target_rules_v3/direct_evaluation_report.html` for all 16 rules and fold details.

## Workflow stages

| Stage | Files | Role |
| --- | --- | --- |
| Frozen split | `prepare_splits.py`, `split_data.py`, `split_refinement.py` | Load/audit rows; historical split construction remains reproducible but is not rerun for v8 comparison. |
| Rule evaluation | `target_rules.py`, `calibrate_target_rules.py` | Fit context-only preprocessing and coefficients, then score held-out development folds. |
| Prior search | `run_optuna.py`, `evaluate_optuna_folds.py` | Train v8 proxy models and compare them on saved development folds. |
| Final training | `train_final.py` | Train the selected v8 configuration and freeze a development-selected checkpoint. |
| Final evaluation | `evaluate_final.py` | Use all development rows as context and evaluate the untouched final-test rows once. |
| Historical variation | `diagnose_target_variation.py`, `run_coefficient_variation.py` | Explicitly v7-only coefficient sensitivity experiment. |

## Artifact layout

- `splits_v2/`: frozen split manifest, lock, assignments, and report.
- `target_rules_v2/`: frozen v7 calibration and training rules.
- `target_rules_v3/`: 24-element direct comparison and v8 training coefficients;
  Optuna loads only the explicit nine-rule promoted subset.
- `optuna_v2/` and `final_v1/...v7/`: retained v7 studies and model.
- `optuna_v3/` and `final_v2/...v8/`: reserved current workflow outputs.
- `checkpoints/epit_pipeline_final_v1/...v7.../`: retained v7 checkpoints.
- `checkpoints/epit_pipeline_final_v2/...v8.../`: current final-training destination.

The v3 comparison contains the retained `al_chloride_temperature` baseline and
the new `al_amphoteric_environment` and `al_composition_environment`
candidates. All three are evaluated only on the 94 Al development rows. The new
rules are direct-evaluation hypotheses and are not available to the synthetic
generator or Optuna.

## Commands

Run the direct rule comparison without replacing v2:

```bash
.venv/bin/python -m scripts.epit_pipeline.calibrate_target_rules
```

The command defaults to `target_rules_v3` and refuses to overwrite existing
artifacts unless `--force` is supplied intentionally.

Run the current staged workflow on a Slurm host:

```bash
sbatch scripts/epit_pipeline/run_optuna.sbatch
sbatch scripts/epit_pipeline/train_final.sbatch
sbatch scripts/epit_pipeline/evaluate_final.sbatch
```

The Optuna launcher defaults to 50 trials, 1,000 training steps per trial,
`target_rules_v3`, the promoted nine-rule subset, and 80% coefficient variation.
The final-training launcher automatically selects the best completed v8 trial
unless `SELECTED_TRIAL_NUMBER` is set explicitly.

Historical supporting evaluation remains available through
`evaluate_baseline_folds.py`, `summarize_checkpoint_comparison.py`, and
`evaluate_checkpoint_comparison.sbatch`. The latter remains pinned to v7 by
design. See `corrosion_datasets/analysis/leakage_policy.md` for data-use limits.
