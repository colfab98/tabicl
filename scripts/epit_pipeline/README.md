# CorrPFN EPIT pipeline

This directory contains the reviewed-data workflow and retained historical workflows.

| Workflow | Status | Composition schema | Base/with-Magpie width | Rule artifacts |
| --- | --- | --- | --- | --- |
| v7 (`epit_pipeline_optuna_empirical_features_scm_target_v7`) | Frozen historical model and results | `epit_dataset_v1`, 17 elements | 21/31 | `target_rules_v2` |
| v8 (`epit_pipeline_optuna_empirical_features_scm_target_v8`) | Previous 28-column workflow | `epit_dataset_v2`, 24 elements | 28/38 | promoted subset of `target_rules_v3` |
| v9 (`epit_pipeline_optuna_empirical_features_scm_target_v9`) | Reviewed inputs and refitted formulas; previous search settings | `epit_dataset_v4`, 24 elements, development only | 70/80 | nine formulas in `target_rules_v4` |
| v10 (`epit_pipeline_optuna_empirical_features_scm_target_v10`) | Current defaults; 100% informed search with broader ranges | `epit_dataset_v4`, 24 elements, development only | 70/80 | nine formulas in `target_rules_v4` |

Historical split, model, and result artifacts were retained. Current launchers
use v10 study subdirectories under `optuna_v3`, `final_v2`, and `epit_pipeline_final_v2`. The legacy
coefficient-variation diagnostic explicitly pins v7 and `epit_dataset_v1`.

## Current 24-element policy

The fixed material order is:

`Fe, Cr, Ni, Mo, W, N, Nb, C, Si, Mn, Cu, P, S, Al, V, Ta, Re, Ce, Ti, Co, B, Mg, Y, Gd`.

The current loader preserved the reviewed values in `epit_reference_v1`, including
accepted zeros, sulfur approximations and the Ti/Fe correction. It did not repeat
the historical 100 ± 0.1 wt.% check or normalize real compositions. All 760 real
rows were retained. Active informed sampling used 255 distinct compositions from
411 Fe and 41 Ni–Cr–Mo development rows.

The current synthetic and real inputs followed the reviewed 70-column schema:
24 composition columns and 46 environment/history/measurement columns. Related
environment values and their missingness were sampled together. Optional Magpie
descriptors continued to use the historical 17-element descriptor subset and
were appended after all 70 base features. The original test-method field was
retained privately for the existing formulas, rather than as a model input.

## Frozen split policy

The existing 608-development/152-final split and its five development folds are
reused exactly. They were constructed with the legacy 17-element grouping. Adding
columns does not add rows and is not a reason to spend a new outer split. The
calibrator accepts this exact frozen legacy schema only when the current dataset
is the exact 24-element schema and records that compatibility in every artifact.
Do not regenerate `splits_v2` for a like-for-like comparison.

## Target-rule evaluation

`target_rules_v4` refitted the same nine promoted formulas using reviewed
composition, `CP_temp`, `CP_Cl` and `CP_pH` values. `formula_data.py` mapped these
values to the formulas' existing input names without changing their definitions.
Each fold fitted preprocessing and coefficients on its applicable context rows
only; the final coefficient centers used all 452 Fe/Ni development rows. The
152 frozen final queries were excluded from fitting and scoring. Mean fold
Spearman remained close to v3; the largest absolute change was 0.0034, and
`method_aware_pren_n` remained highest at 0.5908. The new artifacts recorded the
reviewed input hashes and are the defaults for Python and Slurm training commands.

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

The current runtime retained these nine Fe/Ni rules promoted in v8, in this fixed order:

1. `pren_n_linear` (replaces `pren_linear`)
2. `cr_mow_n_synergy` (replaces `cr_mow_synergy`)
3. `threshold_saturation` (retained)
4. `pren_n_improved_environment` (replaces `improved_environment`)
5. `mo_n_acid_repassivation` (new)
6. `mns_inclusion_penalty` (new)
7. `coupled_breakdown` (retained)
8. `fe_ni_cr_threshold` (retained)
9. `method_aware_pren_n` (replaces `method_aware`)

The loader required exactly this set in `target_rules_v4`; it also retained
support for the explicit promoted subset of historical `target_rules_v3`.
Missing or reordered rules caused an error. The three Al rules remained evaluation-only because the
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
`target_rules_v3/direct_evaluation_report.html` for all 18 rules and fold details.

## Current Optuna search

The default EPIT search fixed the informed share at **100%** and removed it
from Optuna's sampled parameters. The three remaining settings were:

| Setting | Previous range | Current v10 range |
| --- | --- | --- |
| Formula contribution to synthetic targets | continuous 0–1 | continuous 0–1, retained in full |
| MLP share within informed generators | 0, 0.25, 0.50, 0.70, 0.75, 1 | continuous 0–1; the tree share is the remainder |
| Composition perturbation strength | continuous 0–0.15 | continuous 0–0.30 |

Composition perturbation strength controls the existing sampler's variation;
it is not a percentage-point change to an element concentration. The formula
contribution is separate from the fixed informed share: an informed table may
still combine its SCM target with a formula target.

The study fingerprint recorded both the fixed ratio and the new ranges, so
previous studies could not be resumed with these settings. The new policy
applied to the current expanded EPIT `empirical_features_scm_target` mode.
Historical EPIT modes retained their previous search settings. The reviewed
Soccol workflow subsequently adopted the same MLP and formula ranges; its separate
[historical-results review](../../corrosion_datasets/analysis/soccol_pipeline/optuna_range_review_v1/README.md)
led to an approved **0–0.15** composition range for Soccol. EPIT retained
**0–0.30**.

Final training read the fixed ratio from the selected study's fingerprint.
`--override-informed-prior-ratio` (or `OVERRIDE_INFORMED_PRIOR_RATIO` in the
Slurm launcher) remained available for the planned 0.25, 0.50, 0.75 and 1.00
comparison using the same selected settings. Those overrides were recorded
separately from the selected trial and used distinct default output paths.

## Workflow stages

| Stage | Files | Role |
| --- | --- | --- |
| Frozen split | `prepare_splits.py`, `split_data.py`, `split_refinement.py` | Load/audit rows; historical split construction remains reproducible and is not rerun. |
| Rule evaluation | `formula_data.py`, `target_rules.py`, `calibrate_target_rules.py` | Load reviewed inputs, fit context-only preprocessing and coefficients, then score held-out development folds. |
| Prior search | `run_optuna.py`, `evaluate_optuna_folds.py` | Train v10 proxy models and compare them on saved development folds. |
| Final training | `train_final.py` | Train the selected v10 configuration and freeze a development-selected checkpoint. |
| Final evaluation | `evaluate_final.py` | Use all development rows as context and evaluate the untouched final-test rows once. |
| Historical variation | `diagnose_target_variation.py`, `run_coefficient_variation.py` | Explicitly v7-only coefficient sensitivity experiment. |

## Artifact layout

- `splits_v2/`: frozen split manifest, lock, assignments, and report.
- `target_rules_v2/`: frozen v7 calibration and training rules.
- `target_rules_v3/`: 24-element direct comparison and v8 training coefficients;
  Optuna loads only the explicit nine-rule promoted subset.
- `target_rules_v4/`: current nine-rule refit using reviewed inputs, fold scores,
  report, input-change audit and comparison against v3.
- `optuna_v2/` and `final_v1/...v7/`: retained v7 studies and model.
- `optuna_v3/...v10/` and `final_v2/...v10/`: current workflow destinations; earlier outputs were retained.
- `checkpoints/epit_pipeline_final_v1/...v7.../`: retained v7 checkpoints.
- `checkpoints/epit_pipeline_final_v2/...v10.../`: current final-training destination.

The v3 comparison contains the retained `al_chloride_temperature` baseline and
the new `al_amphoteric_environment` and `al_composition_environment`
candidates. All three are evaluated only on the 94 Al development rows. The new
rules are direct-evaluation hypotheses and are not available to the synthetic
generator or Optuna.

## Commands

Build a reviewed-input formula calibration:

```bash
.venv/bin/python -m scripts.epit_pipeline.calibrate_target_rules
```

The command defaults to the nine promoted formulas and `target_rules_v4`, and refuses to overwrite existing
artifacts unless `--force` is supplied intentionally.

Run the current staged workflow on a Slurm host:

```bash
sbatch scripts/epit_pipeline/run_optuna.sbatch
sbatch scripts/epit_pipeline/train_final.sbatch
sbatch scripts/epit_pipeline/evaluate_final.sbatch
```

The Optuna launcher defaults to 50 trials, 1,000 training steps per trial,
100% informed training, `target_rules_v4`, the same nine formulas, and 80%
coefficient variation. Selection uses the same five development folds.
The final-training launcher automatically selects the best completed v10 trial
unless `SELECTED_TRIAL_NUMBER` is set explicitly.

Historical supporting evaluation remains available through
`evaluate_baseline_folds.py`, `summarize_checkpoint_comparison.py`, and
`evaluate_checkpoint_comparison.sbatch`. The latter remains pinned to v7 by
design. See `corrosion_datasets/analysis/leakage_policy.md` for data-use limits.

## PySR symbolic-regression baseline

The standalone CPU baseline uses only Fe, Cr, Ni, Mo, W, N, temperature,
chloride, and pH. It reuses the frozen development folds, masks all 152
final-test targets, and mean-imputes from each fold's context rows. Its
exploratory grammar uses the raw features and allows `+`, `-`, `*`, `/`,
`square`, `sqrt`, `log`, and `exp`. Unary functions cannot be nested within
one another, and the existing size/depth caps remain active. Periodic operators
are excluded because the selected EPIT variables have no periodic mechanism.

Install the pinned analysis dependency and run the timed one-fold smoke test:

```bash
pip install -e '.[corrosion-eval]'
python -m scripts.epit_pipeline.run_pysr --dry-run
```

The dry run defaults to fold 1, 100 iterations, and a ten-minute fit timeout.
It writes the equations, validation predictions, measured runtime, and a rough
five-fold/1,000-iteration projection under
`corrosion_datasets/analysis/epit_pipeline/pysr_v1/`. PySR initializes Julia
and its packages on first import, so an offline CPU server must have both the
Python environment and Julia depot prepared beforehand.

## PySR automatic formula discovery

`run_pysr_discovery.py` is the repository source of truth for the matched
comparison with the strongest Fe/Ni target rule. It evaluates the same 452
Fe/Ni development rows while keeping all final-test rows absent from the
bundle. Candidate physical inputs are all 24 elements plus temperature,
chloride, and pH. PySR's random-forest selector is fitted independently inside
each context fold; test method enters only through additive categorical
offsets. No PREN score or hand-designed corrosion interaction is supplied.

The deployable offline copy lives under `standalone/epit_pysr/`. Its large
wheelhouse and Julia archives are intentionally ignored by Git, while the
canonical runner, exporter, launchers, manifest, and documentation remain in
the main project.

Run the tracked version against an exported bundle with:

```bash
.venv/bin/python -m scripts.epit_pipeline.run_pysr_discovery \
  --bundle-dir standalone/epit_pysr --dry-run
```
