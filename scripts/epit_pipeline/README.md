# EPIT pipeline

This directory replaces the mixed workflow with five separate stages.
Existing scripts remain unchanged so old results stay reproducible.

## Fixed workflow

1. `prepare_splits.py`
   - Group rows by the 17 alloy-composition values visible to the model.
   - Create one fixed split: 608 development rows and 152 final-test rows.
   - Create five validation folds inside the development rows for Optuna trial
     evaluation.
   - Save row indices, composition groups, balance checks, and dataset hashes.

2. `calibrate_target_rules.py`
   - Support multiple target-rule families through one common interface.
   - During Optuna trial evaluation, calibrate each rule using only that fold's
     training rows.
   - Apply each calibrated rule to that fold's held-out development rows and
     compare its scores with the real EPIT targets.
   - Save direct-evaluation Spearman scores, predictions, coefficient checks,
     and a standalone visual HTML report.
   - Fit one coefficient set on all 608 development rows. Stage 3 passes these
     saved coefficients to synthetic training; it does not recalculate them.
   - Never use the 152 final-test targets.

3. `run_optuna.py`
   - Preserve the `pitting_magpie_full_v1` search, proxy-training, and inference
     settings.
   - Load candidate rule scores and all-development coefficients from Stage 2.
   - Sample rule families with `score / sum(scores)`.
   - Evaluate every trial on the same five saved development folds.
   - Explicitly use only `norm_methods=["none"]`; the power transform is off.
   - Never use the 152 final-test rows.
   - Save every trial result and the complete winning configuration.

4. `train_final.py`
   - Train the winning configuration once.
   - Refuse to start while the shared Optuna study still has active trials.
   - Select permanent checkpoints on the same five development folds only.
   - Freeze the selected checkpoint and Stage 5 evaluation configuration.

5. `evaluate_final.py`
   - Use all 608 development rows as context.
   - Evaluate the frozen final model once on the 152 final-test rows.
   - Do not tune, select, or change anything from these results.

## Split artifacts

Run this from the repository root:

`python -m scripts.epit_pipeline.prepare_splits`

It creates:

- `split_manifest.json`: locked machine-readable indices and safeguards.
- `split_assignments.csv`: row-level inspection file. Final-test target values
  are redacted.
- `split_report.html`: standalone visual report with plots and balance checks.

After approving the split, run:

`python -m scripts.epit_pipeline.calibrate_target_rules`

It creates:

- `calibration_summary.json`: direct comparison across registered rules.
- `<rule_name>.json`: fold-local and all-development calibrated coefficients.
- `direct_evaluation_predictions.csv`: development-only out-of-fold targets and
  rule scores.
- `direct_evaluation_report.html`: standalone visual comparison of predicted
  and observed target ordering.

Registered rule families:

- `current_pren`: current all-alloy reference rule.
- `current_pren_fe_ni`: matched current-rule baseline for Fe/Ni rows.
- `pren_linear`: Cr-Mo-W PREN for Fe and Ni-Cr-Mo alloys.
- `cr_mow_synergy`: PREN plus Cr-Mo/W and temperature-chloride interactions.
- `threshold_saturation`: Cr threshold plus saturating Mo/W benefit.
- `improved_environment`: separate log-chloride, high-temperature,
  temperature-chloride, and acidic-pH penalties.
- `coupled_breakdown`: one environment-versus-material breakdown term.
- `fe_ni_cr_threshold`: separate fixed Cr thresholds for Fe and Ni alloys.
- `method_aware`: Cr-Mo/W synergy plus scan-rate and context-only test-method
  correction.

The Fe/Ni candidate rules use the same 452 development rows and are compared
with `current_pren_fe_ni`. Al-specific and HEA-specific rules were discarded.

Synthetic EPIT training receives candidate scores and calibrated coefficients
from the Stage 2 JSON files. The prior normalizes scores as `score / sum(scores)`
and samples one rule family per synthetic task. If these options are omitted,
the old single-PREN generator remains unchanged.

Run Stage 3 on the GPU node with:

`sbatch scripts/epit_pipeline/run_optuna.sbatch`

After every Stage 3 worker has finished, run Stage 4 with:

`sbatch scripts/epit_pipeline/train_final.sbatch`

The default Stage 4 run trains to 10,000 steps, saves permanent checkpoints
every 500 steps, evaluates them on the five development folds, and freezes the
best checkpoint by mean Spearman.

Only after inspecting the frozen Stage 4 manifest, run the one-shot Stage 5:

`sbatch scripts/epit_pipeline/evaluate_final.sbatch`

## Safeguards

- No composition group crosses an outer or Optuna-fold boundary.
- Target values are used to create and audit the one fixed balanced split.
  After it is approved and frozen, final-test targets must not be used until
  final evaluation.
- Every later stage must load the saved manifest instead of making a new split.
- Dataset hashes must match before saved indices are used.
- Final evaluation must require a frozen configuration.
- EPIT inference explicitly passes `--tabicl-norm-methods none`; saved Optuna,
  checkpoint-selection, and final-evaluation provenance is rejected otherwise.

## Implementation tasks

- [x] Define and test composition handling: no normalization, 0.01 wt.%
  rounding, and complete-linkage grouping at 1.0 wt.% total difference.
- [x] Implement the balanced composition-grouped development/final split.
- [x] Implement the five composition-grouped Optuna folds.
- [x] Define the split manifest and dataset-hash format.
- [x] Generate CSV and standalone HTML split audits.
- [ ] Approve and freeze the generated split.
- [x] Define the common target-rule interface.
- [x] Implement fold-local and final-development rule calibration.
- [x] Add development-only direct rule evaluation and a visual report.
- [x] Reuse the latest Optuna workflow without changing the original script.
- [x] Implement fixed-fold Optuna evaluation and trial records.
- [x] Add the shared-study single-GPU Slurm launcher.
- [x] Explicitly disable power transformation in every EPIT evaluation stage.
- [x] Implement final training and development-only checkpoint selection.
- [x] Implement locked one-shot final evaluation.
- [x] Add later-stage leakage and artifact-consistency tests.
