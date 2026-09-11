# Scripts

The final CorrPFN EPIT workflow is in [epit_pipeline/](epit_pipeline/README.md).
Use its `run_optuna.sbatch`, `train_final.sbatch`, and `evaluate_final.sbatch`
entry points for the v7 study and Trial 56. The general project description is
in [main3.tex](../main3.tex).

## Original TabICL stages and training templates

| File | Purpose |
| --- | --- |
| `train_stage1.sh` | Original upstream stage 1 curriculum commands. |
| `train_stage2.sh` | Original upstream stage 2 curriculum and checkpoint-loading commands. |
| `train_stage3.sh` | Original upstream stage 3 curriculum commands. |
| `train_stage1_reg_baseline_ckpt100_2k.sbatch` | Generic SCM regression baseline template: 2,000 steps, checkpoints every 100 steps, and a 10,000-step scheduler horizon. |
| `train_stage2_reg.sbatch` | Additional regression stage 2 template, including restart from a stage 2 checkpoint or model initialization from stage 1. |
| `write_train_params.py` | Reusable recorder for a training command, parameters, Git status, and Slurm metadata. |

The three original `.sh` files are retained unchanged and contain upstream
placeholder paths. The two regression templates are also retained unchanged.
The 2,000-step baseline template is not the full training recipe for the retained
10,000-step baseline run. The stage 2 regression template uses older pitting
prior settings and a historical default stage 1 checkpoint directory; it needs
an explicit valid input checkpoint and an appropriate prior configuration before
use. The reported Trial 56 model uses the stage 1 pipeline in `epit_pipeline/`.

## Current EPIT profiles and analysis

| File | Purpose |
| --- | --- |
| `build_epit_feature_profile.py` | Build the versioned Fe/Ni–Cr–Mo empirical feature profile used by the final informed generator. |
| `analyze_epit_composition_families.py` | Analyze the retained composition-template bank without using target values. |
| `validate_epit_composition_profile.py` | Inspect composition sampling, family frequencies, active elements, and composition sums. |

The current profile assets and their hashes are part of experiment provenance.
For exploratory rebuilds, use separate outputs instead of replacing the assets
used by the frozen model. The standalone composition sampler's defaults describe
the broader composition profile; they are not the final v7 informed-family mix.

## General evaluation and diagnostics

| File | Purpose |
| --- | --- |
| `eval_corrosion_datasets.py` | Shared corrosion evaluator, including fixed EPIT development/final-test partitions, regression metrics, checkpoint comparisons, pretrained TabICL, and CatBoost. |
| `run_corrosion_dataset_eval.sh` | Configurable Slurm wrapper around that evaluator; supports additional evaluator arguments and `DRY_RUN=1`. |
| `eval_checkpoint_standard.py` | General classification-checkpoint checks on standard sklearn datasets, retained for original TabICL work. |
| `investigate_training_efficiency.py` | Inspect a running training process and its logs for performance bottlenecks. |
| `plot_loss_from_err.py` | Extract training-loss curves from logs. |

For v7 evaluation, use the pipeline's fixed-split entry points. The general
corrosion wrapper can accept the corresponding split flags explicitly; its
defaults do not establish the final v7 evaluation protocol.

## Shared compatibility dependencies

These filenames refer to earlier experiments, but retained code still uses them:

- `optuna_pitting_magpie_prior_search.py` supplies training-command construction,
  search constants, and helpers imported by `epit_pipeline/run_optuna.py`.
- `optuna_pitting_fixed_pren_prior_search.py` supplies storage, subprocess,
  evaluation, and parameter helpers used by that module.
- `eval_pitting_repeated_splits.py` remains the evaluator used by the older
  evaluation-command APIs in both shared modules. It also retains general
  repeated-split comparison functionality and CatBoost coverage.

The current v7 pipeline uses `epit_pipeline/evaluate_optuna_folds.py` for its
development folds. Removing the shared compatibility modules requires a separate
refactor of their callers; they are retained here with their existing behavior.

## Historical archive

Obsolete standalone searches, surrogate-based direct-prior calibration,
fixed-PREN direct scoring, and Trial 15/28, material-ablation, and inhibitor
launchers are under `/home/fcolanto/old_tabicl/scripts/` with their original
filenames. This includes the old `train_stage1_reg.sbatch`, whose configuration
was specifically the Magpie Trial 15 experiment despite its generic filename,
and its `.orig` backup. The dedicated old direct-prior test module is preserved
under `/home/fcolanto/old_tabicl/tests/`.

Original copies of the updated command notes and mixed test file are recorded
with the `scripts_cleanup_*.json` archive log. Shared-module regression tests
remain in the active repository; only checks dedicated to archived entry points
were retired.
