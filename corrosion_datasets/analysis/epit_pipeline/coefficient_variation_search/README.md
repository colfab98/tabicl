# Coefficient Variation: Controlled 2,000-Step Experiment

This new Optuna study varies only coefficient strength. It does not append to
v7, change the frozen rule calibration, or evaluate final-test rows.

Default design:

- Strengths: 0, 0.3, 0.5, 0.8.
- Training seeds: 42, 43, 44, crossed with every strength (12 runs).
- Start each model from scratch and stop at 2,000 steps.
- Retain Trial 56's architecture, batch size, lambda, task mixture, composition
  perturbation, and 10,000-step learning-rate schedule.
- Preserve four DataLoader workers, one prior job per worker, and one GPU.
- Save and evaluate checkpoints at steps 1,000 and 2,000 on all five development
  folds. Only the step-2,000 mean fold Spearman is the Optuna objective.
- Compare mean performance and standard deviation across training seeds, plus
  paired gains over the zero-strength run with the same seed. Do not select a
  single lucky seed. No pruning or best-checkpoint selection.

## Commands

Preview the complete plan without creating Optuna trials or model checkpoints:

```bash
.venv/bin/python -m scripts.epit_pipeline.run_coefficient_variation --dry-run
```

Submit training explicitly on the cluster:

```bash
sbatch scripts/epit_pipeline/run_coefficient_variation.sbatch
```

To split the grid across sequential jobs:

```bash
sbatch scripts/epit_pipeline/run_coefficient_variation.sbatch --n-trials 4
```

Repeat the same command/study to run remaining cells; completed cells are not
rerun. Only one runner per experiment directory is allowed. The launcher uses
a new journal under `$OPTUNA_STORAGE_ROOT/$STUDY_NAME/`; standalone Python uses
a journal inside this results directory. Keep the same storage when resuming.

The CLI allows alternate discrete strengths and seed lists. Zero must always be
included. Changing the grid, settings, input hashes, or relevant source code
requires a new study name. `training_seed` is an Optuna grid coordinate only to
schedule repeats, not a hyperparameter whose best value should be selected.

## Files and Safety

Each study has its own subdirectory containing `experiment.json`, per-trial
commands/logs/evaluations, and `summary.csv` / `summary.json`. Checkpoints go to
`checkpoints/epit_coefficient_variation/<study>/trial_<number>/`.
No existing checkpoint or evaluation directory is overwritten. A dry run writes
only the fingerprint and plan. GPU training is never launched by a dry run.

The runner refuses unrelated existing studies and changed fingerprints. A failed
trial remains failed; grid cells are not automatically retried. An interrupted
RUNNING trial must be resolved in Optuna before restarting. Use a new study for
a clean rerun after a failed/interrupted experiment. Incomplete grids are marked
as such and no winning strength is reported until every seed/strength completes.

## Training Integration

`--pitting_coefficient_variation` defaults to zero. At zero, the existing prior
draws no extra random numbers and keeps its original coefficient values. Nonzero
strength requires calibrated upper bounds and uses the same sampler as the
generator-only diagnostic. The bounds are loaded from verified calibration
artifacts, not independently duplicated constants.

Coefficient sampling has a separate seeded stream per DataLoader worker. It
does not consume the existing NumPy/PyTorch/Python streams. The new experiment
also supplies the optional `--python_seed`; old launchers leave that unset.
GPU operations may still have nondeterminism. Changing coefficients can affect
rare target-validity retries, so matching seeds is a paired experimental design,
not a guarantee that every generated table remains identical forever.

The existing failed-SCM handling is not changed in this experiment. Calibration
still uses the original all-development coefficients, so these are development
comparisons, not a new fully nested evaluation. Positive results at 2,000 steps
would need confirmation at the intended longer training budget.
