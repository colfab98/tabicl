# Single-rule informed fine-tuning

All three runs initialize from `checkpoints/tabicl_s1_regression_baseline/step-10000.ckpt`.
Only model weights are loaded. The optimizer and scaler start fresh; step zero saves
the unchanged parent. Checkpoint numbers count **additional S2 updates**, not total
pretraining steps.

| Run | EPIT tables | Soccol tables | Generic tables |
| --- | ---: | ---: | ---: |
| EPIT specialist | 100% | 0% | 0% |
| Soccol specialist | 0% | 100% | 0% |
| Combined | 50% | 50% | 0% |

Every informed table uses the current single-rule sampler: one formula, multiple
retained parents for that formula, and composition perturbation strength 0.3.
The fitted formulas, retained pairs, native schemas, feature shuffling, context-only
imputation and target preprocessing are reused. Targets are 100% formula-generated.
No formula refitting or pair reselection is required.

The existing S2 setup is retained: 2,000 additional updates, global batch 512,
microbatch 1, sequence length 1,024, seed 43, fresh AdamW, learning rate 2e-5 to
5e-6 with the existing quadratic schedule and no warmup. Checkpoints are saved
at step zero and every 100 updates. The schedule horizon remains fixed at 2,000.
The launchers default to one GPU, matching the existing generic S2 control.

## Launch

From the repository root:

```bash
sbatch scripts/epit_pipeline/finetune_single_rule.sbatch
sbatch scripts/soccol_pipeline/finetune_single_rule.sbatch
sbatch scripts/combined_corrosion/finetune_single_rule.sbatch
```

Append `--dry-run` to validate without training, or `--resume` to continue the
same run. Resume restores weights, optimizer, scheduler, scaler and sampling
state. Keep the same resources, paths, parent, recipes and code. A new experiment
needs a new `--run-name`; rerunning without `--resume` refuses existing outputs.
Existing launchers and their runs are unchanged.

New checkpoint directories:

```text
checkpoints/epit_finetuning/generic10k_epit_single_rule100_s2
checkpoints/soccol_finetuning/generic10k_soccol_single_rule100_s2
checkpoints/combined_finetuning/generic10k_combined_single_rule100_s2
```

Logs and plans go under `corrosion_datasets/analysis/{epit_pipeline,soccol_pipeline,combined_corrosion}/single_rule_finetuning/<run-name>/`.
Copy the corresponding checkpoint and analysis directories when moving a run
between hosts, and keep the same resources and installed code for exact resume.

## Exploratory comparison

The evaluation wrapper reuses each dataset's current native preprocessing, five
folds, metrics and checkpoint plots. It compares against the existing matched
generic continuation `checkpoints/epit_finetuning/generic10k_generic_s2`, **not**
the generic pretraining trajectory. Step zero is the unchanged generic 10k model.
Checks reject different parents, schedules, seeds, architectures or GPU counts.

```bash
PYTHONPATH=src:. .venv/bin/python -m scripts.combined_corrosion.evaluate_finetuning_single_rule \
  --prior epit --dataset epit \
  --checkpoint all --checkpoint-scope all --checkpoint-step-interval 0 \
  --max-checkpoint-step 2000 --exploratory-cv-folds 5 --device cpu \
  --output-dir corrosion_datasets/analysis/epit_pipeline/single_rule_s2_through_2000_cv5

PYTHONPATH=src:. .venv/bin/python -m scripts.combined_corrosion.evaluate_finetuning_single_rule \
  --prior soccol --dataset soccol \
  --checkpoint all --checkpoint-scope all --checkpoint-step-interval 0 \
  --max-checkpoint-step 2000 --exploratory-cv-folds 5 --device cpu \
  --output-dir corrosion_datasets/analysis/soccol_pipeline/single_rule_s2_through_2000_cv5

PYTHONPATH=src:. .venv/bin/python -m scripts.combined_corrosion.evaluate_finetuning_single_rule \
  --prior combined --dataset mpea \
  --checkpoint all --checkpoint-scope all --checkpoint-step-interval 0 \
  --max-checkpoint-step 2000 --exploratory-cv-folds 5 --device cpu \
  --output-dir corrosion_datasets/analysis/combined_corrosion/single_rule_s2_through_2000_cv5
```

The output folder gets a dataset suffix (`_epit`, `_soccol`, `_mpea` or `_rebar`).
For combined evaluation on another dataset, change `--dataset`. Use
`--checkpoint-scope common` for only shared checkpoints. Custom training run
names need `--finetune-run <checkpoint-directory>`. An existing evaluation output
is not overwritten; use a fresh output prefix.

These are development comparisons. EPIT/Soccol recipe fitting and retained-pair
selection used all source rows, so their CV scores are not an untouched final-test
estimate. Frozen final-model evaluation must use the frozen outer query partition
through the established final evaluator; this wrapper does not substitute CV for it.

If a new matched generic S2 control is needed (for example, a different GPU
count), use the same launcher with `--prior generic` and pass its directory to
evaluation with `--control-run`:

```bash
sbatch scripts/combined_corrosion/finetune_single_rule.sbatch --prior generic
```
