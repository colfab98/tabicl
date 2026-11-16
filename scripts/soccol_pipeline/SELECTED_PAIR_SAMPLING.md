# Soccol selected-pair experiment

This is the Soccol counterpart of
[EPIT selected-pair training](../epit_pipeline/SELECTED_PAIR_SAMPLING.md).
It is a separate exploratory recipe; the previous Soccol training command,
processed dataset, frozen splits and sampling bank remain available unchanged.

## Matching procedure

1. Refit all nine existing Soccol formula families and the previously selected
   extension on every composition-accepted row. Reuse each formula's existing
   fitting objective, constraints and input transformations. Save coefficients,
   fitted preprocessing and the positive affine conversion to target mV.
2. Obtain one generic step-10000 prediction per query row using the existing
   five-fold exploratory CV, eight estimators, seed 42, native TabICL inference,
   median predictions and CPU. Context never contains the current query.
3. Compare absolute errors for every eligible sample/formula pair. Retain a
   pair only when its formula error is strictly smaller; discard ties.
4. Draw retained pairs uniformly. Each synthetic row uses one parent and one
   formula. Keep that parent's experimental conditions and missingness;
   multiply composition amounts by log-normal noise with strength 0.30 and
   close to 100 wt.%. Zeros stay zero. Evaluate its frozen formula, then apply
   one shared target standardization to the completed synthetic table.
5. Informed targets are 100% formula-based, with no SCM target contribution
   and no coefficient variation. The informed **table proportion** is retained
   pairs / tested pairs, not 100%. EPIT's exact global allocation code carries
   rounding across batches, mixes groups of four tables within each batch,
   and partitions the global batch between GPU processes.

Both datasets use the same generic generator, architecture, training seeds,
optimizer and schedule: 512 tables, microbatch 4, length 1024, CRPS with 999
quantiles, AdamW LR 1e-4, 2% warmup, and cosine LR horizon 10000. The LR horizon
is independent of the stopping step. Training defaults to 10000 steps, temporary
checkpoints every 100 steps, permanent checkpoints every 500, and a 47-hour
Slurm limit. The inherited launcher requires the stopping step to be divisible
by 500. No model evaluation is launched automatically.

## Dataset differences retained explicitly

| Property | EPIT | Soccol |
|---|---|---|
| Real input columns | 70 | 58 |
| Formula-fitting / parent rows | 760 | 3745 composition-accepted rows |
| Existing formula families | Nine EPIT families | Nine Soccol families |
| Previously selected extension | Heat categories, penalty 0.3 | Surface numeric, penalty 1.0 |
| Formula prediction units | mV SCE | Saved E_pit mV Ag/AgCl (3 M KCl) |
| Exploratory CV queries | 760 across five folds | 3755 across five folds |

Soccol's saved preprocessing is unchanged. Its 272 unresolved development
compositions remain excluded; its ten unresolved final queries remain in
evaluation. Neither group is used to fit formulas or create synthetic parents.
The 3745 eligible rows comprise the original 2950 accepted development rows
plus 795 accepted final rows. Their original task IDs are preserved. The
preparation output includes `excluded_rows.csv` listing the same 282 unresolved
compositions, with original IDs and audit fields.

The existing Soccol exploratory CV uses 805 original final queries in fold 1
and partitions accepted development into 738/738/737/737 queries in folds 2–5.
Each fold uses the remaining rows of that same 3755-row population as context.
The ten unresolved final rows consequently also appear in other folds' context,
as in the existing CV implementation. Formula comparisons filter the generic
predictions to the 3745 eligible parents, giving exactly 37450 tested pairs.

The extension name and penalty come from the saved Soccol formula extension
study. No new formula/penalty search or Optuna run is performed. The formula
adapter explicitly enables the established reviewed-data handling, including
the existing rules for unknown environmental values. It does not zero-fill
unknown saved compositions or alter model inputs.

All-data fitting and pair selection use labels from the former test partition;
subsequent CV scores are exploratory, not an independent generalization estimate.
The ordinary frozen-model evaluation path continues to use the fixed outer queries.

## Commands

Run from the updated repository on the machine doing the work. Updating sv1225
does not update sv1223 or sv1221. The reviewed data, generic checkpoint and saved
`formula_extension_study_v1/results.json` must also be present on that machine.
The normal repository environment is used; no new package dependency is added.

Prepare coefficients, generic five-fold predictions and retained pairs on CPU:

```bash
cd /home/fcolanto/projects/tabicl
PYTHONPATH=src:. .venv/bin/python -m scripts.soccol_pipeline.prepare_selected_pairs --dry-run
PYTHONPATH=src:. .venv/bin/python -m scripts.soccol_pipeline.prepare_selected_pairs
cat corrosion_datasets/analysis/soccol_pipeline/selected_pairs_allaccepted_generic10k/summary.json
```

Re-running preparation reuses completed generic folds only when their saved
checksums and the recipe identity match. Changed settings/code require a new
`--output-dir`; pass that same directory as `--recipe-dir` when training.
Preparation does not overwrite the old development-fitted formula artifacts.

Validate the training command without starting a model, then submit training:

```bash
PYTHONPATH=src:. .venv/bin/python -m scripts.soccol_pipeline.train_selected_pairs --dry-run
sbatch scripts/soccol_pipeline/train_selected_pairs.sbatch
```

The default run is `soccol_selected_pairs_allaccepted_generic10k_v1` under
`checkpoints/soccol_selected_pairs/`. Its plan and log are under
`corrosion_datasets/analysis/soccol_pipeline/selected_pair_training/`.
Use `--run-name` for a distinct experiment. As in EPIT, this launcher refuses an
existing run; resuming requires the saved underlying trainer command. The trainer
restores model, optimizer, scheduler, random state and consumed stream position
and checks matching recipe/code/loader configuration and GPU process count.

Evaluate every available checkpoint through step 2000 with five-fold CV on CPU:

```bash
PYTHONPATH=src:. .venv/bin/python -m scripts.soccol_pipeline.evaluate_checkpoint_comparison \
  --candidate-run checkpoints/soccol_selected_pairs/soccol_selected_pairs_allaccepted_generic10k_v1 \
  --candidate-label soccol_selected_pairs \
  --checkpoint all --max-checkpoint-step 2000 --checkpoint-step-interval 0 \
  --exploratory-cv-folds 5 --device cpu \
  --output-dir corrosion_datasets/analysis/soccol_pipeline/selected_pairs_through_2000_cv5
```

After training reaches 10000, compare that explicit step with generic step 10000:

```bash
PYTHONPATH=src:. .venv/bin/python -m scripts.soccol_pipeline.evaluate_checkpoint_comparison \
  --candidate-run checkpoints/soccol_selected_pairs/soccol_selected_pairs_allaccepted_generic10k_v1 \
  --candidate-label soccol_selected_pairs --checkpoint 10000 \
  --exploratory-cv-folds 5 --device cpu \
  --output-dir corrosion_datasets/analysis/soccol_pipeline/selected_pairs_step10000_cv5
```

Use a new output directory per evaluation. `--checkpoint latest` independently
chooses each model's latest checkpoint and can compare unequal training steps.
`all` evaluates each available step; a missing generic step is not fabricated.

## Alignment checks

`tests/test_soccol_selected_pairs.py` exercises real all-accepted formula fits,
saved prediction replay, unchanged conditions/zeros, formula-only targets,
exact generic table equality, global batch allocation, sampling resume and a
small real CPU optimizer/checkpoint resume. It compares parsed EPIT/Soccol
training settings and verifies that Soccol planning restores shared module
configuration before another EPIT plan. Preparation tests mock generic model
predictions and store fixtures in temporary directories; they are not scientific
comparison results. Existing EPIT sampler and shared generic generator source
files are unchanged, preserving their recipe/checkpoint hashes.
