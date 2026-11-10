# CorrPFN EPIT pipeline

## Current experiment: v14

EPIT v14 and Soccol v8 use the original grouped TabICL encoder (group size 3)
for both generic and informed tables, without a missing-value embedding or
artificial feature masking. Both sources shuffle features.

The follow-up audit fixed grouping of padded training tables: circular groups
now wrap within each table's real columns, preserving the original rule for
unpadded tables. Padding from another table no longer changes the relationships
presented to the model. This applies to both sources. Model implementation hashes
are included in study/resume identities. Use fresh studies and retrain every
ratio, including 0%; v13/v7 runs cannot be resumed across this training fix.

The local generic generator uses its original full-batch parameter plan,
preprocessing and rejection sampling. A 0% run is that generic path. Mixed runs
consume a prefix of that same generic batch and an independently generated
informed batch; informed Optuna parameters never enter generic generation.
One original context/query split is sampled per global training batch.
Informed targets are calculated before filling missing inputs from synthetic
context means (zero for a wholly missing context column). The resulting inputs
use the original feature preprocessing, shuffling and constant-column filtering.
The physical feature profiles and fitted formulas are unchanged.

Optuna still searches the three informed-prior parameters at 100% informed.
Final training can override the ratio to 0, .25, .5, .75 or 1 using the existing
`train_final.sbatch`. Grouping and model architecture are not search parameters.

All new TabICL evaluations use Latin permutations and `none`/`power`
normalization, with eight ensemble views by default, standard imputation and
constant-column removal. Optuna/checkpoint selection use development folds;
final evaluation uses the frozen outer queries. Historical missing-aware
checkpoint comparisons require `--legacy-model-inputs`.

Launch the new search from the repository on the cluster:

```bash
sbatch --gpus=1 \
  --export=ALL,STUDY_NAME=epit_pipeline_optuna_empirical_features_scm_target_v14,N_TRIALS=50 \
  scripts/epit_pipeline/run_optuna.sbatch
```

For two GPUs also set `NPROC_PER_NODE=2` and request `--gpus=2`. Final training
already derives its process count from the requested GPU count. Global batch
size and schedule stay fixed. To preserve the original sequential generic RNG,
each rank reproduces the global CPU batch and trains only its assigned slice;
GPU work is divided, but CPU generation does not become cheaper. Resume retains
the saved worker/RNG state and requires the original process count.

New runs use new study/checkpoint directories and reject incompatible old
fingerprints. Existing checkpoints cannot be resumed into v14. There is no
runtime dependency on another checkout of TabICL. Regression support remains
shared across every ratio; historical seeds are not reproduced automatically.

The sections below record the earlier workflow and data preparation. The v14
settings above supersede their training/evaluation settings and launch commands.

This directory contains the reviewed-data workflow and retained historical workflows.

| Workflow | Status | Composition schema | Base/with-Magpie width | Rule artifacts |
| --- | --- | --- | --- | --- |
| v7 (`epit_pipeline_optuna_empirical_features_scm_target_v7`) | Frozen historical model and results | `epit_dataset_v1`, 17 elements | 21/31 | `target_rules_v2` |
| v8 (`epit_pipeline_optuna_empirical_features_scm_target_v8`) | Previous 28-column workflow | `epit_dataset_v2`, 24 elements | 28/38 | promoted subset of `target_rules_v3` |
| v9 (`epit_pipeline_optuna_empirical_features_scm_target_v9`) | Reviewed inputs and refitted formulas; previous search settings | `epit_dataset_v4`, 24 elements, development only | 70/80 | nine formulas in `target_rules_v4` |
| v10 (`epit_pipeline_optuna_empirical_features_scm_target_v10`) | Previous unpaired sampling; 100% informed search with broader ranges | `epit_dataset_v4`, 24 elements, development only | 70/80 | nine formulas in `target_rules_v4` |
| v11 (`epit_pipeline_optuna_empirical_features_scm_target_v11`) | Initial indexed streams; superseded after an infinite-retry defect | `epit_dataset_v4`, 24 elements, development only | 70/80 | nine formulas in `target_rules_v4` |
| v12 (`epit_pipeline_optuna_empirical_features_scm_target_v12`) | Historical missing-aware paired streams | `epit_dataset_v4`, 24 elements, development only | 70/80 | nine formulas in `target_rules_v4` |
| v13 (`epit_pipeline_optuna_empirical_features_scm_target_v13`) | Superseded: grouped training allowed padding to change feature groups | `epit_dataset_v4`, 24 elements, development only | 70 | nine formulas in `target_rules_v4` |
| v14 (`epit_pipeline_optuna_empirical_features_scm_target_v14`) | Current defaults; original generic batches, corrected grouping for padded tables, common evaluation | `epit_dataset_v4`, 24 elements, development only | 70 | nine formulas in `target_rules_v4` |

See [Shared synthetic streams](PAIRED_STREAMS.md) for matching Optuna/final data,
generic controls, checkpoint behavior and unchanged legacy runs.

Historical split, model, and result artifacts were retained. Current launchers
use v14 study subdirectories under `optuna_v3`, `final_v2`, and `epit_pipeline_final_v2`. The legacy
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

| Setting | Previous range | Current v12 range |
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

Final training automatically selects the best completed trial from the shared
Optuna journal. If that trial's worker-local evaluation files or proxy checkpoint
are absent on the training node, it uses the validated journal record by default;
no trial number is required. Existing local files are still checked against their
saved hashes. Incomplete journal records, mismatched inputs and changed files
remain errors. The final manifest records which evidence was available.
The optional `--no-allow-missing-selected-trial-artifacts` flag requires all
local proxy files. This behavior is shared with Soccol final training.

## Workflow stages

| Stage | Files | Role |
| --- | --- | --- |
| Frozen split | `prepare_splits.py`, `split_data.py`, `split_refinement.py` | Load/audit rows; historical split construction remains reproducible and is not rerun. |
| Rule evaluation | `formula_data.py`, `target_rules.py`, `calibrate_target_rules.py` | Load reviewed inputs, fit context-only preprocessing and coefficients, then score held-out development folds. |
| Prior search | `run_optuna.py`, `evaluate_optuna_folds.py` | Train v12 proxy models and compare them on saved development folds. |
| Final training | `train_final.py` | Train the selected v12 configuration and freeze a development-selected checkpoint. |
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
- `optuna_v3/...v12/` and `final_v2/...v12/`: current workflow destinations; earlier outputs were retained.
- `checkpoints/epit_pipeline_final_v1/...v7.../`: retained v7 checkpoints.
- `checkpoints/epit_pipeline_final_v2/...v12.../`: current final-training destination.

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
The final-training launcher automatically selects the best completed v12 trial
unless `SELECTED_TRIAL_NUMBER` is set explicitly.

### Fresh v12 search on sv1225 and sv1224

Use the same committed files on both nodes. Include the new
`src/tabicl/prior/paired_streams.py` and `checkpoint.py` modules when staging:

```bash
git add src/tabicl scripts tests \
  corrosion_datasets/DATASET_PROCESSING.md \
  corrosion_datasets/NEW_DATASET_WORKFLOW.md corrosion_datasets/README.md
```

The reviewed EPIT export, frozen split, formula artifacts and sampling profiles
are already tracked. Checkpoints and local Optuna evaluation outputs do not
need to be copied to start final training on the other node.

From `/home/fcolanto/projects/tabicl`, submit 25 trials per node for 50 new
trials in total. Run the following command once in each terminal. Slurm chooses
the execution node; both jobs use the same v12 journal on `RZ-Dienste`:

```bash
sbatch --export=ALL,STUDY_NAME=epit_pipeline_optuna_empirical_features_scm_target_v12,N_TRIALS=25 scripts/epit_pipeline/run_optuna.sbatch
```

`N_TRIALS` counts additional trials **per job**, not a shared study-wide limit.
The jobs use different Optuna suggestion seeds but identical training/data
seeds. Study initialization is locked across both identity writes so concurrent
workers cannot read a half-written fingerprint. Incompatible settings are
rejected before training.

Wait for the searches to finish, then use `train_final.sbatch` for each model.
It selects the best completed v12 trial and preserves native informed features
and NaNs, the learned missing-value embedding with individual columns, and no
artificial feature masking. Generic tables retain their original recipe and
receive no added NaNs. The final manifest records this input policy; the v12
stream fingerprint is unchanged.

Specify the GPU count and informed ratio directly on the existing script:

```bash
sbatch --gpus=2 scripts/epit_pipeline/train_final.sbatch --override-informed-prior-ratio 0.25
sbatch --gpus=2 scripts/epit_pipeline/train_final.sbatch --override-informed-prior-ratio 0.50
sbatch --gpus=2 scripts/epit_pipeline/train_final.sbatch --override-informed-prior-ratio 0.75
sbatch --gpus=2 scripts/epit_pipeline/train_final.sbatch --override-informed-prior-ratio 1.00
```

Use `--gpus=1` for single-GPU runs. The process count follows the allocated GPUs;
an explicit `NPROC_PER_NODE` or `--nproc-per-node` still overrides it. Arguments
after the script name are forwarded to `train_final.py` and override environment
defaults. For the optional generic control, use ratio `0.0`. This control has the
same architecture, but its missing token receives no training signal from NaNs;
it is not a reproduction of the historical generic checkpoint.

New runs train from scratch for 10,000 steps with global batch 512 and separate
output/checkpoint directories. The script uses the shared v12 journal by
default. Add `--selected-trial-number NUMBER` to pin the same best completed
trial explicitly across runs; if it is no longer best when a queued job starts,
that job stops instead of selecting another trial.

To continue an interrupted run, add `--resume-training` and pin its original
trial. For the single-GPU trial-15 run with 75% informed data:

```bash
sbatch --gpus=1 scripts/epit_pipeline/train_final.sbatch \
  --nproc-per-node 1 --selected-trial-number 15 \
  --override-informed-prior-ratio 0.75 --resume-training
```

Keep the original GPU/process count and all training settings. Resume verifies
the original `train.log`, selects the highest saved checkpoint step, and restores
model, optimizer, scheduler, scaler, RNG and synthetic-loader state. It preserves
the original log and writes a separate `train_resume_step_*.log`. The original
trial may be resumed even if Optuna has since found a better trial. Add
`--dry-run` to validate without training or writing files. Missing or incompatible
state is rejected; no restart from scratch is substituted. A completed run skips
training and resumes development-only checkpoint selection. Frozen models cannot
be overwritten; `--resume-checkpoint-selection` remains available for selection
only after training has finished.

The process-count change is recorded. Two-GPU training requires the seed
correction described in
[shared streams](PAIRED_STREAMS.md#multi-gpu-correction-1-november-2026) on the
execution node. CPU generation and synchronization can limit the speedup.

The default scheduler horizon is 10,000 for both search and final training.
For planned 15,000-step final runs, set `SCHEDULER_TOTAL_STEPS=15000` on **all**
search and final submissions, and `MAX_STEPS=15000` on final submissions only.
Keep the same seeds and worker settings on both nodes. Existing `STORAGE` or
`OPTUNA_STORAGE_ROOT` environment overrides must also point to the same shared
location. Final-query rows remain excluded from Optuna and checkpoint selection.

`evaluate_baseline_folds.py` now uses the reviewed 70 columns and the five
development folds. `evaluate_checkpoint_comparison.py` and its sbatch launcher
use the reviewed columns and all 152 frozen final queries. The candidate is
read from the frozen v12 final-model manifest; an explicit `--candidate-run`
or `--final-model-manifest` selects another reviewed run, including a 25/50/75%
run. The launcher accepts `CANDIDATE_RUN` and `FINAL_MODEL_MANIFEST`. It no longer
defaults to v7/v8. Historical v8 inputs require `--legacy-data` explicitly.

The informed model uses the fixed schema and learned missing-value handling.
The comparison reuses `combined_corrosion.evaluate_final.comparison_command`
and its common options. Generic/pretrained TabICL use native latin shuffling,
default normalization and native constant-feature handling, matching COMB.
The informed models retain fixed order, no extra normalization and learned
missing-value handling. CatBoost uses the same fixed settings as COMB.
The comparison passes the reviewed category columns to CatBoost, retains
each run's available checkpoints and latest endpoint, and validates actual
row membership. `--context-folds 5` changes development context only; all 152
queries remain fixed. Final-query curves must not be used to change the
checkpoint selected on development folds.

For exploratory five-fold evaluation of existing checkpoints, add
`--exploratory-cv-folds 5` to the same comparison command and choose a new
`--output-dir`. Fold 1 uses the original 152 final-test rows; folds 2–5 split
the 608 development rows into four groups of 152 (fixed shuffle seed 42).
Each fold uses the other 608 rows as context. `cv_fold_1/` through `cv_fold_5/`
contain individual results; the top-level `summary.csv` contains five-fold
means, `std_test_*` sample standard deviations, and `n_cv_folds`. Top-level
plots show mean scores. All models use identical folds. This reuses existing
checkpoints without refitting priors, so the average is exploratory: four
query folds influenced informed training. The original split files and default
final-test evaluation are unchanged. This option cannot combine with
`--context-folds` or `--legacy-data`.

Preview without trained checkpoints:

```bash
.venv/bin/python -m scripts.epit_pipeline.evaluate_checkpoint_comparison \
  --candidate-run /path/to/new/epit/run --dry-run
```

See `corrosion_datasets/analysis/leakage_policy.md` for data-use limits.

To plot all five informed ratios together, pass checkpoint directories in
0/25/50/75/100 order. Use `--max-checkpoint-step 5000` to evaluate only saved
checkpoints at or below step 5,000:

```bash
.venv/bin/python -m scripts.epit_pipeline.evaluate_checkpoint_comparison \
  --ratio-runs /path/to/epit0 /path/to/epit25 /path/to/epit50 /path/to/epit75 /path/to/epit100 \
  --max-checkpoint-step 5000 \
  --generic-run /path/to/generic_run --device cpu \
  --output-dir /path/to/new/comparison
```

This produces one comparison with eight curves (`epit_0pct`, `epit_25pct`,
`epit_50pct`, `epit_75pct`, `epit_100pct`, generic, pretrained and CatBoost), the same CSV/JSON
outputs as COMB, and six SVG plots. Defaults evaluate 500-step checkpoints and
each run's latest endpoint within the requested bounds; shorter runs do not
truncate longer ones. The maximum applies to all local runs, including the
generic baseline, and never adds a latest checkpoint above the cap. Omitting it
keeps the previous uncapped behavior. It does not wait for future checkpoints.
The pretrained and CatBoost baselines are evaluated once per context selection
and reused across checkpoint steps. The 0% model keeps the same missing-aware
architecture as the informed models; the historical generic baseline retains
its native preprocessing.

Supply five distinct paths in 0/25/50/75/100 order. Missing directories, empty
runs, and models without checkpoints in the requested range are skipped
automatically, including a missing local generic baseline. Remaining models keep
their original ratio labels. Skipped models are printed and recorded in
`results.json` under `skipped_models`. At least one local model must be available;
model-loading or prediction errors still fail the comparison. `--dry-run`
previews the requested command without checking checkpoint availability.
The existing sbatch script accepts the same
arguments after its filename, including `--ratio-runs` and
`--max-checkpoint-step`. Alternatively set `EPIT0_RUN`, `EPIT25_RUN`, `EPIT50_RUN`,
`EPIT75_RUN`, `EPIT100_RUN` and optional `MAX_CHECKPOINT_STEP` in the environment.

The comparison launcher uses `PROJECT_ROOT`, then `SLURM_SUBMIT_DIR`, then
its current directory. `VENV_PATH` may select an existing environment, and
`PYTHONPATH` selects this checkout's source. Run local comparisons from
`/home/fcolanto/projects/tabicl`; the temporary checkout is no longer required
for evaluation.

Shared preprocessing leaves columns with fewer than two known context values
unscaled, unnormalized and unclipped, retaining their known values and NaNs.
Query values never determine these rules. Restart an older evaluation with a
fresh output directory to use the corrected preprocessing.

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

## Explicit formula extensions without PySR

`discover_formula_extensions.py` compares the nine current Fe/Ni formulas with
28 explicit extensions or alternative equations. Candidate effects cover scan
rate, exposed area, surface preparation, heat treatment, solution conditions,
and material/environment interactions. This is an offline formula study; it
does not change the prior generator or promote equations to training.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
  -m scripts.epit_pipeline.discover_formula_extensions \
  --output-dir corrosion_datasets/analysis/epit_pipeline/formula_extensions_repeat
```

Use a new output directory. The study uses the same 452 Fe/Ni development rows
and frozen five-fold assignments as formula calibration. Coefficients, missing
value fills, scales and supported categories are fitted within each training
split. An inner four-fold comparison selects regularization and formula for
each held-out fold. A second nested comparison holds out entire source
publications. Final-test targets are masked and never evaluated.

The first run (`formula_extensions_v1`) compared 27 candidates; the follow-up
(`formula_extensions_v2`) added ten heat/surface ablations after examining the
development results. These are exploratory comparisons. Individual winning
formula scores have selection optimism; the `nested_new_and_existing` row
assesses the complete selection procedure. Compare it with
`nested_existing_only` and the unchanged `existing_method_aware_pren_n` rule.

Artifacts include `report.html`, `summary.csv`/`fold_scores.csv`/`predictions.csv`
inside each evaluation subdirectory, fold-specific fitted equations,
coefficient stability, feature coverage and full-development equation JSONs.
Full-development fits are candidates for final training only: they must not be
used to generate an Optuna prior evaluated on development rows used to fit them.
Categorical corrections are empirical associations, not established physical
laws. Unknown categories have a neutral correction; missingness itself is not
an added predictor. Numeric term definitions and all preprocessing constants
are saved with the equations.

The November 6 follow-up found a heat-type/cooling correction with mean fold
Spearman 0.648 versus 0.591 for the current strongest formula. Its advantage
did not transfer to publication-held-out folds (0.349 versus 0.378), so no
formula was promoted. The saved follow-up report also includes an independent
artifact audit, selected-candidate uncertainty and `candidate_coefficients.csv`.

## EPIT final-training target v5

The final-training launcher now defaults to `--epit-target-version v5`. It uses
the selected v14 Optuna trial as a starting point and records these changes:

- Informed MLP/tree proportions are 70/30; composition perturbation is 0.30.
- The nine v4 formulas remain. `extend_heat_categories` is added as a tenth
  formula, with selection probability proportional to its development score
  (0.648288), using the same weighting rule as the existing formulas.
- One formula/SCM weight is sampled uniformly per synthetic table. Its mean
  stays at the selected trial's weight. For trial 27, the range is
  0–0.573195 and the mean is 0.286598. It is separate from the informed-table ratio.
  EPIT and Soccol share this weight sampling and target calculation through
  `src/tabicl/prior/target_mixing.py`; each keeps its own formulas and mean.
- The new formula uses its saved development-fitted scales, method offsets and
  heat/cooling coefficients. Missing or unsupported heat/cooling labels add no
  correction. The existing 0.8 coefficient variation also applies to the new
  formula's signed extension coefficients; its underlying base fit stays fixed.

The fitted equation is bundled in `src/tabicl/prior/assets/epit_target_v5.json`.
It was fitted on 452 Fe/Ni development rows, with final-test targets excluded.
This is an experimental final-training override, not an Optuna-tuned v5 result.
The weaker result on unseen papers still applies; improved model performance
has not been established. Optuna keeps its existing recipe. Soccol final training
keeps its own formulas and now varies the weight using the same implementation.

```bash
sbatch --gpus=1 scripts/epit_pipeline/train_final.sbatch \
  --selected-trial-number 27 --override-informed-prior-ratio 0.5
```

Change `--gpus` and the informed ratio in that same command. Checkpoints go to
`checkpoints/epit_pipeline_final_v2/epit_targetv5_trial0027_ratio_0p50_override_<hash>/`.
The hash includes the formula artifact, its runtime code and the shared weight
mixing code. Resume checks include them too, so an old target recipe cannot
silently resume as v5. Use
`--epit-target-version v4` to reproduce the previous final-training recipe.
The Python entry point defaults to v4 for compatibility; pass
`--epit-target-version v5` when invoking it directly.

Sync `scripts/epit_pipeline/train_final.py`, `train_final.sbatch`,
`src/tabicl/train/run.py`, `train_config.py`,
`src/tabicl/prior/epit_target_v5.py`, `src/tabicl/prior/target_mixing.py`,
and `src/tabicl/prior/assets/epit_target_v5.json`
to training nodes. The formula-research outputs are not runtime dependencies.

## Generic 10k continuation with fixed Stage 2 optimization

Both branches load only the weights from
`checkpoints/tabicl_s1_regression_baseline/step-10000.ckpt`. A new AdamW optimizer
uses the original Stage 2 schedule: 2,000 additional steps, learning rate
`2e-5` to `5e-6`, polynomial power 2, no warmup, global batch 512, microbatch 1,
gradient clipping 1, and weight decay 0. Both use seed 43 and the existing
1,024-row tables with context fractions 0.1–0.9. The architecture and regression
head are retained. This does not introduce Stage 2's long-table curriculum.

The informed branch uses 100% EPIT v5 tables, MLP share 0.70, composition
perturbation 0.30, and variable formula weight with trial 27's mean
`0.28659758621052106`. The control uses 100% generic tables. All other training
settings match. The launcher rejects changes to the predeclared recipe.

```bash
sbatch --gpus=1 scripts/epit_pipeline/finetune.sbatch \
  --final-run-name generic10k_epit100_s2 --override-informed-prior-ratio 1

sbatch --gpus=1 scripts/epit_pipeline/finetune.sbatch \
  --final-run-name generic10k_generic_s2 --override-informed-prior-ratio 0
```

Use the same GPU count for both branches. Add `--dry-run` to either command to
validate and print the effective configuration without training. Add
`--resume-training` to the original command to resume an interrupted stage;
this restores its full optimizer, schedule, RNG and sampler state, with the
original worker and process counts. It does not repeat weight-only loading.

Checkpoints are written under `checkpoints/epit_finetuning/<run-name>/`.
`step-0.ckpt` contains the unchanged 10k weights with the fresh optimizer and
sampler, before any training batches are consumed. Subsequent checkpoint steps
count only the new updates, and every 100th step is retained. `finetuning.json`
records the parent hash, fixed recipe, command and split identity.

Step 2,000 is frozen as the predeclared primary endpoint. Training does not run
development checkpoint selection or final-test evaluation. Frozen manifests
are saved under `corrosion_datasets/analysis/epit_pipeline/final_v2/manual/<run-name>/`.

Compare the two trajectories, including step zero, on the frozen 152 outer
queries with all 608 development rows as labeled context:

```bash
sbatch --gpus=1 scripts/epit_pipeline/evaluate_finetuning.sbatch \
  --finetune-run checkpoints/epit_finetuning/generic10k_epit100_s2 \
  --control-run checkpoints/epit_finetuning/generic10k_generic_s2 \
  --output-dir corrosion_datasets/analysis/epit_pipeline/generic10k_s2_comparison
```

The comparison uses the reviewed 70-column schema and the existing eight-view
Latin/none–power evaluation. It checks that the two branches share their parent
and training settings. The result metadata identifies step 2,000 as primary;
the other checkpoints describe the trajectory and do not select a winner.
