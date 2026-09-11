# CorrPFN EPIT pipeline

This directory implements the final v7 workflow described in
[main3.tex](../../main3.tex). The study is
`epit_pipeline_optuna_empirical_features_scm_target_v7`; its selected
configuration is Trial 56, with development-selected checkpoint `step-2500.ckpt`.
The current launchers are `run_optuna.sbatch` and `train_final.sbatch`.

## Workflow and retained implementation

| Stage | Files | Role |
| --- | --- | --- |
| Split construction | `prepare_splits.py`, `split_data.py`, `split_refinement.py`, `split_report.py` | Group the 17 model-visible composition values and construct/audit the fixed 608 development / 152 final-test partition and five development folds. |
| Rule calibration | `target_rules.py`, `calibrate_target_rules.py` | Fit rules on eligible context rows for direct development-fold evaluation, then fit the final candidate coefficients on the 452 eligible Fe/Ni–Cr–Mo development rows. |
| Prior search | `run_optuna.py`, `evaluate_optuna_folds.py` | Train proxy models on synthetic tasks and compare them on the saved development folds. |
| Final training | `train_final.py` | Train the selected configuration, compare permanent checkpoints on development folds, and freeze the selected checkpoint and evaluation configuration. |
| Final evaluation | `evaluate_final.py` | Load the frozen model, use all 608 development rows as context, and evaluate the 152 final-test rows. |
| Supporting comparisons | `evaluate_baseline_folds.py`, `summarize_checkpoint_comparison.py` | Evaluate baselines on development folds and aggregate all-checkpoint fold outputs into tables and plots. |
| Artifact verification | `artifact_hashes.py` | Verify the frozen split, final-model manifest, checkpoint, and associated provenance hashes. |

Composition grouping uses unnormalized composition values rounded to 0.01 wt.%,
with complete-linkage groups bounded by 1.0 wt.% total difference. No composition
group crosses the development/final-test boundary or a development-fold boundary.
Transformer evaluation includes all alloy classes; the Fe/Ni restriction applies
to the informed feature generator and candidate rule calibration.

The v7 search has four active dimensions:

- Informed-task probability (`informed_prior_ratio`).
- Informed MLP probability (`mlp_prob`).
- SCM–EPIT target mixture weight (`informed_target_mix_weight`).
- Composition perturbation (`pitting_composition_perturb_strength`).

Informed tasks use empirical physical features and a standardized mixture of an
SCM target and a calibrated EPIT-rule target. Candidate rules are sampled using
their direct development-fold scores normalized as `score / sum(scores)`.
Magpie descriptors and feature permutation are disabled; feature-block coupling
and Dirichlet sampling are fixed to zero. The generic SCM MLP/tree mixture is
fixed at 0.7/0.3. NumPy/Torch seeds are 42 and prior generation uses one worker.

The seven candidate rules are `pren_linear`, `cr_mow_synergy`,
`threshold_saturation`, `improved_environment`, `coupled_breakdown`,
`fe_ni_cr_threshold`, and `method_aware`. The direct reference rules
`current_pren` and `current_pren_fe_ni` remain useful comparisons; they are not
sampled as candidate target families.

## Current artifacts

Paths are relative to the repository root:

- `corrosion_datasets/analysis/epit_pipeline/splits_v2/`: frozen manifest, lock,
  row assignments, and split report.
- `corrosion_datasets/analysis/epit_pipeline/target_rules_v2/`: calibration
  summary, rule coefficients, and direct development evaluation outputs.
- `corrosion_datasets/analysis/epit_pipeline/optuna_v2/`: retained v7 trial
  records and development evaluations.
- `corrosion_datasets/analysis/epit_pipeline/final_v1/epit_pipeline_optuna_empirical_features_scm_target_v7/`:
  final training log, development checkpoint evaluations, and model manifest/lock.
- `checkpoints/epit_pipeline_final_v1/final_epit_pipeline_optuna_empirical_features_scm_target_v7_trial_0056/`:
  final-training checkpoints, including the selected `step-2500.ckpt`.

The exact final training command and effective parameters are recorded under
`training` in `final_model_manifest.json`. This pipeline does not create the
older separate `model_params/` records. See the
[analysis artifact index](../../corrosion_datasets/analysis/README.md) for the
retained baseline comparisons and historical archive.

## Launch commands

Run from `/home/fcolanto/projects/tabicl` on a host with Slurm access, the project
environment, source dataset, and shared Optuna journal available:

```bash
sbatch scripts/epit_pipeline/run_optuna.sbatch
sbatch scripts/epit_pipeline/train_final.sbatch
sbatch scripts/epit_pipeline/evaluate_final.sbatch
```

These are separate stages, submitted in order after the preceding stage finishes.
The search defaults to v7, one GPU, 1,000 proxy steps, and a 10,000-step scheduler
horizon. Final training pins Trial 56, trains to 10,000 steps, and evaluates
permanent checkpoints every 500 steps. Mean development-fold Spearman determines
the selected checkpoint. Final evaluation uses the v7 manifest's locked settings.

Final training verifies that a pinned trial is the best completed trial in the
study. Active trials currently do not block that check; finish the search before
treating its winner as final. The launcher permits verified Optuna-journal
provenance when selected proxy files are absent, recording which original files
could not be verified locally.

The saved v7 study and trained model already exist. These commands document their
entry points: training refuses to overwrite a frozen model or a nonempty
checkpoint directory. Use a new study name for a new search; the search checks
the study fingerprint before appending trials. Explicit final-training overrides
are recorded and isolated from the selected model's output directory.

Standalone Python defaults now use the v7 study and one prior worker. Search
defaults to `empirical_features_scm_target`. Standalone final training selects
the best completed trial unless `--selected-trial-number` is supplied; the
Slurm launcher supplies 56. Shared Python APIs retain explicit historical modes
for compatibility.

## Supporting evaluations

Run baseline development evaluation through its Python entry point:

```bash
python -m scripts.epit_pipeline.evaluate_baseline_folds --auto-output-dir
```

It compares the retained generic baseline (step 1000 by default), pretrained
TabICL v2, and CatBoost across five development folds. Use `--generic-checkpoint`
to choose another baseline checkpoint.

Regenerate summaries and plots from retained all-checkpoint fold results with:

```bash
python -m scripts.epit_pipeline.summarize_checkpoint_comparison \
  --output-root corrosion_datasets/analysis/epit_pipeline/trial56_checkpoint_folds_66378
```

`evaluate_checkpoint_comparison.sbatch` currently runs all-checkpoint final-test
comparisons for Trial 56 and the baselines. Its outputs are post-selection
diagnostics. The confirmatory CorrPFN result remains development-selected step
2500; final-test checkpoint rankings must not replace that selection.

## Building new split and rule artifacts

Reuse the existing frozen artifacts for the recorded model. To exercise the
construction stages in separate output directories:

```bash
python -m scripts.epit_pipeline.prepare_splits \
  --output-dir corrosion_datasets/analysis/epit_pipeline/splits_rebuild

python -m scripts.epit_pipeline.calibrate_target_rules \
  --split-manifest corrosion_datasets/analysis/epit_pipeline/splits_rebuild/split_manifest.json \
  --output-dir corrosion_datasets/analysis/epit_pipeline/target_rules_rebuild
```

Split construction writes a manifest, assignments CSV, HTML report, and lock.
Calibration writes `calibration_summary.json`, each rule's fold-local and final
development coefficients, `direct_evaluation_predictions.csv`, and
`direct_evaluation_report.html`. Rebuilt artifacts have their own hashes and
must not be substituted into the frozen v7 experiment.

## Data use and dependencies

Final-test targets are masked before rule calibration and excluded from v7
configuration/checkpoint selection. EPIT inference explicitly disables power
normalization with `norm_methods=["none"]`. The full corpus influenced earlier
exploration and empirical profiles include final-test feature information. See
the [data-use guidance](../../corrosion_datasets/analysis/leakage_policy.md) and
`main3.tex` for the scope of this internal composition-separated evaluation.

Keep these shared dependencies even though some filenames describe older work:

- `scripts/optuna_pitting_magpie_prior_search.py`: training-command construction,
  search constants, and parameter helpers imported by `run_optuna.py`.
- `scripts/optuna_pitting_fixed_pren_prior_search.py`: storage, subprocess, and
  evaluation utilities imported by that module.
- `scripts/eval_corrosion_datasets.py` and
  `corrosion_datasets/analysis/scripts/analyze_structure.py`: evaluation and
  dataset-loading dependencies.

The six older v3/v4/v5 launchers are archived under
`/home/fcolanto/old_tabicl/scripts/epit_pipeline/`. Their old generic filenames
now name the v7 launchers in this directory.
