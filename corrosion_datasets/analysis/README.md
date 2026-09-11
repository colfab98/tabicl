# Current analysis artifacts

This directory retains the evidence and shared utilities for the final CorrPFN
EPIT implementation. The project description is in [main3.tex](../../main3.tex);
the historical research narrative remains in [main2.tex](../../main2.tex).
The final study is `epit_pipeline_optuna_empirical_features_scm_target_v7`, with
Trial 56 and development-selected checkpoint `step-2500.ckpt`.

## Final pipeline

Paths below are relative to `epit_pipeline/`.

| Path | Why it is retained |
| --- | --- |
| `splits_v2/` | Frozen composition-group assignments, manifest, lock, and report for the 608 development / 152 final-test partition and five development folds. |
| `target_rules_v2/` | Calibration summary, all seven candidate rules, two direct PREN baselines, and direct evaluation predictions/report. Rule files and hashes are inputs to the current pipeline. |
| `optuna_v2/trials/epit_pipeline_optuna_empirical_features_scm_target_v7/` | All recorded v7 trial configurations and training records, preserving the basis for configuration selection. |
| `optuna_v2/evaluations/epit_pipeline_optuna_empirical_features_scm_target_v7/` | All recorded v7 development evaluations used to compare configurations. |
| `final_v1/epit_pipeline_optuna_empirical_features_scm_target_v7/` | Final training log, development checkpoint evaluations, and frozen model manifest/lock. |
| `trial56_checkpoint_folds_66378/` | Development checkpoint comparisons with the evaluated baselines; these contain information beyond the selected-model manifest. |
| `trial56_final_test_all_checkpoints_11172/` | Final-test comparisons and subsequent checkpoint diagnostics. Step 2500 remains the development-selected CorrPFN result. |

The parent names `optuna_v2` and `final_v1` are still current artifact locations;
their suffixes do not mean that their v7 contents are obsolete. Keep the retained
JSON artifacts unchanged so their provenance hashes continue to verify.

## Supporting analyses and utilities

| Path | Why it is retained |
| --- | --- |
| `scripts/` | Shared dataset loaders and structural/audit utilities. `analyze_structure.py` is imported by the corrosion evaluator and EPIT split loader. |
| `epit_composition_families/` | Composition-family coverage, element ranges, sparsity, and correlation evidence supporting the empirical templates. |
| `feature_group_audit/` | Column roles, missingness, and association audits, including the EPIT source table. |
| `structural_analysis_results.md` and `.json` | Cross-dataset exploratory evidence that is not reproduced by the final experiment outputs. |
| `column_group_map.md` and `feature_group_schema.md` | Broad feature-role references. These historical maps are not the definitive current 21-feature input specification; use `main3.tex` and the implemented EPIT loader for that. |
| `leakage_policy.md` | Current EPIT data-use rules and the parsing/interpretation guidance retained from older planning notes. |
| `eval_results/` | Available output location for general evaluations; its historical contents have been archived. |

The structural reports support interpretation of the datasets. Their associations
and sample counts do not establish numerical prior defaults. The final v7 search
varies informed-task probability, informed MLP probability, SCM–EPIT target mixing,
and composition perturbation. Feature-block coupling is fixed to zero; the older
history/intervention-strength ablation proposals do not describe this model.

## Archived material

Historical outputs are under
`/home/fcolanto/old_tabicl/corrosion_datasets/analysis/`, preserving their paths
relative to this directory. They include the older DatacorTech and pitting-prior
searches, previous general evaluation outputs, earlier splits/rule calibrations,
non-v7 pipeline studies, Trial 25/36 comparisons, and the superseded
`parameter_decision.md` and `prior_implications.md` notes. The original
DatacorTech-oriented `leakage_policy.md` is also preserved there.

The archive is historical evidence, not a source of current parameter defaults.
Move records are stored as `analysis_archive_*.json` in
`/home/fcolanto/old_tabicl/`.
