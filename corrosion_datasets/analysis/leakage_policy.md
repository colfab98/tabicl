# EPIT Data Use and Analysis Guidance

This note describes the final CorrPFN EPIT workflow and preserves the useful
parsing and interpretation guidance from earlier prior-design notes. See
[README.md](README.md) for current artifacts and [main3.tex](../../main3.tex)
for the implemented model and evaluation design.

## Features and structural evidence

- Keep material/composition, environment, process/method, and electrochemical
  responses distinct when interpreting source columns. The current model uses
  17 composition features, three environment features, and one test-method
  feature; a broad historical group map does not override this schema.
- Do not parse identifiers, alloy formulas, process labels, method prose, or
  categorical environment labels as scalar physical measurements for numeric
  correlation or interaction analysis. Use appropriate categorical association
  measures; nominal codes do not imply a physical ordering.
- Exclude same-experiment electrochemical response columns from the current
  EPIT inputs, including Epit minimum/maximum and quantities derived from the
  target. Post-exposure or post-failure observations cannot support a prediction
  made before those outcomes are observed.
- Keep reference IDs, DOI strings, and identifying study text as provenance or
  split-audit information, not predictors. Reference overlap is allowed in the
  current composition-group split, so it is not a publication-held-out test.
- Structural correlations, sample counts, and feature frequencies do not by
  themselves calibrate prior strengths. A directly fitted corrosion rule's
  ranking performance also does not establish a transformer improvement.
- History dependence requires genuinely ordered repeated observations; arbitrary
  row order is not evidence for an autoregressive effect. Process or treatment
  effects need measured inputs and supporting evidence, not universal strengths.
  The old history/intervention ablation ranges are not current model defaults.

## Calibration and selection

1. Use the frozen `epit_pipeline/splits_v2/` composition groups and five
   development folds. The 760 usable rows are partitioned into 608 development
   rows and 152 final-test rows.
2. Exclude final-test targets from target-rule calibration, Optuna selection, and
   checkpoint selection. Direct rule evaluation fits coefficients on eligible
   context rows within each development fold. Final rule coefficients use the
   452 eligible Fe/Ni–Cr–Mo development rows. Transformer evaluation covers all
   alloy classes in the corresponding development or final-test partition.
3. Compare configurations and select the checkpoint using mean development-fold
   Spearman correlation. Report MAE, RMSE, and R-squared as complementary metrics.
   Increasing the informed-task fraction or domain-term strength is not evidence
   of improvement; comparisons need explicit feature representations and
   training budgets, including a generic-prior baseline.
4. Evaluate the frozen selected checkpoint using all 608 development rows as
   context and the 152 final-test rows as queries. The retained all-checkpoint
   final-test results are subsequent diagnostics: they do not justify replacing
   the development-selected Trial 56 checkpoint at step 2500.

## Scope of the reported result

The final workflow reserves final-test labels from calibration and model
selection, but the complete corpus influenced earlier exploratory development.
Empirical feature profiles also contain feature information from the full
corpus, including some final-test compositions as unlabeled templates. This is
an internal assessment with separated labeled composition groups, not an
independent test on compositions absent from every source of pretraining
information.

Only one final training seed was evaluated. Report the recorded results as point
estimates, without claiming stability across training seeds or generalization to
independent laboratories, publications, or datasets.

The original DatacorTech policy and superseded parameter proposals are preserved
under `/home/fcolanto/old_tabicl/corrosion_datasets/analysis/`. Historical
structural reports remain available here as supporting evidence; their proposed
parameter values do not describe the final v7 implementation.
