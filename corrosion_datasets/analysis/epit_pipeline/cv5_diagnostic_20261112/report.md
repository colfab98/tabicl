# EPIT five-fold diagnostic — 12 November 2026

The informed prior provides clear early-training gains, but increasing its share does not reliably improve late-training performance. The strongest diagnosed prior weaknesses are restricted material-family coverage and a heat-category extension that fails to transfer across publications. Neither observation establishes which individual samples the neural checkpoints get wrong: their saved evaluation artifacts contain aggregate scores, not per-sample predictions.

This analysis used saved results, saved formula out-of-fold predictions, reviewed sample metadata, training logs, and runtime code. No checkpoint inference, training, formula refitting, or evaluation rerun was performed. The two targeted runtime-equation tests passed.

## What was compared

The original result paths recorded in the artifacts are under `corrosion_datasets/analysis/epit_pipeline/targetv5_four_ratios_through_8000_cv5/`. The available local copy used here is `targetv5_four_ratios_through_8000_cv5/` at the repository root. These are the v5 pretraining ratio models, not the later fine-tuning experiment.

Every informed result was paired with `generic_baseline` at the **same checkpoint step and fold**. The saved series contain all five folds per available model/checkpoint and no reported errors. Availability differs: 25% through 6000, 50% and 100% through 5500, 75% through 4500, generic through 8000. Cross-ratio trajectory comparisons therefore use the shared 500–4500 window. We do not compare each model's cherry-picked best checkpoint, and checkpoint/fold pairs are not treated as independent statistical replicates.

The experiment has one training seed per trajectory; five folds are not five independent training seeds. Reported fold variability is descriptive. Selection or tuning after looking at these results would require separate confirmation.

## Where the neural models improve or deteriorate

Positive MAE gain means generic MAE minus informed MAE; positive rho gain means informed Spearman minus generic Spearman.

| Step | Informed share | MAE gain, mV | Spearman gain | Folds with lower MAE / higher rho |
|---:|---:|---:|---:|---:|
| 1000 | 100% | +31.42 | +0.0731 | 5/5, 5/5 |
| 2000 | 25% | +9.36 | +0.0102 | 5/5, 4/5 |
| 2000 | 50% | +27.03 | +0.0390 | 5/5, 5/5 |
| 2000 | 75% | +25.66 | +0.0233 | 5/5, 5/5 |
| 2000 | 100% | +20.29 | +0.0192 | 5/5, 5/5 |
| 4500 | 25% | +11.11 | +0.0220 | 5/5, 5/5 |
| 4500 | 50% | −4.41 | +0.0003 | 2/5, 3/5 |
| 4500 | 75% | +9.59 | +0.0121 | 5/5, 5/5 |
| 4500 | 100% | −2.67 | −0.0074 | 1/5, 2/5 |

At step 2000, 50% reduces mean MAE from 185.48 to 158.45 mV (14.6%). At step 4500, 25% reduces 161.69 to 150.58 mV (6.9%). These are useful effect sizes in this experiment, not proof of a universally optimal ratio.

**The late 100% model illustrates the MAE/ranking tradeoff.** At step 3500 it improves MAE in every fold, averaging 8.87 mV better, while making Spearman worse in every fold, averaging −0.0164. At every evaluated step from 3000 through 5500 its mean Spearman is below generic. This is compatible with a prior that helps numerical error while imposing some incorrect ordering, but aggregate metrics cannot identify the mechanism.

**Some apparent gains are baseline fluctuations.** Generic mean MAE worsens from 161.69 at 4500 to 184.00 at 5000, then recovers to 160.33 at 5500. The large informed-model advantages at 5000 should not alone drive a claim of a sudden physics-prior breakthrough.

**The frozen outer fold supports some gains and exposes some losses.** At 2000, 50% gives MAE 188.68 versus generic 226.52 mV and rho 0.756 versus 0.675. At 4500, 25% gives 189.71 versus 209.29 mV and rho 0.763 versus 0.725. But at 5500, 50% gives 218.58 versus 206.57 mV and rho 0.584 versus 0.731. Those last losses are much larger than its five-fold mean rho loss of 0.0270, so averaging obscures an important weakness.

See [checkpoint gains](matched_checkpoint_gains.png), [all paired checkpoint summaries](paired_checkpoint_summary.csv), and [outer-fold results](outer_fold_1_comparison.csv).

## Why the five-fold mean needs qualification

Fold 1 is the original final partition: no query composition group appears in its context, and none of its query rows was used to fit the target formulas. Folds 2–5 partition the old development set and are explicitly exploratory.

| Fold | Queries previously used in formula fitting | Queries with a composition group in context | Queries with a publication in context |
|---:|---:|---:|---:|
| 1 | 0/152 | 0/152 | 135/152 |
| 2 | 113/152 | 107/152 | 150/152 |
| 3 | 109/152 | 113/152 | 149/152 |
| 4 | 116/152 | 116/152 | 150/152 |
| 5 | 114/152 | 118/152 | 152/152 |

These are direct membership counts, not allegations that query labels were accidentally passed as context. The issue is prior construction and split difficulty. Fold 1 is also not a publication-held-out test. Keep its results separate, and use development-only publication/group splits to decide future formula changes. Do not repeatedly choose a final checkpoint based on outer-fold performance.

## Material families: a real coverage gap, not yet a neural error attribution

| Family | All evaluation rows | Formula-fit development rows | Publications in those fit rows |
|---|---:|---:|---:|
| Fe Alloy | 514 | 411 | 41 |
| NiCrMo Alloy | 51 | 41 | 1 |
| Al Alloy | 118 | 0 | 0 |
| HEA | 27 | 0 | 0 |
| Other | 50 | 0 | 0 |

The informed physical sampler uses only Fe and NiCrMo. Al/HEA/Other account for **195/760 = 25.7%** of evaluated rows and receive no corresponding informed material-family examples. Generic synthetic tasks can still support general tabular transfer; absence from the informed sampler does not prove these samples suffer in the neural results. It is nevertheless the clearest coverage hypothesis to test.

The runtime extracts the Fe and NiCrMo entries from the five-family probability vector and renormalizes them. For the logged v5 runs, these become approximately 94.51% Fe and 5.49% NiCrMo. NiCrMo constitutes 9.07% of eligible formula-fit rows but has just 14 composition templates and one source publication. Adding more copies of that source is not equivalent to adding independent chemical evidence.

Extending Fe-based PREN rules blindly to Al/HEA would be the wrong repair. The source dataset paper explicitly discusses their restricted applicability outside the fitted alloy domain. Prefer family-specific hypotheses or retention of a generic component. [Nyby et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7878803/).

## Heat-treatment terms: the strongest existing evidence of poor transfer

These are **formula** diagnostics on the 452 eligible development rows, not neural model predictions. The saved nested development studies compare a method-aware base equation to its heat-category extension:

| Formula | Ordinary development-fold mean rho | Publication-held-out mean rho | Publication-held-out mean MAE, mV |
|---|---:|---:|---:|
| Method-aware base | 0.5908 | 0.3779 | 232.75 |
| Heat categories | 0.6483 | 0.3492 | 252.02 |
| Heat extension without method | 0.6412 | 0.3958 | 251.53 |
| Surface extension | 0.6052 | 0.3984 | 232.03 |

The heat-category gain changes from **+0.0575 rho to −0.0287 rho**. This supports a transfer concern. It does not establish that heat treatment is physically irrelevant, nor that surface terms will improve neural training.

The source associations are strong: all 27 soaking rows and all 18 tempering rows come from one publication each. The fitted heat extension is additive in category indicators; it cannot learn a general time–temperature–composition interaction from those labels alone. Numeric heat fields are also sparse: temperature missing in 53.5%, duration in 45.8%, cooling in 58.2% of eligible development rows.

**The most actionable subgroup is solution treatment.** In publication-held-out formula predictions, its 41 rows across five references have MAE **524.19 mV**, versus **298.14 mV** for the base. Solution annealing is also worse: 329.57 versus 282.42 mV across 27 rows/five references. For source reference 38, solution-treatment MAE grows from 278.66 to 565.27 mV (17 rows); for reference 73, from 188.44 to 505.17 mV (12 rows). These are candidates for source-data review and ablation, not a reason to flip one coefficient globally: other sources in the same subgroup are underpredicted rather than overpredicted.

There is a distinction between wrong ordering and wrong calibration. For 29 solution-treated rows in one publication-held-out fold, base and extension both have within-subgroup Spearman about 0.357; the large mV loss can arise from group shifts/calibration rather than a reversed ordering within that subgroup. Training uses the standardized rank drive, not the formula's fitted mV prediction. The overall held-out rho deterioration remains relevant, but a 226 mV subgroup loss must not be interpreted as a 226 mV neural-model penalty.

See [formula protocol comparison](formula_protocol_scores.csv), [subgroup diagnostics](formula_subgroup_errors.csv), [heat-category comparisons](heat_gain_heat_treatment_type.csv), and [sample-level formula errors with source metadata](formula_heat_vs_base_per_sample.csv).

## Calculation and representation audit

The heat runtime equation matches the saved fitted equation at `rtol=atol=1e-12` on its eligible development inputs; the test also checks split provenance. A second test confirms unknown/missing heat labels contribute neutrally. Both passed. This rules out that particular transcription mismatch; it does not validate every physical assumption, every target-rule family, or historical checkpoint code identity.

The recipe contains nine existing equations plus heat categories. Heat is selected for about **11.76%** of informed tables. Logged training uses a mean formula/SCM mixing weight of 0.2866, sampled uniformly from 0 to 0.5732, and coefficient variation 0.8. Therefore the heat term is not the whole informed prior. Any claim that removing it alone would fix the late-training losses would be unsupported.

Composition and environment/history records are drawn independently. This preserves combinations within one environment/history record but breaks the observed coupling between alloy composition and heat treatment, surface preparation, and test environment. This is a code-confirmed design choice and a plausible distribution mismatch. Whether that independence is beneficial augmentation or harmful requires an ablation with matching seeds and table budgets.

Fourteen raw targets have inequality notation (`>800` or `>1256`); the numeric pipeline uses their threshold values. They are lower-bound observations, not exact measurements. This is a target-semantics issue to audit, especially when examining high-potential residuals. It does not by itself demonstrate the informed/generic difference, and a censored-target sensitivity check should be reported separately rather than silently changing this benchmark.

Method detail also deserves attention: the literature shows Epit depends on the extraction criterion and experimental protocol, with substantial scatter. That supports structured protocol features and source-held-out testing rather than unrestricted offsets for long source-specific method strings. [Coelho et al., 2023](https://www.nature.com/articles/s41529-023-00403-z.pdf).

## What to do next, in priority order

1. **Obtain the missing neural per-sample evidence.** Export predictions for generic and informed models at matched, explicitly exploratory checkpoints (2000 and 4500 are useful contrasting snapshots). Preserve exact preprocessing, context/query indices, estimator settings, checkpoint identity, and row IDs. First reproduce saved aggregate metrics. Then compute paired absolute-error differences by family, source, heat treatment, and environment. Use publication-cluster uncertainty and report small groups. This export was not run.
2. **Test domain coverage.** Compare Fe/NiCrMo and Al/HEA/Other gains separately. If the damage is concentrated outside the informed domain, keep generic tasks or develop appropriate separate priors. Do not infer that concentration from aggregate fold differences.
3. **Ablate the heat extension on development-only source-held-out evaluations.** Compare v4, v5, and a heat-without-method or regularized interaction alternative with the same table counts, seeds, schedule, and mixture. The observed source-transfer reversal justifies this test; it does not justify selecting a replacement using final labels.
4. **Test composition–process coupling.** Contrast independent resampling with joint or composition-conditioned resampling. Check support and physical plausibility before increasing prior strength.
5. **Improve evidence before adding complexity.** NiCrMo needs source diversity; sparse phase labels (roughly 81–98% missing) do not currently justify a rich phase formula. The existing formula study's material-only extension produced almost no improvement. Prioritize the demonstrated source/coverage problems over adding more coefficients indiscriminately.

The available artifacts establish when the models improve and reveal concrete prior weaknesses. They do **not** yet establish which particular neural-model errors are caused by those weaknesses. That last attribution requires per-sample checkpoint predictions and controlled ablations.

Reproduce the artifact analysis from the repository root with:

```bash
.venv/bin/python corrosion_datasets/analysis/epit_pipeline/cv5_diagnostic_20261112/analyze.py
```

This only reads saved artifacts and writes the diagnostic tables/figure in this folder.
