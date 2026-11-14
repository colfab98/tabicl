# Sample-level findings: 100% informed versus generic — 13 November 2026

The strongest actionable findings are (1) losses on held-out Al alloys, partly coupled to information absent from the input features; (2) a specific chromium-rich Fe/acid-chloride series that is severely underpredicted; and (3) a low-Cr/Mo Ni-rich composition consistently overpredicted across temperature. These support targeted data/feature audits and chemistry-specific ablations, not indiscriminate adjustment of all formulas.

## Evidence quality and interpretation

All 760 samples have predictions from both models at the same 11 checkpoints, 500–5500, giving 8,360 paired sample/checkpoint comparisons. The 110 regenerated model/fold/checkpoint scores reproduce the saved evaluation: maximum MAE discrepancy 0.00000798 mV, RMSE 0.00001077 mV, and Spearman discrepancy zero. No model inference or training was run during this interpretation.

For each sample, the penalty is informed absolute error minus generic absolute error, averaged over the 11 checkpoints. Positive means informed loses. This is a trajectory diagnostic, not the performance of a model ensemble or an independently selected final checkpoint. Persistent loss means worse by more than 1 mV in at least 75% of checkpoints (at least 9/11).

Fold 1 is the original outer query set. Folds 2–5 contain development rows that influenced prior construction and are exploratory. Keep them separate. Groups were explored after seeing results; confidence intervals and pattern descriptions are diagnostic, not confirmatory tests. Single-source chemistry series are useful mechanistic hypotheses but not independent validation across publications.

## What the overall mean conceals

| Family | Outer n / sources | Informed MAE | Generic MAE | Penalty mV | Exploratory penalty mV |
|---|---:|---:|---:|---:|---:|
| Al Alloy | 24 / 6 | 347.71 | 307.59 | **+40.12** | −25.19 |
| HEA | 5 / 4 | 348.23 | 281.14 | **+67.09** | −6.13 |
| NiCrMo Alloy | 10 / 1 | 133.37 | 113.35 | **+20.02** | −2.18 |
| Fe Alloy | 103 / 22 | 164.31 | 174.09 | **−9.78** | −5.07 |
| Other | 10 / 1 | 480.42 | 568.16 | **−87.74** | +7.57 |

The outer aggregate advantage is only 2.54 mV. It combines gains in Fe and Other with losses in Al, HEA and NiCrMo. The ten Other samples all come from one source, and both models still have very large errors there. Removing that group changes the remaining outer-sample mean to 3.46 mV worse for informed; this is a sensitivity check, not a proposal to exclude those samples from the benchmark.

There are **31 persistent outer losses**: 14 Fe, 10 Al, 4 NiCrMo and 3 HEA. There are 67 persistent losses among the 608 exploratory samples. Detailed identities are in the existing `persistently_worse_samples.csv` and the selected [priority samples](priority_samples.csv).

## 1. Al: a robust held-out weakness, but not a universal Al failure

The 100% model has larger checkpoint-averaged absolute error on 17/24 outer Al samples. Each of the six source-publication groups has a positive mean penalty. The descriptive publication-cluster interval is approximately **+16.9 to +59.6 mV** (pointwise, after subgroup exploration).

The effect survives removal of the most obvious problematic cluster:

- Excluding reference 32: **+28.20 mV**, 18 rows/five sources.
- Excluding the entire Al69.5–Fe7.99 composition family: **+30.07 mV**, 13 rows/four sources.
- Giving each unique 70-feature input equal weight: **+26.40 mV**, 18 unique inputs.

The weakness grows later: mean outer Al penalty over steps 3000–5500 is **+72.19 mV**, and it is positive at every one of those six checkpoints. In contrast, exploratory Al samples improve by 25.19 mV across all checkpoints and by 40.79 mV late. Even exploratory Al samples without their composition group in context improve on average (−12.66 mV, n=27). Therefore this is not explained solely by the presence/absence of a matching composition in context, and absence from informed sampling is not sufficient proof of causation.

**Data/feature bottleneck:** source sample numbers **703, 779, 780, 781, 782, 783, 784** have identical values in all 70 model-visible inputs, including missingness. Their targets range from **−550 to +980 mV**. They all lie in outer fold 1 and both models produce identical predictions within the group at each checkpoint, as expected. The lowest possible MAE of any common point prediction on these seven observed targets is **318.43 mV**; this is an illustrative post-hoc lower bound, not an estimate fitted for deployment. Actual informed/generic average errors are 518.45/437.16 mV.

These records cannot be distinguished by another deterministic formula operating on the same 70 inputs. Audit the underlying source records, crystallinity, processing/film state and measurement definitions. Some raw records say “Amorphous”, but the feature schema has FCC/BCC/HCP/intermetallic indicators and these rows have all four missing. Other records in this same input group lack even the raw microstructure description. Missing physical information and possible source inconsistencies must be separated from formula error.

The workbook identifies reference 32 as Lucente & Scully, DOI 10.1149/1.2712794; its full text was not retrieved in this investigation. Do not assume all its target spread is explained by any one omitted field without reviewing the paper.

**A separate concrete Al loss:** samples **752 and 753**, reference 80, are Al–Mo compositions with Mo 38.56 and 44.67 wt.% at pH 10 in 0.1 M NaCl. Penalties are **+145.66 and +145.04 mV**, with losses at 9/11 checkpoints. The original study concerns sputter-deposited Al films and reports effects of air aging and pit-growth stability. Those provide specific source variables/mechanisms to audit before inventing a bulk-composition-only Al formula. This does not establish which experimental state belongs to each curated row. [Original authors' abstract at IBM Research](https://research.ibm.com/publications/on-the-pitting-resistance-of-sputter-deposited-aluminum-alloys).

**Decision:** prioritize Al source/feature recovery and then a family-specific prior candidate. Do not just extrapolate the Fe/Ni PREN formulas to Al or claim that all Al samples need the same correction.

## 2. Chromium-rich Fe in 1 M HCl: a localized formula-shape hypothesis

The following original workbook sample numbers are all in outer fold 1, from reference 57, at 25°C, pH 0 and 1 M chloride:

| Sample No. | Cr wt.% | Observed mV | Informed MAE | Generic MAE | Penalty | Checkpoints worse |
|---:|---:|---:|---:|---:|---:|---:|
| 187 | 28 | 910 | 678.64 | 533.64 | +145.00 | 11/11 |
| 188 | 30 | 900 | 662.98 | 491.41 | +171.57 | 11/11 |
| 189 | 40 | 900 | 611.61 | 335.94 | +275.68 | 11/11 |

The error is **underprediction**, not random sign variation. Together they lose **197.42 mV** against generic. They are one source/one composition series, so do not count them as three independent chemistry validations.

The same source contains a useful diagnostic transition: Fe–Cr samples at Cr=15 and 20 have recorded Epit −100 mV, Cr=23 gives 180 mV, while Cr=28–60 gives roughly 900 mV. It also contains Fe–20Cr–25Ni–5Mo samples 183 and 201, which lose 218.70 and 234.81 mV in exploratory folds. Their recorded nitrogen is zero.

**Relevant code:** `threshold_saturation` uses `sigmoid((Cr-12)/2)`, which is already almost saturated by Cr=23, while other formula branches compress standardized material scores with `tanh`. The acid-specific extra Mo/N protection is proportional to `sqrt((Mo + 0.5 W)*N)`, so that extra term vanishes when N=0. Other existing terms still represent Mo: this is not an absence of Mo from the entire recipe.

**Hypothesis:** the prior may not represent the composition-dependent onset of protection under these acid/chloride conditions strongly enough. A candidate is a Cr–Mo interaction/transition conditioned on environment and alloy family, rather than a globally stronger Cr coefficient or weaker acid penalty. The source study itself reports that adequate Cr is necessary for the beneficial effect of Mo in HCl, supporting the interaction hypothesis. Its publisher abstract was accessible via search; full text was not retrieved. [Sugimoto & Sawada, 1977](https://www.sciencedirect.com/science/article/abs/pii/0010938X77900324).

Do not fit a threshold of 23–28% from these inspected outer rows. Use development-only evidence and independent sources to calibrate/test a candidate. The available predictions do not establish which of the ten sampled formula branches caused the error.

## 3. NiCrMo: the loss belongs to one composition, not the whole family

Four held-out samples share **Ni81.45–Cr13.31–Mo5.24 wt.%** and 0.5 M NaCl, pH 7:

| Sample No. | Temperature °C | Observed mV | Error penalty mV | Checkpoints worse |
|---:|---:|---:|---:|---:|
| 602 | 45 | 262.1 | +91.29 | 11/11 |
| 625 | 60 | 299.7 | +77.48 | 9/11 |
| 648 | 75 | 223.8 | +97.69 | 11/11 |
| 671 | 90 | 204.2 | +102.94 | 10/11 |

All four are **overpredicted**. Mean informed MAE is **205.49 mV versus 113.14 mV generic**, a **+92.35 mV** penalty. Several other held-out NiCrMo compositions improve instead (e.g. samples 611, 634, 657 and 659), so blanket down-weighting of the whole NiCrMo family or all Mo protection would risk damaging the successful cases.

All of the NiCrMo data come from reference 44, a 2009 Ohio State dissertation. This is one composition series, not four independent datasets. The most useful test is a composition-conditioned Cr/Mo/Ni protection shape and its interaction with temperature, assessed with held-out compositions within development data and, if possible, additional publications. The shared composition is more diagnostic than a broad “high temperature” label.

## 4. Earlier heat-formula suspicions are not confirmed by the neural errors

The earlier formula-only study found poor source transfer for heat categories. These neural predictions do **not** show a broad solution-treatment failure:

- Solution treatment: **−21.24 mV** penalty in outer rows (n=8, two sources), **−9.13 mV** in exploratory rows (n=41, five sources).
- Annealing: **−23.39 mV** outer (n=25, nine sources).
- Heat treatment: **−26.28 mV** outer (n=20, four sources).

Negative means informed improves. Individual exceptions remain—sample 331 is overpredicted and loses 140.21 mV at every checkpoint—but the subgroup average does not support deleting the heat extension as the first repair. An ablation could still test its role; formula errors and neural errors must not be conflated.

Likewise, broad labels such as “Mo >3” mix Al, Fe, HEA and NiCrMo chemistry. Its outer penalty of +30.41 mV is not evidence for one universally incorrect Mo coefficient.

## Priority decisions

1. **Audit the identical-input Al records and source metadata first.** No formula can reconstruct distinctions absent from all inputs. Preserve the benchmark results and log any corrections rather than silently changing labels.
2. **Test a narrowly specified Cr–Mo/environment interaction candidate.** The acid Fe series provides a clear error direction and direct literature motivation, but its coefficients must be learned/assessed on development-only source/composition splits.
3. **Investigate Ni81.45–Cr13.31–Mo5.24 as a composition extrapolation failure.** Retain the better-performing NiCrMo compositions as counterexamples when defining an ablation.
4. **Consider Al/HEA-specific priors after input audits.** The held-out family losses are real; the improvement on exploratory Al means a universal Al penalty is not an adequate explanation. HEA evidence is only five outer samples, three persistently worse, so avoid fitting a complex rule to that small set.
5. **Do not start by globally reducing heat, Mo, or acid terms.** The observed error directions differ across chemistry and sources. Validate targeted changes through matched-seed, matched-budget development-only ablations, with a fresh confirmation set after this extensive outer-set inspection.

These are now hypotheses grounded in actual paired neural errors. They remain hypotheses about causation until controlled retraining/ablation tests them. No formulas or training settings were changed.

Reproduce tables and the family trajectory figure (no inference):

```bash
PYTHONPATH=src:. .venv/bin/python corrosion_datasets/analysis/epit_pipeline/targetv5_cv5_sample_diagnostic/research_20261113/investigate.py
```
