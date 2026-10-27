# Soccol synthetic sampling compared with EPIT

We checked the current reviewed sampling banks, generated 20,000 rows per
dataset at each composition-noise strength **0, 0.05 and 0.30**, and inspected
the path from sampled values through target generation to model input.
We used seed 606 and kept datasets, coefficients and training settings unchanged.

## Main finding: composition weighting differed

| Quantity | Soccol | EPIT |
|---|---:|---:|
| Distinct source compositions in the active bank | 349 | 255 |
| Sampling unit for composition | Observation | Distinct template |
| Largest individual composition probability | 6.64% | 0.39% |
| Share assigned to the top 10 compositions | 34.31% | 3.92% |
| Share assigned to the top 20 compositions | 52.71% | 7.84% |
| Effective template count from draw weights | 52.26 | 255.00 |

Soccol drew uniformly from 2,950 observation entries. Its most repeated
composition appeared 196 times, so it received 196 times the draw probability
of a composition recorded once. That composition was Fe 69, Cr 17.5, Ni 11 and
Mo 2.5 wt.%. EPIT's current family probabilities and within-family template
draws gave each of its 255 active templates equal probability.

The actual Soccol sample assigned **52.52%** of draws to the 20 most frequent
source compositions, consistent with the expected 52.71%. All 349 distinct
source compositions were represented. The effective count was defined as
`1 / sum(probability²)`; it described concentration of the source weights,
not the number of different perturbed alloys that could be generated.

This was a concrete difference that could reduce coverage of less frequent
Soccol compositions. It did not prove a reduction in trained-model scores.

## What matched and what differed

| Pipeline part | Audit result |
|---|---|
| Source rows | Soccol used the 2,950 accepted development rows; EPIT used its 452 Fe/Ni development observations and 255 composition templates. Neither active sampling bank used frozen final-query rows. |
| Real compositions | Saved reviewed amounts and zeros were preserved in the source banks. |
| Synthetic compositions | Both multiplied positive element amounts by log-normal noise, retained zeros and closed only the generated compositions to 100 wt.%. |
| Environment/history | Both drew every context field and its missing-value pattern together from one observed development row. |
| Composition/environment pairing | Both drew composition independently of the environment/history row. This shared design was not a unique Soccol difference. |
| Numeric missing values | Both retained NaNs for the model and used development means internally to calculate synthetic targets. |
| Categorical missing values | Both reused saved category mappings, used modes internally, and restored the sampled missing-value pattern for model input. A known category code of zero remained distinct from missing. |
| Filling-statistic scope | Soccol used its 2,950 accepted development rows. EPIT retained means/modes from all 608 development rows, while sampling 452 Fe/Ni environment rows. This documented difference involved development data only. |
| Formula/SCM inputs | Soccol passed the matching 58 columns by name. EPIT passed 70 fields to the SCM and a private projection to its existing formulas. EPIT's filled temperature, chloride and pH agreed between its formula and SCM inputs. |
| Final model input | Both standardized finite internal features, retained the fixed column positions and restored the exact sampled NaN pattern. Neither dropped constant columns on this path. |

The Soccol composition and environment source indices had correlation −0.0014
in the sample; the same observation was drawn for both in 0.03% of rows.
The implementation used separate uniform draws. All known context values and
their joint missingness matched the selected bank row exactly.

## Missing and constant columns

On the **25 context columns shared by the two schemas**, missing entries were
**62.69% in Soccol** and **59.91% in EPIT's active environment bank**.
Different full schemas had different overall missing fractions, so this
comparison used matching columns.

Soccol had five entirely missing context columns in accepted development:
`H2SO4_mol_L` and the four second-stage heat-treatment fields. EPIT's active
environment bank had four: `KCl_eq_L`, `NaClO4_mol_L`, `WO4_mol_L` and `CP_SO4`.
These remained missing for the model. Their finite internal placeholders were
constant and contributed zero after column standardization.

After internal filling, 15 of Soccol's 58 columns and 16 of EPIT's 70 columns
were constant across their respective source supports. This included zero-only
composition fields and observed constant context values, not just missing
columns. Missing/constant fields were therefore present in both workflows;
their presence alone did not distinguish Soccol's weaker performance.

## Composition perturbation

For each generated row, we measured the largest absolute change in any element.
The table reported the 95th percentile of that per-row change, in wt.% points.

| Noise strength | Soccol | EPIT |
|---|---:|---:|
| 0.05 | 2.54 | 2.67 |
| 0.30 | 15.63 | 16.08 |

Strength 0.30 was a log-normal noise parameter, not a hard ±30% bound.
Both samplers could generate amounts beyond the observed element ranges; neither
applied a post-perturbation family or original-range filter. The behavior was
shared and had similar magnitude. At strength zero, the only numerical changes
were closure to 100 wt.%; source tables were unchanged. Zeros stayed zero and
all sampled compositions were finite and closed to 100 within numerical precision.

## Recommendation and checks

We recommended aligning Soccol composition weighting with EPIT by sampling
its **349 distinct reviewed compositions equally**, while retaining observation
weighting for the complete environment/history rows. Under that proposal,
any 20 source compositions would receive **20/349 = 5.73%** of draws.
The proposal preserved the accepted material scope and did not imply balancing
alloy families equally. It was documented, not implemented by this audit;
its effect on model performance remained untested.

- **51 sampling checks passed**, including repeated-seed reproduction, exact
  context and internal-value alignment, category codes, retained zeros,
  missingness and composition closure.
- **80 existing focused tests passed**, covering both SCM types, all nine
  Soccol formulas, calibrated formula-input consistency, real/synthetic
  encoding, learned missing-value handling and the EPIT comparison path.
  The suite included tiny CPU training/inference checks. Five existing
  empty-column warnings occurred during the Soccol inference test; it retained
  finite predictions and passed its missingness/schema assertions.
- No Optuna search, full model training or frozen final-query scoring was run.

Reproduce the numerical audit in a fresh directory:

```bash
.venv/bin/python -m scripts.soccol_pipeline.audit_synthetic_sampling --output-dir /tmp/soccol_sampling_audit
```

`soccol_composition_weights.csv` listed all source composition weights and row
IDs. `bank_feature_summary.csv` recorded missingness, constants and source
ranges. The remaining CSVs recorded sampled coverage, perturbation size and
element ranges. Source-entry counts were kept separate from distinct-composition
counts. `summary.json` recorded hashes, seed, sampling comparisons and checks.
