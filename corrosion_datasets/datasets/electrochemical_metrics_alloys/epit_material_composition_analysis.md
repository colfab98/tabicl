# EPIT Evaluation-Dataset Material Composition Analysis

## Purpose

This report performs the first feature-only analysis required for a chemistry-grounded latent material generator for the fixed EPIT prior. It examines the exact rows used by the evaluated task:

`electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg`

The analysis uses EPIT only to reproduce the existing usable-target row filter. No EPIT values or target relationships are used to define alloy families, composition statistics, or latent dimensions.

## Data and evaluated schema

Source workbook:

`raw/CRA_database_Scientific_Data_Publication_12102020.xlsx`

Source sheet: `Pitting Potential`

- Source rows: 810
- Rows with usable average EPIT target: 760
- Source composition columns: 24
- Retained evaluation composition columns: 17
- Other evaluated inputs: temperature, chloride concentration, pH, and test method
- Final evaluated width: 21 columns

The 17 retained material columns are Fe, Cr, Ni, Mo, W, Nb, Al, V, Ta, Re, Ce, Ti, Co, B, Mg, Y, and Gd.

The seven source composition columns omitted from the evaluated schema are N, C, Si, Mn, Cu, P, and S. They were excluded by the general evaluation feature-selection rules because they did not meet the required finite-value coverage. Their omission does not mean that they are chemically absent or unimportant.

All composition analyses below use the raw numeric material entries before evaluation-time mean imputation. Missing entries are therefore kept distinct from explicitly reported zeros. For composition sums only, unreported entries are treated as zero.

## Repeated measurements and distinct compositions

The 760 evaluated rows contain repeated measurements of the same nominal composition under different environments or procedures.

- Exact distinct 24-element numeric composition vectors: 403
- Rows beyond the first exact occurrence: 357
- Largest exact repetition count: 22

The distinct-composition count is sensitive to rounding:

| Composition rounding | Distinct vectors |
|---:|---:|
| 6 decimal places | 403 |
| 4 decimal places | 403 |
| 3 decimal places | 403 |
| 2 decimal places | 401 |
| 1 decimal place | 395 |
| Whole wt.% | 352 |

Thus, 403 is an exact-vector count rather than a definitive count of metallurgically distinct alloys. Nevertheless, row frequencies clearly overweight compositions that were tested repeatedly. Family frequencies intended for synthetic alloy generation should not be estimated directly from the 760 row counts without accounting for this repetition.

Examples of heavily repeated compositions include pure Al (22 rows), pure Fe, Ni, and Mo (10 rows each), Ni-20Cr (10 rows), and one Al-Gd-Fe composition (11 rows).

## Source material-class frequencies

The workbook supplies a `Material class` metadata field. It is not an evaluated model feature, but it provides a more defensible first family definition than inventing Fe-, Ni-, and Co-based families from scratch.

| Source material class | Evaluation rows | Row frequency | Exact distinct compositions | Distinct-composition frequency | Rows per distinct composition |
|---|---:|---:|---:|---:|---:|
| Fe Alloy | 514 | 67.63% | 298 | 73.95% | 1.72 |
| Al Alloy | 118 | 15.53% | 56 | 13.90% | 2.11 |
| HEA | 27 | 3.55% | 19 | 4.71% | 1.42 |
| NiCrMo Alloy | 51 | 6.71% | 17 | 4.22% | 3.00 |
| Other | 50 | 6.58% | 13 | 3.23% | 3.85 |

The row-weighted and composition-weighted frequencies differ materially. For example, NiCrMo alloys account for 6.71% of rows but only 4.22% of exact compositions, while Fe alloys increase from 67.63% to 73.95% after exact-composition weighting.

This dataset does not support a simple Fe/Ni/Co family taxonomy. Co-rich rows occur mainly inside the HEA class. The source taxonomy instead distinguishes Fe alloys, Al alloys, NiCrMo alloys, HEAs, and a heterogeneous `Other` group.

## Composition closure and the fixed 17-element schema

Across all 24 source composition columns, the compositions are essentially closed to 100 wt.%:

| Statistic | Full 24-element sum | Retained 17-element sum | Omitted 7-element contribution |
|---|---:|---:|---:|
| Mean | 100.0002 | 97.9599 | 2.0403 |
| Minimum | 99.5000 | 74.9900 | 0.0000 |
| Median | 100.0000 | 99.7715 | 0.2170 |
| 90th percentile | 100.0000 | 100.0000 | 4.7242 |
| 95th percentile | 100.0000 | 100.0000 | 10.5830 |
| Maximum | 100.5200 | 100.5200 | 24.9900 |

Of the 760 rows, 744 have a full 24-element sum within 0.01 wt.% of 100, and 757 are within 0.1 wt.%.

For the retained 17 columns:

- 408 rows sum to at least 99.5 wt.%.
- 66 rows sum to less than 95 wt.%.
- 46 rows sum to less than 90 wt.%.

The largest missing retained mass is explained by omitted elements such as Mn, Cu, and Si. Nitrogen is also omitted despite being positive in 52.35% of the exact Fe-alloy compositions.

### Consequence for the proposed generator

A Dirichlet distribution applied directly to the 17 observed material columns forces those columns to sum to 100%. That does not match the evaluated feature representation. A chemistry-grounded generator should instead do one of the following:

1. Generate all 24 source elements internally, enforce closure there, and then expose only the evaluated 17 columns; or
2. Generate the 17 observed elements plus an unobserved `other-elements` residual and drop the residual before model input.

The first option preserves element-specific chemistry and is the stronger design, although it requires modelling seven internally generated elements that the transformer will not observe.

## Sparsity and element occurrence

Among the 403 exact distinct compositions, the retained 17-column block contains a median of four positive elements. The full 24-element source composition contains a median of five positive elements.

| Active-element count | Retained 17 | Full 24 |
|---|---:|---:|
| Minimum | 1 | 1 |
| 25th percentile | 3 | 3 |
| Median | 4 | 5 |
| 75th percentile | 4 | 8 |
| 90th percentile | 5 | 10 |
| Maximum | 6 | 13 |

The retained element statistics below are calculated over the 403 exact distinct compositions. Positive-value quantiles exclude zeros and missing values.

| Element | Missing | Positive compositions | Positive frequency | Positive median wt.% | Positive 10th–90th percentile wt.% | Maximum wt.% |
|---|---:|---:|---:|---:|---:|---:|
| Fe | 0 | 330 | 81.89% | 66.163 | 37.130–78.872 | 100.000 |
| Cr | 5 | 344 | 85.36% | 18.335 | 15.000–25.281 | 100.000 |
| Ni | 10 | 283 | 70.22% | 14.380 | 2.140–31.232 | 100.000 |
| Mo | 11 | 259 | 64.27% | 2.640 | 0.080–13.256 | 100.000 |
| W | 6 | 25 | 6.20% | 1.500 | 0.011–10.904 | 27.210 |
| Nb | 12 | 28 | 6.95% | 0.340 | 0.100–22.078 | 64.960 |
| Al | 39 | 85 | 21.09% | 62.200 | 4.370–96.600 | 100.000 |
| V | 0 | 7 | 1.74% | 4.750 | 0.876–16.600 | 25.000 |
| Ta | 0 | 4 | 0.99% | 1.535 | 0.510–2.371 | 2.500 |
| Re | 0 | 2 | 0.50% | 1.745 | 1.141–2.349 | 2.500 |
| Ce | 0 | 7 | 1.74% | 12.890 | 0.382–24.950 | 27.350 |
| Ti | 1 | 11 | 2.73% | 0.910 | 0.120–23.000 | 49.000 |
| Co | 0 | 19 | 4.71% | 22.900 | 10.154–25.360 | 26.000 |
| B | 0 | 3 | 0.74% | 0.050 | 0.050–4.810 | 6.000 |
| Mg | 0 | 2 | 0.50% | 1.970 | 1.602–2.338 | 2.430 |
| Y | 0 | 1 | 0.25% | 11.800 | 11.800–11.800 | 11.800 |
| Gd | 4 | 4 | 0.99% | 23.995 | 10.250–29.683 | 31.480 |

Several rare columns span both trace and major-element regimes. For example, Nb reaches 64.96 wt.%, Ti reaches 49 wt.%, and Gd reaches 31.48 wt.%. A universal trace-element distribution would therefore be inappropriate; their behaviour depends strongly on family.

## Family-level composition profiles

The following summaries use exact distinct compositions and report medians with 10th–90th percentile ranges.

| Class | Number | Characteristic composition summary |
|---|---:|---|
| Fe Alloy | 298 | Fe 66.71 [49.19, 79.46], Cr 18.00 [13.90, 24.09], Ni 10.00 [0, 24.87], Mo 1.04 [0, 4.72] |
| Al Alloy | 56 | Al 76.50 [47.74, 98.34]; Cr and Mo are zero at the median but can reach major fractions in specific Al-based systems |
| HEA | 19 | Fe 22.90 [15.90, 32.14], Cr 21.00 [13.00, 23.90], Ni 24.00 [3.20, 33.96], Co 22.90 [0, 25.36], Al 3.50 [0, 13.20] |
| NiCrMo Alloy | 17 | Ni 66.67 [57.21, 74.59], Cr 19.04 [12.05, 28.91], Mo 11.80 [5.21, 20.01]; Fe is zero in all 17 exact compositions |
| Other | 13 | Heterogeneous collection containing pure metals and mixed compositions; it does not define one coherent template |

The NiCrMo class is especially coherent and appears suitable for a family-specific conditional generator. The Fe and Al classes contain substantially broader internal variation. The `Other` class should not be represented by one average template without further subdivision.

## Co-occurrence structure

Presence/absence associations were calculated on the 403 exact distinct compositions. Selected phi correlations are:

| Element pair | Compositions containing both | Phi correlation | Interpretation |
|---|---:|---:|---|
| Fe–Al | 33 | -0.578 | Strong family separation |
| Cr–Al | 39 | -0.577 | Strong family separation |
| Cr–Ni | 278 | 0.559 | Common joint occurrence |
| Fe–Cr | 311 | 0.534 | Common Fe-alloy structure |
| Fe–Ni | 262 | 0.426 | Common Fe-alloy structure |
| Mo–Al | 32 | -0.287 | Family-dependent separation |
| Ni–Mo | 198 | 0.183 | Positive but not universal association |

These are dataset associations, not causal chemical rules. They are strongly influenced by the family mixture and compositional closure. They should guide candidate family structure, not be inserted directly as universal element correlations.

## Compositional dimensionality

A centered-log-ratio PCA was applied to the 403 exact 24-element compositions after closing each row to 100 wt.% and replacing zeros with a small pseudocount. With a 0.01 wt.% pseudocount:

- First component: 40.8% of variance
- First two components: 54.3%
- Components required for 80%: 6
- Components required for 90%: 9
- Components required for 95%: 12

Across pseudocounts from 0.0001 to 0.1 wt.%, 6–7 components were required for 80% and 9–11 for 90%. The conclusion that the complete dataset is not well represented by one material coordinate is therefore stable, although exact component counts depend on zero handling.

Within-family results at a 0.01 wt.% pseudocount were:

| Class | Exact compositions | Components for 80% | Components for 90% | Components for 95% |
|---|---:|---:|---:|---:|
| Fe Alloy | 298 | 6 | 9 | 11 |
| Al Alloy | 56 | 5 | 6 | 8 |
| HEA | 19 | 5 | 6 | 7 |
| NiCrMo Alloy | 17 | 2 | 2 | 2 |
| Other | 13 | 3 | 5 | 5 |

Two latent material variables may therefore be a useful controlled starting point, but they will not reproduce the full variation of the Fe, Al, or HEA subsets. A practical generator can combine a small number of SCM-connected latents with additional family-conditioned stochastic variation. The NiCrMo family is the clearest case where two composition directions appear sufficient in this dataset.

## Missingness and evaluation preprocessing

The retained material table contains 204 missing cells across 760 rows. The largest missing counts are:

- Al: 76 rows
- Nb: 38 rows
- Mo: 31 rows
- Ni: 22 rows
- W: 15 rows
- Gd: 12 rows

The current evaluation pipeline replaces these missing values with whole-dataset column means. Those imputed values are useful for model input but are not chemically observed compositions and were not used as composition evidence in this report.

A synthetic chemistry generator should create complete compositions before intentionally hiding or corrupting values. It should not treat mean-imputed compositions as physical templates.

## Implications for a chemistry-informed fixed prior

1. **Use the source family taxonomy as the first hypothesis.** Fe Alloy, Al Alloy, NiCrMo Alloy, and HEA are supported directly by the workbook. `Other` needs subdivision or a broad fallback generator.
2. **Do not estimate family probability directly from measurement rows.** Repeated testing changes row frequencies. Exact-composition weighting is a better evaluation-dataset estimate, although it is still not a population prevalence estimate.
3. **Do not close the observed 17 columns to 100%.** Generate the full 24-element composition or include an unobserved residual before dropping the seven non-evaluated elements.
4. **Use family-specific element masks and ranges.** Several globally rare elements become major components in particular families.
5. **Retain more diversity than one latent direction.** Family selection plus two SCM-connected within-family latents is a reasonable initial ablation, but additional stochastic directions are needed for Fe, Al, and HEA compositions.
6. **Keep material-class metadata internal to generation.** The evaluated model does not receive `Material class`; it only sees the fixed element positions.
7. **Treat this as full-feature-table prior calibration.** No target values were analysed, but all 760 evaluated feature rows and source metadata were used.

## Provisional frequency candidates for later experiments

No final family probabilities should be selected until external chemistry evidence is reviewed. The evaluation table provides two candidate distributions:

- Row-weighted: Fe 67.63%, Al 15.53%, NiCrMo 6.71%, HEA 3.55%, Other 6.58%.
- Exact-composition-weighted: Fe 73.95%, Al 13.90%, NiCrMo 4.22%, HEA 4.71%, Other 3.23%.

The second distribution is less affected by repeated measurements and is the better evaluation-dataset starting point. It should still be compared with external alloy-corpus frequencies and a tempered or partially balanced distribution.

## Open questions for the external research phase

- Are the workbook material classes consistent with accepted metallurgical family definitions?
- Should the Al class be subdivided into conventional Al alloys and high-fraction binary/intermetallic systems?
- How should pure metals and binary model systems currently labelled `Other` be represented?
- Should family frequencies follow distinct compositions, external industrial prevalence, or a tempered combination?
- Is using all evaluation features and material-class metadata acceptable for the intended downstream claim, or should templates be constructed only from external sources?
- Should the internal generator model all 24 source elements and drop seven, or use one hidden residual mass?
- How many SCM-connected material latents should be tested initially: two or three?
- How should missing source entries be interpreted when building external templates: true absence, unreported trace content, or unknown composition?

## Recommended next step

Perform the external literature and alloy-database review using the five workbook classes as hypotheses rather than fixed truth. The review should validate family definitions, typical composition ranges, allowed element combinations, and plausible family prevalence before any family templates are implemented.
