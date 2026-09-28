# Soccol source-convention audit

This directory records a high-level review of every `source` in the 2024
Soccol pitting-potential workbook. It separates collection-wide normalization
from source-specific material and composition issues.

## Outputs

- `source_conventions.csv`: one row for each of the 154 sources, including
  sample count, material family, stainless scope, balance-element rule,
  composition coverage, row structure, target/event convention, evidence,
  review status, and recommended handling.
- `source_conventions_summary.json`: counts by material scope and handling
  status.
- `source_reference_metadata.csv`: bibliography, DOI/OpenAlex metadata, and
  workbook alloy designations used during review.
- `openalex_metadata.json`: cached OpenAlex responses.

Rebuild the registry with:

```bash
python3 build_source_conventions.py
```

The script requires `openpyxl` and reads the preserved workbook without
modifying it.

## Collection-wide conventions confirmed from the author manuscript

- Alloy composition is in **wt%** and was taken from a certificate or standard.
- A blank composition cell means missing/unreported information. The paper
  handled missing covariates through multiple imputation; it did not define
  blanks as physical zero.
- The workbook's `E_pit` is the reported breakdown potential `Ebr`, in mV
  versus Ag/AgCl (3 M KCl). It can be actual pitting or another breakdown.
- `event=1` is actual pitting. `event=0` is a competing non-pitting breakdown
  and is right-censored for pitting-potential analysis.
- Raw anion concentrations are molar. The `log10` transform and `10^-5 M`
  value for unmentioned anions belong to the paper's model preprocessing, not
  to the raw workbook.
- Source-level dependence is real; train/test splitting should group by source.

## How to read the balance fields

`balance_element=Fe` says the alloy is Fe-based or specified with Fe as the
nominal remainder. It does **not** say Fe is always the numerically largest
component. `rows_Fe_balance_not_largest_if_blanks_zero` makes that distinction
visible.

The computed Fe remainder is only an audit diagnostic. Because the workbook
omits some elements and leaves others blank, `100 - sum(recorded elements)` is
an approximate benchmark feature, not a certified chemical analysis.

## Recommended benchmark use

For the current informed-vs-generic-vs-traditional model comparison, a
consistent zero-fill benchmark remains defensible if the missingness policy is
explicit and applied to every model. Preserve missingness indicators and split
by source. Before use, repair `2000Russell.Ti`; then either split/filter the four
mixed-family sources and `2009Wong`, or include a material-family/source flag.

A later chemistry-optimization model should recover omitted alloying variables
and verified compositions from the original papers rather than relying on the
zero-filled approximation.
