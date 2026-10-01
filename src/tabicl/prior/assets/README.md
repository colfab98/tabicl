# EPIT composition and feature profiles

The EPIT assets are versioned so historical experiments and the current schema
cannot be confused.

| Asset | Purpose | Model-visible elements | Eligible pretraining templates |
| --- | --- | ---: | ---: |
| `epit_dataset_v1.json` + `epit_dataset_v1.csv` | Frozen v7 compatibility | 17 | 403 |
| `epit_dataset_v2.json` + the same checked CSV | Current v8 schema | 24 | 400 |
| `epit_fe_nicrmo_features_v1.json/.csv` | Target-free joint environment/method rows for Fe and Ni-Cr-Mo families | Uses the selected composition profile | 315 Fe/Ni templates under either profile |

The shared CSV contains 403 exact distinct source compositions with all 24
source element fields. The v2 metadata exposes them in this fixed order:

`Fe, Cr, Ni, Mo, W, N, Nb, C, Si, Mn, Cu, P, S, Al, V, Ta, Re, Ce, Ti, Co, B, Mg, Y, Gd`.

The v2 profile does not interpret column coverage as evidence that a blank means
zero. Coverage is only the fraction of rows with a populated cell. For each
composition, missing element values become structural zeros only when the sum of
reported elements is within 0.1 wt.% of 100. Three distinct templates fail that
closure check and are excluded from pretraining sampling:

- `epit_0314`, HEA, reported sum 100.52 wt.% (source No. 581).
- `epit_0326`, HEA, reported sum 99.8679 wt.% (source No. 601).
- `epit_0390`, Al alloy, reported sum 99.5 wt.% (source No. 773).

No evaluation row is removed. The direct evaluator retains all 760 rows, exposes
all 24 composition columns, zero-fills only closure-supported blanks, and leaves
unresolved blanks missing for train-fitted imputation. The pretraining sampler
uses only the 400 eligible distinct templates, with eligible family counts
298/55/17/17/13 in the documented family order.

Training and sampling never open the source Excel workbook. The static assets are
checksum-validated and cached by
`tabicl.prior.epit_composition_profile.load_epit_composition_profile`. The v2
metadata intentionally reuses `epit_dataset_v1.csv`; duplicating identical data
would create two files that could drift.

The current physical schema has 28 base features: 24 composition, three
environment, and one test-method column. Optional Magpie descriptors still use
the original 17-element lookup subset because the fixed lookup table has not
been expanded; they are appended to the 28 base features for a total of 38.
Magpie remains disabled in the current empirical-feature search configuration.

Explicitly loading `epit_dataset_v1` preserves the historical 17/3/1 feature
path used by the frozen v7 model and coefficient-variation diagnostics. New code
defaults to `epit_dataset_v2` and 24/3/1.
