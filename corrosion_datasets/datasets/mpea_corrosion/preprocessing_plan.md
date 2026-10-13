# MPEA evaluation-candidate preparation v2

Prepared 2026-10-12 by `prepare_evaluation.py`. This is data preparation only:
no prior fitting, imputation, scaling, training, model evaluation, or partition
selection. The raw workbook and paper are unchanged. This is not yet a
drop-in input to the current fixed-position models.

## Source and row selection

The [Mendeley dataset, version 1](https://data.mendeley.com/datasets/nskb9khgsm/1)
describes composition as atomic fractions and empty cells as unreported data.
The local workbook contains 619 rows and 39 columns on `Sheet1`.

There are **335 rows** with a finite numeric `Pitting potential (mV vs SCE)`,
including one genuine numeric zero. Discard the 3 incomplete-composition rows
at the user's request, leaving **332 eligible candidate rows**. The other 284
rows have no usable target. All excluded rows remain in the raw workbook and
audit, not in the candidate table. Do not
fill targets, trim their range, or select rows using model performance.

`row_id=mpea_v1_rXXXXXX` embeds the original Excel row number, including the
header offset: the first data row is `mpea_v1_r000002`. This ID is tied to the
raw workbook hash in the manifest. It is metadata, not a predictor.

The [collection paper](https://www.nature.com/articles/s41529-025-00700-9)
reports **306** pitting-potential entries. The workbook has **335**. Checks
of nonzero targets (334), complete response triplets (311), fully populated
rows (257), complete composition sums (332), and exact duplicate raw rows
(none among the 335) do not explain the difference. It remains unresolved;
no undocumented 29-row deletion was made. The public target label is retained;
individual source measurements have not been rechecked for censoring or
alternative breakdown mechanisms.

## Composition

Retained predictors, supported by at least one of the EPIT/Soccol priors:

`Al B Co Cr Cu Fe Mg Mn Mo Nb Ni Si Ta Ti V W Y`

The workbook also reports `Be Ga Hf La Sn Zn Zr`. These are excluded from
the current candidate predictor list because neither current prior trains on
them. They are **not** deleted from the raw data or the conversion denominator.
This 17-element list is a candidate export, not the future full periodic-table
feature registry. Elements absent from the workbook are not invented as zeros.

For each of the 24 reported elements, convert atomic fraction `x_i` to wt%:

```text
wt%_i = 100 * x_i * atomic_weight_i / sum_all_24(x_j * atomic_weight_j)
```

The processor freezes the [CIAAW abridged standard atomic weights (2024)](https://ciaaw.org/abridged-atomic-weights.htm)
and records them in the manifest. Conversion requires finite, nonnegative
fractions and a total within 0.001 of 1. Explicit workbook zeros remain zeros;
missing composition is not zero-filled. The retained 17 elements are **never
renormalized** to sum to 100%; omitted elements can account for the remainder.

- 332 of the 335 target-bearing rows pass composition conversion and are exported.
- Excel rows 466–468 sum to 0.9375. They are excluded from both candidate
  exports at the user's request, with reason `composition_not_convertible`.
  The raw rows and their target values remain recoverable in the workbook and
  all-row audit; no attempt was made to reconstruct their compositions.
- 67 candidate rows contain positive amounts of an excluded element: Hf in
  10, La in 1, Sn in 14, and Zr in 52 rows, with overlap between categories.
  These rows remain flagged, not silently removed. Be, Ga, and Zn are zero in
  all target-bearing rows.

## Environmental features and units

Only conditions explicitly supported by workbook fields/text are populated.
The seven slots are supported in EPIT and/or Soccol; entirely unobserved slots
are reserved rather than filled. The exact column order is in
`processed/feature_manifest.json`.

| Feature | Unit | Populated candidate rows | Rule |
|---|---|---:|---|
| `CP_temp` | °C | 5 | Extract explicit temperatures; Kelvin minus 273.15. |
| `CP_pH` | pH | 0 | No explicit numeric pH found; keep `NaN`. |
| `CP_Cl` | mol/L | 216 | Simple NaCl solutions with consistent reported molarity. |
| `CP_OH` | mol/L | 5 | Simple NaOH/KOH solutions with consistent reported molarity. |
| `CP_SO4` | mol/L | 0 | Do not equate formal H2SO4 concentration with free sulfate. |
| `CP_NO3` | mol/L | 0 | HNO3 records here have unverified mass%-to-molar conversions. |
| `scan_rate` | mV/s | 18 | Extract explicit 0.1, 1, or 10 mV/s from environment text. |

For NaCl/NaOH/KOH, these are nominal analytical concentrations under simple
salt stoichiometry, not activities. If both a molarity in text and the numeric
column are present, they must agree. Unmentioned ions stay missing, not zero.

Conservative handling of ambiguous information:

- Excel row 21 says `0.6 M NaCl` in the description but `H2SO4` and 0.5 M
  in the other fields. Do not choose a chemistry silently; leave ion features
  missing and flag the conflict.
- 41 rows describe mass-percent solutions. Keep the supplied numeric
  `Concentration in M` in the audit, but do not use it as a verified ion
  concentration. The conversion basis/density is unverified; in particular,
  `5% NaOH` is paired with both 1.32 and 5 M, and 10% HNO3 with both 1.76
  and 2.38 M. These require original-source checks.
- 43 rows use seawater, PBS, or Hanks' solution. Do not infer a complete
  electrolyte recipe or treat a mixture concentration as chloride concentration.
- 26 eligible non-conflicting H2SO4 rows retain the formal concentration as metadata;
  sulfate speciation is not inferred. The parser likewise does not infer
  nitrate speciation from acid concentration.
- Do not infer pH from acid/base concentration. Do not fill room temperature
  from a general paper description: the workbook explicitly includes 353 K,
  310 K, and 37 °C. Explicit temperature or scan-rate values can still be
  used when the electrolyte chemistry is ambiguous.

`NaN` means unknown/unreported, not physical zero. The CSV uses the literal
`NaN`, which pandas reads as missing. Learned missingness vectors, training-time
masking, and the joint model's complete fixed feature order are **not implemented
by this preparation**. Audit flags and missingness provenance are not extra
predictor columns.

## Target and excluded inputs

`E_pit_mV_SCE` is the original numeric pitting potential, unchanged in mV
versus SCE. EPIT uses SCE, but Soccol documents Ag/AgCl (3 M KCl). Their
reference-electrode alignment still needs an explicit policy for combined
training; no unverified fixed offset is applied here. In plain terms, the
voltage readings have different zero points. Use SCE as the common scale and
convert Soccol targets using the documented electrode-potential difference,
with an explicit temperature convention; do not estimate this offset by fitting
MPEA targets. Conversion should precede source-prior fitting and real-context
assembly. Existing separately evaluated models/data are not changed here.

The [Soccol author manuscript, Table 3](https://lirias.kuleuven.be/retrieve/f69750dc-fb1a-4962-8e5c-8982608a4efb)
confirms scan rate is expressed in mV/s, matching the explicit MPEA text.
No scan-rate unit conversion is needed. The paper's log transformation is
a modeling step, not a change to the physical unit used in these raw exports.

Do not use alloy labels, phase/processing fields, raw environment strings,
reference IDs, source-overlap flags, `Ecorr`, current density, or calculated
passive window as predictors. Only the 24 named features in the feature
manifest belong in `X`; `row_id` is metadata and `E_pit_mV_SCE` is `y`.

## Source overlap: audit, not proof of independence

Extract DOI-like strings from MPEA reference text/URLs, normalize case and
URL encoding, and compare with **all reference-sheet DOIs** from the EPIT
and Soccol raw workbooks. The two input workbook hashes are recorded.

| Status among the 332 eligible candidates | Rows | Interpretation |
|---|---:|---|
| `shared_source_doi` | 22 | Shared EPIT publication DOI; possible prior-calibration exposure. |
| `no_shared_doi_found` | 267 | A DOI was extracted but did not match; independence is not established. |
| `unresolved_reference` | 43 | No DOI extracted; source matching remains unresolved. |

No candidate DOI matched the Soccol reference sheet. Shared EPIT sources:

| DOI | Candidate rows |
|---|---:|
| `10.1016/j.intermet.2010.10.008` | 7 |
| `10.1016/j.corsci.2018.01.030` | 6 |
| `10.1016/j.corsci.2010.04.004` | 3 |
| `10.1016/j.matchemphys.2005.01.001` | 3 |
| `10.1179/1743284715y.0000000026` | 3 |

DOI extraction is not DOI resolution: malformed reference entries may not
identify the correct paper. SSRN/ResearchGate links and generic publisher
URLs remain unresolved. Shared publications do not necessarily mean identical
measurements; absent DOI matches do not rule out copied measurements or
alternate publications. Exact cross-dataset row overlap has **not** been
assessed. Do not call the remaining 310 rows a verified non-overlap subset.

The bounded follow-up check searched EPIT/Soccol reference-sheet text for the
four unresolved SSRN IDs (`4032452`, `4128813`, `4156630`, `4157016`) and the
distinctive ResearchGate title phrases (directionally solidified FeCoNiCrAl;
Mo in AlCrFeMnNi). No additional matches were found. These seven unresolved
reference strings represent 34 SSRN rows, 4 ResearchGate rows, and 5 rows
with only a generic publisher URL. They remain unresolved; no extensive
publication-by-publication investigation was attempted.

Two reproducible exports are now available:

- Full eligible pool: 332 rows, including shared-source and unresolved rows.
- Source-screened subset: 267 rows with an extracted DOI and no source-sheet
  match. Excludes the 22 shared-source rows and all 43 unresolved rows. This is
  a conservative source screen, **not a verified measurement-non-overlap set**.

No evaluation partition has been frozen here. The proposed cross-dataset
setup uses labeled EPIT/Soccol rows as context and all 332 eligible MPEA rows
as queries, with the 267-row score reported as an additional subset of the
same predictions. It needs no MPEA context split and supplies no MPEA targets
to the model. All compared models must use the same source context and feature
schema. MPEA targets must not be used for tuning. Actual duplicate source
measurements in context would create direct target leakage, which is stronger
than exposure through fitted prior coefficients alone. The source screen is
only a limited safeguard against that.

When frozen, this complete MPEA query set serves as the outer final-test
partition; context-size diagnostics may vary source context, not query rows.

## Reproduction and artifacts

Run the commands in `README.md`. The processor checks the original MPEA
workbook SHA-256 before generating output. It writes:

- `mpea_evaluation_candidates.csv`: 332 rows, ID + 24 candidate features + target.
- `mpea_source_screened_candidates.csv`: the 267-row source-screened subset,
  with exactly the same feature and target schema.
- `mpea_row_audit.csv`: all 619 rows, raw metadata, conversion/source flags,
  omitted-element mass, and per-feature provenance.
- `mpea_source_audit.csv`: source-level raw/candidate counts and DOI matches.
- `feature_manifest.json`: explicit input schema, units, and excluded metadata.
- `preprocessing_manifest.json`: input and processor hashes, software versions,
  counts, conversion constants, and hashes of the other five outputs.

The generator is deterministic for the same files and dependency versions.
Focused tests cover conversions, preservation of unknowns and zeros, exclusion
of response-derived inputs, ambiguous conditions, DOI parsing, and raw-workbook
counts. These are preparation tests, not model performance evaluations.
