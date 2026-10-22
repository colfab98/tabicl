# Dataset processing guidelines

These rules define how we will prepare datasets for the shared schema. They
start with composition features. Dataset-specific decisions for EPIT, Soccol,
MPEA, and future datasets will be documented separately after checking or fixing
their processing. This guide does not mean that existing exports already follow
every rule.

## Composition features

- **One column per element:** use the same element names across every dataset,
  such as `Fe`, `Cr`, and `Ni`.

- **One unit: weight percent (wt.%).** Check the source units first. Convert
  atomic percentages or atomic fractions using atomic weights. Convert weight
  fractions to percentages. Do not assume an incomplete list of elements is a
  complete composition when converting units.

- **Keep reported amounts**, including explicitly reported zeros.

- **Treat blanks and NaNs as missing initially.** Decide how to fill them using
  the alloy description and dataset context.

- **Fill a missing base element when justified:** for example, for a clearly
  Fe-based alloy whose other reported amounts reasonably describe its alloying
  additions, calculate `Fe = 100 - sum of the other element amounts`. Apply the
  same principle to another base element when appropriate. The source does not
  have to explicitly say "balance Fe" if the alloy description and dataset
  context support that interpretation. A low total alone is not enough reason
  to assume Fe.

- **Set other unreported elements to `0` when we have enough confidence that
  the reported amounts plus any justified balance describe the alloy.** This
  includes missing entries in original element columns and additional element
  columns introduced by the shared schema. These are assumed zeros; preserve
  the distinction from explicitly reported zeros in the processing record.

- **Exclude rows with an unexplained composition total below 90 wt.%.** Use
  **90 wt.% as the current default minimum**. Apply this check after verifying
  units and applying any justified balance calculation. If the total remains
  below 90 wt.% and the dataset provides no explanation or basis for completing
  it, exclude the row from the model-ready data. Keep the original row and
  record the exclusion reason. An explanation requires review; it is not
  automatic acceptance of an incomplete composition.

- **Flag other unclear compositions for review:** if an important amount is
  unknown and the context does not support filling it, do not invent a value.
  A total of 90 wt.% or more does not by itself establish completeness.

- **Check the completed composition:** amounts must be between 0 and 100, and
  the total should be approximately 100%. Allow small differences consistent
  with the source's rounding precision; do not automatically rescale incomplete
  compositions. A total of 100% produced by our balance calculation is not
  independent evidence that the original composition was complete.

- **Record our changes:** preserve the original data and document unit
  conversions, assumed zeros, calculated balance amounts and their justification,
  unresolved rows, and exclusions.

## Examples

- Reported alloying additions total 25 wt.%, and the alloy context supports an
  Fe balance: set `Fe = 75`, then set other unreported elements to `0` if the
  reconstructed composition is accepted as complete.
- Reported elements total 85 wt.%, with no explanation or justified balance
  element: exclude the row and record the unexplained shortfall.
- Reported elements total 96 wt.%, but an important element amount is unclear:
  flag for review; passing the 90 wt.% threshold is not enough to fill the gap.
- A complete composition is mapped to a shared schema containing additional
  elements: set those additional element columns to `0`.
