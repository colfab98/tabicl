# Schema Summary

## Inspected Files

- `raw/CRA_database_Scientific_Data_Publication_12102020.xlsx`
- `raw/s41597-021-00840-y (1).pdf`

## Workbook Structure

Observed workbook sheets:

| Sheet | Observed Data Rows | Max Columns Seen | Main Target/Response |
|---|---:|---:|---|
| `Pitting Potential` | 810 | 39 | `Epit, mV (SCE)` avg/max/min |
| `Repassivation Potential` | 189 | 37 | `Ave. Erp` |
| `Crevice Corrosion Potential` | 31 | 29 | `Ecrev, mV (SCE)` avg/max/min |
| `Pitting Temp` | 117 | 27 | `Tpit, oC` avg/max/min |
| `Crevice Corrosion Temp` | 70 | 33 | `Tcrev, oC` avg/max/min |
| `HEAs_Ecorr, icorr, ipass, Rate` | 51 | 41 | `ECORR`, `iCORR`, corrosion rate, `iPASS` |
| `References` | 85 reference entries | 3 | Source references/DOIs |

Data-row counts above subtract the visible header rows when present. They are an inspection count, not a replacement for the paper's reported dataset statistics.

## Feature Groups

- Material/composition: element wt.% columns such as Fe, Cr, Ni, Mo, W, N, Nb, C, Si, Mn, Cu, P, S, Al, V, Ta, Re, Ce, Ti, Co, B, Mg, Y, Gd.
- Environment: test solution, chloride concentration, pH, test temperature.
- Test/history/procedure: test method, scan rate, heat treatment, microstructures, comments.
- Electrochemical response: pitting, repassivation, crevice potential, pitting/crevice temperature, Ecorr, icorr, ipass, corrosion rate.
- Provenance: reference number, DOI/reference sheet, material class.

## Targets

This workbook is target-rich. Electrochemical variables are measured corrosion responses and should be treated as targets or response descriptors, not generic pre-outcome covariates.

## Leakage Risks

- `Epit`, `Erp`, `Ecrev`, `Tpit`, `Tcrev`, `Ecorr`, `icorr`, `ipass`, and corrosion rate are outcome-like.
- Text comments may include experimental outcome details and should not be used as model features without careful review.
- Reference IDs and material class are provenance/metadata; they can leak dataset/source identity.

## Relevance To Informed Prior

Very high relevance. This is the strongest downloaded source for the corrosion-informed prior because it repeatedly exposes material composition, environment, procedure/history, and electrochemical response blocks in a single harmonized workbook.

Safe prior implications:

- Material and environment blocks should be separate.
- Material-environment interaction is strongly justified.
- Electrochemical response variables form a coherent target block.
- Heat treatment/microstructure can justify a processing/history block, but exact numeric strengths should remain heuristic unless a formal analysis is done.
