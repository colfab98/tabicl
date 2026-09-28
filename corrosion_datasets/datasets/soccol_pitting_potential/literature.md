# Literature and Provenance

## Collection paper

Dimitri Soccol, “An updated Pitting Resistance Equivalent Number by
proportional hazard survival models of reported pitting potentials,”
*Electrochimica Acta* 511, 145355 (2025; online 2024).

- DOI: https://doi.org/10.1016/j.electacta.2024.145355
- Publisher: https://www.sciencedirect.com/science/article/pii/S0013468624015913
- KU Leuven record: https://lirias.kuleuven.be/4204498
- Preserved author manuscript: `raw/Soccol_updated_PREN_author_manuscript.pdf`

The paper analyzes the public collection with Cox and Weibull proportional
hazards models. It uses 20-fold multiple imputation, right-censoring for
competing breakdown, and source-level frailty effects. This is prior predictive
and statistical use of the collection, although it is not a conventional
random-forest/XGBoost benchmark.

The author manuscript is the primary evidence for the collection-wide
conventions recorded here: wt% composition, certificate-or-standard
provenance, mV versus Ag/AgCl (3 M KCl), and `event=1` pitting versus `event=0`
competing breakdown.

## Dataset repository

- Repository: https://github.com/SoccolD/Pitting_potential_database
- Preserved commit: `50f2df0918116e12026f8ea18c32704a5f6822e5`
- Upstream description: “Database with critical pitting potential
  measurements. This file served as a basis for an observational study on
  influential parameters.”

The workbook's `references` sheet is the authoritative mapping between the 154
`source` identifiers and the experimental publications. DOI/OpenAlex metadata
and source-level review fields are preserved in
`../../analysis/soccol_source_conventions/`.

## Prior ML prediction in included studies

- `2009Ramana`: neural-network modeling of AISI 316L pitting behavior.
  DOI: https://doi.org/10.1016/j.matdes.2009.01.039
- `2014Jimenez-Come`: artificial-intelligence prediction of austenitic
  stainless-steel pitting behavior.
  DOI: https://doi.org/10.1016/j.jal.2012.07.005
- `2016Alar`: models for corrosion and pitting potential of AISI 304.
  DOI: https://doi.org/10.20964/2016.09.26

These establish prediction precedent for subsets, not a verified benchmark for
all 4,460 rows.

## Material-scope and source evidence

- `2009Wong`: Fariaty Wong's dissertation, “The Effect of Alloy Composition on
  the Localized Corrosion Behavior of Ni-Cr-Mo Alloys.” The rows are Ni-based,
  and recorded Ni+Cr+Mo is approximately 100 wt%.
  https://etd.ohiolink.edu/acprod/odb_etd/r/etd/search/search-results?clear=1001&p1001_keyword=metastable%2520pitting&request=KEYWORD_SEARCH
- `2006Muwila`: institutional thesis record confirms Fe-based low-Ni
  austenitic experimental alloys with varied Mn, Mo and N.
  https://wiredspace.wits.ac.za/items/78b6192d-9bd7-49fb-a478-9f13de00cb81
- `2007Saithala`: institutional conference record identifies Zeron 100, 2205
  and Ferralium Alloy 255 duplex stainless steels.
  https://shura.shu.ac.uk/view/types/conference%3D5Fitem/2007.date.html
- `2015Blackwood`: the correct DOI is
  https://doi.org/10.14773/cst.2015.14.6.253; the workbook truncates the final
  digit.

The evidence and confidence level for every source are in
`source_conventions.csv` rather than being inferred globally from Fe or Ni.

## Access and licensing

No explicit license was detected in the inspected GitHub repository. The author
manuscript is stored from the KU Leuven institutional record. Check dataset and
manuscript reuse terms separately before public redistribution.
