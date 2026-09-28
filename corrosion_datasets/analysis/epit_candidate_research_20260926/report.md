**Recommended next EPIT dataset: Soccol’s stainless-steel pitting-potential database**

Research date: 2026-09-26. This recommendation follows `main4.tex`, `main4_supplement.md`, the existing corrosion dataset inventory, and a fresh search of publications, public repositories, DOI registries, and downloadable workbooks. It is a dataset-selection study; no prediction model was trained or selected.

**Recommendation and rationale**

Use **Dimitri Soccol’s `Pitting_potential_dataset_2024.xlsx`** as the next main dataset. It preserves the central task in main4—composition and environmental/test conditions → pitting potential—while providing a different compilation, a stainless-steel focus, more observations, explicit censoring, and more surface-preparation and electrolyte detail.

- [Author repository](https://github.com/SoccolD/Pitting_potential_database)
- [Version-pinned Excel download](https://raw.githubusercontent.com/SoccolD/Pitting_potential_database/50f2df0918116e12026f8ea18c32704a5f6822e5/Pitting_potential_dataset_2024.xlsx)
- [Associated paper: An updated Pitting Resistance Equivalent Number by proportional hazard survival models of reported pitting potentials](https://doi.org/10.1016/j.electacta.2024.145355), Electrochimica Acta 511, 145355 (2025; available online in 2024).
- [Local workbook](raw/Pitting_potential_dataset_2024.xlsx)

The paper uses Cox and Weibull proportional-hazards models, including source-level effects and censoring. This establishes prior statistical prediction/regression use, but I did not verify an RF/XGBoost/neural-network benchmark on this exact release. Conventional ML precedent was only a preference in the request; structural suitability and usable data make this the stronger choice.

**What was verified directly**

The paper describes approximately 4,500 measurements from 155 studies. The actual downloaded 2024 workbook contains **4,460 data rows, 43 columns, and 154 source identifiers/reference entries**. These are measured file counts, rather than rounded publication counts.

| Target status in the workbook | Count | Proposed role |
| --- | ---: | --- |
| Numeric `E_pit`, `event == 1` | 4,027 | Initial observed-event regression pool |
| `event == 0` | 398 | Censoring records; 357 have numeric limits |
| Missing `event` | 33 | Quarantine pending interpretation; 24 have numeric potentials |
| `event == 1`, missing `E_pit` | 2 | Exclude from regression |

The event interpretation follows the paper’s observed/censored framework and the workbook convention. The workbook does not supply a separate comprehensive data dictionary, so this convention should be explicitly confirmed during final ingestion. A numeric value alone is not sufficient to select a regression label.

There are no exact full-row duplicates, but this does not establish independent experiments. Labels are also not unique: the 4,460 rows contain 4,448 distinct labels. Preserve workbook row numbers as stable record identifiers. Do not drop rows simply because their label or feature vector repeats; some are repeated experiments.

The repository also includes two older workbooks. I downloaded them for provenance and inspected their structures, but did not merge them. They are alternative revisions, not independent additional datasets. The similarly named `Pitting_database_update_231002.xlsx` has 6,079 parsed rows and is not the selected 2024 release.

**How close is it to main4?**

| Aspect | Current main4 task | Soccol 2024 workbook |
| --- | --- | --- |
| Outcome | Average EPIT, mV vs SCE | `E_pit` plus `event` |
| Materials | Fe, Ni–Cr–Mo, Al, HEA, others | Stainless-steel-oriented compilation |
| Composition | 17 selected element columns | 12 explicit elements: C, N, Si, P, S, Ti, V, Cr, Mn, Ni, Nb, Mo |
| Core environment | Temperature, chloride, pH | `CP_temp`, `CP_Cl`, `CP_pH` |
| Additional solution detail | Primarily collapsed into retained inputs | Br, OH, sulfate, carbonate, nitrate, phosphate, molybdate, chromate, other ions |
| Surface preparation | Not separately retained | Grinding grit, roughness, preparation medium/pH/redox/time |
| Test detail | One method category | Scan rate, test area, conditioning time, aeration, agitation |
| Provenance | Numeric reference field | Source label/ID and separate reference/DOI sheet |

This is a good second dataset for developing the method, but **not a drop-in input table for the frozen 21-column checkpoint**. In particular, Fe and several original elements are absent from the 2024 schema. Do not set unreported element contents to zero or infer Fe by difference when substantial constituents are missing. The source-specific feature schema and prior should be redesigned and validated if retraining CorrPFN.

Its stainless-steel focus is a closer match to main4’s Fe/Ni–Cr–Mo informed branch than the MPEA alternatives. The new surface-treatment and solution variables also expose dependencies that the existing 21-column representation cannot describe. This is my suitability assessment, not evidence of predictive improvement.

**Overlap with Nyby: the central qualification**

A different repository is not proof of independent experimental data. I compared the Soccol reference sheet with the full Nyby reference sheet already in this project, normalizing DOI prefixes, whitespace, case, and byte-order-mark characters.

1. **44 Soccol source identifiers (43 distinct DOIs)** match Nyby references. They account for **632 of the 4,027 observed numeric rows**.
2. Bibliographic comparison identified three more shared sources with missing DOIs: `1991Azuma` ↔ Nyby reference 16, `1986Jargelius` ↔ reference 42, and `2009Wong` ↔ reference 44. These remove **71 additional rows**.
3. Excluding all those source identifiers leaves **3,324 observed numeric rows across 105 source identifiers**. They contain **209 distinct reported composition vectors**, counting different missing-value patterns separately.
4. Of the remaining rows, **2,724 have a source DOI**, across 89 source identifiers; **600 rows from 16 identifiers lack a DOI** and need further provenance review.

These are **provisional source-screened counts**, not a certified independent final test set. DOI errors, alternate publications of the same experiments, and incomplete references can conceal further overlap. Same alloy compositions can occur across independent sources, so this screening also does not establish composition novelty. The conservative screen uses *all* Nyby references, including references used by other sheets. Screening only references appearing in Nyby’s pitting sheet gives a slightly larger pool; the full-reference screen is preferable here.

The audit exports both automatic DOI matches and row-level exclusion flags. No targets were used to choose sources for removal.

**Data quality and remaining preparation**

- **Reference scale:** the associated paper discusses potentials on an Ag/AgCl scale, whereas main4 uses SCE. The raw header is just `E_pit`. Verify the exact reference-electrode convention, units, and conversion metadata before pooling targets or comparing errors in physical units. I have not silently converted the workbook. The paper preview does not establish the precise Ag/AgCl filling-solution convention for every record.
- **Censoring:** using observed events alone permits ordinary regression, but evaluates the subset where pitting was observed. It does not estimate the full uncensored EPIT distribution. Keep the censoring records for a later censoring-aware formulation; do not interpret a transpassive limit as a measured pitting potential.
- **Missing chemistry:** even before detailed cleaning, the source-screened pool has nearly complete temperature/pH/chloride and high scan-rate coverage. Cr/Ni have much better coverage than N, Ti, V, or Nb. Mo has less than 80% coverage. Reusing main4’s 80% column-retention rule blindly would discard useful corrosion variables. Exact current coverage is in `soccol_feature_coverage.csv`.
- **Repeated measurements:** 3,324 rows are not 3,324 independent alloy compositions. Split by source DOI/bibliographic identity, keep experiment replicates together, and run a separate composition-group evaluation if the scientific claim is new-alloy generalization. Some source identifiers share a DOI; do not separate these across source-held-out folds.
- **Outcome predictors:** keep `E_corr`, `event`, IDs, source text, and references out of a composition/environment-only predictor set. `event` is label-status metadata; `E_corr` is a measured response requiring a different prediction-time assumption.
- **Access and reuse metadata:** the files are publicly downloadable. The checked GitHub commit has no explicit license file, and GitHub reports no detected license. Do not describe it as CC BY or as openly licensed without additional evidence. This is relevant if packaging or publishing a redistributed benchmark.

**Alternative candidates and why they rank lower**

| Candidate | Verified/reported usable size and prior use | Assessment for this task |
| --- | --- | --- |
| **MPEA corrosion database**, Birbilis and collaborators | Local published workbook: 619 rows, **335 numeric EPIT values**, 226 in NaCl. The 2025 ML paper reports **306 EPIT entries**, an unresolved preprocessing/version difference. | Best fallback or a later test of transfer to different alloy chemistry. Composition is atomic fraction; includes phase, processing, electrolyte and concentration, but no separate temperature/pH columns. Larger mismatch to the current Fe/Ni–Cr–Mo rules and much smaller EPIT pool. |
| **Rebar chloride-threshold/electrochemical database**, Hou et al., August 2026 | Downloaded Excel: **351 numeric pitting/breakdown values** across four sheets: 283 carbon steel and 68 stainless steel. | Strong newly released tabular alternative, especially for a separate alkaline-environment task. Many more rows exist, but lack the required target. Cement/pore-solution chemistry and inhibitor variables change the task substantially. No prior ML benchmark on this release was verified. |
| **Additively manufactured MPEA database**, V3 | Local archive: **97 rows, 58 numeric EPIT values**. | Useful small supplementary domain, insufficient as the main next benchmark. |
| **316L pitting/passivity descriptors**, Coelho et al. 2023 | Existing local archive: 955 descriptor rows across five concentration/scan-rate settings; ML-based extraction from polarization curves. | Same alloy, little composition variation, and estimated labels. Better for curve/descriptor analysis than the main4 composition-to-EPIT task. |
| **MAP-E**, Persaud et al., 2026 preprint | Public data/code; Gaussian-process-guided pH/chloride experiments on 304 steel, plus 32 benchmark measurements. I did not audit a complete tabular target count. | Interesting controlled external environment-transfer study, but fixed material composition and a narrower problem. |
| **Qiao et al. 2023 stainless-steel XGBR paper** | Direct EPIT prediction, reported R² 0.877 and RMSE 132.51 mV. | Very relevant paper, but I could not verify an accessible raw table, its usable count, or overlap with Nyby. It cannot outrank an inspected downloadable candidate. |
| **AM stainless-steel EPIT ML**, Montes de Oca Zapiain et al., 2026 preprint | Direct pitting-potential prediction; data availability states reasonable request. | Promising process-rich future candidate; raw access not verified here. |
| **Austenitic stainless CPT ML**, 2025 | 123 literature observations, composition and conditions, prior ML use. | Similar structure but predicts critical pitting **temperature**, not EPIT. Only suitable if changing the target is acceptable. |
| **Pipeline steel pitting judgment**, 2021 | 100 observations, composition/solution/environment features, prior ML. | Binary pitting/no-pitting classification, not EPIT regression. |

MPEA also has source overlap: exact DOI matching alone identifies **22 numeric EPIT rows** from sources in Nyby, leaving 313 before deeper provenance/quality checks. Its `Calculated passive window` is directly target-derived and must not be an EPIT input. The ML paper’s generated data are not extra independent measurements.

Primary sources for this comparison:

- MPEA [dataset](https://data.mendeley.com/datasets/nskb9khgsm/1) and [2025 ML study](https://www.nature.com/articles/s41529-025-00700-9).
- Rebar [dataset and usage notes](https://zenodo.org/records/21987599). Notes state potentials are already in volts vs SCE; multiply by 1,000 for mV, rather than performing a reference-electrode conversion. The release is CC BY 4.0.
- AM MPEA [dataset V3](https://data.mendeley.com/datasets/cfz68tkwfk/3).
- 316L [paper](https://www.nature.com/articles/s41529-023-00403-z) and [descriptor data](https://data.mendeley.com/datasets/5x4dmc38bg/1).
- MAP-E [preprint](https://arxiv.org/html/2603.09845v1) and [data/code](https://github.com/dpersaud/MAP_E).
- Qiao [publisher article](https://doi.org/10.1016/j.colsurfa.2023.132274).
- AM stainless-steel [preprint](https://www.researchsquare.com/article/rs-9213876/v1).
- CPT [paper](https://www.nature.com/articles/s41529-025-00563-0).
- Pipeline [paper](https://www.frontiersin.org/journals/materials/articles/10.3389/fmats.2021.733813/full).

**Reused datasets rejected as new benchmarks**

The [2022 deep-learning framework](https://www.nature.com/articles/s41529-022-00281-x) and [2023 NLP/deep-learning paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10421031/) reuse Nyby’s electrochemical metrics dataset. Their 769-record adaptation is not an independent replacement for main4’s 760 numeric targets. The [2024 ensemble-learning study](https://www.sciencedirect.com/science/article/pii/S0010938X23008338) likewise identifies the same 1,274-record multi-metric source database. A different model or feature representation does not make these fresh data.

**Recommended experiment design**

First decide whether the goal is a new dataset-specific model or external evaluation of the frozen Nyby-trained model. For the former, use Soccol’s own richer schema, rebuild profiles/calibration inside the new development partition, and reserve complete source groups for final evaluation. For the latter, first establish a defensible mapping to the frozen 21-column schema and remove shared experimental sources; substantial missing elements currently prevent assuming plug-and-play transfer.

For an initial new-dataset regression task, retain observed numeric EPIT rows, define source-group partitions before fitting anything, and compare generic tabular predictors with the adapted CorrPFN. Fit imputation, feature profiles, coefficient calibration, and model selection within the appropriate training partitions. The supplement’s coefficient-variation setting is another development choice, not a reason to tune against the new final-test labels. Evaluate Spearman alongside MAE/RMSE/R², and account for study clustering when estimating uncertainty. Do not force near-100% row coverage at the expense of meaningful label definitions.

The **3,324-row source-screened pool** is the practical starting point. A stricter preliminary option is the **2,724-row DOI-backed pool**, while the remaining 600 rows undergo bibliographic review. Neither number is promised as the final cleaned training size.

**Saved evidence and reproduction**

- `raw/`: unchanged downloaded Soccol workbook revisions, repository README, rebar workbook and usage notes.
- `raw_manifest.json`: file sizes and SHA-256 hashes.
- `repository_metadata.json`, `rebar_metadata.json`: public repository metadata snapshots.
- `audit_candidates.py`: reproducible counts, coverage and overlap audit; requires pandas/openpyxl.
- `audit_summary.json`: exact totals reported here.
- `soccol_row_audit.csv`: Excel row numbers, event status and source-exclusion flags.
- `overlap_all_nyby.csv`, `overlap_pitting_nyby.csv`: normalized-DOI source matches.
- `soccol_references_missing_doi.csv`, `nyby_references_missing_doi.csv`: remaining bibliographic audit inputs.
- `soccol_feature_coverage.csv`: numeric coverage for the observed pool and source-screened pool.

The Soccol workbook is pinned to GitHub commit `50f2df0918116e12026f8ea18c32704a5f6822e5`; its SHA-256 is `cb255c5a94d90bd207248f6e25727f965d587c29a73b2ffcef6b8c0ced6da1da`. Existing MPEA, AM-MPEA and Nyby files were read from the project's downloaded raw archives. Counts for the 316L candidate follow the previously inspected local schema. No existing split, training configuration, main4 document, or model result was changed.
