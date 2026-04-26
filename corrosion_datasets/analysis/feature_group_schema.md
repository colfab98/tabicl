# Cross-Dataset Feature Group Schema

This schema is intentionally high level. It is meant to support TabICL prior design without copying dataset-specific labels, thresholds, or target distributions.

## Core Groups

| Group | Meaning | Seen In |
|---|---|---|
| Material/composition | Alloy identity, element fractions, binder proportions, material family, UNS/alloy designation. | Electrochemical metrics, MPEA, AM-MPEA, steel mortar, NACE/NIST. |
| Microstructure/phase | Phase indicators, microstructure notes, material class, phase presence. | Electrochemical metrics, MPEA, AM-MPEA. |
| Environment/exposure | Electrolyte, chloride/concentration, pH, temperature, salinity, oxygen, relative humidity, pore solution properties. | Electrochemical metrics, MPEA, AM-MPEA, steel mortar, NACE/NIST, mooring steel. |
| Processing/history | Heat treatment, AM process, processing route, immersion/exposure duration, scan rate/test method where used as context. | Electrochemical metrics, MPEA, AM-MPEA, NACE/NIST, mooring steel, 316L. |
| Electrochemical response | Ecorr, Epit, Erp, Ecrev, Tpit, Tcrev, icorr, ipass, OCP, polarisation curves, corrosion potential/current/rate. | Most datasets, but often as targets or measured responses. |
| Intervention/treatment | Heat treatment, HIP, cold rolling, coatings/inhibitors if present, process changes that plausibly alter corrosion state. | AM-MPEA, MPEA, electrochemical metrics. |
| Provenance/reference | DOI, reference number, paper/source citation, comments. | Electrochemical metrics, MPEA, AM-MPEA, NACE/NIST. |

## Dataset Mapping

| Dataset | Strong Groups | Weak/Missing Groups |
|---|---|---|
| Electrochemical metrics alloys | Material, environment, electrochemical response, heat treatment, microstructure. | True time/history and interventions are limited. |
| MPEA corrosion | Composition, phases, environment, processing, electrochemical response. | Temporal exposure history is mostly absent. |
| AM-MPEA corrosion | AM process, composition, phases, environment, electrochemical response, derived composition descriptors. | Small sample size; history beyond process label is limited. |
| 316L pitting/passivity | Electrochemical curve structure, NaCl concentration, scan rate. | Material variation is absent because alloy is fixed 316L. |
| Steel mortar corrosion | Mixture, material pore solution, environment, electrochemical response. | Alloy composition is fixed/implicit steel; time/history not primary in the sheet schema. |
| NACE/NIST CORR-DATA | Broad material family, environment, concentration, temperature, duration, rate/rating. | Heterogeneous text fields; less clean numerical structure. |
| Mooring steel seawater | Temperature, salinity, oxygen, exposure time, OCP time series. | Single steel grade and small structured table. |

## Safe Structural Signal

The downloaded datasets strongly support representing corrosion tasks with separate but interacting material, environment, processing/history, and electrochemical-response blocks.

They also support keeping material-environment interaction explicit: corrosion behavior is almost always conditional on electrolyte/concentration/pH/temperature or related exposure context.

Electrochemical measurements form a coherent response block, but they should usually be treated as targets or post-treatment measurements, not generic pre-outcome features.
