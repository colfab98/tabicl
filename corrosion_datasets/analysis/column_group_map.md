# Column Group Map

This file records the first-pass grouping used by `scripts/analyze_structure.py`.

The groups are deliberately coarse:

- `material`: composition, alloy/binder chemistry, phases, material family, material descriptors.
- `environment`: electrolyte, chloride/concentration, pH, temperature, humidity, salinity, oxygen, pore-solution exposure context.
- `history`: duration, exposure time, scan/test history, heat treatment, microstructure/procedure fields when they represent sample history.
- `intervention`: processing route, AM process, HIP/heat treatment/process-control labels.
- `electrochem`: measured electrochemical response features when they are not the active target.
- `target`: measured corrosion outcomes or target-like electrochemical responses.
- `metadata`: references, DOI, comments, source identifiers, free text not used for structure estimation.
- `exclude`: local row IDs or unusable fields.

## Dataset Notes

### Electrochemical Metrics Alloys

- Material: element wt.% columns.
- Environment: temperature, test solution, chloride concentration, pH.
- History/intervention: test method, scan rate, heat treatment, microstructure/procedure fields.
- Target: Epit, Erp, Ecrev, Tpit, Tcrev, Ecorr, icorr, corrosion rate, ipass.

### MPEA Corrosion

- Material: phase indicators and element fractions.
- Environment: test environment, concentration, electrolyte.
- Intervention/history: processing.
- Target: corrosion potential, pitting potential, corrosion current density, passive window.
- Metadata: alloy name and reference.

### AM-MPEA Corrosion

- Material: alloy/composition, phases, entropy of mixing, VEC descriptors.
- Environment: electrolyte and concentration.
- Intervention: AM process and post-processing labels.
- Target: corrosion potential, pitting potential, corrosion current density.
- Metadata: DOI/source fields.

### 316L Pitting And Passivity

- Environment: NaCl concentration and scan rate inferred from filenames.
- Target: Epit/Epass descriptors and polarisation curves.
- Metadata: map/trace IDs.
- Material is fixed 316L, so this dataset does not contribute material-composition correlation evidence.

### Steel Mortar Corrosion

- Material: binder proportions, water-to-binder ratio, porosity/material-pore properties.
- Environment: chloride, pH, RH, saturation, water content.
- Target: corrosion potential, resistivity, chloride-to-hydroxide ratio, corrosion rate.

### NACE/NIST CORR-DATA

- Material: material group/family/material/UNS.
- Environment: environment, concentration, temperature.
- History: duration and condition text.
- Target: corrosion rate/rating and localized attack.
- Metadata: references.

### Mooring Steel Seawater

- Environment: salinity, temperature, oxygen condition.
- History: exposure days.
- Target: open-circuit potential.
- Material is fixed R4 steel, so this dataset does not contribute material-composition correlation evidence.
