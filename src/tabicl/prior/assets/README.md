# EPIT composition profiles

`epit_dataset_v1.csv` is a static, versioned template bank containing the 403
exact distinct 24-element compositions from the usable rows of the evaluated
EPIT task. `epit_dataset_v1.json` records its schema, provenance, checksum,
family counts, and dataset-derived default probabilities.

Training and sampling code never opens the source Excel workbook. The static
asset is loaded once per Python process through
`tabicl.prior.epit_composition_profile.load_epit_composition_profile`.

Family probability order is:

1. `fe_alloy`
2. `al_alloy`
3. `hea`
4. `nicrmo_alloy`
5. `other`

The default `pitting_composition_mode=legacy` leaves the existing SCM material
mechanism unchanged. The opt-in `empirical` mode is currently scoped to the
fixed 21-column EPIT schema. With the default two material latents, the SCM
generates six features (two material, three environment, and one process).
The material latents select an alloy family and a real template, receive the
configured small perturbation, and expand into the 17 observed element columns.
Environment and process columns pass through without being expanded.
The masked-Dirichlet controls remain part of legacy mode and are not applied on
top of empirical templates.
