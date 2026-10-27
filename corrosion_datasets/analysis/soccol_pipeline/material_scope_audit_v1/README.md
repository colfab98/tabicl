# Soccol material families and formula coverage

The active Soccol workflow used all **2,950 accepted development observations**,
covering **349 distinct compositions**. It applied no alloy-family filter.
Formula fitting, composition sampling and environment sampling used the same
accepted rows. No frozen final-query predictions were used in this analysis.

| Stored family group | Rows | Distinct compositions | Composition draw share |
|---|---:|---:|---:|
| Stainless steels (`fe_based_stainless`) | 2,745 | 277 | 93.05% |
| Fe-labelled model alloys | 65 | 50 | 2.20% |
| Fe–Ni–Cr Alloy 800 | 64 | 1 | 2.17% |
| Ni–Cr–Mo model alloys | 39 | 14 | 1.32% |
| High-alloy group with broad historical labels | 35 | 7 | 1.19% |
| Nickel Alloy 600 | 2 | 1 | 0.07% |

Composition counts within groups summed to 350 because one composition occurred
under two stored family labels; there were 349 distinct compositions overall.
The sampler drew observations uniformly, so repeated measurements increased a
composition's sampling weight. Composition and environment rows were drawn
independently. The family shares described the source of composition draws,
before random composition perturbation, rather than a classification of every
perturbed alloy.

## Family labels and actual composition

- The old `material_scope_primary` flags contained 2,810 `include`, 99 `review`
  and 41 `exclude` rows within accepted development. These flags were historical
  taxonomy notes. They did not govern the current accepted-row selection.
- The broad stored labels were not a strict largest-element classification.
  Fe was the largest element in 2,901 accepted rows, Ni in 47, and Cr in two.
  Eight rows under `fe_based...` labels had another largest element: six Ni
  and two Cr. Their IDs and compositions were saved in
  `family_label_caveats.csv`. We retained the original labels and separately
  reported this diagnostic.
- The two Cr-largest rows were Fe–Cr model alloys with 60 or 70 wt.% Cr.
  They did not represent an additional Al/HEA-type material population.
- The 272 withheld development rows were excluded for unresolved composition,
  not by a family filter. Counts by frozen partition and acceptance status
  were saved in `partition_family_counts.csv`.

## Formula scores within each family

We re-scored existing held-out development predictions within each family.
The nine formulas retained their globally fitted context coefficients; no
family-specific fitting was performed. Values below were the unweighted mean
of valid within-family fold Spearman correlations. Folds with fewer than three
rows or constant targets/predictions were left unscored.

| Formula | Stainless: 2,745 | Fe-labelled model: 65 | High-alloy: 35 | Alloy 800: 64 | Ni–Cr–Mo: 39 |
|---|---:|---:|---:|---:|---:|
| `pren_n_linear` | 0.3868 | 0.2525 | 0.7224 | 0.7746 | 0.2577 |
| `cr_mow_n_synergy` | 0.4520 | 0.2839 | 0.7224 | 0.8284 | 0.2639 |
| `threshold_saturation` | 0.4193 | 0.0258 | 0.7103 | 0.8284 | 0.2734 |
| `pren_n_improved_environment` | 0.4136 | 0.2021 | 0.7152 | 0.8275 | 0.2639 |
| `mo_n_acid_repassivation` | 0.4141 | 0.1882 | 0.6909 | 0.8275 | 0.2639 |
| `mns_inclusion_penalty` | 0.4557 | 0.2229 | 0.7127 | 0.8275 | 0.2639 |
| `coupled_breakdown` | 0.4513 | 0.3731 | 0.6715 | 0.8107 | 0.0330 |
| `pren_n_coupled_mns` | 0.5601 | 0.2849 | 0.5964 | 0.8107 | 0.0330 |
| `pren_n_coupled_mns_weak_anions` | 0.5592 | 0.2958 | 0.6739 | 0.8275 | 0.0330 |

- Stainless, Fe-labelled model, high-alloy and Ni–Cr–Mo scores used five valid
  folds each. The 39 Ni–Cr–Mo rows came from one source (`2009Wong`), with
  only 8, 14, 8, 6 and 3 validation observations per fold. These estimates were
  sensitive to a few rows; they did not establish performance for nickel
  alloys generally.
- All 64 Alloy 800 rows belonged to one composition and fold 5. Their score
  described ranking under different environments for that composition, rather
  than generalization across Alloy 800 compositions. Nickel Alloy 600 had only
  two rows in one fold, so no meaningful subgroup correlation was reported.
- The strongest formula overall, `pren_n_coupled_mns_weak_anions`, was useful
  on the main stainless group but weak on the 39 Ni–Cr–Mo rows. Its extra Mn/S
  and sulfate/nitrate terms were zero for every one of those nickel rows.
  The same rows had constant pH and chloride, with temperature and composition
  varying. Additional terms therefore did not supply an extra varying signal
  within that nickel subgroup. This observation did not isolate a causal
  explanation for trained-model performance.

## Differences from EPIT

- **Scope:** EPIT's active informed path used 411 Fe and 41 Ni–Cr–Mo
  development rows, excluding its other material classes. Soccol used all
  2,950 composition-accepted development rows, without a separate class gate.
- **Composition weights:** EPIT sampled distinct composition templates,
  yielding 241/255 Fe draws (**94.51%**) and 14/255 Ni–Cr–Mo draws (**5.49%**)
  under its current launcher. Soccol sampled observations, yielding 41/2,950
  nickel-labelled draws (**1.39%**). EPIT's 41/452 nickel environment-row share
  was a different quantity; it was not its current composition draw share.
- **Formula selection:** Soccol applied the same nine rules and global
  coefficient distributions to every material; no Fe/Ni-specific rule was
  promoted. EPIT included `fe_ni_cr_threshold`, which used a Cr threshold of
  12 wt.% for Fe and 15 wt.% for Ni–Cr–Mo. Most other EPIT formulas were also
  shared across its two active material classes.
- **Nickel scores were weaker in EPIT too:** its nine Ni–Cr–Mo subgroup means
  ranged from 0.2917 to 0.4644. The highest came from `fe_ni_cr_threshold`.
  The strongest EPIT formula overall, `method_aware_pren_n`, scored 0.5782
  on Fe and 0.3594 on Ni–Cr–Mo. The full EPIT and Soccol subgroup results were
  saved together in `formula_family_scores.csv`. Their rows, contexts and
  rule sets differed, so these were descriptive comparisons rather than a
  matched experiment attributing a performance difference to family routing.

## Assessment and reproduction

We recommended retaining the current accepted-material scope for the next
Soccol run. The analysis did not justify discarding nickel or automatically
applying the old `exclude` flags. It identified limited nickel support,
observation-versus-template sampling, and the absence of a Fe/Ni-specific rule
as distinct issues to consider when reviewing sampling and formulas. We made
no changes to family labels, dataset values, sampling weights, rules or splits.

We checked the saved bank's row membership, frozen fold membership and saved
prediction targets, and reproduced the overall mean scores for all nine
Soccol and nine EPIT formulas before calculating subgroup metrics.

```bash
.venv/bin/python -m scripts.soccol_pipeline.analyze_material_scope --output-dir /tmp/soccol_material_scope
```

The command refused to overwrite an existing nonempty output directory.
It produced family/partition counts, row-level label caveats, formula scores
and per-fold scores. `summary.json` recorded source/output hashes and the
sampling comparison. No model training, coefficient refitting or final-query
scoring was run.
