# Literature basis for Soccol target-rule candidates

## Terms retained in the recommended rule

The recommended rule combines three structures already present in the latest
project commit:

1. **Nitrogen-aware PREN:** `Cr + 3.3*Mo + 16*N` because Soccol records N and
   does not record W. This replaces the Cr-Mo score inside `coupled_breakdown`.
2. **Coupled environmental breakdown:** chloride and temperature act more
   strongly when material protection is lower. This is the existing
   `coupled_breakdown` structure, not a newly inferred causal equation.
3. **Mn-S susceptibility:** `-sqrt(Mn*S)`. Bulk Mn and S are only a proxy for
   MnS inclusions, but direct microscopy and electrochemical work establishes
   MnS inclusions as common pit-initiation sites in stainless steel:
   [Schmuki et al., 2005](https://doi.org/10.1016/j.corsci.2004.05.023) and
   [Williams et al., 1993](https://doi.org/10.1016/0010-938X(93)90279-P).

The new combination is scientifically conservative: it does not invent a new
effect. It combines the best old environmental structure with two composition
effects that were separate old rules.

## Extra Soccol variables tested

| Variables | Tested effect | Literature basis | Result |
|---|---|---|---|
| `Test_area_cm2` | Larger area lowers measured EPIT: `-log10(area)` | Mean measured pitting potential falls as exposed specimen area increases: [Ernst and Newman, 1996](https://doi.org/10.1016/S0010-938X(97)83146-0). | Rejected; reduced the leading core rule's mean fold Spearman by 0.0370. |
| `Prep_grinding_grit`, `Prep_Ra_micron` | Smoother surface raises EPIT | Rough surfaces increase pit initiation and stable-pit probability: [Tang et al., 2019](https://doi.org/10.3390/ma12050738). | Rejected; area/finish terms reduced mean fold Spearman by 0.0445. |
| `CP_Br` | Effective halide `Cl + 0.5*Br` | Chloride was more aggressive than bromide for a low-Mo austenitic steel, while the relative effect changes with Mo: [Laycock et al., 2000](https://doi.org/10.1016/S0010-938X(99)00056-6). The `0.5` factor is therefore a test heuristic, not a universal constant. | Retained in the compact anion extension; effective halide alone improved mean fold Spearman by 0.0213. |
| `CP_SO4`, `CP_NO3` | Positive weak-inhibitor ratio `log10(1 + (SO4+NO3)/(Cl+0.5*Br+1e-5))` | SO4 and NO3 retarded pitting in chloride solution: [Zuo et al., 2002](https://doi.org/10.1016/S0010-938X(01)00031-2). | Retain as a second candidate. Combined with effective halide, it improved mean fold Spearman by 0.0324 and pooled OOF Spearman by 0.0372 over the core rule, with gains in 3/5 folds. |
| `CP_PO4`, `CP_MoO4`, `CP_CrO4` | Positive strong-inhibitor ratio | PO4 and CrO4 inhibition is reported by [Zuo et al., 2002](https://doi.org/10.1016/S0010-938X(01)00031-2); chromate and molybdate inhibition is also reported by [Ilevbare and Burstein, 2003](https://doi.org/10.1016/S0010-938X(02)00229-9). | Omitted from the retained extension. Its fitted weight in the full anion rule was only 0.0098, and removing it slightly improved the score. |
| `scan_rate` | Both positive and negative log-rate effects were tested | Reported EPIT can depend strongly on potentiodynamic scan rate: [Shibata and Takeyama, 1981](https://doi.org/10.1016/0010-938X(81)90018-4). The direction is not universal across protocols. | Rejected; the positive-rate term reduced the leading core rule's mean fold Spearman by 0.0302. |

## Variables left out of explicit rules

- `Prep_medium`, `Prep_pH`, `Prep_redox`, and `Prep_time`: surface treatment
  effects depend on the treatment. Nitric-acid passivation, for example, shows
  a nonmonotonic optimum rather than a universal linear direction
  ([Burstein and Pistorius, 2000](https://doi.org/10.1016/S0010-938X(00)00052-4)).
- `CP_aeration` and `CP_agitation`: categorical protocol fields without a
  defensible collection-wide signed effect.
- `CP_time`: measurement duration is procedure-dependent.
- `CP_OH`: redundant with measured `CP_pH` in an explicit target rule.
- `CP_anions_info`: duplicates information represented by the numeric ion
  columns and contains heterogeneous text categories.

These columns can still enter the generic synthetic SCM and the real
predictive model. They are excluded only from the explicit analytical target
formula.

## Development-data coverage relevant to interpretation

The frozen development set has 3,222 rows. Nonzero counts are: N 1,228; Mn
1,857; S 1,729; Br 104; sulfate/nitrate in 327 rows from 25 sources; and
phosphate/molybdate/chromate in 108 rows from 9 sources. Area is reported in
1,550 rows, grinding grit in 2,398, roughness in 289, and scan rate in 3,002.

All comparisons use context-fold preprocessing and the frozen composition-group
folds. Final-test targets remain masked.
