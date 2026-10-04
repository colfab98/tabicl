# Soccol target-rule comparison v1

All results use the frozen five composition-group development folds. Final-test
targets were masked before rule fitting and evaluation.

The old set is the nine-rule set enabled by the latest project commit:
`pren_n_linear`, `cr_mow_n_synergy`, `threshold_saturation`,
`pren_n_improved_environment`, `mo_n_acid_repassivation`,
`mns_inclusion_penalty`, `coupled_breakdown`, `fe_ni_cr_threshold`, and
`method_aware_pren_n`. The first eight are evaluated below.
`method_aware_pren_n` is marked unevaluable because Soccol has no comparable
test-method field. Historical formulas outside that set are references only.

| Rule | Type | Mean fold Spearman | Pooled OOF Spearman | Decision |
|---|---:|---:|---:|---|
| `new_pren_n_coupled_mns_weak_anions` | new_candidate | 0.5866 | 0.5712 | keep |
| `new_pren_n_coupled_mns_anions` | new_candidate | 0.5861 | 0.5707 | keep |
| `new_pren_n_coupled_mns_effective_halide` | new_candidate | 0.5755 | 0.5665 | keep |
| `new_pren_n_coupled_mns_strong_anions` | new_candidate | 0.5741 | 0.5629 | keep |
| `new_coupled_mns_anion_competition` | new_candidate | 0.5713 | 0.5614 | keep |
| `new_pren_n_coupled_mns` | new_candidate | 0.5542 | 0.5340 | keep |
| `new_coupled_mns` | new_candidate | 0.5525 | 0.5287 | keep |
| `new_pren_n_coupled_breakdown` | new_candidate | 0.5276 | 0.5087 | keep |
| `new_pren_n_coupled_mns_scan` | new_candidate | 0.5240 | 0.5188 | do_not_keep |
| `new_pren_n_coupled_mns_area` | new_candidate | 0.5171 | 0.5076 | do_not_keep |
| `new_mns_anion_competition` | new_candidate | 0.5105 | 0.4986 | keep |
| `new_pren_n_coupled_mns_surface` | new_candidate | 0.5097 | 0.5058 | do_not_keep |
| `new_mns_effective_halide` | new_candidate | 0.5068 | 0.4958 | do_not_keep |
| `new_full_without_scan` | new_candidate | 0.5028 | 0.4896 | do_not_keep |
| `old_coupled_breakdown` | existing_promoted_rule | 0.5028 | 0.4923 | reference |
| `old_mns_inclusion_penalty` | existing_promoted_rule | 0.4980 | 0.4972 | reference |
| `new_mns_full_with_scan` | new_candidate | 0.4979 | 0.4909 | do_not_keep |
| `new_mns_specimen_area` | new_candidate | 0.4975 | 0.4971 | do_not_keep |
| `old_cr_mo_n_synergy` | existing_promoted_rule | 0.4947 | 0.4885 | reference |
| `new_mns_scan_rate_positive` | new_candidate | 0.4936 | 0.4967 | do_not_keep |
| `new_mns_halide_activation` | new_candidate | 0.4903 | 0.4773 | do_not_keep |
| `new_mns_surface_and_area` | new_candidate | 0.4862 | 0.4889 | do_not_keep |
| `new_mns_acid_halide_activation` | new_candidate | 0.4851 | 0.4744 | do_not_keep |
| `new_anion_competition` | new_candidate | 0.4740 | 0.4682 | keep |
| `new_effective_halide` | new_candidate | 0.4678 | 0.4628 | do_not_keep |
| `new_scan_rate_positive` | new_candidate | 0.4666 | 0.4757 | tentative |
| `old_pren_n_linear` | existing_promoted_rule | 0.4633 | 0.4551 | reference |
| `new_specimen_area` | new_candidate | 0.4600 | 0.4629 | do_not_keep |
| `old_mo_n_acid_repassivation` | existing_promoted_rule | 0.4599 | 0.4658 | reference |
| `old_pren_n_improved_environment` | existing_promoted_rule | 0.4597 | 0.4637 | reference |
| `old_threshold_saturation` | existing_promoted_rule | 0.4596 | 0.4455 | reference |
| `new_scan_rate_negative` | new_candidate | 0.4594 | 0.4636 | do_not_keep |
| `new_surface_and_area` | new_candidate | 0.4521 | 0.4578 | do_not_keep |
| `old_improved_environment` | historical_reference | 0.4484 | 0.4524 | reference |
| `old_current_pren` | historical_reference | 0.4348 | 0.4306 | reference |
| `old_fe_ni_cr_threshold` | existing_promoted_rule | 0.4177 | 0.4082 | reference |

The automatic decision is deliberately conservative: a new rule is kept only
when its mean fold Spearman gain over its stated comparison baseline is at
least 0.01 and it improves at least three of five folds. `tentative` requires
a gain of at least 0.005 in at least three folds. Mn-S extensions are compared
with the existing Mn-S rule; isolated new terms use the existing N-aware rule.

## Recommendation

Keep `new_pren_n_coupled_mns`: the old coupled-breakdown structure with
`Cr + 3.3*Mo + 16*N` and `-sqrt(Mn*S)`. Its mean fold Spearman is 0.5542, versus 0.5028 for the best old rule,
and it improves four of five folds. Area, surface-finish and scan-rate terms
should not be added.

Also keep `new_pren_n_coupled_mns_weak_anions` for the later synthetic-training
comparison. It replaces chloride by `Cl + 0.5*Br` and adds the positive term
`log10(1 + (SO4 + NO3)/(Cl + 0.5*Br + 1e-5))`. It has the best direct
score (0.5866) and improves
three of five folds over the core rule. If only one rule is wanted, use the
core rule because its gain over the old set is more fold-stable.

These are development-screening candidates. The production rule registry and
synthetic generator have not been changed.

Files:

- `comparison_summary.json`: formulas, fold results, fitted coefficients and decisions.
- `fold_metrics.csv`: compact comparison table.
- `oof_predictions.csv`: row-level development predictions for audit.
- `literature_basis.md`: evidence and limits for each added term.
