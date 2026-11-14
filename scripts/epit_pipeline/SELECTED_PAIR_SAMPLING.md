# Selected-pair EPIT sampler

The new sampler is opt-in through `--epit_selected_pair_dir` in
`src/tabicl/train/run.py`. It does not overwrite existing fitted formulas or
change existing samplers when that argument is absent.

It loads the all-760-row fitted recipe and retained pair CSVs. Each synthetic
row draws one retained pair uniformly. The pair fixes both the real parent and
the formula. Composition uses the existing multiplicative log-normal noise and
closure to 100 wt.%; absent elements remain zero. All experimental conditions
and their missingness come from that same parent. There is no second parent.

Use `--pitting_composition_perturb_strength 0.30` to retain the v5 default
perturbation setting. The sampler accepts the configured strength, rather than
silently overriding it.

Targets reuse saved fitting scales, coefficients and per-formula mV conversions.
They are computed from the perturbed composition and unchanged conditions.
Only a single shared standardization of the finished table is applied before
training. There is no coefficient variation, per-rule batch refitting, or SCM
target mixing. Model-input imputation uses context rows only, as before.

Required training options alongside the usual expanded EPIT configuration:

```
--epit_selected_pair_dir corrosion_datasets/analysis/epit_pipeline/selected_pairs_all760_generic10k
--paired_prior_streams true
--original_prior_mix true
--epit_target_version v4
--informed_target_mix_policy fixed
--pitting_coefficient_variation 0
--pitting_composition_perturb_strength 0.30
```

`v4` here only disables the old v5 dispatch. The new sampler evaluates all ten
newly fitted equations, including the heat equation. The supplied old
`informed_prior_ratio` and target mix weight do not control this sampler: the
ratio is calculated from the retained pairs and targets are formula-only.

The current recipe gives 2300 / 7600 = 30.263% informed. Existing batching mixes
whole synthetic tables in groups. The new allocator carries fractional groups
across indexed batches so it approaches that proportion without permanently
rounding it to 25% or 50%. It does not mix unrelated generic and informed rows
inside one context/query table. Generic generation itself is unchanged.
Allocation is calculated over the global batch before splitting it across GPUs;
changing GPU count preserves the global tables and their order.

The selected-pair launcher accepts `--scheduler-total-steps` (default 10000)
independently of `--max-steps`. Shorter runs follow the same learning-rate
schedule up to their stopping step.

Submit final training with `sbatch scripts/epit_pipeline/train_selected_pairs.sbatch`.
This job requests one GPU and a 47-hour wall-time limit.

Parent indices, selected pair indices and formula names are returned by
`SelectedPairSampler.sample` for inspection. File hashes and formula code hashes
are part of the training sampling identity, so changed recipes cannot silently
resume an old run. Run through the repository with `PYTHONPATH=src:.` because
formula evaluation reuses the existing pipeline implementation.

This recipe uses all real samples for fitting and selection. It is exploratory,
not an independent test on those same samples. No training is started by adding
these files.
