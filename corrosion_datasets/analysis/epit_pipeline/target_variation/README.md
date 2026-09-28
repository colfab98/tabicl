# Target Coefficient Variation Diagnostic

This is a generator-only experiment. It does not train models, run Optuna,
refit calibration coefficients, or modify frozen v7 artifacts. The sampler is
also available through an opt-in training setting, disabled by default. The
separate `coefficient_variation_search/` experiment documents that integration;
this diagnostic never launches training.

Run from the repository root using the project environment:

```bash
.venv/bin/python -m scripts.epit_pipeline.diagnose_target_variation
```

Defaults: the saved Trial 56 training settings, 112 base tables (8 per each of
7 formula families and 2 SCM types), 1,024 rows per table, variation levels
0/10/20/30%, and 10 draws per nonzero level. Coverage is balanced for diagnosis;
it is not the production rule-family sampling distribution. The synthetic
context size is fixed at half the table. This does not change target creation.

Each invocation creates a fresh `run_<UTC timestamp>/` here. An explicit output
directory must not already exist. For a quick smoke run:

```bash
.venv/bin/python -m scripts.epit_pipeline.diagnose_target_variation \
  --tables-per-cell 1 --draws 2 --seq-len 128 \
  --output-dir /tmp/target_variation_smoke
```

## Implementation

- Entry point: `scripts/epit_pipeline/diagnose_target_variation.py`.
- Reusable opt-in sampler: `src/tabicl/prior/target_variation.py`.
- Tests: `tests/test_target_variation.py`.

The diagnostic loads the saved training command, verifies calibration artifact
hashes and the empirical feature profile, and calls the existing `SCMPrior`
generation, formula evaluation, standardization, and mixing methods. It records
each base task, then replays its target calculation while changing only the
coefficient vector. It preserves the same features, SCM target, formula family,
method offsets, and lambda. Signed formula terms are inspected through the
existing evaluator using unit coefficient vectors, not copied equations.

The sampler multiplies each coefficient by an independent uniform factor in
`[1-variation, 1+variation]`, normalizes the vector to sum to one, and rejects
draws exceeding the calibration's upper bounds. After 100 unsuccessful attempts
it returns the original coefficients and reports a fallback. Zeros stay zero.
Zero variation consumes no random numbers and leaves coefficients unchanged.
Normalization and rejection may shift the mean; variation is not an exact
variance. Coefficient randomness is separate from task-generation randomness.

## Outputs

- `config.json`: settings, seeds, dependency versions, source and artifact hashes.
- `status.json`: running, complete, or failed; check this before using results.
- `tables.csv`: base-task hashes, SCM/rule correlations, and failed/constant SCM flags.
- `term_correlations.csv`: signed within-formula term correlations.
- `measurements.csv`: paired rule and mixed-target Pearson/Spearman correlations,
  absolute changes, rejection counts, and fallbacks for every coefficient draw.
- `coefficients.csv`: every sampled coefficient and its reference value.
- `summary.csv`: per-family, per-SCM, per-variation quantiles; failed SCM cases
  are kept in separate groups. Correlations of constant signals are undefined.
- `coefficient_summary.csv`: realized coefficient spread and mean drift.
- `report.html` and `target_changes.png`: readable tables and plots.

Target differences are measured after the same standardization used in training.
Draws sharing a base table are not independent observations. The existing SCM
failure behavior is recorded, not fixed or silently filtered by this experiment.
No real EPIT target labels or model checkpoints are read. Existing empirical
feature profiles are reused with their original provenance; this is not a new
held-out predictive evaluation. Predictive benefit requires later training and
validation experiments.
