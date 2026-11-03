# Shared synthetic streams for EPIT and Soccol

This document describes historical EPIT v12 / Soccol v6. Current runs use
[EPIT v14 / Soccol v8](README.md#current-experiment-v14), which restore original
batch-level sampling and the grouped encoder. Do not use the policy below to
resume old checkpoints into the new experiment.

Implemented on 30 October 2026; retry handling corrected on 31 October 2026.
New searches use EPIT **v12** and Soccol **v6**.
The reviewed data, frozen splits, formulas, coefficient variation and search
ranges were retained. Historical runs and checkpoints were not converted.

## What was shared

Each run consumed two ordered streams: generic tables G1, G2, … and informed
tables I1, I2, …. A table's random seed depended on its source and index, not
the mixture ratio, Optuna trial number, worker assignment or stopping step.

At the standard batch size of 512:

| Informed share | Generic tables per step | Informed tables per step |
|---|---:|---:|
| 0% | 512 | 0 |
| 25% | 384 | 128 |
| 50% | 256 | 256 |
| 75% | 128 | 384 |
| 100% | 0 | 512 |

For example, the 50% run consumed G1–G256 in its first batch. The 75% run
consumed G1–G128, then G129–G256 in its second batch. The corresponding generic
tables had identical features, targets and context/query split positions.
The same relationship held for informed tables when their generator settings
matched. Tables were not stored on disk: their indexed seeds regenerated them.

Groups of four retained shared SCM hyperparameters, sequence length and split
position for compatible micro-batches. Whole groups were shuffled independently
of table generation. Each source's group count was exact, not a Bernoulli draw.
An incompatible ratio/batch/group combination was rejected instead of rounded.
Groups used their own split positions, replacing the historical one-position-
per-batch sampling so a table kept its identity when reached in another batch.

## Generic recipe

The generic stream used the ordinary generic SCM recipe: 70% MLP / 30% tree,
2–100 initial features, category-generation probability 0.2 and random feature
permutation. It did not inherit the specialist's fixed width, disabled category
generation, disabled permutation or tuned MLP probability. Mixed batches were
padded only for processing; each table retained its own active feature count.

Both datasets used this same generic stream. A matching pure-generic control
could use an explicit `--override-informed-prior-ratio 0.0` in the same final
pipeline, with the same selected trial, seeds and other settings. The low-level
trainer also supported `--paired_prior_streams true --prior_type mix_scm`.
Previously trained generic checkpoints did not retroactively acquire these
indexed tables. Existing evaluation conventions were not changed by this work.

The agreed final-run policy (1 November 2026) keeps native informed NaNs and the
learned missing-value embedding, with feature grouping disabled for the whole
model. Generic samples receive no added NaNs, and no artificial feature masking
is applied to either source. A 0% informed run therefore shares the architecture
but does not train the missing token on missing examples. Different informed
datasets retain their own native feature sets when trained as separate tasks;
this does not introduce a shared column-name identity into the model. The old
combined 50-column sampler is a separate historical workflow.

Use the existing `train_final.sbatch` with `sbatch --gpus=1` or `--gpus=2` and
`--override-informed-prior-ratio 0.25`, `0.50`, `0.75`, `1.0`, or optional `0.0`
after the script name. Its process count follows the allocated node GPUs unless
explicitly overridden. Each final manifest records `input_policy`; the v12
generator and stream fingerprint are unchanged.

## Optuna and final training

- Training NumPy, Torch and Python seeds defaulted to 42; the separate
  `--prior-stream-seed` also defaulted to 42. The launchers exposed
  `NP_SEED`, `TORCH_SEED`, `PYTHON_SEED` and `PRIOR_STREAM_SEED`.
- Optuna's `RANDOM_SEED` controlled parameter suggestions, not training data.
- All trials shared physical draws wherever their settings allowed it. The
  random SCM target computation used a separate random stream, so choosing an
  MLP versus tree did not shift composition/environment draws or formula
  coefficient draws. EPIT environment sampling was separated from composition
  perturbation, including the zero-perturbation endpoint.
- Different formula weights still changed targets. Different perturbation
  strengths still changed composition values. Common random draws did not
  erase the parameter differences being optimized.
- Final training requires the selected study's stream policy, data seed,
  training seeds, scheduler horizon and loader-worker settings. The process
  count may change when the global batch and source groups divide exactly.
  Training duration and explicit prior overrides may also differ. Without
  prior overrides, the final model's initial tables repeat the selected trial's
  tables, including when partitioned across GPUs.
- Plan the scheduler horizon before Optuna. If final runs are intended to reach
  15,000 steps, use `SCHEDULER_TOTAL_STEPS=15000` for the 1,000-step search and
  for final training; a newly stretched schedule is a different optimization
  experiment.
- Study fingerprints recorded the stream policy and generator implementation
  hashes. New workers could not silently join an incompatible older study.
- Shared-journal startup was locked across study creation and both fingerprint
  writes. Concurrent node launches could not observe a partly written identity.
  The Optuna launcher exposed the same loader-worker, prefetch and evaluation
  ensemble settings as final training. The README documented two 25-trial jobs
  and four separate ratio-specific final submissions.

## Checkpoints and limits

New checkpoints stored the paired-stream configuration, profile/category/source
identities and the consumed worker boundary. Prefetched but unconsumed batches
were regenerated when resuming. Checkpoint writes became atomic. The low-level
trainer rejected legacy checkpoints or changed stream settings for a paired
training resume; loading weights alone remained a separate operation.

The high-level `train_final.sbatch` still did **not** gain a training-resume
flag in this change. Its `--resume-checkpoint-selection` mode remained evaluation
recovery only. A separate deferred implementation of launcher resume/writer
locking was not applied. Do not describe those flags as available.

Small CPU tests compared actual EPIT and Soccol features, targets and NaN masks,
and uninterrupted versus resumed training with spawned workers. Shared tables
reduce sampling noise; they do not guarantee identical GPU arithmetic across
hardware, or identical learned weights after changing the mixture.

## Retry correction, 31 October 2026

The original paired sampler reset the SCM seed on every rejection attempt.
An invalid MLP could therefore be reconstructed indefinitely. EPIT v11 trials
4 and 7 stalled after step 459; source group 58799 included the failing table
with seed `11044767929213912588`. Its raw MLP output was non-finite and its
processed target was constant. A CPU reproduction isolated this from GPU speed.

The corrected sampler retained the original first-attempt seed. Each subsequent
SCM attempt used a seed derived from the same table seed and the attempt index.
Retries stayed reproducible and independent of other tables, mixture ratios and
worker scheduling. Every paired table, generic or informed, was limited to 100
attempts; exhaustion raised an error identifying the table seed and dimensions.
No invalid table was returned and no formula-only fallback was introduced.
Legacy non-paired generation retained its existing behavior.

The regression test used the actual calibrated EPIT configuration and failing
group. The failing table succeeded on its second attempt and matched exactly
across 50% and 100% informed streams. Separate tests forced rejection and checked
that EPIT, Soccol and generic tables stopped at the configured attempt limit.

The stream version changed to `paired_generic_informed_v2`. Optuna and final
launchers moved together to EPIT v12 and Soccol v6. Existing v11/v5 journals,
results and checkpoints were retained, and their identity checks were not
relaxed. Running processes retain the old implementation: stop the affected old
workers, synchronize the corrected code on every node, and launch the new study.
Do not append corrected runs to the old study or resume its checkpoints as if
the generator were unchanged. Final training must use the corrected study too.

## Multi-GPU correction, 1 November 2026

The trainer previously added the GPU rank to the formula-coefficient seed,
even for paired streams. The second GPU therefore generated different targets
for the same table IDs. Paired training now uses the unchanged coefficient seed
on every rank. Legacy non-paired training retains its historical rank offset.
The generator files and stream policy were not changed: existing single-process
EPIT v12/Soccol v6 studies, tables and checkpoint identities remain compatible.
Corrected multi-process studies record `paired_coefficient_seed_policy` in their
fingerprint; final training rejects older multi-process studies without it.

Final training may repartition the global batch of 512 into 256 tables on each
of two GPUs, retaining micro-batches of four and the same learning-rate schedule.
Both the launcher and trainer reject partitions requiring rounding or partial
source groups. Process-count overrides are recorded and receive separate default
run destinations. Paired training raises on OOM instead of skipping tables.
Checkpoint resume still requires the saved process count and loader settings;
the high-level launcher still has no training-resume flag.

Regression tests compare actual EPIT/Soccol tables by source/index across all
five mixture ratios through the trainer. Two-process CPU/Gloo tests compare
optimizer updates with single-process training and resume both ranks from one
checkpoint. These establish data equality and numerical agreement, not identical
GPU arithmetic or a measured twofold speedup.
