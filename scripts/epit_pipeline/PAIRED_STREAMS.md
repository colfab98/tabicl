# Shared synthetic streams for EPIT and Soccol

Implemented on 30 October 2026. New searches use EPIT **v11** and Soccol **v5**.
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
- Final training required the selected study's stream policy, data seed,
  training seeds, scheduler horizon and process/worker settings. Training
  duration and explicit prior overrides could differ. Without overrides, the
  final model's initial tables repeated the selected trial's tables.
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
