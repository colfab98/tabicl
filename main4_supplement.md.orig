# Supplement to main4.tex: target mixing and coefficient variation

Recorded: 2026-09-26. Working notes for a later merge into [main4.tex](main4.tex).
Sources: the Codex conversation **“Review main4 and presentation.md”**
(2026-09-23–25, thread `01a0cde2-bea1-7b32-81c2-2ac6045ae94f`), the subsequent
launch/concurrency discussion, and the repository artifacts linked below.

This document separates measured findings, implemented functionality, and proposed
experiments. It does not replace the frozen v7 experiment or establish that the
new coefficient sampler improves prediction. Some explanations already appear
in main4; reconcile them during the merge instead of adding duplicate sections.

## 1. Status at this snapshot

| Item | Status |
| --- | --- |
| Review of target standardization and the meaning of λ | Completed; retain the existing mixing design. |
| Coefficient-variation diagnostic | Implemented and run, including 50% and 80% variation. |
| Optional coefficient sampling during training | Implemented; default strength is zero. |
| Separate 2,000-step Optuna experiment | Implemented, tested, and dry-run plan generated. |
| Predictive results from the new GPU study | None recorded in the reviewed chat or local experiment outputs. |
| Several nodes running the same new study concurrently | **Not supported by the current runner.** Shared journal storage exists, but the runner permits only one active process. |
| Failed/constant SCM component rejection | Discussed, but not implemented by this experiment. |
| Adaptive λ, grouped bootstrap, joint λ/variation tuning | Discussed only. |

The earlier chat reports **64 passing tests** and tiny CPU training checks with
zero and four DataLoader workers. These are historical verification results,
not tests rerun while writing this document. The subsequent launch check also
successfully generated a 12-run dry-run plan without starting training.

## 2. What target mixing means

For each informed synthetic table, the existing generator uses

\[
y=Z\left[(1-\lambda)Z(y_{\mathrm{SCM}})
              +\lambda Z(y_{\mathrm{rule}})\right],
\qquad Z(v)=\frac{v-\bar v}{\operatorname{std}_{\mathrm{population}}(v)}.
\]

The SCM receives standardized input features. The analytical rule uses physical
features, with several rule terms standardized internally. Each target component
is separately standardized across all rows of that synthetic table, including
context and query, before mixing. Constant components become zero. The final
standardization applies the same scale factor to both weighted components.

For Trial 56, λ = 0.2048264040. A clear description is:

> The two target components are standardized separately and combined using
> weights of approximately 80% for SCM and 20% for the corrosion rule.

This is a weight in the addition, not a claim that the rule explains 20% of
target variance or influences every row equally. For example, standardized
values of SCM = 0.1 and rule = 2.0 produce contributions of 0.08 and 0.40 at
λ = 0.2; the rule dominates that row despite its lower weight.

For uncorrelated, unit-variance components, the rule's share of the pre-final-
normalization variance would be

\[
\frac{\lambda^2}{(1-\lambda)^2+\lambda^2}\approx 6.2\%.
\]

Correlation introduces a covariance term, so that expression is conditional,
not an empirical decomposition of Trial 56. It also does not measure how much
physical information the transformer learns. λ is applied **once**: the earlier
rule calculation supplies an already-weighted term to the final mixer.

Decision from the discussion: retain separate standardization and fixed λ as the
baseline. A learned row-dependent gate has no established training signal or
demonstrated benefit here. Coefficient variation is the simpler first experiment.

### Failed SCM draws remain an unresolved edge case

The chat's initial diagnostic generated 64 tables of 1,024 rows, balanced between
MLP and tree SCMs. One SCM draw returned the constant `-100` failure sentinel.
After standardization, its contribution vanished, so the accepted mixed target
became entirely rule-driven for positive λ. The chat also recorded an extreme
tree case in which the rule contribution was larger on 91% of rows despite the
small λ; equal overall standard deviations do not guarantee similar tails.

These are observations from that small diagnostic, not estimates of historical
training frequency. The later saved 112-table diagnostic independently records
one failure-sentinel table, separated from normal summary statistics. The
coefficient experiment preserves existing SCM failure handling to avoid adding
another change to the comparison. Rejection/resampling of failed or degenerate
SCM components remains a separate proposed fix.

Implementation references: [dataset.py](src/tabicl/prior/dataset.py) and
[MLP SCM](src/tabicl/prior/mlp_scm.py). Whole-table synthetic standardization
also differs from real inference, where target scaling is fitted on context
labels; that distinction remains relevant when describing the method.

## 3. Earlier evidence that motivated the experiment

### Training duration and prior mismatch

The saved five-fold **development** results are:

| Training step | Trial 56 Spearman | Generic baseline Spearman |
| ---: | ---: | ---: |
| 1,000 | 0.6778 | 0.5745 |
| 2,500 | 0.6966 | 0.6129 |
| 7,000 | 0.6645 | 0.6723 |
| 10,000 | 0.6487 | 0.6676 |

Source: [checkpoint comparison summary](corrosion_datasets/analysis/epit_pipeline/trial56_checkpoint_folds_66378/summary.csv).
Trial 56's 1,000-step value above belongs to final retraining; its original
Optuna proxy score was 0.6903.

Interpretation: a 1,000-step search can favor priors that learn useful patterns
quickly. The subsequent decline could reflect mismatch between the synthetic
prior and real corrosion, but the curves do not prove that mechanism or prove
that more generic tasks would improve later training. Lowering informed-task
probability ρ changes feature and target coverage; lowering λ changes the target
mixture inside informed tasks. They are different interventions.

Proposed, not implemented here: compare ρ = 0.75, 0.50, 0.25 through the same
10,000-step schedule across seeds, and measure fresh synthetic validation losses
separately for informed and generic tasks. A scheduling experiment should also
include a constant-mixture baseline with matching total generic exposure.

### Original Optuna study's λ trend

The 2026-09-23 analysis covered 72 completed trials in
`epit_pipeline_optuna_empirical_features_scm_target_v7`. Scores are mean
five-fold development Spearman after 1,000 updates.

| Trial | λ | ρ | Development Spearman |
| ---: | ---: | ---: | ---: |
| 56 | 0.2048 | 0.75 | 0.6903 |
| 53 | 0.2290 | 0.75 | 0.6886 |
| 63 | 0.3116 | 0.75 | 0.6884 |
| 54 | 0.2018 | 0.75 | 0.6868 |
| 19 | 0.2724 | 0.50 | 0.6863 |

All ten best trials used λ between 0.1296 and 0.3508. Eight trials below 0.10
had a best score of 0.6607; eight above 0.60 had a best score of 0.5808.
The search increasingly concentrated around 0.2–0.3. This supports a useful
region at that budget, not an exact optimum or a causal effect of λ alone:
other parameters varied simultaneously.

Provenance: the analysis used a snapshot of
`/home/fcolanto/RZ-Dienste/hpc-user/fcolanto/optuna/epit_pipeline_optuna_empirical_features_scm_target_v7/study_journal_v1.log`.
Its exported table and analysis are currently in `/tmp/tabicl_optuna_lambda/`
(`trials.csv`, `analyze.py`, `study_journal_snapshot.log`). The table was checked
when writing this supplement. These temporary files should be archived or
regenerated before publication; the counts describe that snapshot.

## 4. Calibration and evaluation clarifications

Formula coefficients are fitted before synthetic training. Calibration uses
constrained regression with SciPy SLSQP, minimizing squared error against
standardized EPIT ranks plus a penalty for departing from predefined anchors.
Coefficients are nonnegative, sum to one, and respect configured upper bounds;
the formula already determines each term's sign. There is no coefficient-fitting
MLP. See [calibration implementation](scripts/epit_pipeline/calibrate_target_rules.py).

For direct formula evaluation, fit preprocessing and coefficients on eligible
rows in four development folds, then apply them to the unused fifth fold.
For synthetic generation, refit once on all 452 eligible Fe/Ni–Cr–Mo development
rows. Those coefficients generate training targets; they are not refitted or
explicitly evaluated by the transformer during final inference. The latter uses
608 real development rows as context to predict the 152 final-test rows.

Consequently, direct formula evaluation is out of fold, but the transformer's
development scores are not fully held out from formula calibration: those
validation labels contributed to the all-development coefficients. The new
variation experiment inherits that limitation. Final-test labels are not used
by its training-selection procedure. Existing broader data-use qualifications
remain in force; see [data-use guidance](corrosion_datasets/analysis/leakage_policy.md).

The discussion also proposed investigating the different development/final-test
gaps through frozen-model context-size checks and additional composition-group
partitions. Such diagnostics need no retraining but remain exploratory if query
rows previously influenced calibration or selection. Evaluating the full method
on new outer splits would require rebuilding CorrPFN's profiles, calibration,
training, and inner selection within each outer training partition. Existing
generic checkpoints can be reused if checkpoint selection stays inside that
partition. Additional training seeds address a separate source of uncertainty.
These follow-up evaluations were proposed, not run as part of this work.

## 5. Implemented coefficient sampler

The selected design uses one variation strength `v` for all formula families.
For the selected family's fitted coefficients `w`, independently draw positive
multipliers and normalize:

\[
u_j\sim\operatorname{Uniform}(1-v,1+v),\qquad
w'_j=\frac{w_j u_j}{\sum_k w_k u_k},\qquad 0\leq v<1.
\]

One coefficient vector is shared by all context and query rows of the synthetic
table. It changes the relationship being learned rather than adding independent
coefficient noise to each row. The sampler:

- Preserves nonnegative coefficients and their sum-to-one constraint.
- Keeps originally zero coefficients at zero.
- Rejects draws exceeding calibrated upper bounds.
- Returns the original valid vector after 100 unsuccessful attempts and records
  the fallback.
- Returns the original coefficients exactly at zero strength, consuming no
  additional random numbers.
- Uses a separate coefficient random stream; training gives DataLoader workers
  distinct reproducible streams.

`v` is a multiplier range, **not an exact variance**. Renormalization and rejection
can shift the mean and alter the realized spread. Multiplying every coefficient
by the same factor would be ineffective because target standardization removes
common scale; relative weights must change.

Grouped bootstrap refitting was considered as a way to estimate sensitivity to
the calibration data, but was not implemented. The chosen sampler instead asks
how much variation is useful for training. Neither its distribution nor an
Optuna-selected strength should be presented as estimated physical uncertainty.

## 6. Completed generator diagnostics

The diagnostic reuses existing feature generation, formula evaluation,
standardization, and target mixing. Each comparison keeps the table, SCM output,
formula family, method offsets, and λ fixed while changing coefficients.

Both runs use 112 base tables: eight tables for each of seven rule families and
two SCM types, with 1,024 rows per table and ten draws at every nonzero strength.
Coverage is balanced for diagnosis, not weighted by production rule probabilities.
The first run contains 3,472 comparisons; the extended run contains 5,712.

The following are medians across paired comparisons with `scm_status == ok`
(111 base tables). MAE is the per-comparison mean absolute change in the final
standardized mixed target, followed by a median across comparisons.

| Variation | Rule Spearman vs. original | Mixed-target Spearman vs. original | Median mixed-target MAE |
| ---: | ---: | ---: | ---: |
| 0% | 1.000000 | 1.000000 | 0.00000 |
| 10% | 0.999711 | 0.999952 | 0.00463 |
| 20% | 0.998870 | 0.999833 | 0.00942 |
| 30% | 0.997445 | 0.999630 | 0.01461 |
| 50% | 0.993420 | 0.999064 | 0.02326 |
| 80% | 0.982675 | 0.997671 | 0.03783 |

These values were recalculated from the saved `measurements.csv` files for this
supplement. The extended run reproduces the original levels' aggregate values.
One failure-sentinel table is recorded separately in each run. Across all tables,
approximately 1.06% of attempted draws at 50% and 3.78% at 80% were rejected;
no fallback occurred. Accepted coefficient sets respected the constraints.
Repeated draws on one base table are not independent base tasks.

Interpretation: increasing variation changes rule targets, but mixing with the
dominant SCM contribution reduces the final-target changes. At 80%, coefficient
means can shift as well as spreads. These results justify a controlled predictive
comparison; they do not demonstrate improved prediction. Low target correlation
is not itself the objective.

Saved evidence:

- [Initial report](corrosion_datasets/analysis/epit_pipeline/target_variation/run_20260925T160755_387637Z/report.html).
- [Extended report](corrosion_datasets/analysis/epit_pipeline/target_variation/run_20260925T161921_148867Z/report.html).
- [Extended measurements](corrosion_datasets/analysis/epit_pipeline/target_variation/run_20260925T161921_148867Z/measurements.csv).
- [Realized coefficient spread and drift](corrosion_datasets/analysis/epit_pipeline/target_variation/run_20260925T161921_148867Z/coefficient_summary.csv).

## 7. New Optuna training experiment

| Setting | Implemented value |
| --- | --- |
| Study | `epit_coefficient_variation_2k_v1` |
| Sampler | Optuna `GridSampler` |
| Variation strengths | 0, 0.3, 0.5, 0.8 |
| Training seeds | 42, 43, 44 for every strength |
| Total | 12 fresh training runs |
| Training duration | 2,000 steps, retaining the 10,000-step learning-rate schedule |
| Saved/evaluated checkpoints | 1,000 and 2,000 steps |
| Trial objective | Mean five-development-fold Spearman at step 2,000 |
| Strength comparison | Mean and sample standard deviation across seeds; paired gains over zero variation with the same seed |
| Fixed settings | Trial 56's λ, task mixture, architecture, batch size, and composition perturbation |
| Resources | One GPU, four DataLoader workers, one prior job per worker |
| Final-test evaluation | Excluded |

The seed coordinate schedules repeats; it is not a hyperparameter whose best
value should be selected. There is no pruning or best-checkpoint selection.
The summary reports a winning strength only once every grid cell completes.
The new runner explicitly seeds Python, NumPy, and PyTorch, in addition to the
separate coefficient stream. Thus its zero-strength runs are paired baselines,
not promises to reproduce the historical Trial 56 run bit for bit. GPU
nondeterminism and rare target-validity retries can also affect pairing.

Provenance checks bind the study to its grid, source hashes, calibration/split
artifacts, feature profile, training template, and output paths. Changed settings
or relevant source code require a new study name. Frozen v7 artifacts and the
original launchers remain separate.

### Files added or integrated

| File | Role |
| --- | --- |
| [target_variation.py](src/tabicl/prior/target_variation.py) | Reusable constrained sampler. |
| [diagnose_target_variation.py](scripts/epit_pipeline/diagnose_target_variation.py) | Generator diagnostic and reports. |
| [run_coefficient_variation.py](scripts/epit_pipeline/run_coefficient_variation.py) | Study, commands, evaluation, provenance, and summaries. |
| [run_coefficient_variation.sbatch](scripts/epit_pipeline/run_coefficient_variation.sbatch) | Slurm launcher using shared journal storage. |
| [dataset.py](src/tabicl/prior/dataset.py) | Optional table-level coefficient sampling. |
| [train_config.py](src/tabicl/train/train_config.py), [run.py](src/tabicl/train/run.py) | Optional sampling settings and Python seeding. |
| [test_target_variation.py](tests/test_target_variation.py) | Sampler constraints and diagnostic replay tests. |
| [test_coefficient_variation_search.py](tests/test_coefficient_variation_search.py) | Command, default-off behavior, worker seeds, resume, summaries, and evaluation checks. |

The earlier diagnostic README says sampling is not connected to training. That
describes its initial implementation stage and is now outdated: integration is
available explicitly, while the default remains zero. The current
[experiment README](corrosion_datasets/analysis/epit_pipeline/coefficient_variation_search/README.md)
documents the training stage.

### Launch and output locations

From a host with Slurm access and the project/shared storage mounted:

```bash
cd /home/fcolanto/projects/tabicl
sbatch scripts/epit_pipeline/run_coefficient_variation.sbatch
```

Limit a job to one new trial:

```bash
sbatch scripts/epit_pipeline/run_coefficient_variation.sbatch --n-trials 1
```

Preview commands without training or creating Optuna trials:

```bash
.venv/bin/python -m scripts.epit_pipeline.run_coefficient_variation --dry-run
```

The dry run does write the experiment fingerprint and plan. The recorded
[12-run plan](corrosion_datasets/analysis/epit_pipeline/coefficient_variation_search/epit_coefficient_variation_2k_v1/dry_run_plan.json)
exists, but is not evidence that those runs completed.

Results are stored under
`corrosion_datasets/analysis/epit_pipeline/coefficient_variation_search/<study>/`;
checkpoints under `checkpoints/epit_coefficient_variation/<study>/trial_<number>/`.
The results directory contains provenance, trial logs/configurations/evaluations,
and `summary.csv`/`summary.json` once the study runs.

The Slurm launcher defaults to
`/home/fcolanto/RZ-Dienste/hpc-user/fcolanto/optuna/<study>/study_journal.log`.
Standalone Python defaults to a journal in the study's local results directory.
Use the same storage when resuming. Slurm logs use
`/home/<user>/tmp/epit_coefficient_variation_2k.<jobid>.out` and `.err`.

### Concurrency limitation confirmed after implementation

Unlike the original Optuna workflow, submitting this new launcher on another
node while it is running does **not** currently add another worker:

- `experiment_lock` holds an exclusive lock for the full run.
- Startup also refuses any study containing RUNNING or WAITING trials.
- Successful partial studies can resume through sequential jobs with the same
  configuration and storage. Failed cells are not automatically retried, and
  interrupted RUNNING trials require explicit resolution before restart.

Shared storage alone does not make this runner support concurrent execution.
Before enabling it, implement coordinated grid-cell allocation, short-lived
initialization/summary locks, safe summary writes, and defined recovery for
interrupted or failed workers. Verify that concurrent workers neither duplicate
seed/strength cells nor overwrite outputs. These changes are **pending**; no
parallel-launch claim should be merged into the manuscript or usage guidance.

### Interrupted trial and exact-source recovery (2026-09-28)

On 2026-09-28, trial 2 crossed a partial worktree update that introduced the
24-element v8 composition schema while the coefficient study still required v7.
The source mismatch raised a missing-argument error before trial 2 reached its
first training batch; it did not produce a checkpoint or evaluation result.

The preserved study state was:

- trial 0: complete, variation 0.8, seed 43, Spearman 0.6867086703509501;
- trial 1: complete, variation 0.8, seed 42, Spearman 0.7055425776426066;
- trial 2: failed, variation 0.0, seed 42, before training.

The deferred v8 work is preserved in the
[tracked snapshot](corrosion_datasets/analysis/epit_pipeline/snapshots/epit24_v8_20260928/README.md).
The archive contains the code, tests, documentation, and v3 rule-evaluation
artifacts, rather than relying on unpushed files under ignored `checkpoints/`.

Every source hash recorded in the coefficient study's `experiment.json` was
restored exactly. The original runner must not be used for this recovery because
it subtracts all historical trials, including failures, from the grid size; the
failed trial would therefore leave one grid cell uncompleted.

Resume with the same external Optuna storage through:

```bash
sbatch scripts/epit_pipeline/resume_coefficient_variation.sbatch
```

The recovery runner validates the frozen directory and Optuna fingerprints,
queues only fixed grid cells, and prioritizes the failed cell. With the state
above, Optuna trial 3 retries variation 0.0/seed 42 and subsequent trials cover
all still-missing cells. A successful finish has 12 unique completed cells plus
one retained failed historical trial (13 trial records in total). No failed
trial directory or study history is deleted, and the frozen fingerprint is not
rewritten.

## 8. What would count as improvement, and later merge points

The immediate question is whether variation improves step-2,000 development
performance consistently across paired seeds. Selecting a higher λ later, or
merely producing more diverse targets now, would not establish improvement.
If variation helps, a subsequent comparison can tune λ with equal budgets for
fixed and variable coefficients, then confirm results at the intended longer
training duration and under a stronger evaluation design.

| main4 destination | Material to incorporate later |
| --- | --- |
| Target-mixture explanation | Amplitude-weight interpretation, conditional variance example, failed-SCM edge case. |
| `sec:coefficient-objective`, `sec:final-rule-refit` | Clarify calibration ordering; introduce optional coefficient sampling as a later extension. |
| `sec:selected-prior-configuration`, `sec:final-training-and-selection` | Historical λ trend and training-duration evidence, clearly marked as development analysis. |
| `sec:physical-limitations`, `sec:next-experiments` | Diagnostic findings, possible prior mismatch, sampling assumptions, pending SCM handling and longer-budget checks. |
| New experimental subsection | The 12-run design, paired-seed results when available, and exact artifact provenance. |
| Implementation/reproducibility notes | Entry points, default-off behavior, test evidence, and current single-runner limitation. |

Before merging, retain the distinction between frozen v7's fixed coefficients
and this optional extension; add actual training results only when available;
archive the temporary λ analysis; and update concurrency/test status if the
implementation changes. Keep measured outcomes separate from proposed mechanisms.
