# Supplement to main4.tex: target mixing, combined training, and diagnostics

Recorded: 2026-09-26; updated with completed training results on 2026-10-01;
Soccol extension added on 2026-10-04; frozen-query context-size diagnostic
implementation and the corresponding checkpoint diagnostics added on
2026-10-11; combined EPIT--Soccol training and frozen MPEA-query workflow
added on 2026-10-13; combined results, implementation audit, and the V2
missingness policy documented on 2026-10-15.
Working notes for a later merge into [main4.tex](main4.tex).
Sources: the Codex conversation **“Review main4 and presentation.md”**
(2026-09-23–25, thread `01a0cde2-bea1-7b32-81c2-2ac6045ae94f`), the subsequent
launch/concurrency discussion, the linked combined EPIT--Soccol/MPEA design and
implementation conversations continued on 2026-10-13--15, and the repository
artifacts linked below.

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
| Separate 2,000-step Optuna experiment | Completed: 12 unique strength/seed cells, plus one retained failed historical trial. |
| Predictive result | The 80% setting has the highest mean at 2,000 steps, but its small and seed-dependent gain does not establish a reliable improvement over fixed coefficients. |
| Several nodes running the same new study concurrently | **Not supported by the current runner.** Shared journal storage exists, but the runner permits only one active process. |
| Failed/constant SCM component rejection | Discussed, but not implemented by this experiment. |
| Adaptive λ, grouped bootstrap, joint λ/variation tuning | Discussed only. |
| Combined EPIT--Soccol context model | Original run trained beyond step 5,900, with saved checkpoint comparisons through step 5,500; the intended step-10,000 endpoint was not reached in the audited run. Prior mixture remains 25% generic, 25% EPIT-informed, and 50% Soccol-informed. |
| MPEA evaluation | Corrected checkpoint comparisons and specialist-transfer results are recorded on the same 267 fixed SCE queries; these are diagnostic results, not a newly selected final model. See Section 15. |
| Missing-query-column inference bug | Fixed: ordinary missing values no longer trigger SHAP-style feature deletion. Earlier affected comparisons require reevaluation. |
| Combined V2 training | Implemented and checked: additional query-column masking and protection of observed composition values, with a fresh-run launcher. No V2 performance result is recorded here. |
| Electrode-conditioned synthetic targets | Limitation identified; generating reference labels and electrode-dependent targets remains deferred. |

The earlier chat reports **64 passing tests** and tiny CPU training checks with
zero and four DataLoader workers. These are historical verification results,
not tests rerun while writing this document. A 12-run dry-run plan was verified
before GPU submission. The completed grid was checked directly from the shared
Optuna journal on 2026-10-01.

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

## 7. Completed Optuna training experiment

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

### Completed results

The shared journal contains 12 completed unique grid cells and the one failed
historical trial described below. The failed trial has no value and is excluded
from every result. The predefined objective is the mean Spearman correlation
across the five development folds at step 2,000.

| Coefficient variation | Mean Spearman at 1,000 | Mean Spearman at 2,000 | SD across seeds at 2,000 | Mean paired change from fixed at 2,000 |
| ---: | ---: | ---: | ---: | ---: |
| 0% | **0.6612** | 0.6917 | 0.0143 | — |
| 30% | 0.6435 | 0.6855 | 0.0065 | −0.0062 |
| 50% | 0.6553 | 0.6760 | 0.0055 | −0.0157 |
| 80% | 0.6589 | **0.6967** | 0.0095 | +0.0050 |

The paired step-2,000 scores expose the interaction with training seed:

| Seed | Fixed | 30% | 50% | 80% | 80% minus fixed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.6879 | 0.6794 | 0.6792 | **0.7055** | +0.0177 |
| 43 | 0.6797 | **0.6924** | 0.6696 | 0.6867 | +0.0070 |
| 44 | **0.7076** | 0.6847 | 0.6791 | 0.6978 | −0.0097 |

The 50% setting is worse than the paired fixed baseline for all three seeds.
The 30% setting improves only seed 43 and is worse on average. The 80% setting
has the largest mean, but it improves seeds 42 and 43 and harms seed 44. Its
mean gain of 0.0050 is small relative to its paired-difference standard deviation
of 0.0138. With only three seeds, this does not demonstrate a reliable benefit.
There is also no monotonic relationship in which increasing variation steadily
improves performance.

At step 1,000, fixed coefficients still have the best mean. The nominal advantage
of 80% variation appears only at the later checkpoint, so any follow-up must use
the intended longer training duration rather than infer a general improvement
from the 2,000-step result.

Seed effects are material. Averaged across strengths, the step-2,000 means are
0.6880, 0.6821, and 0.6923 for seeds 42, 43, and 44, respectively. The best seed
also changes with strength: seed 44 is best for fixed coefficients, seed 43 for
30%, and seed 42 for 50% and 80%. Seeds are therefore replicates for estimating
uncertainty, not another hyperparameter to select.

The current conclusion is to retain fixed coefficients as the supported method.
If this extension is pursued, the only setting justified for a follow-up is an
80%-versus-0% paired comparison with more seeds and the full training duration.
The present conclusion applies to Trial 56's fixed λ of approximately 0.205 and
this 2,000-step development-fold experiment; no final-test rows were used.

Primary result source:
`/home/fcolanto/RZ-Dienste/hpc-user/fcolanto/optuna/epit_coefficient_variation_2k_v1/study_journal.log`.

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
records the intended grid; completion is established by the shared journal.

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

Recovery completed successfully on 2026-10-01. Trials 3–12 filled every missing
cell, producing the 12 unique completed combinations analyzed above. Trial 2
remains in the journal as a failed historical record and is not treated as an
additional replicate.

## 8. Verdict and later merge points

The experiment does not show a consistent predictive improvement from coefficient
variation. The generator diagnostic establishes that the sampler changes the
formula targets as intended, while the training grid shows that this added
diversity is not automatically beneficial. The 80% setting is a candidate for
confirmation, not a replacement for the fixed-coefficient method. Selecting a
higher λ later, or merely producing more diverse targets, would likewise not be
evidence of improvement without paired predictive results.

| main4 destination | Material to incorporate later |
| --- | --- |
| Target-mixture explanation | Amplitude-weight interpretation, conditional variance example, failed-SCM edge case. |
| `sec:coefficient-objective`, `sec:final-rule-refit` | Clarify calibration ordering; introduce optional coefficient sampling as a later extension. |
| `sec:selected-prior-configuration`, `sec:final-training-and-selection` | Historical λ trend and training-duration evidence, clearly marked as development analysis. |
| `sec:physical-limitations`, `sec:next-experiments` | Diagnostic findings, possible prior mismatch, sampling assumptions, pending SCM handling and longer-budget checks. |
| New experimental subsection | The completed 12-run design, aggregate and paired-seed results, verdict, and exact journal provenance. |
| Implementation/reproducibility notes | Entry points, default-off behavior, test evidence, and current single-runner limitation. |

Before merging, retain the distinction between frozen v7's fixed coefficients
and this optional extension; archive the temporary λ analysis; and update
concurrency/test status if the implementation changes. Keep the generator-target
similarity evidence separate from predictive performance, and keep measured
outcomes separate from proposed mechanisms.

## 9. 24-element Al-only rule extension (2026-10-01)

### Scope and leakage controls

The deferred v8 work was restored after the v7 coefficient study completed. The
model-visible composition schema now contains 24 elements. A blank composition
cell is treated as a structural zero only when the other reported elements close
to 100 wt.% within 0.1; otherwise the real row remains available for evaluation
but is not eligible as an empirical pretraining template. This leaves all 760
real rows in evaluation and 757 eligible as empirical templates.

The frozen v7 row and fold assignments were reused without resplitting. The
Al-only direct evaluation uses 94 development rows. The 24 final-test Al rows
remain masked and unused. Existing Fe/Ni-Cr-Mo rules and synthetic routing were
not changed.

### Fixed candidate formulas

All terms are standardized using each fold's context rows. Their calibrated
coefficients are constrained to be nonnegative and sum to one.

The retained matched Al baseline, `al_chloride_temperature`, uses:

- `r_Cl = -log10(max([Cl-], 1e-12))`;
- `r_T = -max(T_C - 30, 0)`.

The new `al_amphoteric_environment` candidate adds:

- `r_acid = -max(5 - pH, 0)`;
- `r_alkaline = -max(pH - 9, 0)`.

Its coefficient anchor is `[0.55, 0.15, 0.15, 0.15]` for chloride,
temperature, acidic pH, and alkaline pH, respectively. The temperature
coefficient is capped at 0.50 and each pH coefficient at 0.30. The two pH
thresholds are fixed exploratory choices, not fitted breakpoints.

The new `al_composition_environment` candidate retains the baseline
environment terms and adds:

- `r_CuMn = log1p(max(Cu,0) + max(Mn,0))`;
- `r_MgSi = -log1p(max(Mg,0) + max(Si,0))`.

Its anchor is `[0.50, 0.15, 0.20, 0.15]`; the Cu-Mn and Mg-Si coefficients
are capped at 0.30 and 0.20. These are deliberately limited bulk-composition
proxies. The dataset cannot distinguish solid solution from precipitates or
heat-treatment state. Cu/Mn is nonzero in only 24 of the 94 development rows,
and Mg/Si in 12, so this rule must remain exploratory.

### Development-fold results

Mean fold Spearman is the primary comparison because every score is first
computed inside its held-out fold. Pooled OOF Spearman concatenates
fold-standardized predictions and is reported only as a secondary diagnostic;
it can disagree with the mean when fold composition and difficulty differ.

| Al-only rule | Rows | Mean fold Spearman | Pooled OOF Spearman | Fold Spearman values |
| --- | ---: | ---: | ---: | --- |
| `al_chloride_temperature` | 94 | 0.1869 | 0.1083 | 0.0000, 0.7628, -0.0635, 0.1044, 0.1306 |
| `al_amphoteric_environment` | 94 | 0.0870 | 0.3739 | -0.2920, 0.0398, 0.0088, 0.5427, 0.1355 |
| `al_composition_environment` | 94 | **0.2368** | 0.4052 | 0.0000, 0.7478, 0.2227, 0.1044, 0.1091 |

The baseline scores exactly reproduce the archived pre-extension artifact. On
all 94 development rows, the final coefficient centers are:

| Rule | Full-development coefficient center |
| --- | --- |
| `al_chloride_temperature` | chloride 0.4712; temperature 0.5288 |
| `al_amphoteric_environment` | chloride 0.2264; temperature 0.3237; acidic pH 0.3000; alkaline pH 0.1499 |
| `al_composition_environment` | chloride 0.2922; temperature 0.3513; Cu-Mn 0.1565; Mg-Si 0.2000 |

The pH rule is worse than the baseline by 0.0999 mean fold Spearman and is not
supported. The composition rule improves the mean by 0.0499, but the gain comes
mainly from one fold: it improves one fold, ties two, and is slightly worse in
two. This is not sufficiently stable evidence to add Al synthetic routing or to
use either new rule in Optuna. The existing synthetic generator therefore
remains Fe/Ni-Cr-Mo only.

### Literature basis and artifacts

The fixed signs follow published Al corrosion mechanisms: log-chloride and pH
kinetics ([McCafferty 1995](https://doi.org/10.1016/0010-938X(94)00150-5)),
the temperature transition near 30 C
([Corrosion Science 2011](https://doi.org/10.1016/j.corsci.2010.09.046)), and
potential-pH-chloride passivation behavior
([Corrosion Science 1978](https://doi.org/10.1016/0010-938X(78)90054-9)).
The composition hypotheses use evidence for Cu state
([Al-Cu study](https://doi.org/10.1016/0010-938X(77)90044-0)), Mn in solid
solution ([Al-Mn study](https://doi.org/10.1016/j.corsci.2020.108749)), and
Mg-Si/Si-rich constituent effects
([Al-Mg-Si study](https://doi.org/10.1016/j.corsci.2013.06.035);
[AlSi10Mg study](https://doi.org/10.1016/j.corsci.2019.03.010)). These sources
do not make bulk wt.% a phase-state measurement, which is why the composition
terms are capped and not promoted.

Complete JSON results, OOF predictions, and the HTML report are under
`corrosion_datasets/analysis/epit_pipeline/target_rules_v3/`. The implementation
is in `scripts/epit_pipeline/target_rules.py`; the final-test masking and
calibration protocol remain in `calibrate_target_rules.py`.

## 10. V8 Optuna protocol: promoted rules and coefficient variation (2026-10-01)

The new study uses the 24-element input schema and a deliberately fixed subset
of nine Fe/Ni target-rule families. Four historical formulas are replaced by
their nitrogen-aware versions: `pren_linear` by `pren_n_linear`,
`cr_mow_synergy` by `cr_mow_n_synergy`, `improved_environment` by
`pren_n_improved_environment`, and `method_aware` by `method_aware_pren_n`.
`threshold_saturation`, `coupled_breakdown`, and `fe_ni_cr_threshold` are
retained. `mo_n_acid_repassivation` and `mns_inclusion_penalty` are added as the
two new Fe/Ni rules. The three Al rules remain excluded from synthetic training.

The calibrated coefficients are centers rather than constants. For each
synthetic task and each nonzero coefficient `w_i`, the generator independently
draws `u_i ~ Uniform(0.2, 1.8)` and computes
`w'_i = w_i u_i / sum_j(w_j u_j)`. Draws that violate the calibrated
rule-specific upper bounds are rejected, with at most 100 attempts; exhausted
draws fall back to the original calibrated vector and record that fallback.
Zero coefficients remain zero. Consequently, "80% variation" denotes a
relative multiplier range around each coefficient, not statistical variance or
an absolute 80-percentage-point change. A dedicated RNG stream uses seed 42 and
worker-derived seeds so this augmentation is reproducible.

The choice of 0.8 follows the completed three-seed development experiment. Its
mean Spearman was 0.6967 versus 0.6917 for fixed coefficients, with paired seed
differences of +0.0177, +0.0070, and -0.0097. This is promising but not
conclusive evidence; the purpose in v8 is to expose the model to plausible
within-rule coefficient changes and test whether that improves transfer. One
important implementation detail is that the calibrated Mo-N-acid interaction
coefficient is zero, so multiplicative variation leaves that term at zero.

The new Optuna study is isolated under
`epit_pipeline_optuna_empirical_features_scm_target_v8` and `optuna_v3`. It runs
50 trials at 1,000 steps each. The study fingerprint includes the 24-element
profile, promoted rule order, coefficient centers and bounds, variation strength
0.8, and seed 42, preventing accidental resume with a different configuration.

## 11. Soccol dataset and second EPIT pipeline (2026-10-04)

### Scope and relationship to the earlier EPIT work

Soccol is the second pitting-potential prediction dataset. Its implementation
reuses the later EPIT-v8 empirical-feature SCM-target workflow while replacing
the 28-column EPIT runtime schema with a Soccol-specific 37-column schema. It
does not recompute or replace the earlier 21-column v7 result described in
`main4.tex`, and it does not replace the EPIT-v8 launchers or artifacts.

The retained methodology is:

- empirical-feature synthetic pretraining;
- analytical corrosion rules for the informed synthetic target;
- composition-grouped development folds;
- Optuna configuration selection;
- development-only checkpoint selection; and
- a separate one-time evaluation on the untouched outer final-test targets.

Soccol-specific entry points live under
[`scripts/soccol_pipeline/`](scripts/soccol_pipeline/), while shared training
and evaluation behavior is reused from `scripts/epit_pipeline/` through thin
wrappers.

### Dataset provenance and regression task

The source is Dimitri Soccol's public
[Pitting Potential Database](https://github.com/SoccolD/Pitting_potential_database),
preserved at commit `50f2df0918116e12026f8ea18c32704a5f6822e5`. The primary
input is the 2024 workbook; two earlier workbooks are retained as historical
upstream snapshots. The associated paper is D. Soccol, “An updated Pitting
Resistance Equivalent Number by proportional hazard survival models of
reported pitting potentials,” *Electrochimica Acta* 511, 145355
([DOI](https://doi.org/10.1016/j.electacta.2024.145355)). The paper uses
censor-aware proportional-hazards models; the present ordinary-regression
benchmark is a separate task.

The `pitting_potentials` sheet contains 4,460 observations and 43 fields. The
`references` sheet maps 154 source identifiers to publications. `E_pit` is the
reported breakdown potential in mV versus Ag/AgCl (3 M KCl): `event=1` denotes
actual pitting and `event=0` a competing non-pitting breakdown, right-censored
for latent pitting potential in the collection paper.

The primary regression subset requires numeric `E_pit` and `event=1`, leaving
4,027 usable rows. A separate survival table retains 4,384 rows with numeric
breakdown potential and a defined event indicator; it is not used by this
regression pipeline. `E_corr`, `event`, identifiers, source, alloy designation,
references, audit flags, and missingness indicators are excluded from model
inputs. In particular, alloy names cannot circumvent the composition-group
split.

The original workbooks remain unchanged. Repairs and encodings occur only in
generated processed tables. No explicit license was detected in the preserved
repository, so redistribution terms remain separate from scientific
provenance. See the [schema summary](corrosion_datasets/datasets/soccol_pitting_potential/schema_summary.md)
and [literature record](corrosion_datasets/datasets/soccol_pitting_potential/literature.md).

### Source audit, material families, and composition repair

All 154 source identifiers were reviewed. The registry at
[`source_conventions.csv`](corrosion_datasets/analysis/soccol_source_conventions/source_conventions.csv)
records material family, composition coverage, balance-element convention,
source-specific problems, evidence, confidence, and recommended row handling.

| Audit category | Sources | Raw rows |
| --- | ---: | ---: |
| Usable Fe-based | 131 | 3,706 |
| Usable with a composition caveat | 14 | 265 |
| Material-taxonomy caveat | 3 | 204 |
| Mixed material families | 4 | 169 |
| Separate non-Fe family | 1 | 92 |
| Repair required | 1 | 24 |

No family is silently removed from the broad benchmark. Row-level include,
review, and exclude flags remain available for sensitivity analysis.

The confirmed `2000Russell` error affects all 24 titanium cells. Their values
equal `Cr + 3.3*Mo + 20*N`, a PRE expression rather than Ti composition. The
processed tables discard those values, encode `Ti=0` under the benchmark's
missing-composition convention, and retain `Ti_missing=1` for audit. Eleven of
the affected rows occur in the 4,027-row event-1 regression subset.

A blank composition cell means missing or unreported, not a verified physical
zero. The benchmark nevertheless uses a common zero encoding for composition
blanks, while retaining the per-element missingness columns for audit only.

Fe was added as a thirteenth composition feature only where the row-level
material classification and reported major composition support an Fe balance:

\[
\mathrm{Fe}_{\mathrm{approx}}
=100-\sum_{e\ne\mathrm{Fe}}x_e.
\]

Fe is not reconstructed when major composition is unreported, the remainder is
materially negative, or the row belongs to an incompatible or uncertain
high-Ni family. Unsupported rows receive `Fe=0, Fe_missing=1`. For example,
the `2009Wong` Ni-Cr-Mo rows already close to approximately 100 wt.% in their
three reported elements and receive no invented Fe. The derived Fe value is an
approximation because the workbook can omit source-specific alloying elements.

### Fixed 37-feature schema and fold-local preprocessing

Every Soccol model receives exactly 37 columns, with informed-prior block
allocation `(13,20,4,0,0,0,0,0,0)`. Magpie descriptors are disabled.

| Block | Count | Fixed columns |
| --- | ---: | --- |
| Composition, wt.% | 13 | `Fe`, `C`, `N`, `Si`, `P`, `S`, `Ti`, `V`, `Cr`, `Mn`, `Ni`, `Nb`, `Mo` |
| Continuous preparation/procedure | 10 | `Prep_grinding_grit`, `Prep_Ra_micron`, `Prep_pH`, `Prep_redox`, `Prep_time`, `CP_time`, `CP_temp`, `CP_pH`, `Test_area_cm2`, `scan_rate` |
| Ion concentration, M | 10 | `CP_Cl`, `CP_Br`, `CP_OH`, `CP_SO4`, `CP_CO3`, `CP_NO3`, `CP_PO4`, `CP_MoO4`, `CP_CrO4`, `CP_ion_other` |
| Categorical procedure | 4 | `Prep_medium`, `CP_aeration`, `CP_agitation`, `CP_anions_info` |

Composition and ion blanks use the fixed zero encoding. For every development
fold or final evaluation, means for the other continuous columns are fitted on
the labeled context rows only and applied to both context and query rows.
Category-to-integer mappings are likewise fitted on context rows only; missing
context values and missing or unseen query categories become `-1`. The
evaluator verifies that all 37 supplied values are finite.

The real-data path adds no logarithmic ion transform, universal numeric
sentinel, missingness-indicator predictor, or measured response feature.
Negative values remain valid in fields such as preparation redox or pH. The
exact contract is recorded in the
[preprocessing plan](corrosion_datasets/datasets/soccol_pitting_potential/preprocessing_plan.md)
and [feature manifest](corrosion_datasets/datasets/soccol_pitting_potential/processed/feature_manifest.json).

### Composition-grouped outer split and development folds

The split applies the previous EPIT grouping method to the 13 processed
composition values. Values are rounded to 0.01 wt.%, and two rows are linked
when

\[
d(i,j)=\sum_{e=1}^{13}|x_{i,e}-x_{j,e}|\leq1.0\ \mathrm{wt.\%}.
\]

Connected components are indivisible. The 4,027 rows form 339 groups from 474
rounded composition keys. The largest connected-component diameter is 4.84
wt.% because chains of close rows can span more than the direct-link threshold;
the minimum distance between distinct components is 1.01 wt.%.

| Partition | Rows | Role |
| --- | ---: | --- |
| Development | 3,222 | Rule development, Optuna selection, and checkpoint selection |
| Final test | 805 | One final evaluation after model freezing |
| Development validation folds | 645 / 645 / 644 / 644 / 644 | Five grouped rotations |

No composition group crosses the outer boundary or a development fold. Every
development row is a validation query once. Subject to group integrity and
fixed row counts, the assignment balances pitting-potential deciles, material
family, composition isolation, temperature, chloride, pH, and source. The old
test-method balancing block is omitted because Soccol has no comparable single
method field.

Source is used for balance and audit, not as an atomic group: 101 sources are
development-only, 14 final-only, and 37 occur on both sides. The result is a
composition-held-out, mixed-source benchmark, not a source-held-out claim about
unseen publications or laboratories.

The [manifest](corrosion_datasets/datasets/soccol_pitting_potential/processed/splits_v1/split_manifest.json),
[assignments](corrosion_datasets/datasets/soccol_pitting_potential/processed/splits_v1/split_assignments.csv),
[report](corrosion_datasets/datasets/soccol_pitting_potential/processed/splits_v1/split_report.html),
and [lock](corrosion_datasets/datasets/soccol_pitting_potential/processed/splits_v1/split_lock.json)
are frozen together by hashes. The public assignments omit final-test target
values.

### Development-only rule investigation

The later EPIT rule set was first replayed on the five frozen Soccol
development folds. Eight formulas were directly evaluable. The method-aware
formula was not carried over because Soccol has no field comparable to the old
test-method category. Historical names remain for artifact compatibility; for
example, `cr_mow_n_synergy` contains no W term because the Soccol schema has no
W column.

Every rule fold fits its preprocessing, scaling, and constrained coefficients
on four development folds and evaluates the fifth. Final-test targets are
replaced by missing values before the fitting code receives the table.

The investigation additionally tested specimen area, grinding grit, roughness,
scan rate, bromide, sulfate, nitrate, phosphate, molybdate, and chromate. Area,
surface-finish, and scan-rate additions reduced the leading rule's mean fold
Spearman and were rejected. The retained extension uses bromide and a
sulfate-nitrate ratio. Phosphate, molybdate, and chromate were omitted because
their fitted contribution was small and did not improve the compact rule.

The first new rule combines

\[
M=\mathrm{Cr}+3.3\,\mathrm{Mo}+16\,\mathrm{N}
\]

with the existing coupled environmental-breakdown structure and a
susceptibility penalty proportional to

\[
\sqrt{\max(\mathrm{Mn},0)\max(\mathrm{S},0)}.
\]

Bulk Mn and S are only a proxy for MnS inclusions. The second new rule replaces
chloride by the heuristic effective halide

\[
H=\mathrm{Cl}+0.5\,\mathrm{Br}
\]

and adds

\[
\log_{10}\left(1+
\frac{\mathrm{SO_4}+\mathrm{NO_3}}{H+10^{-5}}\right).
\]

The 0.5 bromide factor is a dataset-screening heuristic, not a universal
chemical equivalence. Literature and limitations for all added terms are
recorded in
[`literature_basis.md`](corrosion_datasets/analysis/soccol_target_rules_v1/literature_basis.md).

Seven adapted EPIT families and the two new Soccol families were promoted for
synthetic target generation:

| Promoted family | Mean fold Spearman | Pooled OOF Spearman |
| --- | ---: | ---: |
| `pren_n_linear` | 0.4633 | 0.4551 |
| `cr_mow_n_synergy` | 0.4947 | 0.4885 |
| `threshold_saturation` | 0.4596 | 0.4455 |
| `pren_n_improved_environment` | 0.4597 | 0.4637 |
| `mo_n_acid_repassivation` | 0.4599 | 0.4658 |
| `mns_inclusion_penalty` | 0.4980 | 0.4972 |
| `coupled_breakdown` | 0.5028 | 0.4923 |
| `pren_n_coupled_mns` | 0.5542 | 0.5340 |
| `pren_n_coupled_mns_weak_anions` | **0.5866** | **0.5712** |

These are direct analytical-rule development scores, not trained-transformer
or final-test results. `pren_n_coupled_mns` improves four of five folds over
the best old rule. The weak-anion extension improves three of five folds over
that core new rule.

The full-development coefficient center for `pren_n_coupled_mns` is 0.0274
material passivity, 0.5726 coupled breakdown, 0.1500 acidic-pH aggressiveness,
and 0.2500 Mn-S susceptibility. For
`pren_n_coupled_mns_weak_anions`, the corresponding weights are 0.0027,
0.5468, 0.1500, 0.2080, and 0.0925 for the weak-inhibitor ratio. Each vector is
nonnegative, respects recorded upper bounds, and sums to one.

The comparison outputs are under
[`soccol_target_rules_v1/`](corrosion_datasets/analysis/soccol_target_rules_v1/),
and the nine production artifacts under
[`soccol_pipeline/target_rules_v1/`](corrosion_datasets/analysis/soccol_pipeline/target_rules_v1/).
The final paragraph of the comparison directory's `README.md` still says the
production registry and generator were unchanged. That sentence describes the
earlier screening snapshot and is now stale: the two new families have been
promoted and the Soccol production registry is active. The frozen production
artifacts and implementation are authoritative.

### Target-free empirical feature profile and synthetic generation

The immutable
[`soccol_pitting_features_v1`](src/tabicl/prior/assets/soccol_pitting_features_v1.json)
profile contains the 37 processed input columns for all 4,027 regression rows,
stable row identifiers, preprocessing metadata, and hashes. It contains no
`E_pit`, `event`, or other target column.

For each synthetic row, the generator samples a composition-template row and
an environment/procedure row independently with replacement. Positive
composition entries receive multiplicative log-normal perturbations; exact
zeros remain zero. The 13-component composition is then closed to 100 wt.%
when its total is positive, and the independently sampled context supplies the
remaining 24 columns.

The complete target-free profile supplies synthetic continuous means and
categorical mappings. Real fold and final-test evaluation does not reuse that
state: it fits means and mappings from the current labeled context rows.

The profile includes feature covariates from the 805 final-test rows. The
holdout is therefore target-blind, not completely covariate-blind. Final-test
targets do not enter rule fitting, coefficient calibration, rule probabilities,
Optuna objectives, checkpoint selection, or the feature profile, but final-test
feature values contribute to the empirical sampling distribution. This matches
the earlier EPIT empirical generator and must be disclosed with any result.

For each informed synthetic task, one of the nine families is sampled with
probability proportional to its positive development score. The standardized
analytical response is mixed with the generic MLP/tree SCM response using the
searched `lambda`, followed by final standardization. Each nonzero rule
coefficient receives the v8 multiplier variation `Uniform(0.2, 1.8)`, the
vector is renormalized, rule-specific upper-bound violations are rejected, and
zero coefficients remain zero. The dedicated coefficient seed is 42.

### Soccol Optuna search

The dedicated launcher preserves the previous search protocol:

| Setting | Soccol value |
| --- | --- |
| Trials requested by one invocation | 50 |
| Startup trials | 10 |
| Training steps per trial | 1,000 |
| Scheduler horizon | 10,000 |
| Development validation folds | 5 |
| Estimators per fold | 8 |
| Coefficient variation | 0.8, seed 42 |
| Objective | Mean development-fold Spearman |

The four searched quantities are informed-task probability
`rho ∈ {0.25, 0.50, 0.75, 1.00}`, MLP share within informed SCMs
`{0, 0.25, 0.50, 0.70, 0.75, 1}`, analytical-target mixture
`lambda ∈ [0,1]`, and composition perturbation strength `tau ∈ [0,0.15]`.
Magpie is fixed off, direct feature-block coupling is zero, and the legacy
Dirichlet material mixture is disabled.

The pipeline fingerprint binds the processed-data hash, split manifest and
lock, feature-profile hashes, nine rule artifacts, scores, coefficients,
bounds, search space, and evaluation policy. Workers with a different identity
cannot silently join the study.

`--n-trials 50` is passed to Optuna per launcher invocation; it is not a global
study cap. Two concurrent launchers would each request 50 trials. Workers must
therefore be coordinated or given divided counts if the intended global total
is exactly 50.

Generated worker-local trial and development-evaluation directories below
`corrosion_datasets/analysis/soccol_pipeline/optuna_v1/` are ignored by version
control. Previously tracked examples were removed from the index without
deleting the running jobs' files from disk. They are operational output, not
frozen source artifacts.

The study has produced partial development results, but the planned search and
frozen final workflow are not complete. No interim trial is presented here as
the selected configuration, and no final-test performance is reported.

### Compatibility correction and final workflow

The implementation audit found that the two Soccol-only rules had initially
been inserted into the shared EPIT rule registry. That made the legacy EPIT
evaluator capable of selecting formulas requiring Soccol-only ion columns.
The registry was split as follows:

- `EPIT_TARGET_RULE_COEFFICIENTS` contains only legacy-schema-compatible rules;
- `SOCCOL_TARGET_RULE_COEFFICIENTS` extends it with the two Soccol families;
- the EPIT launcher selects the EPIT registry; and
- the Soccol launcher explicitly selects the Soccol registry.

This restores legacy behavior while retaining all nine Soccol rules.

The final workflow now has separate Soccol entry points:

1. [`train_final.py`](scripts/soccol_pipeline/train_final.py) verifies the
   study fingerprint and selected trial, reconstructs the effective settings,
   trains for 10,000 steps, and evaluates permanent checkpoints from step 500
   through 10,000 at 500-step intervals on the five development folds.
2. The checkpoint with the highest mean development-fold Spearman is frozen
   with its study, rule, split, command, checkpoint, and inference identities.
3. [`evaluate_final.py`](scripts/soccol_pipeline/evaluate_final.py) uses all
   3,222 development rows as labeled context and predicts the 805 final-test
   rows once with 37 features, eight estimators, median output, no feature
   shuffle, no power normalization, and no uncertainty inference.

The associated Slurm entry points are
[`train_final.sbatch`](scripts/soccol_pipeline/train_final.sbatch) and
[`evaluate_final.sbatch`](scripts/soccol_pipeline/evaluate_final.sbatch).
Soccol manifests and evaluation output use
`corrosion_datasets/analysis/soccol_pipeline/final_v1/`; checkpoints use
`checkpoints/soccol_pipeline_final_v1/`. These roots do not overwrite EPIT
outputs.

The only required change to the shared old final evaluator was to replace its
embedded 608/152 EPIT row-count assertions with configurable constants. Their
defaults remain 608/152, so the old pipeline behaves as before. The Soccol
wrapper overrides them with 3,222/805 and sets the model label to
`final_soccol_model`; the old inference policy was otherwise left unchanged.

The final Soccol workflow is implemented and tested but has not been executed
on the final-test labels. There is therefore no Soccol held-out result yet.

### Reproducibility checks and limits

The processed tables, split, rule comparison, target-free profile, and promoted
rule artifacts have reproducible builders. Rebuilding the profile CSV and
rule artifacts reproduced the frozen versions byte for byte. The NumPy
calibration and Torch synthetic formulas were also compared for every promoted
family on all development rows: correlations were 1.0 and maximum absolute
differences were below `1.3e-7`.

Tests cover the 37-column schema, finite synthetic sampling, context-only
preprocessing, rule-registry separation, grouped final split, real rule
artifact loading, the Soccol final wrappers, and unchanged legacy EPIT
behavior. At this implementation checkpoint the complete repository suite
passed 303 tests with two skips.

The result must retain these qualifications:

- zero-filled composition blanks and approximate Fe balances are benchmark
  encodings, not recovered or certified alloy chemistry;
- the 13-element representation omits source-specific elements for which the
  workbook has no column;
- ordinary regression conditions on actual-pitting events and does not model
  `event=0` competing breakdowns as censored observations;
- composition grouping prevents close-alloy leakage, but 37 publications occur
  in both outer partitions;
- final-test covariates, though not targets, contribute to the empirical
  feature profile; and
- possible overlap with the older Nyby collection must be removed or audited
  before Soccol is called a fully independent external-validation dataset.

Soccol is therefore a second grouped pitting-potential prediction task with an
untouched target holdout, not yet a completed external-validation result.

## 12. Frozen-query context-size diagnostic and fixed-schema inference (2026-10-11)

### Purpose and evaluation boundary

The earlier development/final-test comparison changed both the labeled context
size and the query rows, so it could not isolate sensitivity to sample scarcity.
The implemented diagnostic changes only which development rows are supplied as
labeled context. The frozen outer final-test partition remains the query set in
every run: 152 rows for EPIT and 805 rows for Soccol. A development fold is
never substituted for the final-model query set.

Each of the five saved development folds is used separately as labeled context.
This gives 121--122 EPIT context rows or 644--645 Soccol context rows per run,
while the corresponding outer query rows remain fixed. The complete frozen
model is reused without retraining or checkpoint selection. The diagnostic is
therefore a context-size sensitivity analysis, not a new cross-validation
estimate and not an additional model-selection stage.

Metrics are calculated independently for all five context runs and then
summarized by their mean and sample standard deviation. Predictions are not
averaged across context folds. Every requested fold must complete for every
model; otherwise the aggregate comparison fails rather than silently averaging
an incomplete set. Results from this diagnostic must not be used to replace a
development-selected checkpoint or configuration.

### Fixed preprocessing state

For this diagnostic, preprocessing is fitted once on the covariates of the full
development partition and reused for every reduced labeled-context fold. No
development target outside the selected context fold enters inference. This
keeps numeric imputation, categorical mappings, and the supplied feature order
fixed, so the comparison changes labeled context availability rather than also
changing the preprocessing state.

For Soccol, the full-development reference supplies means for the non-composition
continuous columns and mappings for the four categorical procedure columns.
Composition and ion blanks retain the previously defined zero encoding. For
EPIT, nullable categorical missing values are normalized before applying the
fixed numerical encoding. Both context and query matrices are checked for a
finite, unchanged feature count.

### Preserving the trained feature schema

Reduced context can make a feature constant even when it varied during training.
The default TabICL preprocessing removes constant context columns, which would
shift later columns and violate the positional meaning of the fixed physical
schema. Fixed-schema inference therefore disables constant-feature removal for
these evaluations. All trained columns are retained in their original order:
28 for EPIT-v8 and 37 for Soccol.

The evaluator records both the supplied and model-visible feature counts and
fails if a fixed-schema model drops a column. This behavior is opt-in for named
fixed-schema models in ordinary evaluations and is enforced for TabICL models
in the frozen-query context-fold diagnostic. The generic estimator default
remains unchanged outside this workflow.

Implementation references are
[`eval_corrosion_datasets.py`](scripts/eval_corrosion_datasets.py),
[`preprocessing.py`](src/tabicl/sklearn/preprocessing.py), and
[`regressor.py`](src/tabicl/sklearn/regressor.py). The EPIT and Soccol
checkpoint-comparison wrappers expose the diagnostic through
`--context-folds 5`.

## 13. EPIT-v8 and Soccol frozen-query checkpoint diagnostics (2026-10-10--11)

### Scope and selection status

Four comparison runs evaluate the EPIT-v8 Trial 17 and Soccol Trial 19 final
training trajectories. Each contains 25 saved checkpoints: steps 500 through
9,000 at 500-step intervals, followed by steps 9,400 through 10,000 at 100-step
intervals. The local generic baseline is compared at the same step. Pretrained
TabICL V2 and fixed CatBoost do not change with checkpoint step.

These runs inspect the frozen outer query labels at every checkpoint. They are
therefore post-selection diagnostics, following the same restriction as the
later-checkpoint analysis in `main4.tex`: an outer-test maximum cannot select or
rename the final model. At this snapshot, the local EPIT-v8 and Soccol analysis
roots do not contain completed `final_model_manifest.json` files. The
best-looking checkpoints below are descriptive maxima from the recorded outer
test sweeps, not development-selected checkpoints and not new final claims.

The full-context evaluations use all 608 EPIT or 3,222 Soccol development rows
as labeled context. The reduced-context evaluations use each saved development
fold separately: 121--122 EPIT or 644--645 Soccol labeled rows. The query sets
remain the same 152 EPIT or 805 Soccol outer final-test rows in every run.

### EPIT-v8 results

With all 608 development rows as context, the largest recorded outer-test
Spearman value for CorrPFN-v8 occurs at step 6,500. Values in this table are
single evaluations on the fixed 152-row query set.

| Model at step 6,500 | Spearman | MAE (mV) | RMSE (mV) | R2 |
| --- | ---: | ---: | ---: | ---: |
| CorrPFN-v8 Trial 17 | **0.7082** | 204.5 | **302.3** | **0.4045** |
| Generic local baseline | 0.5268 | 241.0 | 340.6 | 0.2440 |
| Pretrained TabICL V2 | 0.6355 | **199.4** | 321.8 | 0.3249 |
| CatBoost | 0.6008 | 224.4 | 326.7 | 0.3043 |

The same CorrPFN-v8 model at its final step 10,000 has Spearman 0.7021, MAE
208.5 mV, RMSE 296.2 mV, and R2 0.4282. Thus, the largest Spearman and the best
error/R2 values do not all occur at one checkpoint. The outer-test trajectory
must not be used to choose among them.

In the five reduced-context runs, the largest recorded mean Spearman for
CorrPFN-v8 occurs at step 8,500. Values are mean plus or minus sample standard
deviation across the five context folds; predictions are not averaged.

| Model at step 8,500 | Spearman | MAE (mV) | RMSE (mV) | R2 |
| --- | ---: | ---: | ---: | ---: |
| CorrPFN-v8 Trial 17 | **0.5409 +/- 0.1081** | **255.1 +/- 19.9** | **340.5 +/- 27.2** | **0.2402 +/- 0.1219** |
| Generic local baseline | 0.4688 +/- 0.1200 | 267.5 +/- 19.1 | 357.9 +/- 22.7 | 0.1624 +/- 0.1068 |
| Pretrained TabICL V2 | 0.4563 +/- 0.2094 | 266.4 +/- 47.8 | 389.2 +/- 87.7 | -0.0274 +/- 0.4585 |
| CatBoost | 0.4293 +/- 0.1010 | 271.2 +/- 16.4 | 367.5 +/- 31.0 | 0.1147 +/- 0.1532 |

At step 10,000, the corresponding CorrPFN-v8 fold means are Spearman
0.5297 +/- 0.1151, MAE 257.1 +/- 19.1 mV, RMSE 343.0 +/- 27.6 mV, and R2
0.2294 +/- 0.1247. Relative to its full-context step-10,000 result, mean
Spearman is lower by 0.1724, while MAE and RMSE are higher by 48.7 and 46.8 mV.
This is a sensitivity comparison on the same queries, not an estimate from five
independent test samples.

The saved EPIT reduced-context run predates the evaluator fields that record
full-development preprocessing and the model-visible fixed feature count. Its
rows confirm the fixed outer query set and 28 supplied features, but they do not
prove that constant-column removal was disabled inside every model fit. Treat
the numerical fold comparison above as provisional and rerun it with the
current fixed-schema evaluator before using it as formal evidence about sample
scarcity.

### Soccol results

With all 3,222 development rows as context, the largest recorded outer-test
Spearman for Soccol Trial 19 occurs at step 8,500 on the fixed 805-row query
set.

| Model at step 8,500 | Spearman | MAE (mV) | RMSE (mV) | R2 |
| --- | ---: | ---: | ---: | ---: |
| Soccol Trial 19 | **0.6897** | 158.5 | **210.7** | **0.4545** |
| Generic local baseline | 0.5566 | 184.5 | 249.5 | 0.2349 |
| Pretrained TabICL V2 | 0.6272 | 158.7 | 221.2 | 0.3989 |
| CatBoost | 0.6605 | **152.0** | 210.8 | 0.4541 |

At step 10,000, Soccol Trial 19 has Spearman 0.6830, MAE 161.4 mV, RMSE
214.0 mV, and R2 0.4374. Trial 19 has the best rank correlation at its
descriptive step-8,500 maximum, while CatBoost retains the smallest MAE and is
nearly tied in RMSE and R2. The differences do not establish dominance on all
metrics.

In the five fixed-schema reduced-context runs, the largest recorded mean
Spearman for Soccol Trial 19 occurs at step 9,400.

| Model at step 9,400 | Spearman | MAE (mV) | RMSE (mV) | R2 |
| --- | ---: | ---: | ---: | ---: |
| Soccol Trial 19 | **0.5752 +/- 0.0839** | **183.8 +/- 14.5** | **242.2 +/- 13.3** | **0.2774 +/- 0.0791** |
| Generic local baseline | 0.4448 +/- 0.0883 | 208.1 +/- 21.0 | 276.4 +/- 27.3 | 0.0537 +/- 0.1858 |
| Pretrained TabICL V2 | 0.4758 +/- 0.0656 | 201.0 +/- 7.8 | 259.8 +/- 10.3 | 0.1696 +/- 0.0665 |
| CatBoost | 0.5644 +/- 0.0532 | 188.6 +/- 8.9 | 246.5 +/- 10.6 | 0.2525 +/- 0.0646 |

At step 10,000, the Soccol fold means are Spearman 0.5727 +/- 0.0860, MAE
184.1 +/- 14.7 mV, RMSE 242.5 +/- 13.3 mV, and R2 0.2754 +/- 0.0792. Relative
to the full-context step-10,000 result, mean Spearman is lower by 0.1103, while
MAE and RMSE are higher by 22.7 and 28.6 mV. The saved manifest records all
five completed folds, the unchanged outer query partition, preprocessing fitted
on the full development covariates, preservation of the 37-feature schema, and
zero evaluation errors.

### Artifact and plot locations

The four recorded comparison directories are:

| Evaluation | Result and table files | Plot directory |
| --- | --- | --- |
| EPIT-v8, full development context | [`v8_final_test_cpu_comparison/`](corrosion_datasets/analysis/epit_pipeline/v8_final_test_cpu_comparison/) contains `results.json`, `rows.csv`, `rows_wide.csv`, and `rows_summary.csv`. | [`plots/`](corrosion_datasets/analysis/epit_pipeline/v8_final_test_cpu_comparison/plots/) |
| EPIT-v8, five reduced context folds | [`v8_final_test_context_folds/`](corrosion_datasets/analysis/epit_pipeline/v8_final_test_context_folds/) contains the aggregate files plus one directory per context fold. | [`plots/`](corrosion_datasets/analysis/epit_pipeline/v8_final_test_context_folds/plots/) |
| Soccol, full development context | [`final_test_full_development_comparison/`](corrosion_datasets/analysis/soccol_pipeline/final_test_full_development_comparison/) contains `results.json`, `rows.csv`, `wide.csv`, and `summary.csv`. | [`plots/`](corrosion_datasets/analysis/soccol_pipeline/final_test_full_development_comparison/plots/) |
| Soccol, five reduced context folds | [`final_test_context_folds_comparison/`](corrosion_datasets/analysis/soccol_pipeline/final_test_context_folds_comparison/) contains the aggregate files plus one directory per context fold. | [`plots/`](corrosion_datasets/analysis/soccol_pipeline/final_test_context_folds_comparison/plots/) |

Every plot directory contains these six SVG files:

- `trend_mean_test_spearman.svg`
- `trend_mean_test_mae.svg`
- `trend_mean_test_r2.svg`
- `trend_mean_test_pearson.svg`
- `trend_mean_test_nmae_iqr.svg`
- `trend_mean_test_nrmse_iqr.svg`

These paths should be used when figures are later copied into the manuscript.
The separate `epit_pipeline/v8_final_test_cpu/` directory is an earlier
CorrPFN-only run and is not the complete EPIT full-context comparison.

## 14. Combined EPIT--Soccol training with frozen MPEA queries (2026-10-13)

### Target and reference-electrode decision

The combined task uses one response,
`reported_pitting_potential_mV`, meaning the reported potential in mV on the
scale identified by the categorical predictor `reference_type`. It does not
create separate SCE and Ag/AgCl targets and does not use a numeric sentinel such
as `-1` for an unknown reference. The two known reference categories are `SCE`
and `AgAgCl`; an unverified reference is represented by an actual missing value,
not a third physical electrode category.

The standalone Soccol section above records the collection-level Ag/AgCl
convention. For combined training, the audit adopted a more conservative
source-level rule: 565 Soccol rows have verified SCE provenance, 112 have
verified Ag/AgCl provenance, and 3,350 of 4,027 do not have enough retained
primary-source evidence to assign either scale. Those 3,350 rows are retained
with `reference_type=NaN`, while `reference_type_evidence` records
`source_reference_not_verified` for auditing. No electrode conversion is
applied, because a blanket conversion would falsely claim a known reference for
most Soccol rows. This keeps the observed target and exposes known measurement
types as inputs. It does not establish that the model has learned the physical
reference-electrode relationship: the informed generators currently leave
`reference_type` missing, as clarified in Section 15.

This is input uncertainty, not a missing-target problem. All labeled context
rows retain a finite reported potential. A row with an actually missing target
could not be context or contribute to a score. The MPEA scoring targets are SCE
and remain separate from the query features.

### Derived data and shared feature contract

The original EPIT, Soccol, and MPEA files are unchanged. The derived directory
[`combined_epit_soccol_mpea/`](corrosion_datasets/datasets/combined_epit_soccol_mpea/)
contains:

| Artifact | Contents |
| --- | --- |
| `epit_soccol_context.csv` | 4,787 labeled rows: 760 EPIT and 4,027 Soccol. Across both sources, 1,325 have verified SCE, 112 verified Ag/AgCl, and 3,350 have a missing reference type with separate audit evidence. |
| `mpea_queries.csv` | 267 fixed source-screened MPEA query rows with no target column. |
| `mpea_scoring_targets.csv` | The 267 row-ID-aligned SCE targets, held separately for final scoring. |
| `shared_feature_registry.json` | Canonical names, order, units, aliases, source mappings, and reference categories. |
| `context_missingness_profile.json` | Context-only joint missingness patterns: 9 EPIT patterns and 265 Soccol patterns, including missing reference type. |
| `combined_prior_config.json` | Frozen prior regimes, source commands, source hashes, formulas, coefficients, and mixture policy. |

The shared input has exactly 50 named slots: 24 composition columns, 20
continuous preparation/environment columns, and six categorical columns
(`Prep_medium`, `CP_aeration`, `CP_agitation`, `CP_anions_info`, `test_method`,
and `reference_type`). Fifty features means 50 stable semantic positions; a
missingness mask sets individual cells to `NaN` and does not delete columns.
Features unsupported by a source remain `NaN`. Soccol composition zeros are
also restored to `NaN` where its retained audit flag says that the original
cell was unreported. The two known `2016Alar` value discrepancies are recorded
but preserved rather than silently changing an existing processed dataset.

Incoming DataFrames are aligned by exact canonical column names plus explicit
approved aliases. Columns can arrive in any order; absent columns are filled
with `NaN`, and unknown columns are reported or rejected. The aligner performs
no fuzzy matching and does not infer units. The estimator also reindexes exact
canonical names at prediction time. Fixed-schema inference retains all 50
columns, disables constant-feature removal, and uses no feature shuffle.

### Learned missingness and source-aware masks

The combined model opts into a learned missing-value token through
`--col_missing_value_embedding True`. Actual `NaN` values are preserved through
the scikit-learn path instead of being replaced by a universal numeric
sentinel. The implementation uses one learned embedding vector shared across
columns, not a separate missing token for each named feature. Column feature
grouping is disabled for this path, as required by the current implementation;
this is an architecture difference from the generic and specialist checkpoints.
The ordinary generic model keeps its historical feature grouping and all new
behavior is opt-in; existing generic configurations and checkpoints are not
rewritten. Canonical names align the inputs; they are not text descriptions
encoded by a language model.

In the original training policy, joint masks sampled from the 4,787 context
rows are applied to composition as well as non-composition inputs, never to the target. EPIT-
informed tables receive EPIT masks, Soccol-informed tables receive Soccol masks,
and generic tables sample EPIT or Soccol masks in proportion to their empirical
context row counts. At the default augmentation probability of 0.8, at least
two originally observed composition values are retained per synthetic row when
possible. Internal source IDs route these masks without becoming model inputs.
MPEA queries and MPEA targets contribute to neither the mask profile nor
pretraining. Section 15 records the V2 change to composition protection and
additional query-only masking.

### Frozen combined prior

The prior does not average the EPIT and Soccol formulas or coefficients. It
first gives the two selected source-model regimes equal weight. The EPIT-v8
model was historically 50% generic and 50% informed, whereas the selected
Soccol model was 100% informed. This yields the frozen combined mixture below.

| Regime | Share | MLP/tree | Categorical generation | Retained source settings |
| --- | ---: | ---: | ---: | --- |
| Generic | 0.25 | 0.7/0.3 | `cat_prob=0.2` | Generic 50-feature SCM; no informed corrosion formula. |
| EPIT-informed | 0.25 | 0.7/0.3 | Random categorization off | EPIT profile, `(24,3,1,0,0,0,0,0,0)` block allocation, target-mix weight 0.2642320686, family probabilities `(0.745,0.1375,0.0425,0.0425,0.0325)`, perturbation 0.1496753492, and the selected EPIT formulas and coefficient dictionaries. |
| Soccol-informed | 0.50 | 1.0/0.0 | Random categorization off | Soccol profile, `(13,20,4,0,0,0,0,0,0)` block allocation, target-mix weight 0.6629552825, family probabilities `(1,0,0,0,0)`, perturbation 0.03576597424, and the selected Soccol formulas and coefficient dictionaries. |

Both informed regimes retain coefficient variation 0.8, its calibrated bounds,
and seed 42. Their schema-defined categorical fields remain categorical even
though generic random category conversion is disabled. EPIT's 28 and Soccol's
37 generated source features are mapped into the canonical 50 slots by name;
unsupported slots remain `NaN`. The generic regime directly generates all 50
slots. The configuration builder
[`prepare_prior_config.py`](corrosion_datasets/datasets/combined_epit_soccol_mpea/prepare_prior_config.py)
freezes the selected EPIT Trial 17 and Soccol Trial 19 training commands and
hashes, so the exact rule scores, formula coefficients, coefficient bounds, and
other selected settings are auditable in
[`combined_prior_config.json`](corrosion_datasets/datasets/combined_epit_soccol_mpea/processed/combined_prior_config.json).

The implementation is opt-in through
[`combined_corrosion.py`](src/tabicl/prior/combined_corrosion.py),
[`missingness.py`](src/tabicl/train/missingness.py), and the combined-prior flag
in [`run.py`](src/tabicl/train/run.py). Checkpoints record both the frozen-prior
and missingness-profile provenance.

### Final training and evaluation boundary

The final run is fixed at 10,000 steps with seed 42, batch size 512,
micro-batch size 4, sequence length 1,024, 999 regression quantiles, and the
standard 128-dimensional TabICL architecture. It saves every 100 steps and
permanent checkpoints every 500 steps, but the intended final model is the
fixed step-10,000 endpoint; MPEA labels must not be used for checkpoint or
configuration selection. The Slurm entry point is
[`train_final.sbatch`](scripts/combined_corrosion/train_final.sbatch), launched
with:

```bash
sbatch scripts/combined_corrosion/train_final.sbatch
```

The final frozen model must use all 4,787 labeled EPIT--Soccol rows as context
and the same 267 MPEA rows as the query partition. Development folds may be
used only for development-time selection or diagnostics; they cannot replace
the MPEA query set. The separated MPEA targets are used only for scoring, not
as query inputs or pretraining data. The later checkpoint and input-handling
diagnostics also scored these same targets; their interpretation is recorded
in Section 15.

The final evaluator uses the same structure as the existing EPIT and Soccol
evaluators: the thin combined wrapper
[`evaluate_final.py`](scripts/combined_corrosion/evaluate_final.py) delegates
task construction, model execution, metrics, and output tables to the shared
[`eval_corrosion_datasets.py`](scripts/eval_corrosion_datasets.py). Its Slurm entry point is
[`evaluate_final.sbatch`](scripts/combined_corrosion/evaluate_final.sbatch):

```bash
sbatch scripts/combined_corrosion/evaluate_final.sbatch
```

By default it compares every checkpoint step currently shared by the combined
CorrPFN and local generic TabICL runs, together with official pretrained TabICL
V2 and CatBoost, on the same frozen query partition. An explicit
`--checkpoint <step>` can restrict the comparison to one saved step; evaluation
does not require training to reach step 10,000. All methods receive the same canonical 50 raw
columns. Only the combined checkpoint retains `NaN` through preprocessing and
uses the learned missing token. The local generic and official
pretrained checkpoints keep original TabICL behavior (Latin feature shuffling,
the default normalization ensemble, numeric mean imputation, and categorical
missing/unseen code `-1`). CatBoost handles numeric `NaN` natively and maps
categorical `NaN` to its fixed missing-category string. The evaluator verifies
the context/query row boundary and refuses a combined checkpoint without the
trained missing-value mode or a generic/pretrained checkpoint that enables it.

At the original 2026-10-13 implementation snapshot, the combined path had
passed focused prior, missingness, estimator-regression, and one-step training
checks. Training and MPEA checkpoint diagnostics subsequently ran; the results,
corrections, and V2 status are recorded below.

## 15. Combined-model results, missingness audit, and V2 fixes (2026-10-14--15)

### Evaluation scope and previous results

All comparisons below keep the outer query partition fixed: 4,787 labeled
EPIT--Soccol context rows and the same 267 MPEA queries. The combined,
generic, pretrained, and CatBoost comparison uses the canonical 50 raw
features. Targets, row identities, and context/query boundaries agree across
models. TabICL regression predictions use the median. Its target scaling is
fitted only on context labels, with mean 337.8953 mV and standard deviation
310.3503 mV.

The corrected checkpoint comparison was saved on 2026-10-14 after the
missing-column inference fix described below. Its latest plotted step is
5,500. At that step:

| Model | Test Spearman | Test RMSE (mV) |
| --- | ---: | ---: |
| Combined CorrPFN, original missingness training | 0.1576 | 733.7 |
| Local generic TabICL | 0.5592 | 596.8 |
| Official pretrained TabICL V2 | 0.3278 | 652.9 |
| CatBoost | 0.5311 | 564.3 |
| EPIT specialist, native 28-feature projection | 0.4005 | 636.3 |
| Soccol specialist, native 37-feature projection | 0.6058 | 570.0 |

The specialist rows come from the separate transfer evaluation saved on
2026-10-13. They use the same context/query rows, but project them to each
specialist's native schema and apply its existing imputation/encoding. They
therefore demonstrate transfer with the mixed-source context, without being
a controlled test of changing only the prior mixture.

The combined curve rises from Spearman 0.1197 at step 500 to 0.2869 at step
1,500, then reaches 0.1892 at step 4,000 and 0.1576 at step 5,500. Its largest
observed value is a descriptive diagnostic, not an authorized checkpoint
selection. At the specialists' fixed step-10,000 endpoints, EPIT scores 0.3924
and Soccol 0.5821 on this transfer task.

The original combined training log ends around step 5,935 with a Slurm
termination, before the planned 10,000 steps. Mean logged training CRPS falls
from 0.5685 over the first 100 logged steps to 0.1766 over the last 100.
This is evidence of learning the synthetic tasks; it does not establish good
real-data transfer or explain why the transfer curve declines.

Sources: [corrected comparison table](corrosion_datasets/analysis/combined_epit_soccol_mpea/checkpoint_comparison_cpu/summary.csv),
[comparison Spearman plot](corrosion_datasets/analysis/combined_epit_soccol_mpea/checkpoint_comparison_cpu/plots/trend_mean_test_spearman.svg),
[specialist transfer table](corrosion_datasets/analysis/combined_epit_soccol_mpea/specialist_transfer_cpu/summary.csv),
[specialist Spearman plot](corrosion_datasets/analysis/combined_epit_soccol_mpea/specialist_transfer_cpu/plots/trend_mean_test_spearman.svg),
and [training-loss audit](corrosion_datasets/analysis/combined_epit_soccol_mpea/implementation_audit_20261014T154809/training_loss_check.json).

### Confirmed inference bug and implemented correction

Previously, ordinary prediction treated any query column containing only
`NaN` as a SHAP exclusion request for generic/pretrained checkpoints. It then
removed that feature from both context and queries, even when context had
useful observations. For this MPEA input, the earlier audit traced 21 actual
model inputs despite an evaluator count of 50. The missing-aware combined
checkpoint retained all 50. Which columns disappeared could also depend on
which query rows were predicted together.

Ordinary prediction now retains entirely missing query columns and uses each
checkpoint's usual missing-value processing. Generic/pretrained models impute
or encode them using context-fitted preprocessing; combined preserves `NaN`
for its learned missing representation. Feature exclusion requires an explicit
boolean `feature_mask`; the SHAP adapters now pass that mask deliberately.
The fix covers regression and classification. Preprocessing also handles a
categorical column that arrives as an all-missing numeric-typed column.
The evaluator records the feature count actually passed during prediction.

The current 50-feature comparison already includes this correction. It
improves the validity of evaluation but does not modify or retrain the old
combined checkpoint. It does not add MPEA-only features to the fixed schema.

Implementation: [regressor](src/tabicl/sklearn/regressor.py),
[classifier](src/tabicl/sklearn/classifier.py),
[preprocessing](src/tabicl/sklearn/preprocessing.py),
[explicit SHAP masks](src/tabicl/shap/_masking.py), and
[regression tests](tests/test_prediction_missing_columns.py).

### Inference settings and controlled missing-value checks

Beyond missing-value handling, the default evaluator gives generic/pretrained
TabICL Latin feature shuffling and the `none`/`power` transform ensemble.
Combined uses fixed feature order and `none`. Although all request eight
estimators, combined's identical configurations reduce to one distinct pass,
whereas the native baselines use eight. Here `none` means no additional
power/quantile transform; standard scaling and outlier processing still occur.

At step 5,500, matching fixed order, the `none` setting, and one distinct pass
gives Spearman 0.4488 for generic, 0.2567 for pretrained, and 0.1576 for
combined. Missing-value processing and trained architectures still differ.
Thus the inference settings account for part of the plotted gap, but not all
of it. These controls did not change the default evaluator configuration.

The combined inputs contain approximately 48.8% missing cells in context and
62.5% in queries. A second diagnostic holds the combined checkpoint, all rows,
and all observed values/category codes fixed. It fills only missing cells,
using context-derived numeric means and categorical code `-1`:

| Combined input treatment | Test Spearman | Test RMSE (mV) |
| --- | ---: | ---: |
| Preserve original missing values | 0.1576 | 733.7 |
| Fill query missing values only | 0.1531 | 656.5 |
| Fill context missing values only | 0.2507 | 647.7 |
| Fill missing values in both | 0.3389 | 661.3 |
| Change observed categorical codes to legacy encoding, retain all NaNs | 0.1577 | 733.6 |

The first imputation trial scored 0.3365 but also changed some observed
categorical codes. The controlled 0.3389 result removes that confound.
Missing-value handling therefore matters substantially, including in context;
the categorical-code difference alone has negligible effect in this check.
Query-only filling improves RMSE but not Spearman. These results do not prove
that the learned missing token is incorrectly implemented or that additional
query masking will solve the transfer gap. Imputation also changes the input
distribution and the fitted feature scales.

The audit found no further confirmed feature/target alignment or evaluation
bug. A four-table synthetic check spanning all three source regimes produced
exactly equal finite outputs through the model's training and inference paths.
These checks cannot rule out every implementation issue. Combining three
useful priors remains a possible source of transfer mismatch, not an established
explanation: combined also changes missing-value representation, feature
grouping, and per-source training exposure relative to the specialists.

Sources: [matched-setting results and input traces](corrosion_datasets/analysis/combined_epit_soccol_mpea/implementation_audit_20261014T154809/results.json),
[controlled missing-value results](corrosion_datasets/analysis/combined_epit_soccol_mpea/missingness_audit_20261015T000223/results.json),
[forward-path check](corrosion_datasets/analysis/combined_epit_soccol_mpea/implementation_audit_20261014T154809/forward_parity.json),
and [audit findings](corrosion_datasets/analysis/combined_epit_soccol_mpea/implementation_audit_20261014T154809/audit_summary.json).
Both audit directories include `reproduce.py` and per-query predictions.

### What missingness means in training and inference

There are separate sources of missing inputs. The informed generators use
preprocessed source profiles; they do not simply copy raw real rows with all
their original NaNs. Mapping their native generated schemas into 50 columns
leaves unsupported features missing. On top of that, the original augmentation
artificially hides values using joint missingness patterns from real context
rows. An 80% per-table draw enables that augmentation; enabled tables sample
patterns per row for both synthetic context and synthetic queries.

In real inference, missing values come from the real context and query data
or absent schema columns. No training augmentation is applied during
prediction. The V2 change below adds a further training-only mechanism to
represent a feature being available in context but absent from an entire query
dataset. A column such as pH remains in the shared schema even when all its
query values are unknown.

### Implemented V2 training policy and launch status

Here V2 names the second combined training recipe; it is distinct from the
official pretrained TabICL V2 baseline. The agreed policy changes the original
training augmentation as follows:

| Input part | V2 treatment |
| --- | --- |
| Observed composition values, in context and queries | Protected from both empirical and additional query masks. Already missing or unsupported values remain missing. |
| Non-composition context features | Keep the original source-derived artificial masking; receive no additional query-column mask. |
| Non-composition query features | Keep source-derived masking, then independently hide each column with probability 0.8 across all query rows in that synthetic table. |
| Synthetic targets | Generated before masking and unchanged by it. |

All 26 non-composition slots are eligible, including environmental,
preparation, and categorical features. The extra 0.8 probability is per
table/column, not per cell, and is separate from the existing 0.8 per-table
empirical augmentation probability. Context is therefore not guaranteed to be
complete. Observed composition protection applies to both masking mechanisms.

The flags are `--missingness_query_column_probability 0.8` and
`--missingness_preserve_observed_composition True`. The trainer passes the
actual context/query boundary, logs context and query masking fractions, and
records the new settings in checkpoints. The prior mixture, formulas, target
standardization, and model architecture are unchanged. This changes two
masking aspects together, so it does not isolate the effect of query masking
from composition protection.

Launch a fresh run with:

```bash
sbatch scripts/combined_corrosion/train_final_v2.sbatch
```

The [V2 launcher](scripts/combined_corrosion/train_final_v2.sbatch) uses
`checkpoints/combined_epit_soccol_mpea/final_v2_seed42`, refuses a directory
already containing step checkpoints, and delegates to the original launcher
with the new flags enabled. The original `train_final.sbatch` still defaults
to the old masking policy. V2 retains the planned 10,000-step seed-42 schedule.
For evaluation, explicitly pass the V2 directory as `--candidate-run` and a
separate `--output-dir`; the evaluator otherwise defaults to the original run.
The outer MPEA queries must stay fixed.

The readiness check recorded **56 passing tests**, including a small actual
training step with the new masking policy, checkpoint provenance, missing-value
inference, and the combined evaluation boundaries. The launcher dry-run and
shell syntax checks passed. These were CPU and launch-configuration checks,
not a completed V2 training experiment. No V2 checkpoint or performance result
was present when this update was written; no Slurm submission is claimed here.

Implementation: [augmentation](src/tabicl/train/missingness.py),
[trainer integration](src/tabicl/train/run.py),
[training flags](src/tabicl/train/train_config.py), and
[masking and training tests](tests/test_missingness_augmentation.py).

### Deferred changes and limits of interpretation

`reference_type` is available for 1,437 real context rows, but both informed
generators leave that slot missing. Generic SCM tasks do not encode a physical
reference-electrode offset either. Including the column therefore does not
establish learned SCE/AgAgCl conditioning. Generating reference labels and
making synthetic targets depend on them remains deferred, as agreed. Existing
analytical rule scores and target components are standardized; this update
does not treat their intermediate scores as calibrated mV or insert an
uncalibrated mV offset into those formulas.

MPEA-only features remain excluded, and the 50-column source-supported schema
is retained. No electrode, formula, prior-weight, or feature-grouping change
was made as part of V2. The missing-value experiments are diagnostics, not an
adopted imputation policy or a demonstrated cure for underperformance.

All reported scores use the frozen outer queries rather than substituting
development folds. Their labels have now been inspected for checkpoint and
input-handling diagnostics. V2 was motivated after that inspection, so a later
comparison on the same MPEA rows is a follow-up result, not independent
confirmation on a previously untouched test set. No checkpoint or input
treatment is promoted as the final model from the diagnostic maxima above.
