# Supplement to main4.tex: target mixing and coefficient variation

Recorded: 2026-09-26; updated with completed training results on 2026-10-01;
Soccol extension added on 2026-10-04.
Working notes for a later merge into [main4.tex](main4.tex).
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
| Separate 2,000-step Optuna experiment | Completed: 12 unique strength/seed cells, plus one retained failed historical trial. |
| Predictive result | The 80% setting has the highest mean at 2,000 steps, but its small and seed-dependent gain does not establish a reliable improvement over fixed coefficients. |
| Several nodes running the same new study concurrently | **Not supported by the current runner.** Shared journal storage exists, but the runner permits only one active process. |
| Failed/constant SCM component rejection | Discussed, but not implemented by this experiment. |
| Adaptive λ, grouped bootstrap, joint λ/variation tuning | Discussed only. |

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
