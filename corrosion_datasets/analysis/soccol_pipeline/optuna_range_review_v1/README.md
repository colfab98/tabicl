# Historical Soccol Optuna range review

We reviewed `soccol_pipeline_optuna_empirical_features_scm_target_v1` using a
temporary copy of its shared journal. The original journal was not modified.
There were **50 completed trials** and five entries still marked `RUNNING`;
only completed trials entered this analysis. All 50 shared one pipeline
fingerprint. We independently matched the 24 locally available result files
to journal scores, parameters and file hashes.

The objective was mean Spearman over the five development folds after 1,000
training steps. No final-query scores were used. The old run used 37 columns,
3,222 development rows for evaluation and observation-weighted sampling from
the old 4,027-row bank. Its settings therefore did not directly establish the
optimum for the corrected workflow.

## Results

The best trial was **19**, with mean development Spearman **0.604646**:
100% informed, MLP share 1.0, formula contribution 0.662955, and composition
perturbation 0.035766. Trial 20 scored 0.604573, so trial 19 was not an isolated
winning configuration.

| Setting | Top 10 completed Soccol trials | Approved new range |
| --- | --- | --- |
| Informed share | All 10 used 100% | Fixed 100%, already configured |
| MLP share | Eight used 0.75 or 1.0; the other two used 0.25 and 0.50 | Continuous 0–1 |
| Formula contribution | 0.581274–0.836408 | Continuous 0–1 |
| Composition perturbation | 0.018839–0.045396 | Continuous 0–0.15 |

The broad MLP/formula ranges allowed the corrected data and generator to
favour different settings. Restricting them to the old winners would assume
more stability than the old search demonstrated. In particular, none of the
30 completed 100%-informed trials used formula contribution below 0.25, so
poorer low-contribution scores from mixed-ratio trials did not establish its
effect under the new fixed ratio.

For composition perturbation, even within the 30 100%-informed trials, the
0.025–0.05 interval contained the best results. Only five such trials tested
values above 0.05, so we recommended retaining the full old 0–0.15 range
rather than concentrating narrowly around 0.036. The approved range reduced
the previous 0–0.30 range; it did not exclude any completed historical trial.

The historical EPIT v8 journal showed a different trend: its best trial used
0.149675, close to the old 0.15 limit, and its top 10 used 0.095352–0.149675.
That supported exploring above 0.15 for EPIT. Soccol's old results did not
provide the same reason for the 0–0.30 extension.

These were descriptive results from an adaptive search with jointly varying
parameters, not controlled measurements of each parameter's effect. Sampling
weights, accepted rows, columns, missing-value handling and fitted coefficients
had since changed. We did not claim an optimum for the new workflow.

## Saved evidence and status

- `completed_trials.csv`: all 50 completed trials, sorted by development score.
- `parameter_summary.csv`: ranges and quantiles for all trials, the 30
  100%-informed trials, and the top 10.
- `summary.json`: journal hashes, historical search space, matching local
  result hashes, EPIT comparison, and current/recommended ranges.

The review initially recorded a recommendation only. After approval, the
Soccol search range was changed to **0–0.15**, while EPIT retained **0–0.30**.
The actual Optuna distribution, saved fingerprint and trial metadata used the
same dataset-specific range. A study bound to the earlier 0–0.30 range could
not silently resume. Final training retained the selected trial's noise value
at all four informed ratios. Coefficient variation remained fixed at 0.8 with
seed 42; the MLP and formula ranges remained 0–1. No training was launched.
