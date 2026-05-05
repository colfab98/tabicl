# TabICL Corrosion-Informed Prior Report

## Method: Generic Versus Informed Prior

The original generic prior treats tabular prediction as a broad synthetic
learning problem. It samples artificial datasets from generic structural causal
models:

- `mlp_scm`
- `tree_scm`

The `mix_scm` setting is not a third generator. It is a probabilistic selector
over `mlp_scm` and `tree_scm`. With the current default fixed hyperparameters,
generic `mix_scm` samples `mlp_scm` and `tree_scm` with probabilities
`(0.70, 0.30)`.

For each generated dataset, the continuous synthetic target is converted into a
classification target through `Reg2Cls`. The dataset is then filtered and
retried if it fails basic validity checks, including feature filtering through
`delete_unique_features(...)` and class/train-test sanity checks through
`sanity_check(...)`. The effective training distribution is therefore the raw
SCM distribution plus target conversion plus post-generation filtering.

This generic setup is useful for broad tabular generalization, but it does not
encode the recurring structure of corrosion datasets: material chemistry,
exposure environment, electrochemical response, history, process state, and
interventions.

The informed prior keeps the same generic machinery but adds a corrosion
inspired transformation before `Reg2Cls`. The base synthetic task is still
produced by the original SCM prior. The generated features and target are then
modified so that some tasks contain corrosion-like dependency structure.
Informed mode can also use its own MLP/tree sampling probabilities. The current
default `informed_mix_probs` are `(0.90, 0.10)`, although experiments can
override this through CLI arguments.

Relevant implementation points:

- `src/tabicl/prior/dataset.py`
  - `apply_informed_structure(...)`
  - `apply_informed_physical_marginals(...)`
  - `prior_type` behavior for `mix_scm`, `informed_scm`, and `hybrid_scm`
- `src/tabicl/prior/prior_config.py`
  - default informed-prior hyperparameters
- `src/tabicl/train/train_config.py`
  - CLI exposure for informed-prior arguments
- `src/tabicl/train/run.py`
  - forwarding of informed-prior arguments into SCM fixed hyperparameters

There are three relevant training modes:

| mode | behavior |
|---|---|
| `mix_scm` | Generic baseline. Samples from the original generic MLP/tree synthetic prior mixture. |
| `informed_scm` | Samples an MLP/tree base prior using informed-mode probabilities, then always applies the informed corrosion structure. |
| `hybrid_scm` | For each synthetic subgroup, samples whether informed mode is active with probability `--informed_prior_ratio`; generic subgroups use `mix_probs`, informed subgroups use `informed_mix_probs` and receive the informed structure. |

The ratio in `hybrid_scm` is stochastic. It controls the expected fraction of
informed synthetic subgroups, not an exact fixed fraction of all generated
tasks.

The code also supports direct `mlp_scm`, direct `tree_scm`, and `dummy` modes,
but the corrosion-prior experiments discussed in this report focus on
`mix_scm`, `informed_scm`, and `hybrid_scm`.

CLI overrides for informed-prior behavior are applied when training generates
prior data on the fly. If `--prior_dir` is set and pre-generated prior data is
loaded from disk, those generation-time prior overrides are not used for new
data generation in that run.

## Informed Prior Components

### Feature Block Structure

Corrosion features are not exchangeable anonymous columns. Alloy composition,
pH, chloride concentration, exposure duration, inhibitor dose, and
electrochemical potential represent different functional families. The informed
prior therefore splits synthetic columns into coarse conceptual blocks:

- `material`
- `environment`
- `electrochem`
- `history`
- `intervention`

The allocation is controlled by `informed_block_allocation`, with weights given
in material, environment, electrochem, history, intervention order. The weights
are normalized internally, so they define relative allocation rather than
absolute counts. Zero weights are allowed, and blocks with zero assigned width
are not created. For small synthetic tables, low-weight blocks may also be
absent after rounding.

The block widths are approximate. The implementation computes fractional target
counts from the normalized weights, floors them, and then assigns remaining
columns by largest fractional remainder. This means the realized block
composition is close to the requested allocation, but not an exact percentage
for every synthetic task.

Within each block, the prior injects shared latent variation:

```text
X_block = (1 - alpha) * X_block + alpha * z_block
```

`alpha` is controlled by `informed_feature_block_strength`. This represents the
idea that variables inside a corrosion family often move together or act as
proxies for a shared latent condition. Environmental variables may jointly
reflect solution aggressiveness. Electrochemical variables may jointly reflect
passive-film stability or corrosion tendency. Material descriptors may jointly
reflect alloy family or microstructural resistance.

This shared latent coupling is applied only to blocks with more than one
feature. A one-column block can still participate in later target interactions,
but it does not receive within-block correlation because there is no within-block
pair to correlate.

In simple terms, each block receives a hidden shared condition. At
`alpha = 0`, the columns receive no shared block signal. At `alpha = 0.20`, each
column remains mostly itself, but about 20% of its value comes from the shared
block condition. Very large `alpha` values would make features inside the block
too similar, so the audited recommendation is soft-to-moderate rather than
strong.

`alpha` is clipped by the implementation to stay below `1.0`, preventing the
shared latent component from fully replacing the original feature values.

The block structure is injected before `Reg2Cls`. During `Reg2Cls`, features are
standardized and randomly permuted by default. Therefore the final model input
can still contain block-induced correlations, but the correlated columns are not
guaranteed to remain contiguous in input order.

### Material-Environment Interaction

Corrosion resistance is conditional. A material that performs well in one
environment may fail in another; chloride concentration, pH, temperature, and
electrolyte chemistry can change the ranking of materials. The informed prior
therefore can add an extra material-environment interaction term to the
synthetic target.

This term is used only when both material and environment blocks are present.
The block summaries are simple arithmetic means over the synthetic columns in
each block, computed after the within-block latent coupling step. The prior does
not identify specific columns as pH, chloride, alloy content, or temperature;
those names are conceptual interpretations of the synthetic blocks.

```text
coupled = tanh(material_mean * environment_mean)
y = y + beta * coupled
```

`beta` is controlled by `informed_interaction_strength`. The product creates an
additional target contribution that depends on the combination of material-like
and environment-like block states. It does not replace the base SCM target, and
it does not imply that the whole synthetic task is interaction-only.

The `tanh(...)` bounds the raw coupled signal to `[-1, 1]`, which prevents the
material-environment product itself from exploding. The final magnitude of the
added term is still controlled by `beta`, which is not clipped by the
implementation. If `beta = 0`, this extra interaction term has no effect.

The intended learning signal is that important corrosion behavior may live in
cross-block relationships. For example, material-like features may matter more
under aggressive environment-like conditions than under mild ones, even though
the synthetic columns are not explicitly labeled with real physical names.

### Electrochemical Proxy Behavior

The informed prior also allows the material-environment interaction to perturb
the electrochemical block. This encodes the idea that electrochemical
measurements are often downstream observations of the material-environment
system.

In implementation, this is not a separate electrochemical mechanism. When
material and environment blocks are both present, the prior computes the same
`coupled` signal used for the target interaction. If an electrochemical block is
also present, that signal is added uniformly to all electrochemical columns:

```text
X_electrochem = X_electrochem + 0.5 * beta * coupled
```

The target receives the coupled material-environment term directly; the
electrochemical block is not used as a mediator that then causes the target.
The implementation also does not use the computed electrochemical block mean for
this perturbation.

This perturbation only exists when material, environment, and electrochemical
blocks are all present. If the electrochemical block is absent because its
allocation is zero or rounded away, no electrochemical columns are perturbed. If
`informed_interaction_strength` is zero, this perturbation also disappears
because it uses the same strength parameter as the target interaction.

The prior does not implement electrochemical equations or different formulas
for potential, current density, resistance, EIS, Tafel behavior, or passive film
kinetics. It only makes electrochemical-like columns statistically connected to
the material-environment state. If physical marginals are enabled, this
perturbation occurs before the electrochemical columns may be remapped into
broad potential/current/resistance-like marginal shapes.

### History Dependence

Corrosion can be path-dependent. Exposure duration, previous surface state,
previous corrosion rate, previous potential, and inhibitor depletion can all
matter. The informed prior therefore supports autoregressive history behavior:

```text
h_t = rho * h_(t-1) + (1 - rho) * h_t
```

This transform is applied to the synthetic row order, not to explicit physical
time stamps or grouped time-series conditions. It runs only when a history block
exists and the generated table has enough rows for an autoregressive update.

`rho` is controlled by `informed_history_strength` and clipped by the
implementation to stay below `1.0`. Larger values make each history row depend
more strongly on the previous synthetic row.

After the history features are smoothed, the prior adds a fixed target
contribution:

```text
y = y + 0.2 * mean(history_features)
```

There is no separate target-strength parameter for this `0.2` coefficient.
`informed_history_strength` controls the autoregressive smoothing of the
history features; it does not directly scale the history contribution added to
the target.

The corrected dataset audit supports keeping history as a possible corrosion
motif, but not as a strong universal assumption. Most constructed benchmark
tasks are ordinary row-wise tabular tasks, not true time-series tasks. The only
usable time-series evidence came from the mooring steel OCP table and was
limited.

### Conditional Intervention Behavior

Interventions such as inhibitors, coatings, treatments, and process controls
should not be modeled as simple global main effects. Their effect depends on
the environment in which they are applied. The informed prior therefore supports
a conditional intervention term:

```text
y = y - gamma * sigmoid(intervention) * relu(environment)
```

This term is applied only when both intervention and environment blocks exist.
The `intervention` and `environment` values in the formula are simple block
means over synthetic columns, computed before any optional physical marginal
transforms. They are not explicitly identified inhibitor, coating, dose, pH, or
chloride columns.

`gamma` is controlled by `informed_intervention_strength`. The implementation
does not clip this value or force it to be positive. Under the intended positive
setting, the term reduces the target when the environment block mean is
positive. If `gamma` were set negative, the same formula would instead increase
the target.

If the environment signal is mild or negative, `relu(environment)` is zero and
the intervention term does not change the target:

```text
mild environment:
relu(environment) = 0
y = y - gamma * sigmoid(intervention) * 0
y = y
```

In an aggressive environment, `relu(environment)` is positive and the
intervention gate can reduce the target under positive `gamma`:

```text
aggressive environment:
relu(environment) > 0
y = y - gamma * sigmoid(intervention) * positive_value
```

`sigmoid(intervention)` is a continuous gate, not a binary no/yes inhibitor
indicator. The intended qualitative behavior is:

```text
mild environment + intervention-like signal:
target is approximately unchanged by this term

aggressive environment + weak intervention-like signal:
target changes little from this term

aggressive environment + strong intervention-like signal:
target is reduced by this term
```

The corrected dataset audit found intervention/process evidence to be sparse,
so intervention should remain weak and should not receive a large default
feature allocation.

### Physical Marginal Feature Distributions

The first informed-prior version encoded dependency structure but did not
impose realistic one-feature value distributions. It did not force pH into
`0-14`, chloride into positive/log-scaled ranges, or composition-like material
columns to sum to roughly 100%.

A second optional layer adds broad physical marginal feature shapes:

```text
--informed_physical_marginal_prob
--informed_physical_marginal_profile
```

This layer runs only inside informed-mode synthetic datasets, after the
structural informed-prior transformations. The default probability is `0.0`,
preserving previous behavior unless the feature is explicitly enabled.
`informed_physical_marginal_prob` is clipped to `[0, 1]`. Profiles named
`false`, `none`, `off`, or `disabled` skip the transform, while unknown profile
names raise an error.

The broad `corrosion_broad` profile can create feature shapes such as:

- pH constrained to `0-14`
- chloride/salinity as positive log-scale variables
- temperature in broad Celsius-like ranges
- electrochemical potentials in broad voltage-like ranges
- current density and resistance as positive log-scale variables
- exposure time and cycle count as positive, skewed variables
- material composition-like groups that sum to approximately 100%
- material fractions/properties
- prior damage values in `[0, 1]`
- intervention binary/category/dose-like variables

These are broad general-knowledge distributions, not fitted empirical
quantiles from the benchmark datasets. They are implemented as rank-based
remappings of existing synthetic columns: each column is converted to empirical
quantiles and then mapped into a broad physical-looking marginal family. This
preserves the column's ordering while changing its scale and marginal shape; it
does not draw independent physical measurements from fitted data distributions.

Material composition behavior is also conditional. A composition-like subset is
created only when the material block has at least two columns and a random
condition passes. The selected subset is softmax-scaled to sum to `100` before
later processing, while the rest of the material block receives
fraction/property-like marginals.

The ordering matters. Material-environment, history, and intervention target
terms are computed before physical marginal remapping, so those target effects
use the pre-marginal synthetic values. The physical marginal layer changes the
feature values afterward.

After these marginals are created, `Reg2Cls` can still convert some columns to
categorical form, remove outliers, standardize every feature column, randomly
permute features, and pad to the maximum feature count. Therefore this layer
does not preserve literal physical units such as pH `0-14` or original chloride
concentration at model input. It mainly injects broad rank, skew, boundedness,
positivity, discreteness, and compositional structure before the final TabICL
feature preprocessing.

### Difference From `col_feature_group`

The informed prior's feature groups are synthetic semantic blocks in the data
generator: material, environment, electrochemical, history, and intervention.
The model-side `col_feature_group` mechanism is an architectural embedding
mechanism that groups feature values before creating column embeddings.

These mechanisms are related but not identical:

- The informed prior says: generate tasks where columns have block-like
  corrosion structure.
- The model column grouping says: represent algorithmically grouped feature
  values jointly during embedding.

The model-side grouping is not semantic. It does not know which generated
columns are material, environment, electrochemical, history, or intervention
columns, and it is not given labels such as "this column is pH" or "this column
is alloy composition".

The default model behavior uses `col_feature_group="same"` with group size `3`.
In that mode, grouping is based on circular shifted feature views, not simple
adjacent-column chunks. A separate `"valid"` mode can group by padding and
reshaping, but that is not the same thing as the informed-prior semantic block
allocation.

The ordering also matters. Informed-prior blocks are created during synthetic
data generation, before `Reg2Cls`. `Reg2Cls` randomly permutes features by
default, so columns that came from the same informed block are not guaranteed to
remain adjacent or grouped together when the model-side column embedder runs.

The two mechanisms can still be compatible in a broad sense: the prior can
create correlated/block-structured feature behavior, and the model can represent
small groups of feature values jointly. But any complementarity is indirect and
depends on the final feature order, grouping mode, and grouping size.

## External Dataset Analysis

The external corrosion datasets were analyzed in:

```text
projects/tabicl/corrosion_datasets/analysis/
```

Relevant files:

- `feature_group_schema.md`
- `column_group_map.md`
- `leakage_policy.md`
- `structural_analysis_results.md`
- `structural_analysis_results.json`
- `parameter_decision.md`
- `prior_implications.md`
- `scripts/analyze_structure.py`

The analysis did not train TabICL, did not tune against DatacorTech test
performance, and did not select a final model from benchmark metrics. It did
fit simple diagnostic target probes, such as ridge CV interaction probes and
target-association summaries, but those were used as structural diagnostics
rather than as model-selection evidence. The analysis asked higher-level
questions:

- Which feature groups appear often?
- How much within-block numeric dependence is present?
- Is material/environment grouping supported?
- Are intervention/process labels target-associated?
- How much evidence exists for path/history effects?

The datasets were used in a limited way:

1. Columns were manually mapped, using the shared schema, into coarse groups:
   `material`, `environment`, `history`, `intervention`, `electrochem`,
   `target`, and `metadata`.
2. The analysis counted how often those groups appeared across downloaded
   corrosion tables.
3. Within each group, numeric dependence was estimated using pairwise Spearman
   correlations.
4. Cross-block numeric dependence was estimated for selected group pairs,
   including material-environment.
5. Material-environment coupling was probed by comparing simple ridge models
   using material and environment features against ridge models that also
   included material x environment product terms.
6. Numeric target associations were estimated with Spearman correlations, and
   categorical target associations were estimated with eta squared.
7. History/path dependence was checked using the available time-series mooring
   steel OCP data and lag-1 autocorrelation.
8. Intervention/process effects were checked through associations from
   processing or treatment-like fields. In the corrected result file, the
   usable intervention association evidence comes from AM-MPEA `AM process`
   associations with three electrochemical targets.

The broad pre-audit findings were:

- The datasets supported separate material, environment, electrochemical
  response, history/process, and intervention-like blocks.
- Within-block dependence was usually moderate rather than extreme.
- Material-environment coupling was scientifically sensible and sometimes
  useful, but simple interaction probes were mixed.
- History/path dependence is scientifically important, but the downloaded
  time-series evidence was narrow.
- Intervention/process variables were present and sometimes informative, but
  too sparse and dataset-specific to justify a large universal intervention
  strength.
- `informed_prior_ratio` cannot be estimated directly from these datasets
  because it controls how often informed synthetic tasks appear during training.

The interaction probes should be read conservatively. They select a small set
of material and environment columns using target association before
cross-validation, then compare ridge models with and without product terms.
They are useful diagnostics for whether interaction structure is plausible, but
they are not clean predictive validation and do not calibrate an exact
interaction strength.

The physical-marginal layer is also separate from this analysis. The structural
analysis mostly informs block structure, interaction, history, intervention,
and prior-ratio ablation ranges. It does not estimate empirical physical
marginal distributions for pH, chloride, temperature, current density,
composition, or other corrosion-like feature families.

The conclusion from this stage was that the informed prior should remain
moderate rather than stronger everywhere.

## Dataset-Derived Informed-Prior Values

The corrected structural-analysis values are stored in:

```text
projects/tabicl/corrosion_datasets/analysis/structural_analysis_results.json
```

They should be read as dataset-derived design values for ablation, not as
calibrated physical constants and not as DatacorTech-selected hyperparameters.
The external corrosion datasets estimate broad structural evidence: block
dependence, material-environment coupling, limited history evidence, and sparse
intervention/process association.

### Corrected Structural Defaults

| parameter | dataset-derived default | suggested range | evidence from collected datasets | reading |
|---|---:|---:|---|---|
| `informed_feature_block_strength` | `0.25` | `0.20-0.35` | within-block numeric dependence: `n=18`, mean `0.343`, median `0.302`, q75 `0.397` | Use a soft-to-moderate shared block signal. The evidence supports block structure but not aggressive coupling. |
| `informed_interaction_strength` | `0.25` | `0.10-0.35` | material-environment correlation: `n=8`, mean `0.213`, median `0.147`, q75 `0.240`; ridge interaction delta: `n=14`, mean `-0.046`, median `-0.000`, q75 `0.025` | Material-environment interaction is scientifically sensible, but the diagnostic gain is mixed and near zero on median. Keep interaction moderate. |
| `informed_history_strength` | `0.25` | `0.00-0.50` | absolute lag-1 Spearman from mooring OCP: `n=3`, mean `0.194`, median `0.068`, q75 `0.282` | History is a corrosion motif, but the usable time-series evidence is narrow and weak. Avoid a strong universal autoregressive prior. |
| `informed_intervention_strength` | `0.10` | `0.05-0.20` | intervention/process target association: `n=3`, mean `0.236`, median `0.225`, q75 `0.264` | Intervention/process labels can be informative, but the evidence is sparse and categorical. Keep the conditional intervention effect weak. |
| `informed_prior_ratio` | `0.50` | `0.25-0.75` | no direct structural estimate: `n=0` | The dataset collection does not estimate how often informed synthetic tasks should appear. Treat this as an ablation knob. |

The dataset evidence argues against simply making the informed prior stronger.

### Values Not Directly Estimated

The structural-analysis file does not directly estimate every informed-prior
setting.

`informed_block_allocation` is not derived from the strength table above. It is
driven mainly by feature-composition evidence: the collected and constructed
corrosion tasks are material/composition heavy, with smaller environment and
history/process components, very sparse intervention inputs, and electrochemical
values often appearing as targets or downstream measurements. The final
material-heavy allocation is therefore reported in the later
**Dataset-Derived Recommended Setting** section.

`informed_physical_marginal_prob` is also not estimated from empirical
corrosion-feature marginal distributions. The physical marginal layer is a
hand-specified broad prior over pH-like, concentration-like, time-like,
electrochemical-like, compositional, and intervention-like shapes. Its
probability should be treated as a conservative design choice rather than a
measured value.

## Benchmark And Task Quality

The current benchmark contains 12 primary corrosion tasks. In the current
`6660.out` comparison, every checkpoint evaluates 12 tasks across 8 models, and
all 12 tasks succeed for every model. The saved JSON artifact for this run has
zero task errors.

The current task construction uses explicit scalar parsing, sparse-feature
filtering, and task-quality flags inside `scripts/eval_corrosion_datasets.py`.
Sparse numeric-like features are dropped when their finite ratio is too low;
sparse categorical features are dropped when their non-missing ratio is too
low; high-cardinality categorical features are also filtered. This makes weak
or shortcut-prone tasks visible in the output instead of silently treating all
tasks as equally informative.

The clearest current example is:

```text
electrochemical_metrics_alloys__crevice_corrosion_temp__tcrev_oc_max
```

In the current run, this task completes with 11 retained features and the
`small_n` quality flag. Historical versions of the task had 23 retained
features and failed with `TypeError("Invalid value '0.0' for dtype 'str'")`,
but those earlier failures are superseded for the current result file.

### Current Task Set

The 12 tasks in the common-checkpoint comparison are:

| task | n | features | target | quality flags |
|---|---:|---:|---|---|
| `electrochemical_metrics_alloys__pitting_potential__epit_mv_sce_avg` | 760 | 21 | `Epit, mV (SCE) Avg.` | none shown |
| `electrochemical_metrics_alloys__repassivation_potential__ave_erp_vsce` | 151 | 19 | `Ave. Erp VSCE` | none shown |
| `electrochemical_metrics_alloys__pitting_temp__tpit_oc_avg` | 105 | 12 | `Tpit, oC Avg.` | none shown |
| `electrochemical_metrics_alloys__crevice_corrosion_temp__tcrev_oc_max` | 67 | 11 | `Tcrev, oC Max` | `small_n` |
| `electrochemical_metrics_alloys__heas_ecorr_icorr_ipass_rate__ecorr_mv_sce_avg` | 51 | 14 | `ECORR, mV (SCE) Avg.` | `high_cardinality_categorical_features`, `small_n` |
| `mpea_corrosion__sheet1__pitting_potential_mv_vs_sce` | 335 | 32 | `Pitting potential (mV vs SCE)` | none shown |
| `am_mpea_corrosion__am_mpea_corrosion_database_v3__pitting_potential_mv_vs_sce` | 58 | 20 | `Pitting potential (mV vs. SCE)` | `high_cardinality_categorical_features`, `small_identity_like_material_table`, `small_n` |
| `steel_mortar_corrosion__01_carbonation__corrosion_rate_of_steel` | 180 | 12 | `Corrosion Rate of Steel` | none shown |
| `steel_mortar_corrosion__02_chloride__corrosion_rate_of_steel` | 95 | 11 | `Corrosion Rate of Steel` | none shown |
| `316l_pitting_passivity__epit_epass_descriptors__epit_x` | 955 | 2 | `Epit_x` | `few_unique_feature_patterns`, `fixed_material_low_condition_grid`, `very_few_features` |
| `nace_nist_corr_data__corr_data_database__rate_mm_yr_or_rating` | 2000 | 1 | `Rate (mm/yr) or Rating` | `coarse_heterogeneous_corpus`, `few_unique_feature_patterns`, `very_few_features` |
| `mooring_steel_seawater__ocp_r4_s31_omax__ocp_v` | 62 | 2 | `ocp_V` | `small_n`, `small_time_series_condition_table`, `very_few_features` |

### Task Caveats

Not all tasks should be treated as equally informative.

`am_mpea_corrosion__...__pitting_potential` is small (`n=58`) and includes
identity-like fields such as alloy name, alloy formula, AM process, and phase
labels. In the current all-common-checkpoint result, its AUROC is exactly
`1.0` in 99 of 104 model/checkpoint rows, including every model at step 3350.
This task is useful as a sensitivity check but should not dominate model
ranking.

`316l_pitting_passivity__...__epit_x` has many rows (`n=955`) but only two
features and only 5 unique feature patterns. Under the current random split,
all 239 test feature patterns are also present in training. This is a
low-condition-grid task, not a broad corrosion generalization test.

`mooring_steel_seawater__...__ocp_v` is a very small time/condition table
(`n=62`) with two retained features. Under the current random split, all 16
test feature patterns are also present in training. It is useful for discussing
history/path dependence, but random row splitting can make it too easy.

`nace_nist_corr_data__...__rate_mm_yr_or_rating` is large (`n=2000`) but very
coarse after safe feature filtering, with only one retained feature in the
current task. The retained feature has only 10 unique feature patterns, and all
500 test feature patterns are also present in training under the current random
split. It should not dominate weighted aggregate metrics simply because it has
many rows.

The strongest model-ranking signal should come from tasks with enough samples,
enough feature diversity, and no obvious identity or repeated-pattern shortcut.
Weak or suspect tasks should be reported with explicit caveats.

## Dataset-Derived Recommended Setting

If the informed setting is chosen from the corrected corrosion dataset insights
rather than from selecting the best checkpoint result, the best default is a
conservative, material-heavy hybrid prior.

The strongest dataset-derived signal is feature composition, not exact numeric
strength. The downloaded corrosion tables and constructed corrosion tasks are
dominated by material/composition descriptors, with smaller environment and
history/process components. Electrochemical measurements are often targets or
downstream responses, so they should not occupy a large default input-feature
block. Intervention/process variables exist, but they are sparse and mostly
categorical.

Recommended setting:

| parameter | recommended value | reason |
|---|---:|---|
| `prior_type` | `hybrid_scm` | Keep both generic and corrosion-shaped synthetic tasks. |
| `informed_prior_ratio` | `0.50` | Not estimated by datasets; use a balanced design default, not a claimed calibrated value. |
| `mix_probs` | `0.70 0.30` | Keep the generic MLP/tree mixture unchanged. |
| `informed_mix_probs` | `0.70 0.30` | Avoid adding an extra uncalibrated preference for MLP-only informed tasks. |
| `informed_block_allocation` | material `0.70`, environment `0.18`, electrochem `0.10`, history `0.02`, intervention `0.00` | Matches the material-heavy benchmark feature composition and avoids giving intervention a default block. |
| `informed_feature_block_strength` | `0.25` | Corrected within-block dependence is moderate: mean about `0.343`, median about `0.302`. |
| `informed_interaction_strength` | `0.20` | Material-environment interaction is scientifically sensible, but corrected interaction-probe gains are mixed and near zero on median. |
| `informed_history_strength` | `0.10` | Path dependence is a corrosion motif, but usable time-series evidence is narrow and weak. |
| `informed_intervention_strength` | `0.05` | Intervention/process evidence is sparse; keep this as a weak conditional effect. |
| `informed_physical_marginal_prob` | `0.20` | Use broad physical marginal shapes occasionally, but do not let hand-specified ranges dominate training. |
| `informed_physical_marginal_profile` | `corrosion_broad` | Use the existing broad profile if marginal transforms are enabled. |

Launch arguments:

```text
--prior_type hybrid_scm
--informed_prior_ratio 0.5
--mix_probs 0.7 0.3
--informed_mix_probs 0.7 0.3
--informed_block_allocation 0.70 0.18 0.10 0.02 0.0
--informed_feature_block_strength 0.25
--informed_interaction_strength 0.20
--informed_history_strength 0.10
--informed_intervention_strength 0.05
--informed_physical_marginal_prob 0.20
--informed_physical_marginal_profile corrosion_broad
```

This setting is not selected because `v12`, `v13`, `v14`, `v16`, or `v18` won a
specific evaluation checkpoint. Past results only show that stronger informed
structure is not obviously necessary and that softer/material-heavy variants
are plausible. The dataset-based argument is:

```text
material/composition should dominate the synthetic feature blocks;
material-environment coupling should exist but remain moderate;
electrochemical inputs should be much smaller than material inputs because they are often targets or downstream measurements;
history should be weak because most benchmark tables are static row-wise data;
intervention should have no default feature block because intervention columns are rare;
physical marginal shapes should be occasional, broad, and non-calibrated.
```

Corrected-analysis supported ranges:

| parameter | corrected-analysis reading |
|---|---|
| `informed_feature_block_strength` | soft-to-moderate, around `0.20-0.35`; corrected default suggestion `0.25` |
| `informed_interaction_strength` | moderate, around `0.10-0.35`; corrected default suggestion `0.25`, with the recommended material-heavy setting using `0.20` |
| `informed_history_strength` | weak or ablated, around `0.00-0.50`; corrected default suggestion `0.25`, with `0.00` worth testing and the recommended setting using `0.10` |
| `informed_intervention_strength` | weak and conditional, around `0.05-0.20`; corrected default suggestion `0.10`, with the recommended setting using `0.05` |
| `informed_prior_ratio` | not estimable from these datasets; keep as an ablation knob, e.g. `0.25-0.75` |

### Primary Metric Choice

The primary aggregate should be the unweighted mean across tasks. In the
evaluation script, `mean_*` metrics are plain task averages, while
`weighted_mean_*` metrics are weighted by each task's full `n_samples`. The
weighted mean is useful as a sensitivity metric, but row count is not the same
as task reliability in this benchmark.

Large tasks can be structurally weak. NACE has many rows but very coarse
features. The 316L task has many rows but very few unique feature settings.
Small tasks such as AM-MPEA or mooring can be too easy for structural reasons.
Weighting by row count can therefore emphasize dataset size rather than
benchmark quality. In the current 12-task benchmark, NACE alone contributes
`2000 / 4819 = 41.5%` of the sample-weighted aggregate. NACE plus 316L
contribute `2955 / 4819 = 61.3%`, even though both are flagged for few unique
feature patterns.

The current comparison shows that weighting can change the headline. At step
3350, unweighted mean balanced accuracy ranks `v12` first (`0.8372`), while
weighted mean balanced accuracy ranks `v18` first among all models (`0.7088`).
The weighted result is useful, but it should not replace the task-level
unweighted mean as the primary ranking.

## Experiment Versions

The baseline is a reproduced generic `mix_scm` run. It keeps the original
generic MLP/tree prior mixture and disables all informed-prior structure. The
informed tests then add corrosion-derived structure in stages, so the effect of
each informed component can be interpreted more cleanly.

All new runs use the same core small-model training setup:

```text
--max_steps 10000
--batch_size 512
--micro_batch_size 4
--lr 1e-4
--scheduler cosine_warmup
--warmup_proportion 0.02
--gradient_clipping 1.0
--device cuda
--dtype float32
--np_seed 42
--torch_seed 42
--embed_dim 128
--col_num_blocks 3
--row_num_blocks 3
--icl_num_blocks 12
```

All runs also use 2 GPUs through `torchrun --standalone --nproc_per_node=2`,
CPU prior generation, `prior_n_jobs=8`, `dataloader_num_workers=4`, and
`dataloader_prefetch_factor=4`.

### New Test Matrix

| run | prior type | purpose | informed structure |
|---|---|---|---|
| `tabicl_s1mini_generic_baseline` | `mix_scm` | Reproduced generic baseline. | No informed-prior structure. |
| `tabicl_s1mini_generic_test1` | `hybrid_scm` | Block-structure-only test. | Adds material-heavy corrosion-like feature blocks with moderate shared block signal. |
| `tabicl_s1mini_generic_test2` | `hybrid_scm` | Main structural informed-prior test. | Adds block structure plus material-environment interaction. |
| `tabicl_s1mini_generic_test3` | `hybrid_scm` | Full structural prior without physical marginals. | Adds block structure, material-environment interaction, weak history, and weak intervention. |
| `tabicl_s1mini_generic_test4` | `hybrid_scm` | Full dataset-derived config with physical marginals. | Adds the complete recommended dataset-derived setting, including occasional broad physical marginal shapes. |
| `tabicl_s1mini_generic_test5` | `hybrid_scm` | Stronger informed-contribution stress test. | Uses the dataset-derived structure but increases block, interaction, and physical-marginal contribution. |

### Baseline: Reproduced Generic Prior

`tabicl_s1mini_generic_baseline` is the clean reproduced generic baseline. It
uses the original `mix_scm` prior and keeps the generic MLP/tree mixture at:

```text
--prior_type mix_scm
--mix_probs 0.7 0.3
```

All informed-prior controls are disabled:

```text
--informed_prior_ratio False
--informed_feature_block_strength False
--informed_interaction_strength False
--informed_history_strength False
--informed_intervention_strength False
--informed_physical_marginal_prob False
--informed_physical_marginal_profile False
```

This run should be treated as the reference for whether any corrosion-informed
synthetic structure improves downstream corrosion-task performance over the
generic prior.

### Test 1: Block Structure Only

`tabicl_s1mini_generic_test1` tests the most basic informed-prior hypothesis:
whether corrosion-like feature blocks help by themselves.

It uses `hybrid_scm` with a balanced expected informed-task ratio:

```text
--prior_type hybrid_scm
--informed_prior_ratio 0.5
--mix_probs 0.7 0.3
--informed_mix_probs 0.7 0.3
```

The informed feature allocation is material-heavy:

```text
--informed_block_allocation 0.70 0.18 0.10 0.02 0.0
```

This corresponds to:

| block | allocation weight |
|---|---:|
| material | `0.70` |
| environment | `0.18` |
| electrochem | `0.10` |
| history | `0.02` |
| intervention | `0.00` |

Only the shared feature-block signal is active:

```text
--informed_feature_block_strength 0.25
--informed_interaction_strength False
--informed_history_strength False
--informed_intervention_strength False
--informed_physical_marginal_prob False
--informed_physical_marginal_profile False
```

This isolates whether the material-heavy corrosion block structure is useful
without adding target interactions, history, intervention, or physical marginal
feature remapping.

### Test 2: Block Structure Plus Material-Environment Interaction

`tabicl_s1mini_generic_test2` adds the main cross-block corrosion motif:
material-environment interaction.

It keeps the same hybrid setup and material-heavy allocation as Test 1:

```text
--prior_type hybrid_scm
--informed_prior_ratio 0.5
--mix_probs 0.7 0.3
--informed_mix_probs 0.7 0.3
--informed_block_allocation 0.70 0.18 0.10 0.02 0.0
```

It enables moderate block structure and moderate material-environment coupling:

```text
--informed_feature_block_strength 0.25
--informed_interaction_strength 0.20
```

History, intervention, and physical marginals remain disabled:

```text
--informed_history_strength False
--informed_intervention_strength False
--informed_physical_marginal_prob False
--informed_physical_marginal_profile False
```

This test asks whether adding corrosion-relevant material-environment target
coupling improves over block structure alone. The interaction strength is kept
at `0.20`, rather than made stronger, because the dataset analysis supports the
interaction motif but does not provide strong calibration evidence for a large
effect.

### Test 3: Full Structural Prior Without Physical Marginals

`tabicl_s1mini_generic_test3` tests the full structural informed prior while
leaving the physical marginal layer disabled.

It uses the same hybrid setup and material-heavy allocation:

```text
--prior_type hybrid_scm
--informed_prior_ratio 0.5
--mix_probs 0.7 0.3
--informed_mix_probs 0.7 0.3
--informed_block_allocation 0.70 0.18 0.10 0.02 0.0
```

The structural parameters are:

```text
--informed_feature_block_strength 0.25
--informed_interaction_strength 0.20
--informed_history_strength 0.10
--informed_intervention_strength 0.05
--informed_physical_marginal_prob False
--informed_physical_marginal_profile False
```

This test isolates the incremental effect of weak history and weak intervention
on top of the main block-plus-interaction structure.

One caveat is important: because the intervention block allocation is `0.00`,
`--informed_intervention_strength 0.05` will usually have no practical effect.
That is intentional and consistent with the dataset-derived recommendation,
which avoids giving intervention a default input block because intervention
features are rare in the constructed corrosion tasks.

### Test 4: Full Dataset-Derived Config With Physical Marginals

`tabicl_s1mini_generic_test4` is the full dataset-derived recommended setting.

It keeps the balanced hybrid prior and material-heavy allocation:

```text
--prior_type hybrid_scm
--informed_prior_ratio 0.5
--mix_probs 0.7 0.3
--informed_mix_probs 0.7 0.3
--informed_block_allocation 0.70 0.18 0.10 0.02 0.0
```

It uses the recommended soft-to-moderate structural strengths:

```text
--informed_feature_block_strength 0.25
--informed_interaction_strength 0.20
--informed_history_strength 0.10
--informed_intervention_strength 0.05
```

It also enables the broad physical marginal layer conservatively:

```text
--informed_physical_marginal_prob 0.20
--informed_physical_marginal_profile corrosion_broad
```

This test asks whether occasional broad corrosion-like feature shapes help when
added to the structural informed prior. The physical marginal probability is
kept low because the physical marginal profile is hand-specified and not fitted
from empirical corrosion-feature marginal distributions.

### Test 5: Stronger Informed-Contribution Stress Test

`tabicl_s1mini_generic_test5` uses the same basic design as Test 4, but increases
the informed contribution.

It keeps the same hybrid ratio, MLP/tree mixture, and material-heavy allocation:

```text
--prior_type hybrid_scm
--informed_prior_ratio 0.5
--mix_probs 0.7 0.3
--informed_mix_probs 0.7 0.3
--informed_block_allocation 0.70 0.18 0.10 0.02 0.0
```

The stronger settings are:

```text
--informed_feature_block_strength 0.35
--informed_interaction_strength 0.30
--informed_history_strength 0.10
--informed_intervention_strength 0.05
--informed_physical_marginal_prob 0.30
--informed_physical_marginal_profile corrosion_broad
```

This is not the primary dataset-derived recommendation. It is a stress test for
whether stronger informed structure helps or starts to overconstrain the
synthetic prior. It should be interpreted against Test 4, not as a replacement
for the conservative dataset-derived setting.



## Limitations And Caveats

The informed prior encodes plausible corrosion motifs, but it can also inject
the wrong feature proportions or overemphasize structures that are rare in the
benchmark inputs.

The main mismatch found in the audit was feature allocation. The older informed
prior split synthetic features approximately as:

| block | older informed-prior allocation |
|---|---:|
| `material` | 27.7% |
| `environment` | 27.0% |
| `electrochem` | 20.0% |
| `history` | 15.0% |
| `intervention` | 10.3% |

The constructed corrosion evaluation tasks had aggregate input-feature
composition:

| block | eval-task feature count | eval-task share |
|---|---:|---:|
| `material` | 146 | 73.0% |
| `environment` | 29 | 14.5% |
| `history` | 24 | 12.0% |
| `intervention` | 1 | 0.5% |
| `electrochem` | 0 | 0.0% |

This mismatch gives a plausible explanation for why informed variants can
underperform the clean generic baseline at some checkpoints:

- too little material/composition capacity relative to the benchmark tasks
- too much electrochemical-feature capacity despite electrochemical variables
  usually being targets or leakage-prone post-measurements
- too much intervention structure despite almost no intervention features in
  the evaluation tasks
- too much row-order/history structure for mostly non-time-series tables
- interaction strength chosen from weak, noisy interaction evidence

History evidence is especially weak. The only true time-series signal came
from the mooring steel OCP table:

| condition | n | lag-1 Spearman |
|---|---:|---:|
| `S=31 T=2 Omax` | 21 | 0.495 |
| `S=31 T=32 Omax` | 21 | 0.018 |
| `S=31 T=17 Omax` | 20 | -0.068 |

The median absolute lag-1 Spearman is about `0.068`. This supports history as a
possible motif, but not a strong universal autoregressive component.

Interaction probes are directionally useful but not strong calibration
evidence. The corrected median interaction improvement is approximately
`0.000` R2. This supports testing material-environment interaction as a motif,
but it does not prove that any specific `informed_interaction_strength` is the
right magnitude.

The physical marginal layer is hand-specified, not inferred from empirical
benchmark marginals. Because `Reg2Cls` standardizes features afterward, the
model mostly sees distribution-shape information, not literal physical units.

The training jobs are checkpoint comparisons under a 24-hour Slurm limit, not
completed 10000-step training runs.

## Training Reproducibility And 2-GPU Efficiency Check

A separate short training-log check was run on three `s1mini_generic` jobs:

- `v1` versus `v2`: repeated single-GPU generic training, used as a determinism/reproducibility check.
- `v1` versus `v3`: single-GPU versus 2-GPU parallel training, used to check whether parallel training changes optimization efficiency at the same optimizer step.

This check should be read as a CE-loss trajectory diagnostic, not as benchmark model-selection evidence. The question is whether the training dynamics remain comparable before looking at downstream corrosion-task metrics.

| comparison | compact metric | reading |
|---|---:|---|
| `v1` vs `v2` | mean CE difference `v2 - v1 = +0.0021`; 50-step rolling CE MAE `0.0047` | The run is not bitwise/replay deterministic, but the smoothed CE trajectory is effectively reproduced. |
| `v1` vs `v3` | mean CE difference `v3 - v1 = -0.0026`; last-100 mean CE `0.8080` for `v3` versus `0.8153` for `v1` | The 2-GPU run does not converge more slowly at equal optimizer steps; if anything, it is slightly lower in late training. |
| `v1` vs `v3` wall-clock | mean `train_time` per step, excluding first step: `15.72s` for `v1` versus `7.67s` for `v3` | The 2-GPU run is about `2.05x` faster per optimizer step in this log. |

The 2-GPU curve is only slightly worse at the very beginning: over steps `1-100`,
mean CE is `1.1934` for `v3` versus `1.1766` for `v1`, a difference of
`+0.0167`. This early difference disappears by mid-training. From steps
`201-1000`, the 100-step CE averages for `v3` are consistently below the
corresponding `v1` averages. Using a 50-step rolling CE, the `0.85` threshold is
reached at step `476` for `v1` and step `402` for `v3`.

Therefore, there is no evidence from these logs that switching from single-GPU
training to 2-GPU parallel training reduces optimization efficiency when CE is
compared at the same optimizer step. The parallel run preserves the loss
trajectory while substantially reducing wall-clock time.

All three logs contain two isolated non-finite CE entries:

| run | isolated `ce=nan` steps |
|---|---|
| `v1` | `221`, `984` |
| `v2` | `317`, `662` |
| `v3` | `474`, `915` |

These isolated non-finite entries do not derail the surrounding CE trajectory,
but they should still be fixed or explained before treating longer training
curves as final evidence.


The downstream evals are more important for interpreting future config
experiments. Across 84 task/checkpoint rows from steps `250, 500, 700, 750,
800, 850, 900`, the three nominally equivalent models produce different
task-level metrics. The best model by balanced accuracy is:

| model | balanced-accuracy wins |
|---|---:|
| `v1` | 16 |
| `v2` | 16 |
| `v3` | 52 |

For MCC, `v3` is also most often best, with 51 wins. AUROC is less aligned:
`v2` is best in 36 rows, `v3` in 29 rows, and `v1` in 19 rows. Therefore, the
evidence does not support treating `v3` as a universally better model. The
safer conclusion is that the three runs define a non-trivial replicate-variance
band for this training and evaluation setup.

This variance should be considered when selecting the best model after changing
training configs. A config variant should not be called better merely because
it wins one checkpoint or improves a small number of task metrics. Improvements
need to be judged against the variance observed among same-config replicates.

The replicate spread is largest at early checkpoints and smaller later:

| checkpoint | mean balanced-accuracy SD across tasks |
|---:|---:|
| 250 | `0.054` |
| 500 | `0.031` |
| 700 | `0.019` |
| 750 | `0.024` |
| 800 | `0.013` |
| 850 | `0.014` |
| 900 | `0.016` |

This implies that early-checkpoint wins are especially noisy. Around steps
`800-900`, a balanced-accuracy difference of only one or two points can still
be within ordinary replicate variation, especially if it comes from a small
number of unstable tasks. Larger, repeated gains across checkpoints, metrics,
and robust tasks are more meaningful.

The largest replicate variance appears mostly on small or structurally fragile
tasks, including crevice corrosion temperature, repassivation potential,
chloride mortar corrosion, pitting temperature, AM-MPEA, mooring OCP, and the
low-condition-grid 316L task. These tasks are useful sensitivity checks, but
they can amplify small training differences and should not dominate conclusions
about a config change.





## Superseded Notes

Earlier step-2000 reporting used an output where the crevice corrosion task
failed for every model, leaving 11 successful tasks. That result is superseded
for primary reporting by the `6660.out` comparison, where the crevice task is
fixed and all 12 tasks succeed.

Earlier audit notes considered step-250, step-500, and the old step-2000
comparisons informative. They remain useful historical checks, but the results
section in this report uses the corrected all-common-checkpoint `6660.out`
comparison.

Earlier prose suggested that v14 was the best local softer-prior variant at a
2500-step comparison. The corrected common-checkpoint comparison gives the
step-2500 win to `v13` by unweighted mean balanced accuracy (`0.8342`), with
`v14` close behind (`0.8245`) and `v16` close behind (`0.8239`). The older note
should be treated as historical context only.

Earlier dataset-derived parameter values should be read as heuristic ablation
values, not calibrated estimates. The corrected parser audit softened the
evidence for exact strengths, especially history and intervention.

The original `v18` description as a code-level block-allocation change remains
important for reproducibility, but the allocation is now exposed as a CLI
argument through `--informed_block_allocation`.
