# Project Report: TabICL Background


## TabICL In One Sentence

TabICL is a tabular foundation model that predicts test targets from a labeled training table in a single forward pass:

```text
y_test = model(X_train, y_train, X_test)
```

The model weights are not updated for the downstream dataset. Instead, the training rows are part of the model input. This is the meaning of in-context learning in TabICL: the model uses the labeled examples in the context to infer the feature-target relationship for the current table.

This makes TabICL different from standard per-dataset training. A gradient-boosted tree or a task-specific neural network learns new parameters for each dataset. TabICL has already learned a general prediction procedure during pretraining, and it applies that procedure to a new dataset through its input context.

In the scikit-learn interface, `fit(X_train, y_train)` mostly stores and preprocesses the context. The main learning-from-context step happens when `predict(X_test)` or `predict_proba(X_test)` runs the pretrained model on the combined training and test table.

## Pretraining On Synthetic Tasks

TabICL is pretrained on many synthetic supervised learning tasks. Each task is a full tabular dataset with features, targets, and a train/test split. During pretraining, the model receives `X_train`, `y_train`, and `X_test`, but not `y_test`. It predicts the held-out targets, and the loss updates the shared model weights.

Across many such tasks, the model learns an algorithm-like behavior: given a small or medium-sized labeled context, identify useful feature-target relationships and apply them to query rows.

The synthetic data generator is therefore central. It defines the distribution of tasks that the model practices on. In prior-data fitted networks, this generator is usually called the prior. It is a prior over whole datasets or tasks, not just a prior over scalar parameters.

A useful prior should generate tasks with realistic diversity:

- different numbers of rows and features;
- continuous and categorical-like columns;
- nonlinear feature-target relationships;
- redundant, irrelevant, and correlated features;
- varying train/test splits and task difficulties.

The exact downstream dataset is not expected to appear in pretraining. The goal is broader: expose the model to enough task structure that it learns a reusable inference procedure.

## Structural Causal Models In The Generic Prior

A structural causal model, or SCM, is one way to generate synthetic tabular tasks. The basic idea is to create hidden causes, transform them through random mechanisms, and then select some variables as observed features and one variable as the target.

The SCM generator first creates a few starting columns. Each starting column has one value per row. These starting columns are sampled directly from random distributions, for example normal or uniform distributions. They are called root variables because they are the first columns in the synthetic data-generation process.

The generator then applies random functions to these starting columns. In TabICL, these functions can be MLP layers or tree-based transformations. The functions create new columns from the starting columns.

Depending on the sampled generator path, the final feature and target selection
can happen in two closely related ways. In causal-mode SCMs, the generator
chooses some generated intermediate columns as the input features `X` and one
generated column as the target `y`. In the current tree-based path, the sampled
causes are used as `X` and the final tree/ensemble transformation output is used
as `y`. In both cases, the synthetic task is defined by random mechanisms that
connect features and target.

In the TabICL code used here, the generic SCM prior has two main generator families:

- MLP-based SCMs, where random neural-network transformations create nonlinear dependencies;
- tree-based SCMs, where random tree or ensemble transformations create piecewise and tree-like dependencies.

The `mix_scm` setting samples between these generator families. This matters because MLP-style and tree-style synthetic tasks encourage different inductive biases. The MLP prior tends to generate smooth nonlinear relationships. The tree prior tends to generate split-based, piecewise relationships that are closer to the behavior of decision trees and gradient-boosted trees.

The SCM generator does not simply sample independent columns. It creates variables that can share latent causes, depend on intermediate variables, and affect the target through nonlinear paths. After generation, the data is postprocessed: features are scaled, some numerical columns can be converted to categorical-like values, uninformative columns can be removed, feature order can be shuffled, and invalid train/test splits can be rejected or retried.

The SCM generator does not only create unrelated random columns. It first creates
starting columns, then uses random MLP or tree functions to create outputs from
them. In causal-mode paths, several selected feature columns can share generated
ancestors. In direct predictive paths, the target is still produced by a random
nonlinear transformation of the feature columns. Either way, the features and
target can have nonlinear relationships.

For classification pretraining, a continuous synthetic target is converted into class labels. For regression pretraining, the target remains continuous. In both cases, the model sees many tasks where the training context contains enough information to predict held-out rows.

## Bayesian-Style Interpretation

The Bayesian interpretation is useful, but it should be stated carefully.

In Bayesian prediction, a prior defines which data-generating processes are plausible before observing data. After observing the training set, the posterior gives more weight to processes that explain the observed examples. Prediction averages over those plausible processes:

```text
p(y_test | X_test, X_train, y_train)
```

TabICL does not explicitly sample from a posterior at inference time. It does not fit an SCM to the downstream dataset, and it does not update its weights. Instead, pretraining teaches a neural network to approximate the posterior-predictive mapping for tasks drawn from the synthetic prior.

This is why the usual description is amortized approximate Bayesian inference. Amortized means the expensive part is learned once during pretraining. Approximate means the transformer forward pass is a learned approximation, not an exact Bayesian computation. Bayesian-style means the prediction is conditioned on the observed context in a way that resembles posterior prediction under the pretraining task prior.

This distinction is important for the project. If the task prior changes, the Bayesian-style inference behavior that the model learns can also change.

## Architecture: From Cells To Row Vectors

TabICL starts from a normal table. Each entry in the table is a cell value:

```text
row i, column j -> one raw value x_ij
```

A raw value can be a number or an encoded category. The model does not keep this value as a single scalar. It maps it into a learned vector, called an embedding:

```text
x_ij -> embedding vector for cell (i, j)
```

This vector is not computed from the cell value alone. TabICL uses the row and column structure of the table when building it. The full architecture can be summarized as:

```text
raw cell values
-> cell embeddings that use column context
-> one vector for each row
-> attention across rows for prediction
```

So yes, part of the architecture is a learned compression step. The many cell values in one row are eventually compressed into one vector for that row.

### Step 1: Make Cell Embeddings Using Column Context

Consider one column, for example column `j`. It contains one value for every row:

```text
x_1j, x_2j, x_3j, ..., x_nj
```

TabICL processes this column across rows. This means the embedding of a cell value such as `x_ij` can depend on the other values in the same column. The model can therefore represent whether the value is typical, extreme, missing, categorical-like, continuous-like, or strongly associated with the target in the training rows.

The result is still cell-level. After this step, each cell has a vector:

```text
raw table:        n rows x d columns x 1 value
embedded table:   n rows x d columns x embedding_dim
```

The word column-wise means that this embedding step is applied by looking down columns, not by treating each cell as an isolated number.

In TabICLv2, several feature values can also be grouped before embedding. This grouping is architectural. The model is not given semantic column names or domain labels. It simply receives grouped feature values so that a cell embedding can include information from a small feature group instead of only one scalar feature.

For training rows, the known target `y_train` is also embedded and added during this stage. This is called target-aware embedding. It helps the model represent how a feature column relates to the target in the current training context. Test rows do not provide target values.

### Step 2: Turn Each Row Into One Vector

After Step 1, every row still contains multiple vectors: one vector per feature cell. For row `i`, this looks like:

```text
cell embedding (i, 1)
cell embedding (i, 2)
...
cell embedding (i, d)
```

The row-wise transformer lets these feature-cell embeddings interact inside the same row. This is how the model can represent feature combinations. For example, feature A may only matter when feature B is large.

Then TabICL summarizes the whole row into one fixed-size vector:

```text
all cell embeddings in row i -> one row vector r_i
```

This is the main compression step. A row with many feature values becomes one learned row vector. That row vector is supposed to keep the information that is useful for prediction.

TabICL uses learnable `CLS` tokens to create this summary. These tokens attend to the feature-cell embeddings in the row. Their outputs are combined to form the final row vector.

After this step, the table is no longer represented as `n rows x d columns`. It is represented as a sequence of row vectors:

```text
r_1, r_2, ..., r_n
```

### Step 3: Use Row Vectors For In-Context Learning

The final transformer works across rows. It receives:

```text
training row vectors + training targets
test row vectors without targets
```

The test row vectors attend to the training row vectors. In simple terms, each test row can compare itself with the labeled training rows and use them as context for prediction.

For classification, the output is class logits or probabilities. For regression, TabICLv2 outputs many target quantiles. These quantiles can be summarized as a mean or median prediction and can also express uncertainty.

The important idea is:

```text
individual cells become cell embeddings;
all cell embeddings in one row become one row vector;
test row vectors use training row vectors to predict targets.
```


## Parameter Updates

During pretraining, TabICL optimizes the parameters of:

- the cell/column embedding layers
- the row attention layers
- the dataset-level train/test attention layers
- the output head that produces class logits or regression quantiles

So the model learns:

```text
how to turn cells into useful vectors
how to combine cells into a row vector
how test rows should attend to training rows
how to output the final prediction
```

The training objective is:

```text
given X_train, y_train, X_test
predict y_test
```

The prediction error on synthetic `y_test` updates all these parameters.

## Prediction For A New Dataset

At prediction time, TabICL receives the preprocessed training rows, their targets, and the test rows. The training set acts as the context. The test rows act as queries. The pretrained model performs one forward pass, or several forward passes if ensembling over feature shuffles is enabled, and returns predictions for the test rows.

No downstream gradient updates are required. Optional KV caching can store parts of the computation for the training context, which speeds up repeated prediction on the same training set.

The practical result is that TabICL behaves like a learned inference engine for tabular data. The input context specifies the current task, while the pretrained weights encode what was learned across synthetic pretraining tasks.

## Why The Prior Matters

The architecture defines how TabICL processes a table. The prior defines what kinds of table structures it learns from during pretraining.

This makes the prior a direct way to influence the model inductive bias. If the generic synthetic prior contains mostly broad, domain-agnostic structures, the model learns a broad tabular inference procedure. If a downstream domain repeatedly contains additional structure, the prior can be modified to include that structure during pretraining.

The later project-specific sections build on exactly this point: changing the synthetic task generator changes the kinds of patterns that the in-context learner is trained to recognize.

## Informed Prior: Activation And Ratio

First control parameter: `informed_prior_ratio`. This parameter matters when
`prior_type` is `hybrid_scm`.

There are three relevant modes:

```text
mix_scm
  generic baseline; no informed corrosion transform

informed_scm
  every generated subgroup uses the informed corrosion transform

hybrid_scm
  each synthetic subgroup randomly chooses generic or informed mode
```

For `hybrid_scm`, `informed_prior_ratio` is the probability that a synthetic
subgroup uses informed mode. Example:

```text
informed_prior_ratio = 0.5

for each subgroup:
    50% chance -> generic SCM task
    50% chance -> generic SCM task + informed corrosion transform
```

This is an expected fraction, not an exact fixed count. Because the choice is
random, one batch can contain slightly more or fewer informed tasks.

Why this exists: the informed prior is meant to bias training toward corrosion
structure without completely removing generic tabular diversity. A ratio of zero
recovers generic behavior. A ratio of one makes every relevant generated task
informed. Values between zero and one mix both distributions.

Important implementation detail: informed mode still starts from the generic SCM
base. The ratio controls whether the extra informed transformation is applied
after `X, y` generation. It does not switch to a separate hand-built corrosion
simulator.

## Informed Prior: MLP/Tree Base Mixture

Next sampling controls: `mix_probs` and `informed_mix_probs`. These decide which
generic SCM family is used as the base task before any informed transform.

```text
mlp_scm
  random neural-network-style SCM base

tree_scm
  random tree/ensemble-style SCM base
```

For generic `mix_scm` tasks, `mix_probs` controls the probability of using
`mlp_scm` versus `tree_scm`. For informed tasks, `informed_mix_probs` can control
that same choice separately.

Implementation idea:

```text
if task is generic:
    choose mlp_scm or tree_scm using mix_probs
    generate X, y

if task is informed:
    choose mlp_scm or tree_scm using informed_mix_probs
    generate X, y
    apply informed corrosion transform
```

This is not itself a corrosion mechanism. It only controls what kind of generic
synthetic task the informed structure is added to. Keeping `informed_mix_probs`
close to `mix_probs` means the informed prior changes corrosion structure
without also changing the MLP/tree balance. Changing it lets informed tasks lean
more toward smooth neural-style synthetic functions or tree-style synthetic
functions.

## Informed Prior: Task Family Sampling

Next sampling control: `informed_task_family_probs`. This applies only after a
task has already been selected as informed. It chooses which informed corrosion
family the generator should create.

```text
normal_corrosion
  material/environment/process/exposure style corrosion task

inhibitor_agent
  descriptor-heavy inhibitor or treatment-agent task
```

Implementation idea:

```text
if informed mode is active:
    choose normal_corrosion or inhibitor_agent using informed_task_family_probs

    if normal_corrosion:
        split columns using informed_normal_block_allocation

    if inhibitor_agent:
        split columns using informed_inhibitor_block_allocation
```

Why this exists: corrosion datasets are not all shaped the same way. Some tables
are mostly material and environment variables. Other tables are dominated by
molecular descriptors of an inhibitor or treatment agent. The task-family choice
lets the informed prior represent both broad patterns without forcing one block
composition onto all informed tasks.

The exact sampling probabilities are a design/analysis choice, not part of this
mechanism explanation.

## Informed Prior: Insertion Point And Block Structure

First thing to document: the informed prior is not a separate generator. The
base synthetic table is still generated by the generic SCM machinery. The
informed logic is inserted after generic `X, y` generation and before the final
`Reg2Cls` / preprocessing step:

```text
generic SCM generates X, y
-> if informed mode is active, assign X columns to corrosion semantic blocks
-> later informed mechanisms use those blocks
-> Reg2Cls / feature preprocessing runs afterward
```

The first informed implementation piece is the normal-corrosion block split.
For a normal corrosion task, the generator takes the anonymous synthetic feature
columns and assigns them to internal semantic blocks:

```text
material
environment
process_history
exposure_duration
temporal_history
direct_intervention
molecular_descriptor
electrochem_control
electrochem_downstream
```

The parameter to explain first is `informed_normal_block_allocation`. It is a
vector of relative weights, in the block order above. The implementation
normalizes the weights, converts them into approximate feature counts for the
current synthetic table, floors the counts, and gives leftover columns to the
largest fractional remainders. So the allocation controls the expected block
composition, but it is not an exact percentage guarantee for every generated
task. Small synthetic tables may miss low-weight blocks entirely.

These blocks are generator-internal. The model does not receive semantic labels
and does not know that a column came from the material or environment block. The
blocks only decide which columns later informed transformations are allowed to
treat together. This matters because the later additions need block membership:
within-block coupling needs to know which columns share a latent block signal,
material-environment target structure needs material and environment blocks,
history behavior needs a temporal-history block, and intervention behavior needs
a direct-intervention block.

Difference from the generic prior: generic SCM columns are anonymous throughout
generation. In the informed normal-corrosion path, the columns are still
anonymous to the model, but the generator temporarily treats them as
corrosion-role blocks before final preprocessing. That is the minimal structural
change that makes the rest of the informed prior possible.

## Informed Prior: Within-Block Coupling

Next parameter: `informed_feature_block_strength`. After the normal-corrosion
block split exists, this parameter controls whether columns inside the same
semantic block share a latent component.

Implementation idea:

```text
for each block with more than one column:
    sample one latent block signal z_block
    X_block = (1 - alpha) * X_block + alpha * z_block
```

Here `alpha` is `informed_feature_block_strength`. At `alpha = 0`, the block
assignment exists but does not change feature values. As `alpha` increases,
columns in the same block contain more of the same latent block signal. The
implementation clips the strength below `1.0`, so the shared signal cannot fully
replace the original SCM features.

The key clarification is that this is meant to spread block-level information
across several columns. It is not mainly saying that one named feature should
increase whenever another named feature increases. Instead, several columns in
the same block become partially informative about the same hidden condition.

Examples: material columns can share information about alloy family or material
state; environment columns can share information about exposure aggressiveness;
process/history columns can share information about treatment or surface state.
Because the signal is spread across a block, later target mechanisms can depend
on a block-level condition rather than on one isolated independent column.

Pairwise correlation is a side effect of this shared latent signal, not the main
claim. The current implementation is deliberately simple and should not be read
as a calibrated physical covariance model between named variables.

This is the first informed step that numerically changes `X`. The previous
block-allocation step only labels columns internally for the generator. Blocks
with only one assigned column are skipped for this step. A one-column block can
still be used later in target mechanisms, but it cannot have within-block
correlation because there is no second column to correlate with.

Difference from the generic prior: generic SCM generation can already produce
correlated features by chance or through shared synthetic causes. The informed
version makes one specific kind of dependence more deliberate: columns assigned
to the same corrosion role softly share latent information before final
preprocessing. The model still does not receive block labels; it only sees the
resulting statistical structure in the table.

## Informed Prior: Block-Level Corrosion Target Mechanism

Next parameter: `informed_interaction_strength`. After columns have been assigned
to corrosion blocks, the informed prior can modify the synthetic target using
block-level corrosion signals.

The generic SCM already creates a target `y`. The informed prior does not replace
that target. It adds an extra corrosion-like target component on top of it.

Implementation idea:

```text
project material block -> material_susceptibility
project environment block -> environment_aggressiveness
project exposure block -> exposure_effect
project process/history block -> process_or_history_modifier

corrosion_drive =
    material_susceptibility
  + environment_aggressiveness
  + material_susceptibility * environment_aggressiveness
  + exposure_effect
  + process_or_history_modifier

y = y + beta * standardized(corrosion_drive)
```

Here `beta` is `informed_interaction_strength`. A larger value makes the
corrosion-drive term contribute more strongly relative to the original generic
SCM target.

The block projections are random projections of the synthetic columns inside a
block. This is important: the generator is not using named columns or a fixed
physical equation. It does not need to know that a specific column is pH,
chloride, chromium content, or exposure time. It only uses the internal block
roles created earlier.

The main hypothesis is that normal corrosion targets often depend on block-level
conditions and cross-block interactions. The material state matters, the
environment state matters, and the effect of the environment can depend on the
material. Exposure and process/history can also shift the response. This term
encourages some synthetic tasks to practice that structure.

Difference from the generic prior: generic SCM targets can depend on features in
arbitrary synthetic ways. The informed target mechanism biases some tasks toward
corrosion-style dependencies: especially material/environment effects and their
interaction. The original SCM target remains present, so the task is still a
broad synthetic tabular task rather than a hand-coded corrosion simulator.

## Informed Prior: Target Family Selection

Next parameter: `informed_target_family`. This selects the interpretation of the
extra informed target component. The default is:

```text
informed_target_family = generic_corrosion
```

In the generic-corrosion family, higher `y` means a stronger corrosion-like
response. The material/environment/exposure/history drive raises the target, and
protective intervention or inhibitor effects subtract from it. This is the
behavior described in the previous target-mechanism section and in the
intervention sections below.

The newer supported setting is:

```text
informed_target_family = pitting_potential
```

This creates an Epit-like pitting-potential target. The sign convention is
different: higher `y` now means a higher breakdown threshold, or stronger
pitting resistance. Protective or passivating material chemistry raises the
target, while aggressive chloride-/pH-/temperature-like environment signals,
material susceptibility under aggressive environment, exposure, and accumulated
history lower it.

Implementation idea:

```text
epit_drive =
    material_passivity
  - environment_aggressiveness
  - material_susceptibility * environment_aggressiveness
  - exposure_effect
  + process_offset
  - history_damage

y = y + beta * standardized(epit_drive)
```

For pitting-potential targets, direct interventions and inhibitor-agent effects
are added when they look protective, because protection raises the synthetic
breakdown threshold:

```text
y = y + gamma * standardized(protection_effect)
```

So the same block system can represent two different corrosion target semantics:
generic corrosion severity, where protection lowers the target, and pitting
potential, where protection raises the target. The model still does not see the
target-family label. It only sees the resulting feature-target statistics in the
synthetic pretraining tasks.

## Informed Prior: Temporal History Behavior

Next parameter: `informed_history_strength`. This applies only when the synthetic
task has a `temporal_history` block. It is meant to add a weak path-dependence
motif to some informed tasks.

Implementation idea:

```text
for temporal-history features:
    h_t = rho * h_(t-1) + (1 - rho) * h_t

y = y + small_history_contribution
```

Here `rho` is `informed_history_strength`. At `rho = 0`, the history block is not
autoregressively smoothed. As `rho` increases, each row in the temporal-history
block depends more on the previous synthetic row. The implementation clips `rho`
below `1.0`, so the previous row cannot fully overwrite the current row.

This is different from ordinary within-block coupling. Within-block coupling
shares information across columns in the same row. Temporal-history behavior
shares information across rows within the history block.

The target also receives a small contribution from the smoothed history features.
So history can matter both as part of the feature table and as a weak target
signal. The history contribution is intentionally small because this is only a
broad path-dependence motif.

Important boundary: this is not a full time-series corrosion simulator. The
synthetic rows are smoothed in their generated row order. The implementation
does not create explicit specimens, grouped trajectories, timestamps, depletion
kinetics, or physical state evolution. It only gives the model practice with the
idea that previous condition/state information can matter for a corrosion
response.

Difference from the generic prior: generic SCM tasks can contain arbitrary row
patterns, but they do not explicitly reserve a history-like block or apply a
corrosion path-dependence transform. The informed prior adds that motif when a
temporal-history block exists.

## Informed Prior: Direct Intervention Effect

Next parameter: `informed_intervention_strength`. This uses the
`direct_intervention` block.

An intervention is something done in the experiment to change the corrosion
response. Examples: adding an inhibitor, adding a coating, changing heat
treatment, polishing the surface, or changing the processing route. An inhibitor
is one specific intervention: a chemical added to slow corrosion.

The generator does not have named columns like `chloride`, `pH`, or
`inhibitor_dose`. Instead it creates latent high/low scores from block
projections:

```text
environment block -> random projection -> environment_score -> sigmoid -> environment_drive
direct_intervention block -> random projection -> intervention_score -> sigmoid -> intervention_gate
```

So yes, after projection this becomes a higher/lower latent-score mechanism:

```text
higher environment_drive = more aggressive synthetic environment
higher intervention_gate = stronger synthetic intervention/protection signal
```

But this is not a raw-column rule. It is not saying one particular input column
is inhibitor dose and larger values of that column always mean more inhibitor.
The score is a synthetic summary of the intervention block.

For normal corrosion tasks under the default `generic_corrosion` target family,
the intervention effect is applied to the target as:

```text
protection_effect = intervention_gate * (0.60 + 0.40 * environment_drive)

y = y - gamma * standardized(protection_effect)
```

Here `gamma` is `informed_intervention_strength`. Environment and intervention
do not need to be correlated with each other. They are two inputs whose
combination affects the target. A more aggressive environment can raise the main
corrosion drive, while a stronger intervention signal subtracts a protective
effect. The protection term is environment-modulated because an intervention can
be more visible under aggressive conditions than under mild conditions.

For the `pitting_potential` target family, the same protection signal has the
opposite sign in `y`, because a protective intervention should raise the
synthetic breakdown threshold rather than lower a corrosion-severity response.

Simple reading for the default generic-corrosion target:

```text
high environment, low intervention  -> high corrosion-like target
high environment, high intervention -> environment pushes up, intervention pushes down
low environment, any intervention   -> lower corrosion pressure, intervention may matter less
```

This is a weak synthetic bias, not a physical law. It teaches that action-like
columns can matter through their combination with exposure conditions, while the
original generic SCM target remains present.

## Informed Prior: Electrochemical Proxy Behavior

Simple idea first: electrochemical-like columns can be proxy measurements of the
same corrosion state that affects the target. They are not always independent
causal inputs. If one synthetic row has a high latent corrosion drive, the
electrochemical proxy columns for that row can shift with that state.

Relevant blocks:

```text
electrochem_control
electrochem_downstream
```

Concrete implementation:

```text
1. Build corrosion_drive for each row
   using material/environment/exposure/process/history blocks.

2. Convert that row-wise drive into a bounded signal:

   electro_signal = tanh(standardized(corrosion_drive))

3. If electrochemical blocks exist, add that signal to those columns:

   X[:, electrochem_block] = X[:, electrochem_block]
                         + 0.5 * beta * electro_signal
```

Here `beta` is `informed_interaction_strength`. The same parameter that controls
how strongly the corrosion drive affects `y` also controls how strongly that
drive appears in electrochemical proxy features.

More concretely:

```text
high corrosion_drive row -> electrochemical proxy columns shift one way
low corrosion_drive row  -> electrochemical proxy columns shift the other way
```

The original electrochemical feature values are not replaced. They are perturbed
by an added row-wise signal. `standardized(...)` keeps the signal scale stable,
and `tanh(...)` bounds extreme values.

This is not saying a specific column is `Ecorr`, `Icorr`, or impedance. The
generator only knows that a column belongs to an electrochemical proxy block.
The model still sees ordinary feature columns with no semantic labels.

Boundary: this is not an electrochemistry simulator. It does not model Tafel
slopes, EIS spectra, passivation kinetics, mixed potentials, or film growth. It
only adds a statistical link between electrochemical-like features and the
latent corrosion state.

Difference from the generic prior: generic SCM generation has no concept of
electrochemical proxy columns. The informed prior can create columns that are
partly downstream/proxy signals of the same latent corrosion drive used in the
target mechanism.

## Informed Prior: Physical Marginal Remapping

Next parameters: `informed_physical_marginal_prob` and
`informed_physical_marginal_profile`. This part is different from the earlier
structural mechanisms. It changes the one-column value distributions of informed
features.

Simple idea: after the informed block and target structure has been added, some
columns can be remapped to look more like broad corrosion variables. The current
`corrosion_broad` profile can use families like these:

```text
environment columns
  -> pH-like bounded values
  -> chloride-like positive log-scale values
  -> concentration-like positive log-scale values
  -> salinity-like positive log-scale values
  -> temperature-like bounded values
  -> bounded or categorical environment variables

material columns
  -> composition-like groups that sum to about 100
  -> fraction-like values
  -> bounded material values
  -> material-property-like positive values
  -> low-cardinality material categories

process / exposure / history columns
  -> process binary variables
  -> process categories
  -> process score-like values
  -> exposure-time-like positive skewed values
  -> cycle-count-like positive integer values
  -> prior-damage-like bounded values

electrochemical columns
  -> potential-like bounded values
  -> current-density-like positive log-scale values
  -> resistance-like positive log-scale values

direct intervention columns
  -> intervention binary variables
  -> intervention categories
  -> dose-like bounded values

molecular descriptor columns
  -> standard-normal-like descriptor values
  -> positive log-scale descriptor values
  -> count-like descriptor values
  -> bounded descriptor values
```

`informed_physical_marginal_prob` controls whether this remapping runs for a
given informed task. `informed_physical_marginal_profile` selects the profile;
the current broad corrosion profile is `corrosion_broad`.

More concrete examples from the implementation:

```text
pH-like
  rank -> piecewise value in roughly 0 to 14
  lower ranks map to acidic values, middle ranks near neutral, upper ranks alkaline

chloride-like
  rank -> log-uniform-like positive range from about 1e-3 to 1e5

generic concentration-like
  rank -> log-uniform-like positive range from about 1e-6 to 1e1

salinity-like
  rank -> log-uniform-like positive range from about 1e-3 to 3.5e1

temperature-like
  rank -> linear bounded range from about -10 to 120

electrochemical potential-like
  rank -> linear bounded range from about -1.5 to 1.5

current-density-like
  rank -> log-uniform-like positive range from about 1e-9 to 1e-1

resistance-like
  rank -> log-uniform-like positive range from about 1e-2 to 1e6

exposure-time-like
  rank -> log-uniform-like positive range from about 1e-2 to 1e5

cycle-count-like
  rank -> log-uniform-like positive integer/count range from about 1 to 1e6

composition-like material group
  selected material columns -> softmax-scaled so the group sums to about 100
```

Implementation idea:

```text
for selected informed feature columns:
    take the existing synthetic column
    convert values to ranks / empirical quantiles
    map those quantiles into a broad physical-looking marginal family
```

So the transform preserves row ordering within a column, but changes the scale
and shape of the values. For example, a ranked synthetic column can be mapped
into a positive log-scaled concentration-like variable, a bounded pH-like
variable, a binary/category-like intervention variable, or a composition-like
material group.

Important boundary: this does not make the target mechanism a physical equation.
The corrosion target mechanisms are computed before this marginal remapping. So
the generator is not literally computing corrosion from final pH, chloride, or
temperature units. This layer mainly injects broad marginal properties such as
positivity, boundedness, skew, discreteness, and composition-like behavior.

Another boundary: final `Reg2Cls` / preprocessing can standardize, transform,
permute, or pad features afterward. So exact physical units are not preserved as
the final model input. The useful signal is that informed tasks sometimes
contain corrosion-like distribution shapes, not calibrated physical units.

Difference from the generic prior: generic SCM columns have generic synthetic
marginals. The informed physical marginal layer can make some columns look more
like corrosion variables before final preprocessing, while still keeping the
underlying task synthetic and broad.

## Informed Prior: Inhibitor-Agent Tasks

For `inhibitor_agent` tasks, the important allocation parameter is
`informed_inhibitor_block_allocation`. Compared with normal corrosion tasks, this
allocation gives much more feature capacity to the `molecular_descriptor` block.

Why this exists: inhibitor datasets often contain many columns describing the
chemical agent itself. These can be molecular descriptors, electronic
descriptors, atom counts, topology descriptors, or other computed chemistry
features. In that kind of table, the intervention is not just a simple dose or
yes/no treatment. The identity and properties of the inhibitor molecule matter.

Implementation idea:

```text
molecular_descriptor block -> descriptor_efficacy
direct_intervention block  -> dose/control signal
environment block          -> environment modifier

inhibitor_effect = descriptor_efficacy * dose * environment_modifier

y = y - gamma * standardized(inhibitor_effect)
```

Here `gamma` is still `informed_intervention_strength`. The effect is subtracted
from the target because the synthetic assumption is that an effective inhibitor
can reduce the corrosion-like response in the default `generic_corrosion` target
family. In the `pitting_potential` target family, the descriptor-dependent
protection term is added instead, because stronger inhibitor protection raises
the synthetic pitting-potential threshold.

The important difference from the normal direct-intervention case is that the
protective effect can depend on the molecular descriptor block. In plain words:

```text
not all inhibitors are equally effective
features describing the inhibitor molecule can help determine efficacy
dose/control and environment still modulate the effect
```

This is still not a molecular corrosion simulator. The generator does not model
adsorption, charge transfer, solubility, surface coverage, or real quantum
chemistry. It creates a synthetic descriptor-dependent efficacy signal from
random projections of the descriptor block.

Difference from the generic prior: generic SCM generation has no concept of a
descriptor-heavy inhibitor table. The inhibitor-agent informed path creates some
synthetic tasks where many columns describe the intervention agent, and where
those descriptors can matter for the target through a conditional protective
effect.
