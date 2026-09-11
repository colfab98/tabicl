# Test coverage

The active tests cover the final CorrPFN v7 implementation, its shared
dependencies, and the original TabICL interfaces. See
[the pipeline guide](../scripts/epit_pipeline/README.md) and
[main3.tex](../main3.tex) for the model and evaluation protocol.

## Final EPIT workflow

| File | Coverage |
| --- | --- |
| `test_epit_pipeline_empirical_scm_target.py` | Empirical physical features, SCM input standardization, MLP/tree target heads, target mixing, regression routing, four-parameter search, current launchers, profile-integrity checks, the single-worker requirement, and isolated training overrides. |
| `test_epit_pipeline_optuna.py` | Study and artifact checks, selected-trial provenance, fixed development folds, baseline/final evaluation, normalization, and checkpoint locks. |
| `test_epit_pipeline_splits.py` | Composition grouping and the current split geometry. |
| `test_epit_target_rule_calibration.py` | Rule coefficients, valid feature terms, and exclusion of held-out targets from method-aware transformations. |
| `test_epit_feature_profile.py` | The current feature assets, missingness, method categories, and asset-integrity checks. |
| `test_epit_feature_generator.py` | Shared physical-feature sampling, family selection, and the retained empirical-feature generator paths. |
| `test_epit_composition_profile.py` | Composition assets, sampling and closure, family probabilities, shared target-rule formulas, and configuration parsing. |

Some of these files contain compatibility checks alongside current coverage.
They stay intact because their shared functions and supported code paths remain
in the repository. The v7 tests directly verify the final SCM–EPIT behavior.

## Shared regression, evaluation, and optional features

- `test_regression_prior.py` covers continuous regression targets and retained
  prior functions, including compatibility branches for earlier corrosion tasks.
- `test_catboost_corrosion_eval.py` covers the retained CatBoost adapter,
  evaluator arguments, repeated-split wrapper, and model-specific routing.
- `test_magpie_features.py` covers optional descriptors and their disabled path.
  Magpie is disabled in the selected v7 model but remains supported by the code.
- `test_optuna_pitting_magpie_prior_search.py` and
  `test_optuna_pitting_fixed_pren_prior_search.py` cover the shared helper modules
  still used by the current pipeline. Their filenames do not make all their
  coverage obsolete.

## Original TabICL tests

`test_numpy_inputs.py`, `test_string_input.py`, `test_sklearn.py`, and
`__init__.py` are retained unchanged during this cleanup. They cover the original
input handling and sklearn interfaces. These tests instantiate TabICL estimators;
running the complete suite requires the corresponding model weights and runtime
resources. The preprocessing-only check can run without loading a model.

## Running the current local checks

From the repository root, use the project environment:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_epit*.py tests/test_optuna*.py \
  tests/test_catboost_corrosion_eval.py tests/test_regression_prior.py \
  tests/test_magpie_features.py \
  tests/test_sklearn.py::test_transform_to_numerical_preserves_dataframe_column_order
```

This covers the project checks and original preprocessing without running the
weight-dependent estimator suite. To run the complete suite in an appropriately
provisioned environment, use `.venv/bin/python -m pytest tests/`.

## Archived tests

The separate v4 Fe/Ni softmax and v5 EPIT-only pipeline suites are preserved at:

- `/home/fcolanto/old_tabicl/tests/test_epit_pipeline_fe_ni.py`
- `/home/fcolanto/old_tabicl/tests/test_fe_ni_softmax_composition.py`
- `/home/fcolanto/old_tabicl/tests/test_epit_pipeline_empirical_features.py`

Before archiving the v5 suite, its useful profile-integrity and worker-setting
checks were moved into the v7 suite and adapted to the current mode. Historical
direct-prior tests were archived with their implementation in the preceding
scripts cleanup. Older implementation source remains where it is still shared;
pruning those branches later should include reviewing their remaining tests.
