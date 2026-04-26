# Leakage Policy

This project uses external corrosion datasets to inform broad prior structure, not to tune against DatacorTech performance.

## Safe To Use

- Existence of recurring feature groups: material/composition, environment, processing/history, electrochemical response.
- Qualitative cross-group logic: material-environment interactions, processing/history effects, and within-block correlations.
- Broad schema statistics such as which groups are common across datasets.
- Conservative prior ranges chosen before looking at DatacorTech test performance.

## Risky Or Usually Excluded

- Electrochemical descriptors measured from the same experiment when the target is corrosion status/rate.
- Calculated targets such as passive window if derived directly from Epit and Ecorr.
- Dataset-specific thresholds, pitting labels, or corrosion-rate rating boundaries.
- Source-reference IDs, paper comments, DOI strings, and text that can identify a study rather than a physical condition.
- Post-exposure microscopy or post-failure observations when predicting corrosion risk before outcome.

## Forbidden For Model Selection

- Choosing informed-prior strengths because DatacorTech test AUROC, balanced accuracy, or MCC improves.
- Repeatedly training candidate priors and selecting the winner on the DatacorTech test split.
- Encoding a target distribution or label threshold from DatacorTech into the synthetic prior.

## Recommended Workflow

1. Document external datasets and feature groups.
2. Decide a small prior-ablation grid from cross-dataset structure only.
3. Use validation/CV for threshold and model selection.
4. Preserve one final DatacorTech test comparison for reporting.
