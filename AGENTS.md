# Communication

- Keep answers very short, simple, and directly scoped to the question. Lead with the answer, avoid repetition, and include only essential evidence. Do not add testing plans, staged workflows, implementation details, or adjacent recommendations unless explicitly requested.

# Evaluation boundary

- A frozen final model must always be evaluated on the frozen outer final-test/query partition. Development folds may be used as evaluation queries only for Optuna or development-time model selection.
- Context-size diagnostics may change which development rows are supplied as labeled context, but they must keep the outer final-test/query rows fixed and must never substitute a development fold as the final-model query set.

# MPEA workflow alignment

- Keep future MPEA splitting, formula fitting, Optuna, training, and evaluation aligned with the established EPIT and Soccol procedures. Check their implementations before adapting a step.
- Use the frozen MPEA split in `corrosion_datasets/datasets/mpea_corrosion/processed/splits_v1/` for the MPEA specialist workflow. Formula fitting and model selection use development rows only.
