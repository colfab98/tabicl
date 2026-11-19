# Combined single-rule selected pairs

Training reuses the current EPIT and Soccol single-rule generators and original generic stream. Each informed table chooses one rule, draws from that rule's retained parents, and uses the standalone perturbation strength of 0.3. EPIT retains 70 input columns and Soccol 58; both retain their existing preprocessing, missingness handling and feature shuffling.

The mixture halves each specialist's informed proportion: **15.13% EPIT, 14.55% Soccol, 70.32% generic**. Source groups are mixed within batches; rows from the two informed datasets are not combined into one table. Architecture, optimizer, batch sizes and schedule come from the standalone launch plans, with mismatches outside dataset-specific parameters rejected. The learning-rate horizon (`--scheduler-total-steps`, default 10000) is independent of the stopping step (`--max-steps`, default 10000). Resume checks the original recipes, configuration and source identities and restores stream state.

From the repository root:

```bash
sbatch scripts/combined_corrosion/train_selected_pairs_single_rule.sbatch
```

The job requests two GPUs and 47 hours. The launcher starts one distributed training process per GPU; the global batch size stays at 512 tables. The default checkpoint directory is `checkpoints/combined_selected_pairs/combined_selected_pairs_single_rule_v1`. To resume that run:

```bash
sbatch scripts/combined_corrosion/train_selected_pairs_single_rule.sbatch --resume
```

Evaluation selects **one dataset at a time**, with context and queries from that dataset only. MPEA is the default. EPIT/Soccol delegate directly to their existing evaluators. All use five-fold exploratory CV, native TabICL inference, the same three baselines (generic, pretrained TabICL v2, CatBoost), checkpoint snapshots/deletion handling, fold summaries and trend plots.

```bash
PYTHONPATH=src:. .venv/bin/python -m scripts.combined_corrosion.evaluate_selected_pairs \
  --dataset mpea \
  --checkpoint all --checkpoint-scope all --max-checkpoint-step 3000 \
  --device cpu \
  --output-dir corrosion_datasets/analysis/combined_corrosion/selected_pairs_through_3000_cv5
```

Use `--dataset epit` or `--dataset soccol` for secondary evaluations. The selected dataset is appended to the output folder name, e.g. `selected_pairs_through_3000_cv5_mpea`. `--checkpoint-scope common` restricts evaluation and plots to shared steps; `all` retains each model's available steps. The default step interval is 500; use `--checkpoint-step-interval 0` for every saved checkpoint. `--include-latest-checkpoint` includes each run's latest checkpoint within the requested range even when it falls outside the interval. Output directories must be empty; use a new output root when repeating an evaluation.

MPEA reads the hash-locked 267-row input export referenced by `mpea_corrosion/processed/splits_v1/split_manifest.json`. The 50 recorded feature columns and their missing values are preserved on load; native inference removes constant columns and fits preprocessing on context. No EPIT/Soccol rows supply MPEA context. Fold 1 uses all 214 development rows as context and the frozen 53 final-test rows as queries. Folds 2–5 partition development with seed 42, using the complement as context, matching the current EPIT/Soccol exploratory protocol. These diagnostic folds are not the frozen Optuna folds. **The five-fold mean is exploratory, not a frozen final-test score**; only fold 1 preserves that final evaluation partition. No split, coefficient fit or retained-pair selection is changed by evaluation.

`--dataset rebar` uses the reviewed v2 inputs (65 predictors, including 31 composition columns) and the same exploratory CV convention as Soccol/MPEA: the 56 original outer queries plus four seed-42 folds of 38 accepted development rows. Final evaluation uses only the frozen 56 queries with 152 reviewed context rows. The output suffix is `_rebar`. See [the dataset audit and processing rules](../../corrosion_datasets/datasets/rebar_pitting/README.md). Original-paper verification and source-overlap screening remain incomplete, so results cannot yet establish independent transfer.
