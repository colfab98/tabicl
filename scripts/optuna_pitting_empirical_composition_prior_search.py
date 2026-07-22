#!/usr/bin/env python
"""Run the fixed-PREN Optuna study with empirical alloy compositions only.

This is a strict entry point for the new EPIT study. The implementation remains
in optuna_pitting_fixed_pren_prior_search.py so the legacy and empirical modes
share training, evaluation, and result handling without duplicating that code.
"""

from __future__ import annotations

try:
    from scripts import optuna_pitting_fixed_pren_prior_search as search
except ModuleNotFoundError:
    import optuna_pitting_fixed_pren_prior_search as search


DEFAULT_STUDY_NAME = "pitting_fixed_pren_empirical_composition_v1"


def parse_args(argv: list[str] | None = None):
    args = search.parse_args(
        argv,
        default_n_trials=50,
        default_scheduler_total_steps=10000,
    )
    if args.pitting_composition_mode != "empirical":
        raise ValueError(
            "This launcher only supports --pitting-composition-mode empirical. "
            "Use optuna_pitting_fixed_pren_prior_search.py for legacy studies."
        )
    if args.study_name is None:
        args.study_name = DEFAULT_STUDY_NAME
    return args


def main() -> None:
    search.run_search(parse_args())


if __name__ == "__main__":
    main()
