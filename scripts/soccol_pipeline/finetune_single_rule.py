"""SOCCOL single-rule, 100% informed continuation of generic 10k."""
from scripts.combined_corrosion.finetune_single_rule import main

if __name__ == '__main__':
    main(default_prior='soccol')
