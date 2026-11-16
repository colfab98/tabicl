"""Opt-in 100% informed ablation; preserve the existing mixture's identity."""
import hashlib
from pathlib import Path

from .epit_selected_pairs import SelectedPairPriorMixture


class InformedOnlySelectedPairPrior(SelectedPairPriorMixture):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_ratio = 1.
        self._config.update(
            selected_pair_effective_informed_ratio=1.,
            selected_pair_allocation='all global groups informed; same retained-pair sampler',
            selected_pair_informed_only_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
