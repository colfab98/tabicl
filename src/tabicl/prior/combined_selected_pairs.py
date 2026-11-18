"""Three native-schema streams built from the existing single-rule specialists."""
import hashlib
import math
from pathlib import Path

from torch.utils.data import IterableDataset

from .original_mixture import OriginalPriorMixture
from .selected_pairs_single_rule import SingleRuleSelectedPairSampler

VERSION = 'combined_selected_pairs_single_rule_v1'


class CombinedSelectedPairPrior(OriginalPriorMixture):
    """Equal mixture of the two specialist recipes, including their generic mass.

    Inherit original batching verbatim. Each source emits the same full-plan
    prefix as its standalone counterpart at the same global step/seed/shape.
    Only source group counts and the resulting group shuffle change.
    """
    def __init__(self, epit, soccol):
        IterableDataset.__init__(self)
        for source in (epit, soccol):
            if not isinstance(source, OriginalPriorMixture) or source._next_batch is not None:
                raise ValueError('Combine fresh original-mixture datasets before iteration.')
            if not isinstance(source.priors[1].sampler, SingleRuleSelectedPairSampler):
                raise ValueError('Both informed sources must use one rule per table.')
        for attr in ('batch_size', 'group_size', 'seed', 'rank', 'world_size', 'return_source_ids'):
            if getattr(epit, attr) != getattr(soccol, attr):
                raise ValueError(f'Specialist stream settings differ: {attr}.')
            setattr(self, attr, getattr(epit, attr))
        for key in ('shape_settings', 'generic_fixed', 'generic_sampled', 'policy'):
            if epit._config[key] != soccol._config[key]:
                raise ValueError(f'Specialist generic/split settings differ: {key}.')
        if (epit.priors[1].max_features, soccol.priors[1].max_features) != (70, 58):
            raise ValueError('Expected native EPIT 70-feature and Soccol 58-feature priors.')
        if self.batch_size*self.world_size//self.group_size < 2:
            raise ValueError('Combined allocation requires at least two global source groups.')
        self.priors = [epit.priors[0], epit.priors[1], soccol.priors[1]]
        self.epit_ratio = epit.selected_ratio / 2
        self.soccol_ratio = soccol.selected_ratio / 2
        self.selected_ratio = self.epit_ratio + self.soccol_ratio
        if not 0 < self.selected_ratio < 1:
            raise ValueError('Combined recipe must contain generic and both informed sources.')
        self.group_counts = None  # Indexed cumulative rounding, not a fixed per-batch count.
        self._next_batch = None
        self.last_batch_ids = []
        self._config = {
            'version': VERSION, 'rank': self.rank,
            'source_ids': {'generic': 0, 'epit': 1, 'soccol': 2},
            'source_probabilities': {'generic': 1-self.selected_ratio,
                                     'epit': self.epit_ratio, 'soccol': self.soccol_ratio},
            'allocation': 'half of each standalone informed ratio; cumulative floor per source; generic remainder',
            'sources': {'epit': epit.checkpoint_identity(), 'soccol': soccol.checkpoint_identity()},
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }

    def _global_group_counts(self, step):
        if step < 0:
            raise ValueError('Batch index must be nonnegative.')
        groups = self.batch_size*self.world_size//self.group_size
        counts = [math.floor((step+1)*groups*r)-math.floor(step*groups*r)
                  for r in (self.epit_ratio, self.soccol_ratio)]
        generic = groups-sum(counts)
        if generic < 0:
            raise ValueError('Insufficient source groups for the combined allocation.')
        return (generic, *counts)

    def sampling_state_dict(self):
        return {'version': VERSION, 'next_batch': self._next_batch}

    def load_sampling_state_dict(self, state):
        if state['version'] != VERSION:
            raise ValueError('Cannot resume another prior policy as the combined selected-pair stream.')
        self._next_batch = state['next_batch']
