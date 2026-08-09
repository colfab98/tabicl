from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

from scripts.epit_pipeline.split_data import (
    COMPOSITION_DISTANCE_THRESHOLD,
    build_composition_groups,
    cluster_composition_vectors,
    load_epit_dataset,
)


def test_threshold_components_keep_every_close_pair_together() -> None:
    vectors = np.asarray([[0.0], [0.6], [1.2]])
    labels = cluster_composition_vectors(vectors, distance_threshold=1.0)

    assert len(set(labels)) == 1
    distances = pdist(vectors, metric="cityblock")
    assert distances.max() > 1.0


def test_current_epit_composition_groups() -> None:
    dataset = load_epit_dataset()
    grouping = build_composition_groups(dataset)

    assert dataset.n_rows == 760
    assert grouping.rounded_key_count == 397
    assert grouping.n_groups == 301
    assert grouping.sizes.sum() == 760
    assert grouping.sizes.max() == 45
    assert grouping.diameters.max() == 5.0
    assert grouping.isolation_distances.min() > COMPOSITION_DISTANCE_THRESHOLD
