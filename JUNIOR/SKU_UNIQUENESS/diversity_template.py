"""Template for user."""
from typing import Tuple
from sklearn.neighbors import KernelDensity

import numpy as np


def kde_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on KDE.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group 

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    kde = KernelDensity(kernel='gaussian').fit(embeddings)
    likelihood = np.exp(kde.score_samples(embeddings))
    return 1/likelihood


def group_diversity(embeddings: np.ndarray, threshold: float) -> Tuple[bool, float]:
    """Calculate group diversity based on kde uniqueness.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group
    threshold: float :
       group deversity threshold for reject group

    Returns
    -------
    Tuple[bool, float]
        reject
        group diverstity

    """
    diversity = np.mean(kde_uniqueness(embeddings))
    if diversity < threshold:
        return (True, diversity)
    else:
        return (False, diversity)
