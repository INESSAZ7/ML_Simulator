from typing import List

import numpy as np


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    score = 0
    iscore = 0
    sort_relevance = sorted(relevance,reverse=True)
    if method=='standard':
        for i, rel in enumerate(relevance[:k]):
            score += rel/(np.log2(i+2)) 
        for i, sort_rel in enumerate(sort_relevance[:k]):
            iscore += sort_rel/(np.log2(i+2))           
    elif method=='industry':
        for i, rel in enumerate(relevance[:k]):
            score += (2**rel-1)/(np.log2(i+2))
        for i, sort_rel in enumerate(sort_relevance[:k]):
            iscore += (2**sort_rel-1)/(np.log2(i+2))   
    else:
        raise ValueError
    return score/iscore
