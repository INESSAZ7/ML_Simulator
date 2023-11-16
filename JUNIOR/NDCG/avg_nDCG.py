from typing import List

import numpy as np

def avg_ndcg(list_relevances: List[List[float]], k: int, method: str = 'standard') -> float:
    """average nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
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
    list_score = []
    if method=='standard':
        for relevance in list_relevances:
            score = 0
            iscore = 0
            sort_relevance = sorted(relevance,reverse=True)
            for i, rel in enumerate(relevance[:k]):
                score += rel/(np.log2(i+2)) 
            for i, sort_rel in enumerate(sort_relevance[:k]):
                iscore += sort_rel/(np.log2(i+2))    
            list_score.append(score/iscore)       
    elif method=='industry':
        for relevance in list_relevances:
            score = 0
            iscore = 0
            sort_relevance = sorted(relevance,reverse=True)
            for i, rel in enumerate(relevance[:k]):
                score += (2**rel-1)/(np.log2(i+2))
            for i, sort_rel in enumerate(sort_relevance[:k]):
                iscore += (2**sort_rel-1)/(np.log2(i+2))
            list_score.append(score/iscore)   
    else:
        raise ValueError
    return sum(list_score)/len(list_score)
