from scipy.stats import binom, norm, stats
from numpy import random
import numpy as np

def cpc_sample(
    n_samples: int, conversion_rate: float, reward_avg: float, reward_std: float
) -> np.ndarray:
    """Sample data."""
    
    cvr = random.binomial(n=1, p=conversion_rate, size=n_samples)
    cpa = random.norm(loc=reward_avg, scale=reward_std, size=n_samples)
    
    return cvr*cpa

def t_test(cpc_a: np.ndarray, cpc_b: np.ndarray, alpha=0.05
) -> Tuple[bool, float]:
    """Perform t-test.

    Parameters
    ----------
    cpc_a: np.ndarray :
        first samples    
    cpc_b: np.ndarray :
        second samples
    alpha :
         (Default value = 0.05)

    Returns
    -------
    Tuple[bool, float] :
        True if difference is significant, False otherwise
        p-value
    """
    
    _, pvalue = stats.ttest_ind(cpc_a, cpc_b)
    accept = bool(pvalue<alpha)
    return (accept, pvalue)
