from typing import List, Tuple
from scipy import stats


def ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Two-sample t-test for the means of two independent samples"""
    _, p_value = stats.ttest_ind(control, experiment)
    result = bool(p_value < alpha)
    return p_value, result

