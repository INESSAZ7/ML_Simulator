from typing import List
from typing import Tuple

import numpy as np
from scipy.stats import ttest_ind


def quantile_ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
    quantile: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[float, bool]:
    """
    Bootstrapped t-test for quantiles of two samples.
    """
    bs_quantile_control = []
    bs_quantile_experiment = []

    for _ in range(n_bootstraps):
        bs_control = np.random.choice(control, size=len(control), replace=True)
        bs_quantile_control.append(np.quantile(sorted(bs_control), quantile))
        bs_experiment = np.random.choice(experiment, size=len(experiment), replace=True)
        bs_quantile_experiment.append(np.quantile(sorted(bs_experiment), quantile))

    p_value = ttest_ind(bs_quantile_control, bs_quantile_experiment)[1]
    result = bool(p_value < alpha)

    return p_value, result
