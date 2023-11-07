import numpy as np


def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    """Loss function. Penalizes for under-forecasting"""
    res = np.sum(((y_pred - y_true)/y_pred)**2)
    return res
