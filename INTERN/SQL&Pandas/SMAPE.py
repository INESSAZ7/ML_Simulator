import numpy as np


def smape(y_true: np.array, y_pred: np.array) -> float:
    """Calculate sMAPE metric"""
    denominator = np.abs(y_true) + np.abs(y_pred)
    return np.mean(2 * np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator))
