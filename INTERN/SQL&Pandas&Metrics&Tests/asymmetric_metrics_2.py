
import numpy as np

def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    diff = y_true-y_pred
    diff = np.multiply(np.abs(diff), np.where(diff >= 0, 0.25, 0.75))
    error = np.sum(diff)
    return error
