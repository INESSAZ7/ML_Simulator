import numpy as np
from typing import Tuple


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean squared error loss function and gradient."""
    loss = float(np.sum(np.square(y_pred - y_true)) * 1/y_true.shape)
    grad = y_pred - y_true
    return loss, grad


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean absolute error loss function and gradient."""
    loss = float(np.sum(np.abs(y_pred - y_true)) * 1/y_true.shape)
    grad = np.sign(y_pred - y_true)
    return loss, grad
