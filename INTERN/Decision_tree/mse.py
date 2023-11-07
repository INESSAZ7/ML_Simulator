import numpy as np

def mse(y: np.ndarray) -> float:
    """Compute the mean squared error of a vector."""    
    return np.square(y - np.mean(y)).mean()


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute the weighted mean squared error of two vectors."""
    mse_left = mse(y_left)
    mse_right = mse(y_right)
    n_left = len(y_left)
    n_right = len(y_right)
    return (mse_left*n_left + mse_right*n_right) / (n_left+n_right)
