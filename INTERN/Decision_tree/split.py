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

def split(X: np.ndarray, y: np.ndarray, feature: int) -> float:
    """Find the best split for a node (one feature)"""
    feature_list = X[:, feature]
    best_metric = mse(y)
    f_size = y.shape[0]
    for j in range(f_size-1):
        metric = weighted_mse(y[0:j+1], y[j+1:f_size])
        if metric < best_metric:
            best_metric = metric
            threshold = j
    return feature_list[threshold]
