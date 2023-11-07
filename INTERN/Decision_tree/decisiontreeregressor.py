from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass
class Node:
    """Decision tree node."""
    feature: int = None
    threshold: float = None
    n_samples: int = None
    value:int = None
    mse: float = None   
    left: Node = None
    right: Node = None


@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""
    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """Build a decision tree regressor from the training set (X, y)."""
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mse criterion for a given set of target values."""
        return np.square(y - np.mean(y)).mean()

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weithed mse criterion for a two given sets of target values"""
        mse_left = self._mse(y_left)
        mse_right = self._mse(y_right)
        n_left = len(y_left)
        n_right = len(y_right)
        return (mse_left*n_left + mse_right*n_right) / (n_left+n_right)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        best_idx = None
        best_thr = None
        best_metric = None
        f_size = y.shape[0]
        for i in range(X.shape[1]):
            feature_list = X[:, i]
            for feat in np.unique(feature_list):
                c = feature_list<=feat
                left = y[c]
                right = y[~c]
                if left.shape[0] < self.min_samples_split or right.shape[0] < self.min_samples_split:
                    continue
                metric = self._weighted_mse(left, right)
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_thr = feat
                    best_idx = i
        return best_idx, best_thr

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        if depth > self.max_depth:
            return
        
        feature, threshold = self._best_split(X, y)
        
        if feature is None:
            return
        
        c = X[:, feature] <= threshold
            
        return Node(feature,
                    threshold,
                    X.shape[0],
                    round(y.mean()),
                    self._mse(y),
                    self._split_node(X[c], y[c], depth=depth+1),
                    self._split_node(X[~c], y[~c], depth=depth+1))
