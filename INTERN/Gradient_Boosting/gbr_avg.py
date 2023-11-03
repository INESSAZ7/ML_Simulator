import numpy as np


class GradientBoostingRegressor():
    """Gradient boosting regressor."""
    def __init__(self):
        self.base_pred_ = None

    def fit(self, X, y):
        """Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
            
        """
        self.base_pred_ = np.mean(y)
  
    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.
            
        """
        predictions = np.full(X.shape[0], fill_value=self.base_pred_)

        return predictions