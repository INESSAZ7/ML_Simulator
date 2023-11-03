import numpy as np

from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    """Class for gradient boosting"""
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        loss="mse",
        verbose=False,
    ):
        self.base_pred_ = None
        self.trees_ = []
        if loss == "mse":
            self.loss = self._mse
        else:
            self.loss = loss 
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.verbose=verbose

    def _mse(self, y_true, y_pred):
        loss = np.mean(np.square(y_pred - y_true))
        grad = y_pred - y_true
        return loss, grad

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        self.base_pred_ = np.mean(y)
        y_pred = np.full(X.shape[0], fill_value=self.base_pred_)

        for _ in range(self.n_estimators):
            loss_, grad_ = self.loss(y, y_pred)
            b_i = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            b_i.fit(X, (-grad_))
            y_pred+= self.learning_rate*b_i.predict(X)
            self.trees_.append(b_i)
            if self.verbose:
                print(loss_)


    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.
            
        """
        predictions = np.full(X.shape[0], fill_value=self.base_pred_)
        for tree_i in self.trees_:
            predictions += self.learning_rate*tree_i.predict(X)
        
        return predictions
