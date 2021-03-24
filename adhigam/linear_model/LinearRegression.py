from typing import Union
import numpy as np


class LinearRegression(object):
    """Linear regression using least squares

    """

    def __init__(self) -> None:
        self.weights = None
        self.bias = None

    def fit(self, x: Union[list, np.array], y: Union[list, np.array]) -> None:
        """Fit the linear regression model on training data

        Args:
            x: training data
            y: target values

        Returns: None

        """
        self.weights = np.zeros(x.shape)
        self.bias = np.zeros(len(x))

        y_pred = self.weights*x + self.bias



    def predict(self, x: Union[list, np.array]) -> Union[list, np.array]:
        """Predict the given input values from the training linear model

        Args:
            x: input samples to do the prediction
        """
        pass
        if not self.weights:
            raise ValueError("Model not fitted to the training data, please call `fit` before this")
