r"""k-nearest neighbors is a non-parametric method used for regression and classification problems in classical machine learning.
In regression, the output is property of entity calculated by averaging the values of k-nearest neighbors.
In classification, the output is type of a class, and the entity is classified based on the voting from k-nearest neighbors.
"""
import types
from typing import Union, Callable

import numpy as np


class Classifier(object):
    def __init__(
        self, k: int, weights: Union[str, Callable[[], float]] = "uniform"
    ) -> None:
        """k-nearest neighbor classifier.

        Args:
            k: Number of neighbors.
            weights: weight function for the prediction {"uniform", "distance"} or a callable function.
        """
        if k <= 0:
            raise ValueError(f"Value of `k` cannot be non-positive.")
        self.k = k
        self.weight_func = None
        if isinstance(weights, str):
            if weights == "normalize":
                self.weight_func = self._normalized_weights
            elif weights == "distance":
                self.weight_func = self._distance_weights
            else:
                raise ValueError(f"Unknown weighting function type {weights}")
        elif isinstance(open, types.FunctionType):
            self.weight_func = weights
        else:
            raise ValueError(f"Unknown weighting parameter {weights}")

    def _normalized_weights(self):
        """Uniform weights, all points in the neighborhood are weighted equally.
        """
        pass

    def _distance_weights(self):
        r"""Weights are calculated using inverse distance from the neighbor.
        Closer neighbors will have greater influence than the farther neighbors.
        """
        pass

    def fit(self, x: Union[list, np.array], y: Union[list, np.array] = None) -> None:
        """Fit the k-nearest neighbor classifier using training dataset.

        Args:
            x: Input training data
            y: Target training values (Optional)

        Returns: None

        """
        pass

    def predict(self, x: Union[list, np.array]) -> Union[list, np.array]:
        pass
