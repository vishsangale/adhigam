import numpy as np


def L1Loss(x: np.array, y: np.array) -> np.array:
    """Mean Absolute Error Loss"""
    return np.abs(x - y).mean()


def MSELoss(x: np.array, y: np.array) -> np.array:
    """Mean Square Error Loss"""
    return np.square(x - y).mean()


def MSELogLoss(x: np.array, y: np.array) -> np.array:
    """Mean Square Log Error Loss"""
    return np.square(np.log(x + 1.0) - np.log(y + 1.0)).mean()
