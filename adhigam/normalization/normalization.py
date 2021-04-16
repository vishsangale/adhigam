import numpy as np


def normalize(x: np.array) -> np.array:
    # subtract the mean
    x -= np.mean(x, axis=0)
    # normalize by standard deviation
    x /= np.std(x, axis=0)
    return x