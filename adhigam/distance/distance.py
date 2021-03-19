from typing import Union

import numpy as np


def manhattan(
    x: Union[list, np.array], y: Union[list, np.array]
) -> Union[float, list, np.array]:
    """Calculate manhattan distance between two points.
    The distance between two points measured along axes at right angles.

    Args:
        x: Point x
        y: Point y

    Returns:

    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    if x.shape != y.shape or x.size != y.size:
        raise ValueError(
            f"Shape or size of x and y does not matches, x shape {x.shape}, y shape {y.shape}"
        )
    return np.abs(x[0] - y[0]) + np.abs(x[1] - y[1])
