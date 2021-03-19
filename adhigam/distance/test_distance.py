import numpy as np
import pytest

from adhigam.distance.distance import manhattan


def test_manhattan_distance_unequal_shape_size():

    a = [1]
    b = [2, 3]
    with pytest.raises(ValueError):
        manhattan(a, b)
    with pytest.raises(ValueError):
        manhattan(b, a)


def test_manhattan_distance():
    # 2D Coordinates with list
    x = [2, 3]
    y = [4, 5]
    assert manhattan(x, y) == 4

    x = [-2, 3]
    y = [4, -5]
    assert manhattan(x, y) == 14

    # 2D Coordinates with np array
    x = np.array([2, 3])
    y = np.array([4, 5])
    assert manhattan(x, y) == 4

    x = np.array([-2, 3])
    y = np.array([4, -5])
    assert manhattan(x, y) == 14

    # symmetry
    assert manhattan(x, y) == manhattan(y, x)
