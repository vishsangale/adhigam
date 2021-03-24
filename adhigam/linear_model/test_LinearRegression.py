import matplotlib.pyplot as plt
import numpy as np

from adhigam.linear_model.LinearRegression import LinearRegression


def test_linear_regression():
    nr_samples = 50
    x = np.arange(0, nr_samples, step=2)

    noise = np.random.normal(-3.0, 3.0, x.shape)

    # y = mx + noise
    m = 2.3
    y = m * x + noise

    # plt.scatter(x, y)
    # plt.show()

    lr = LinearRegression()

    lr.fit(x, y)

    pred = lr.predict([100.0, 200.0])

    print(pred)
