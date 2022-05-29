import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def test_iris_prediction(save: bool = False) -> None:
    torch.manual_seed(123)

    iris = load_iris()
    x_orgin = iris.data
    y_origin = iris.target

    assert x_orgin.shape == (150, 4)
    assert y_origin.shape == (150,)

    # choose only class0 or class1 data
    indexes = (y_origin == 0) | (y_origin == 1)
    x_data = x_orgin[indexes, :2]
    y_data = y_origin[indexes]
    assert np.count_nonzero((y_data == 0) | (y_data == 1)) == 100

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        train_size=70,
        test_size=30,
    )

    if save:
        x0 = x_train[y_train == 0]
        x1 = x_train[y_train == 1]
        plt.scatter(x0[:, 0], x0[:, 1], marker="x")
        plt.scatter(x1[:, 0], x1[:, 1], marker="*")
        plt.xlabel("sepal_length")
        plt.ylabel("sepal_width")
        plt.legend()
        plt.savefig("sepal.png")
