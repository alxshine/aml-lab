from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from os.path import join
import copy


def load_mnist():
    """Load and prepare MNIST dataset

    Returns:
        tuple of np.ndarray -- x_train, y_train, x_test, y_test in this order
    """
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize to [0, 1] and reshape to (-1, 28, 28, 1)
    x_train = (x_train/255).reshape((-1,28,28,1))
    x_test = (x_test/255).reshape((-1,28,28,1))

    # convert labels to categorical one-hot encoding
    # label 5 -> [0,0,0,0,0,1,0,0,0,0]
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    
    print(y_train[0])
