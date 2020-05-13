from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D


def build_mnist_MLP():
    """Build a fully connected network for classifying MNIST images

    Returns:
        tf.keras.models.Sequential -- the built model
    """
    model = Sequential()
    model.add(Input((28, 28, 1)))

    return model


def build_mnist_CNN():
    """Build a convolutional neural network for classifying MNIST images

    Returns:
        tf.keras.models.Sequential -- the built model
    """
    model = Sequential()
    model.add(Input((28, 28, 1)))

    return model


if __name__ == "__main__":
    print('MNIST network summary:')
    build_mnist_MLP().summary()
