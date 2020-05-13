# fix all seeds
from numpy.random import seed
seed(1337)
import tensorflow
tensorflow.random.set_seed(1337)

import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
from tensorflow import GradientTape

import data
import models


def train_mnist(epochs, optimizer, loss, metrics=['acc']):
    """Train a model for classifying MNIST images using model.fit

    Arguments:
        epochs {int} -- number of epochs to train for
        optimizer {tensorflow.keras.optimizers.Optimizer} -- optimizer to use for training
        loss {tf.keras.losses.Loss} -- loss function

    Keyword Arguments:
        metrics {list} -- evaluation metrics (default: {['acc']})
    """
    # get training data
    # build model

    # compile model with optimizer, loss, and metrics

    # fit model to data
    # evaluate model

    # print()
    # print(f"Validation loss: {val_loss}")
    # print(f"Validation accuracy: {val_acc}")

    # save model


def train_mnist_manual(epochs, optimizer, loss, metrics=['acc']):
    """Manually train a model for MNIST using tf.GradientTape

    Arguments:
        epochs {int} -- number of epochs to train for
        optimizer {tf.keras.optimizers.Optimizer} -- optimizer to use for training
        loss {tf.keras.losses.loss} -- loss function

    Keyword Arguments:
        metrics {list} -- evaluation metrics (default: {['acc']})
    """
    #get training data
    # build model
    # compile model with optimizer, loss, and metrics

    # repeat for all epochs:
    with GradientTape() as tape:
        pass
        # get predictions for entire x_train
        # get loss
    
    # calculate gradients for trainable weights in model
    # apply gradients using optimizer
    
    # evaluate model

    # print()
    # print(f"Validation loss: {val_loss}")
    # print(f"Validation accuracy: {val_acc}")

    # save model


if __name__ == "__main__":
    train_mnist(1, None, None)
