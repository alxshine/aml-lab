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
    x_train, y_train, x_test, y_test = data.load_mnist()
    # build model
    model = models.build_mnist_CNN()

    # compile model with optimizer, loss, and metrics
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # fit model to data
    model.fit(x_train, y_train, epochs=epochs)
    # evaluate model
    val_loss, val_metrics = model.evaluate(x_test, y_test)

    print()
    print(f"Validation loss: {val_loss}")
    print(f"Validation metrics: {val_metrics}")

    # save model
    model.save('/tmp/mnist.h5')


def train_mnist_manual(epochs, optimizer, loss, metrics=['acc']):
    """Manually train a model for MNIST using tf.GradientTape

    Arguments:
        epochs {int} -- number of epochs to train for
        optimizer {tf.keras.optimizers.Optimizer} -- optimizer to use for training
        loss {tf.keras.losses.loss} -- loss function

    Keyword Arguments:
        metrics {list} -- evaluation metrics (default: {['acc']})
    """
    # get training data
    x_train, y_train, x_test, y_test = data.load_mnist()
    # build model
    model = models.build_mnist_CNN()
    # compile model with optimizer, loss, and metrics
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    for e in range(epochs):
        print(f"epoch: {e+1}")
        # repeat for all epochs:
        with GradientTape() as tape:
            pass
            # get predictions for entire x_train
            predictions = model(x_train)
            # get loss
            epoch_loss = loss(y_train, predictions)
        
        # calculate gradients for trainable weights in model
        # dtarget/dsources
        weights = model.trainable_weights
        gradients = tape.gradient(epoch_loss, weights)
        # apply gradients using optimizer
        optimizer.apply_gradients(zip(gradients, weights))
    
    # evaluate model
    val_loss, val_metrics = model.evaluate(x_test, y_test)

    print()
    print(f"Validation loss: {val_loss}")
    print(f"Validation metrics: {val_metrics}")

    # save model
    model.save('/tmp/mnist_manual.h5')

if __name__ == "__main__":
    train_mnist_manual(3, optimizers.Adam(), losses.CategoricalCrossentropy())
