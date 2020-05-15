from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import GradientTape, constant
import tensorflow.keras.losses as losses

from copy import deepcopy

import data
import visualize


def fgsm(x, y, model, loss, eta=0.1):
    """Perform an untargeted fast gradient method attack

    Arguments:
        x {np.ndarray} -- target sample
        y {np.ndarray} -- true label
        model {tf.keras.Model} -- target model
        loss {tf.keras.losses.Loss} -- target loss function

    Keyword Arguments:
        eta {float} -- step size for FGM (default: {0.1})
    """
    # convert x to tf.Tensor
    x_t = constant(x)
    model.trainable = False

    with GradientTape() as tape:
        # explicitly add input tensor to tape
        tape.watch(x_t)
        # get prediction
        pred = model(x_t)
        # calculate loss
        l = loss(y, pred)
    

    # calculate dloss/dx
    gradients = tape.gradient(l, x_t)
    visualize.gradients(x,y,pred,gradients)

    # perturb input sample: x_adv = x + eta * sign(gradient)
    x_adv = x + eta * np.sign(gradients)
    # clip x_adv to valid range
    x_adv = np.clip(x_adv, 0, 1)

    # predict on adversarial sample
    pred_adv = model(x_adv)
    # visualize prediction
    visualize.prediction(x_adv,y, pred_adv)


if __name__ == "__main__":
    # get data
    _, _, x_test, y_test = data.load_mnist()

    sample_index = 1234
    x = x_test[sample_index:sample_index+1]
    y = y_test[sample_index:sample_index+1]

    model = load_model('/tmp/mnist.h5')

    fgsm(x,y,model,losses.CategoricalCrossentropy(), eta=0.2)