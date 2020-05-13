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

    with GradientTape() as tape:
        # explicitly add input tensor to tape
        # get prediction
        # calculate loss
        pass

    # calculate dloss/dx
    # visualize.gradients(x,y,pred,gradients)

    # perturb input sample: x_adv = x + eta * sign(gradient)
    # clip x_adv to valid range

    # predict on adversarial sample
    # visualize prediction


if __name__ == "__main__":
    pass
