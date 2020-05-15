import matplotlib.pyplot as plt
import numpy as np


def gradients(x, y, pred, gradients):
    """Visualize gradients for sample along with sample itself

    Arguments:
        x {np.ndarray} -- Input sample
        y {np.ndarray} -- True label as one-hot encoding
        pred {np.ndarray} -- Model predictions
        gradients {np.ndarray} -- dloss/dx
    """
    fig = plt.figure()
    fig.suptitle(f"True label: {np.argmax(y)}, Prediction: {np.argmax(pred)}, Confidence: {np.max(pred)}")
    plt.subplot(121)
    if len(gradients.shape) == 4:
        vis_gradients = gradients[0,:,:,0]
    else:
        vis_gradients = gradients
    plt.imshow(vis_gradients, cmap='coolwarm')
    plt.title("Gradients")

    plt.subplot(122)
    plt.imshow(x[0,:,:,0], cmap='gray')
    plt.title("Sample")
    plt.show()

def prediction(x,y,pred):
    """Visualize prediction

    Arguments:
        x {np.ndarray} -- Input sample
        y {np.ndarray} -- True label as one-hot encoding
        pred {np.ndarray} -- Model predictions
    """
    fig = plt.figure()
    fig.suptitle(f"True label: {np.argmax(y)}, Prediction: {np.argmax(pred)}, Confidence: {np.max(pred)}")
    plt.imshow(x[0,:,:,0], cmap='gray')
    plt.show()
