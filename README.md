# Adversarial Machine Learning Lab
This is the AML Lab for the Network Security proseminar in the UIBK Computer Science Master's programme.
This lab will cover the following
- Downloading and preparing a dataset for use in machine learning
- Building and training a neural net with TensorFlow
- Manually computing gradients to train a neural net
- Creating adversarial samples using the [Fast Gradient Method](https://arxiv.org/abs/1412.6572)
- The effectiveness of poisoning attacks on training data

## Preparation
You will require the following:
- python 3.7
- `virtualenv`

All other required packages will be installed later.

### Setting up the virtual environment
To separate your python system install from packages required for this lab, it is recommended to set up a virtual environment.
This can be done using the command:
```
virtualenv -p python3.7 venv-aml-lab
```

The above command will create a directory called `venv-aml-lab` where all future packages will be installed.

To activate the virtual environment use the command:
```
source venv-aml-lab/bin/activate
```
Your command line prompt should now start with `(venv-aml-lab)`.

### Installing required packages
Required python packages are contained in `setup.py`.
To install them first activate the virtualenv, and execute the following line:
```
pip install -e .
```
This executes `setup.py` and installs the required packages.
The `-e` flag will allow you to make changes to the code.

### Acquiring the datasets
For this lab we use [MNIST](http://yann.lecun.com/exdb/mnist) and an email spam detection dataset.
MNIST can be downloaded automatically using TensorFlow utility functions.
The email spam detection dataset must be downloaded from [the UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/Spambase) and extracted to `datasets`.

# The Lab Exercise
To make it easy to follow we split the entire exercise into small individual steps.

## Cleaning up the data
Start in `data.py`.
While we can get the training and test sets directly as `numpy.ndarray`, they will be in the range [0,255].
That is not good for neural nets, so we need to normalize that data to [0,1]
We also need to reshape it for TensorFlow.
The labels come as integers, which we will convert to a categorical one-hot encoding.

## Building a neural net using TensorFlow and Keras
The file `models.py` houses the functions to build the models we need for training and attacking.
These models currently only contain an input layer.
We will need to add Dense Layers to make the model actually do something meaningful.
The required layers are already imported.

For now focus on the MNIST model, we will cover the spam model later.

## Training a neural net using TensorFlow built-in functions
Open `train.py`, and take a look at the function `train_mnist`
It takes a loss function, an optimizer, and a number of epochs and then trains the model.
Take a look at [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) and its functions, the comments should help you find the right functions.

## Building a convolutional neural net
All state-of-the-art image recognition classifiers use convolutional neural nets.
Go to the `build_mnist_CNN` function in `models.py` and fill the gaps.
The training function is already built, so you can test your model right away.

## Manually computing gradients and training a neural net
We can also manually compute gradients using `tf.GradientTape`.
This will come in handy for generating adversarial samples later on.
The GradientTape will monitor variables during execution and can be used for automatic differentiaton.
It can generate the gradients *dloss/dw* for all weights *w* of the network.
These can be used to optimize the network for training data.

Take a look at the `train_mnist_manual` function and fill in the gaps.

## Creating adversarial samples
Just like we used *dloss/dw* to get the gradients for all weights, we can calculate *dloss/dx* to calculate the gradients of the **input image**.
With these gradients we can change the image such that is misclassified by the network.

The code for this can be found in `attack.py`.

# Feedback
I hope you have enjoyed this lab and learned something while completing it.
If you have feedback or encountered a bug, feel free to open up an issue.
When you do, please be so kind and let me know what your background with machine learning is and how you came across this lab.