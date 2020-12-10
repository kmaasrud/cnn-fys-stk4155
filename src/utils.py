import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Different functions for use as activation functions
def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid_derivative(x)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """ Derivative of the sigmoid function. """
    return sigmoid(x) * (1 - sigmoid(x))

def ReLU(x, derivative=False):
    if derivative:
        return heaviside(x)
    return x if x > 0 else 0

def heaviside(x):
    return 1 if x >= 0 else 0

def leaky_ReLU(x, derivative=False):
    """ The Leaky ReLUfunction. """
    if derivative:
        return leaky_ReLU_derivative(x)
    idx = np.where(x <= 0)
    x[idx] = 0.01 * x
    return x

def leaky_ReLU_derivative(x):
    """ Derivative of the Leaky ReLU function. """
    idx1 = np.where(x < 0)
    x[idx1] = 0.01
    idx2 = np.where(x > 0)
    x[idx2] = 1.0
    return x

def MSE(x, y):
    """The mean squared error function.
    The result is divided by 2 to make sure the derivative of the cost function is easily written as just (x - y)"""
    assert len(x) == len(y), "The arrays need to have the same length"
    for xval, yval in zip(x, y):
        s += (xval - yval)**2
    return s / (2 * len(x))

def accuracy(y, y_pred):
    y=np.ravel(y)
    y_pred=np.ravel(y_pred)
    numerator=np.sum(y == y_pred)
    return numerator/len(y)

def R2(x, y):
    """Evaluates the R2 score of two lists/arrays"""
    deno = MSE(x, np.full(len(x), mean_value(y)))
    R2 = 1 - MSE(x, y) / deno
    return R2


def mean_value(y):
    """Evaluates the mean value of a list/array"""
    return sum(y) / len(y)

"""Function to calculate and visualise the confusion matrix using sklearn
to calculate the confusion matrix, and seaborn to plot the heatmap"""
def heatmap(y_test, y_pred, dataset=None):
    if dataset=='fashion':
        axis_labels=['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal',
                        'shirt', 'sneaker', 'bag', 'boots']
    elif dataset=='mask':
        axis_labels=['Mask', 'No mask']
    else:
        axis_labels=np.unique(y_test)

    sns.set()

    #Scikit learn to calculate the confusion matrices
    test_samples=confusion_matrix(y_test, y_pred)
    test_accuracy=confusion_matrix(y_test, y_pred, normalize='true')
    test_accuracy=np.round(test_accuracy, 2)

    #Plotting the confusion matrix by number of samples
    """
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_samples, xticklabels=axis_labels, yticklabels=axis_labels, annot=True, ax=ax, cmap="viridis", fmt='g')
    ax.set_ylabel("Solution")
    ax.set_xlabel("Prediction")
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()
    """

    #Plotting the confusion matrix by accuracy
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, xticklabels=axis_labels, yticklabels=axis_labels, annot=True, ax=ax, cmap="viridis")
    ax.set_ylabel("Solution")
    ax.set_xlabel("Prediction")
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.show()

    return
