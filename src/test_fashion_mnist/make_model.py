from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from data import Data
from cnn import CNN


# Load data and inspect
data = Data(*fashion_mnist.load_data(), onehot=["y_train", "y_test"])
data.inspect()


# Reshaping of input data
data.X_train = data.X_train.reshape(-1, 28, 28, 1)
data.X_test = data.X_test.reshape(-1, 28, 28, 1)

# Scaling the input data
data.scale(X=1/255)

# Splitting into training and validation data sets (80% and 20% respectively)
data.make_validation_set(val_size=0.2)


cnn = CNN([
    Conv2D(32, kernel_size=(3,3), activation="linear", input_shape=(28, 28, 1), padding="same"),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2), padding="same"),
    Conv2D(64, (3, 3), activation="linear", padding="same"),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), padding="same"),
    Conv2D(128, (3, 3), activation="linear", padding="same"),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), padding="same"),
    Flatten(),
    Dense(128, activation="linear"),
    LeakyReLU(alpha=0.1),
    Dense(data.n_classes, activation="softmax")
])

cnn.train(data.X_train, data.y_train, validation_data=(data.X_val, data.y_val))