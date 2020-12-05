from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from make_model import make_data

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from cnn import CNN

data = make_data()

cnn = CNN([
    Conv2D(32, kernel_size=(3,3), activation="linear", input_shape=(28, 28, 1), padding="same"),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2), padding="same"),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation="linear", padding="same"),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), padding="same"),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation="linear", padding="same"),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2), padding="same"),
    Dropout(0.4),
    Flatten(),
    Dense(128, activation="linear"),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(data.n_classes, activation="softmax")
])

cnn.train(data)

cnn.dump("models/fashion_mnist_cnn_dropout")