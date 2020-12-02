from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from inspect_data import inspect_data

(X_train, y_train_orig), (X_test, y_test_orig) = inspect_data(*fashion_mnist.load_data())

# Preprocessing
# -------------

# Reshaping of input data
X_train = X_train.reshape(-1, 28,28, 1)
X_test = X_test.reshape(-1, 28,28, 1)

# One-hot encoding of output data
y_train_oh = to_categorical(y_train_orig)
y_test_oh = to_categorical(y_test_orig)

# Scaling the input data
X_train = X_train / 255
X_test = X_test / 255

# Splitting into training and validation data sets (80% and 20% respectively)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train_oh, test_size=0.2, random_state=13)

# Preprocessing done
# ------------------