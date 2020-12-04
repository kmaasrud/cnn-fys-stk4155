import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from data import Data
from cnn import CNN

OUT_DIR = os.path.join(os.path.dirname(parentdir), "doc/assets")

cnn = CNN.load("cnn1")

epoch = list(range(len(cnn.train_history["loss"])))

plt.plot(epoch, cnn.train_history["loss"], label="Loss")
plt.plot(epoch, cnn.train_history["accuracy"], label="Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "fashion_mnist_train_history"))

plt.clf()
plt.plot(epoch, cnn.train_history["val_loss"], label="Loss")
plt.plot(epoch, cnn.train_history["val_accuracy"], label="Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "fashion_mnist_val_history"))