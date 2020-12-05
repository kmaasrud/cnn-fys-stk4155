import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from data import Data
from cnn import CNN

OUT_DIR = os.path.join(os.path.dirname(parentdir), "doc/assets")

cnn = CNN.load("models/fashion_mnist_cnn")
cnn_dropout = CNN.load("models/fashion_mnist_cnn_dropout")

epoch = list(range(len(cnn.train_loss)))

plt.plot(epoch, cnn.train_loss, label="Loss")
plt.plot(epoch, cnn.train_accuracy, label="Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "fashion_mnist_train_history"))

plt.clf()
plt.plot(epoch, cnn.val_loss, label="Loss")
plt.plot(epoch, cnn.val_accuracy, label="Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "fashion_mnist_val_history"))

plt.clf()
plt.plot(epoch, cnn_dropout.train_loss, label="Loss")
plt.plot(epoch, cnn_dropout.train_accuracy, label="Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "fashion_mnist_train_history_dropout"))

plt.clf()
plt.plot(epoch, cnn_dropout.val_loss, label="Loss")
plt.plot(epoch, cnn_dropout.val_accuracy, label="Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "fashion_mnist_val_history_dropout"))