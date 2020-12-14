import matplotlib.pyplot as plt

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from cnn import CNN

OUT_DIR = os.path.join(os.path.dirname(parentdir), "doc/assets")

cnn = CNN.load("models/facemask_cnn2")

epoch = list(range(len(cnn.train_loss)))

plt.plot(epoch, cnn.train_loss, label="Loss")
plt.plot(epoch, cnn.train_accuracy, label="Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "facemasks_train_history"))

plt.clf()
plt.plot(epoch, cnn.val_loss, label="Loss")
plt.plot(epoch, cnn.val_accuracy, label="Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "facemasks_val_history"))