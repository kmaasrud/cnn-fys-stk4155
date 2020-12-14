# No warning or info messages from Tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import matplotlib.pyplot as plt
from make_model import make_data

import sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from cnn import CNN

OUT_DIR = os.path.join(os.path.dirname(parentdir), "doc/assets")

cnn = CNN.load("models/fashion_mnist_cnn_dropout")

data = make_data()

predictions = data.postprocess(cnn.predict(data.X_test))
print(f"Test set accuracy: {data.accuracy(predictions)}")

label = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

plt.figure(figsize=(10,10))
for i, precition in enumerate(predictions[:8]):
    p = plt.subplot(2,4, i + 1)

    p.imshow(data.X_test[i].reshape(28,28), cmap=plt.cm.gray, interpolation='bilinear')
    p.set_xticks(()); p.set_yticks(()) # remove ticks

    p.set_title(f"Prediction: {label[precition]} \nSolution: {label[data.y_test_orig[i]]}")

    #fig.tight_layout()
plt.show()