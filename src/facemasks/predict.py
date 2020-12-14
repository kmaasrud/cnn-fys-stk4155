import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from skimage.io import imread
from data_facemasks import Data

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from cnn import CNN

OUT_DIR = os.path.join(os.path.dirname(parentdir), "doc/assets")

cnn = CNN.load("models/facemask_cnn2")

data = Data.load_from_npy("data/preprocessed")
predictions = data.postprocess(cnn.model.predict(data.X_test))
print(f"Test set accuracy: {data.accuracy(predictions)}")

data, img_paths = Data.load_from_imgs("real_pics", return_img_paths=True)
predictions = data.postprocess(cnn.model.predict(data.X_test))
print(f"Real pictures accuracy: {data.accuracy(predictions)}")

fig, axes = plt.subplots(1, 4, figsize=(6, 6))
ax = axes.ravel()

for ax_i, i in enumerate([1, 3, 5, 8]):
    pic = imread(img_paths[i])
    pic = np.asarray(pic)
    pic = resize(pic, (224, 224), preserve_range=True, mode='constant')
    pic /= 255
    ax[ax_i].imshow(pic)
    ax[ax_i].title.set_text(f"Solution = {data.y_test_orig[i]}\nPredicted = {predictions[i]}")

fig.tight_layout()
plt.show()