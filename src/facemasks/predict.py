import sys
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from data_facemasks import Data

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from cnn import CNN


cnn = CNN.load("models/facemask_cnn2")

data = Data.load_from_npy("data/preprocessed")
predictions = data.postprocess(cnn.model.predict(data.X_test))
print(f"Test set accuracy: {data.accuracy(predictions)}")

data = Data.load_from_imgs("real_pics")
predictions = data.postprocess(cnn.model.predict(data.X_test))
print(f"Real pictures accuracy: {data.accuracy(predictions)}")