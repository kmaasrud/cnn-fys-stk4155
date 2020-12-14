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

data = Data.load_from_imgs("real_pics")

cnn = CNN.load("models/facemask_cnn2")

predictions = data.postprocess(cnn.model.predict(data.X_test))

data.accuracy(predictions, print_report=True)