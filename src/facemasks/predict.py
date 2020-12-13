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

# if sys.argv[1]:
#     img_path = sys.argv[1]
# else:
#     img_path = input("Insert path to directory of images you want to assess: ")

# imgs = []
# for dirname, dirs, filenames in os.walk(img_path):
#     for filename in filenames:
#         if os.path.splitext(filename)[1] in [".png", ".jpg", ".jpeg"]:
#             img = preprocess_input(img_to_array(load_img(os.path.join(dirname, filename), target_size=(224, 224))))
#             imgs.append(np.array(img))

data = Data.load_from_npy("data/preprocessed")

cnn = CNN.load("models/facemask_cnn")

prediction = cnn.model.predict(data.X_test[:50], batch_size=50)

print(f"\nPredicted value: {prediction}")