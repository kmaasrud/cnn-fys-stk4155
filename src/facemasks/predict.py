from data_facemasks import Data

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from cnn import CNN

data = Data.load_from_npy("data/preprocessed")

cnn = CNN.load("models/facemask_cnn")

prediction = cnn.predict(data.X_test[0:5], eval_data=data.y_test[0:5])

print(f"\nPredicted test value: {prediction[3]}")
print(f"Actual test value: {data.y_test[3]}")