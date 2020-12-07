from data_facemasks import Data

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from cnn import CNN

data = Data.load_from_npy("data/preprocessed")

cnn = CNN.facemask_model(data.n_labels)

cnn.train(data, from_flow=True)

cnn.dump("models/facemask_cnn")