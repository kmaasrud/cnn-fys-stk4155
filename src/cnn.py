import pickle
import numpy as np
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

class CNN:
    def __init__(self
                batch_size,
                epochs):
        pass
        
    def dump(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            cnn = pickle.load(f)
        
        return cnn