import pickle
import numpy as np

class CNN:
    def __init__(self):
        pass
        
    def dump(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            cnn = pickle.load(f)
        
        return cnn