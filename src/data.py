import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

class Data:
    def __init__(self, train, test, onehot=None):
        self.train = train
        self.test = test
        self.X_train, self.y_train = train
        self.X_test, self.y_test = test

        self.classes = np.unique(self.y_train)
        self.n_classes = len(self.classes)
        
        self.onehot = onehot
        if onehot:
            for attr_name in onehot:
                setattr(self, attr_name + "_orig", getattr(self, attr_name))
                setattr(self, attr_name, to_categorical(getattr(self, attr_name)))
        
    def scale(self, X=None, y=None):
        if X:
            self.X_train = self.X_train * X
            self.X_test = self.X_test * X
        if y:
            self.y_train *= y
            self.y_test *= y
            
        self.train = (self.X_train, self.y_train)
        self.test = (self.X_test, self.y_test)
        
    def make_validation_set(self, val_size=0.25):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=val_size)
        
    def inspect(self):
        print(f"Shape of X's: {self.X_train[0].shape}")
        print(f"Shape of y's: {self.y_train[0].shape}")
        print(f"Number of training datapoints: {self.X_train.shape[0]}")
        print(f"Number of testing datapoints: {self.X_test.shape[0]}")
        print(f"Number of different outputs: {self.n_classes}")
        print(f"Output classes: {self.classes}")