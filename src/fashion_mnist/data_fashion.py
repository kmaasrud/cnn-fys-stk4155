import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

class Data:
    """Class that contains the fashion dataset and supports different preprocessing/postprocessing operations on it.
    
    Parameters
    ==========
        train: (numpy.ndarray, numpy.ndarray) -- Tuple containing the input and output of the training data
        test: (numpy.ndarray, numpy.ndarray) -- Tuple containing the input and output of the testing data
        
    Optional parameters
    ===================
        onehot: tuple -- Tuple of which parts of the dataset to one-hot encode. Tuple can contain the values:
                            - "X_train"
                            - "X_test"
                            - "y_train"
                            - "y_test"
                        The original data is accessible through an attribute named <name>_orig, e.g.: Data.X_train_orig
    """
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

        self.X_shape = self.X_train[0].shape
        self.y_shape = self.y_train[0].shape
        self.n_training = self.X_train.shape[0]
        self.n_testing = self.X_test.shape[0]
        
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
        self.n_training = self.X_train.shape[0]
        self.n_validation = self.X_val.shape[0]
        
    def __str__(self):
        return f"""==========================================================================
| Shape of X's:\t\t\t\t | \t{self.X_shape}
| Shape of y's:\t\t\t\t | \t{self.y_shape}
| Number of training datapoints:\t | \t{self.n_training}
| Number of testing datapoints:\t\t | \t{self.n_testing}
| Number of different outputs:\t\t | \t{self.n_classes}
| Output classes:\t\t\t | \t{self.classes}
=========================================================================="""