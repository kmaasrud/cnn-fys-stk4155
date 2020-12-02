import numpy as np

def inspect_data(train, test):
    X_train, y_train = train
    X_test, y_test = test

    print(f"Shape of X's: {X_train[0].shape}")
    print(f"Shape of y's: {y_train[0].shape}")
    print(f"Number of training datapoints: {X_train.shape[0]}")
    print(f"Number of testing datapoints: {X_test.shape[0]}")
    
    classes = np.unique(y_train)
    n_classes = len(classes)
    print(f"Number of different outputs: {n_classes}")
    print(f"Output classes: {classes}")
    
    return train, test