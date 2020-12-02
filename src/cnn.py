import pickle
import numpy as np
from keras.models import Sequential, Input, Model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

class CNN:
    """Wrapper class over the tf.keras.Sequential class.
    Makes interactions easier for our purposes and implements some extra practical features."""
    def __init__(self, layers):
        """Initializes the CNN wrapper class.
        
        Parameters
        ==========
            layers: list -- List of Keras layers, that will be added to the model."""
        self.model = Sequential()
        for layer in layers:
            self.model.add(layer)
            
        self.model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
        
    def train(self, X_train, y_train, batch_size=50, epochs=20, verbose=1, validation_data=None):
        model_train = self.model.fit(
            X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=validation_data
        )
        self.train_history = model_train.history
        
    def predict(self, X, eval_data=None):
        if eval_data:
            evaluate = self.model.evaluate(X, eval_data)
            print(f"Test loss: {evaluate[0]}")
            print(f"Test accuracy: {evaluate[1]}")
            
        return self.model.predict(X)
        
    def dump(self, path):
        """Saves generated model to a pickled binary file. This can then be loaded again."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path):
        """Loads a previously saved model and retuns it."""
        with open(path, "rb") as f:
            cnn = pickle.load(f)
            
        assert cnn == CNN, "Loaded model must be of class CNN"
        
        return cnn
    
    def __str__(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        return "\n".join(stringlist)