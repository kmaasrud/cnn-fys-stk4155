import pickle
import os
import numpy as np
from keras.models import Sequential, Input, Model, load_model
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
        
    def train(self, data, batch_size=50, epochs=20, verbose=1):
        try:
            validation_data = (data.X_val, data.y_val)
        except NameError:
            validation_data = None

        model_train = self.model.fit(
            data.X_train, data.y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=validation_data
        )

        self.train_accuracy = model_train.history["accuracy"]
        self.train_loss = model_train.history["loss"]
        self.val_accuracy = model_train.history["val_accuracy"]
        self.val_loss = model_train.history["val_loss"]
        
    def predict(self, X, eval_data=None):
        if eval_data:
            evaluate = self.model.evaluate(X, eval_data)
            print(f"Prediction loss: {evaluate[0]}")
            print(f"Prediction accuracy: {evaluate[1]}")
            
        return self.model.predict(X)
        
    def dump(self, filename):
        """Saves generated model to a pickled binary file. This can then be loaded again."""
        try:
            history = (self.train_accuracy, self.train_loss, self.val_accuracy, self.val_loss)
            with open(filename + "_history.pickle", "wb") as f:
                pickle.dump(history, f)
        except NameError:
            pass
        finally:
            self.model.save(filename + ".tf")
            
    @classmethod
    def load(cls, filename):
        """Loads a previously saved model and returns it."""
        model = load_model(filename + ".tf")

        cnn = CNN([])
        cnn.model = model 
        
        if os.path.isfile(filename + "_history.pickle"):
            with open(filename + "_history.pickle", "rb") as f:
                history = pickle.load(f)
            cnn.train_accuracy, cnn.train_loss, cnn.val_accuracy, cnn.val_loss = history
        
        return cnn
    
    def __str__(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        return "\n".join(stringlist)