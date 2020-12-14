import pickle
import os
import numpy as np
from keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2


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
            
        self.compile()
            

    def compile(self, loss=categorical_crossentropy):
        self.model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])
        

    def train(self, data, batch_size=50, epochs=20, verbose=1, from_flow=False):
        try:
            validation_data = (data.X_val, data.y_val)
        except AttributeError:
            validation_data = (data.X_test, data.y_test)

        if from_flow:
            model_train = self.model.fit(
                data.gen.flow(data.X_train, data.y_train, batch_size=batch_size), steps_per_epoch=len(data.X_train) // batch_size, epochs=epochs, validation_data=validation_data)
        else:
            model_train = self.model.fit(
                data.X_train, data.y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=validation_data
            )

        self.train_accuracy = model_train.history["accuracy"]
        self.train_loss = model_train.history["loss"]
        try:
            self.val_accuracy = model_train.history["val_accuracy"]
            self.val_loss = model_train.history["val_loss"]
        except KeyError:
            pass
        

    def predict(self, X, eval_data=None):
        # if len(eval_data):
        #     evaluate = self.model.evaluate(X, eval_data)
        #     print(f"Prediction loss: {evaluate[0]}")
        #     print(f"Prediction accuracy: {evaluate[1]}")
            
        return self.model.predict(X)
        

    def dump(self, filename):
        """Saves generated model to a pickled binary file. This can then be loaded again."""
        try:
            history = (self.train_accuracy, self.train_loss, self.val_accuracy, self.val_loss)
            with open(filename + "_history.pickle", "wb") as f:
                pickle.dump(history, f)
        except AttributeError:
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
    

    @staticmethod
    def facemask_model(n_labels):
        cnn = CNN([])
        mobile = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        outmodel = Dense(units=n_labels, activation="softmax")(
            Dropout(0.25)(
                Dense(units=256, activation="relu")(
                    Flatten()(
                        AveragePooling2D(pool_size=(7, 7))(
                            mobile.output
                        )
                    )
                )
            )
        )

        cnn.model = Model(inputs=mobile.input, outputs=outmodel)

        for layer in mobile.layers:
            layer.trainable = False
            
        cnn.compile()
        
        return cnn
    
    def __str__(self):
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        return "\n".join(stringlist)
