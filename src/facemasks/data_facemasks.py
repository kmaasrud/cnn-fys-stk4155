import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

class Data:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.enc = LabelEncoder().fit(y_test)

        if X_train:
            self.X_train = X_train
            self.y_train = y_train
            self.train_enc = LabelEncoder.fit(self.y_train)
            self.y_train = to_categorical(self.enc.transform(self.y_train))

        self.X_test = X_test
        self.y_test_orig = y_test
        
        # One-hot encoding of y data
        self.y_test = to_categorical(self.enc.transform(self.y_test_orig))
        self.n_labels = len(self.enc.classes_)

        self.gen = ImageDataGenerator(
            rotation_range=10, zoom_range=0.10, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, fill_mode="nearest"
        )
        

    def postprocess(self, predictions):
        prediction_objects = []
        for array in predictions:
            prediction_objects.append(Prediction(array, self.enc))

        return prediction_objects
    

    def accuracy(self, predictions, print_report=False):
        s = 0
        for prediction, real in zip(predictions, self.y_test_orig):
            if print_report:
                print(f"Prediction:\t{prediction}\nConfidence:\t{prediction.confidence[prediction.val]}\nReal:\t\t{real}\n")
            s += int(prediction == real)
            
        return s / len(predictions)


    @staticmethod
    def load_from_npy(dir):
        X_train = np.load(os.path.join(dir, "X_train.npy"))
        X_test = np.load(os.path.join(dir, "X_test.npy"))
        y_train = np.load(os.path.join(dir, "y_train.npy"))
        y_test = np.load(os.path.join(dir, "y_test.npy"))
        
        return Data(X_train, X_test, y_train, y_test)


    @classmethod
    def load_from_imgs(cls, data_dir):
        img_paths = []
        data = {}
        X = []
        y = []
        for dirname, dirs, filenames in os.walk(os.path.join(data_dir)):
            for filename in filenames:
                if os.path.splitext(filename)[1] in [".png", ".jpg", ".jpeg"]:
                    img_paths.append(os.path.join(dirname, filename))
                
        for img_path in img_paths:
            label = img_path.split(os.path.sep)[-2]
            img = preprocess_input(img_to_array(load_img(img_path, target_size=(224, 224))))
            X.append(img)
            y.append(label)
            
        return cls(None, np.array(X), None, np.array(y))
                

    @staticmethod
    def save_from_imgs(data_dir):
        img_paths = []
        X = []
        y = []
        for set_type in ["train", "test"]:
            for dirname, dirs, filenames in os.walk(os.path.join(data_dir, set_type)):
                for filename in filenames:
                    if os.path.splitext(filename)[1] in [".png", ".jpg", ".jpeg"]:
                        img_paths.append(os.path.join(dirname, filename))
                    
            for img_path in img_paths:
                label = img_path.split(os.path.sep)[-2]
                img = preprocess_input(img_to_array(load_img(img_path, target_size=(224, 224))))
                X.append(img)
                y.append(label)
                
            np.save(os.path.join(data_dir, os.path.join("preprocessed", "X_" + set_type)), np.array(X))
            np.save(os.path.join(data_dir, os.path.join("preprocessed", "y_" + set_type)), np.array(y))
            
            
class Prediction:
    def __init__(self, array, encoder):
        max_val = 0
        for i, val in enumerate(array):
            if val > max_val:
                max_val = val
                max_i = i
                
        self.val = encoder.inverse_transform([max_i])[0]
        
        self.confidence = {}
        for i, class_ in enumerate(encoder.classes_):
            self.confidence[class_] = array[i]
        
    def __str__(self):
        return self.val

    def __eq__(self, other):
        return self.val == other

        
if __name__ == "__main__":
    data = Data.save_from_imgs("data")
