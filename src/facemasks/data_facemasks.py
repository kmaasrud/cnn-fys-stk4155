import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

class Data:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # One-hot encoding of y data
        enc = LabelEncoder()
        self.y_train = to_categorical(enc.fit_transform(self.y_train))
        self.y_test = to_categorical(enc.fit_transform(self.y_test))
        self.n_labels = len(y_train[0])

        self.gen = ImageDataGenerator(
            rotation_range=10, zoom_range=0.10, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, fill_mode="nearest"
        )

    @classmethod
    def load_from_npy(cls, dir):
        X_train = np.load(os.path.join(dir, "X_train.npy"))
        X_test = np.load(os.path.join(dir, "X_test.npy"))
        y_train = np.load(os.path.join(dir, "y_train.npy"))
        y_test = np.load(os.path.join(dir, "y_test.npy"))
        
        return Data(X_train, X_test, y_train, y_test)

    @staticmethod
    def save_from_imgs(data_dir):
        self.data_dir = data_dir
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
            
        
if __name__ == "__main__":
    data = Data.save_from_imgs("data")