import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class Data:
    @classmethod
    def save_from_imgs(self, data_dir):
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