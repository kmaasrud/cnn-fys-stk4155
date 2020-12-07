#Import modules
import sklearn.linear_model as lm
from sklearn.utils import shuffle

from skimage import io
from skimage.transform import resize

import numpy as np
import time

from utils import heatmap

# load and show an image with Pillow
from PIL import Image
import glob
from matplotlib import image, pyplot
import random
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

random.seed(2020)

#Defining lists
train_mask_list = []
train_nomask_list = []
test_mask_list = []
test_nomask_list = []

#Extract the images from the folders and makes them into 100x100sizes for faster
#handling of the images, can also try with thumbnails(Keeping the ratio)

for filename in glob.glob('facemasks/data/train/with_mask/*'): #assuming gif
    im2=io.imread(filename)
    im2 = resize(im2, (128, 128), preserve_range=True, mode='constant')
    im2 /= 255.
    train_mask_list.append(im2)

for filename in glob.glob('facemasks/data/train/without_mask/*'): #assuming gif
    im2=io.imread(filename)
    im2 = resize(im2, (128, 128), preserve_range=True, mode='constant')
    im2 /= 255.
    train_nomask_list.append(im2)

for filename in glob.glob('facemasks/data/test/with_mask/*'): #assuming gif
    im2=io.imread(filename)
    im2 = resize(im2, (128, 128), preserve_range=True, mode='constant')
    im2 /= 255.
    test_mask_list.append(im2)

for filename in glob.glob('facemasks/data/test/without_mask/*'): #assuming gif
    im2=io.imread(filename)
    im2 = resize(im2, (128, 128), preserve_range=True, mode='constant')
    im2 /= 255.
    test_nomask_list.append(im2)


#How to resize imread???
"""
fig, axes = plt.subplots(4, 5, figsize=(20, 20))
ax = axes.ravel()
images = data.lfw_subset()
for i in range(20):
    ax[i].imshow(images[90+i], cmap=plt.cm.gray)
    ax[i].axis('off')
fig.tight_layout()
plt.show()
"""

#Saving the lists in arrays
"""Might have to concatenate train and test, and split it up into a bigger testset"""

X_train_mask = np.asarray(train_mask_list)
X_train_nomask = np.asarray(train_nomask_list)
X_test_mask = np.asarray(test_mask_list)
X_test_nomask = np.asarray(test_nomask_list)
#X_test_nomask=test_nomask_list

#Making y values: 1= mask, 0=no masks
y_train_mask=np.ones(len(X_train_mask))
y_train_nomask=np.zeros(len(X_train_nomask))
y_test_mask=np.ones(len(X_test_mask))
y_test_nomask=np.zeros(len(X_test_nomask))

#Setting up the training and test sets
X_train = np.concatenate((X_train_mask, X_train_nomask), axis=0)
X_test = np.concatenate((X_test_mask, X_test_nomask), axis=0)
y_train= np.concatenate((y_train_mask, y_train_nomask), axis=None)
y_test= np.concatenate((y_test_mask, y_test_nomask), axis=None)


#Shuffling the images
X_train_scaled, y_train = shuffle(X_train_scaled, y_train, random_state=0)
X_test_scaled, y_test = shuffle(X_test_scaled, y_test, random_state=0)

# Reshape the data, and convert to grayscale
#X_test_nomask= rgb2gray(X_test_nomask)
print(X_test_nomask.shape)
image_shape = X_test_nomask.shape[1:]
IMG_HEIGHT = image_shape[0]
IMG_WIDTH = image_shape[1]

print(IMG_HEIGHT)

#Converting to greyscale?
x_train = rgb2gray(x_train)
x_test = rgb2gray(x_test)

colours=3

#Ravels and reshaping
X_train.ravel()
X_test.ravel()

#or is the colours allready in the array, so we dont need to multiply with 3??
X_train_reshape=X_train.reshape(X_train.shape[0],image.shape[1]*image.shape[2])
X_test_reshape= X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])

np.save('X_train.npy', X_train_reshape)
np.save('X_test.npy', X_test_reshape)
np.save('t_train.npy', y_train)
np.save('y_test.npy', y_test)
