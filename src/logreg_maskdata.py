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

size=128

#Defining lists
train_mask_list = []
train_nomask_list = []
test_mask_list = []
test_nomask_list = []

#Extract the images from the folders and makes them into 100x100sizes for faster
#handling of the images.


for filename in glob.glob('facemasks/data/train/with_mask/*'): #assuming gif
    im2=io.imread(filename)
    im2 = resize(im2, (size, size), preserve_range=True, mode='constant')
    im2 /= 255.
    train_mask_list.append(im2)

for filename in glob.glob('facemasks/data/train/without_mask/*'): #assuming gif
    im2=io.imread(filename)
    im2 = resize(im2, (size, size), preserve_range=True, mode='constant')
    im2 /= 255.
    train_nomask_list.append(im2)

for filename in glob.glob('facemasks/data/test/with_mask/*'): #assuming gif
    im2=io.imread(filename)
    im2 = resize(im2, (size, size), preserve_range=True, mode='constant')
    im2 /= 255.
    #im2 = np.asarray(im2)
    test_mask_list.append(im2)

for filename in glob.glob('facemasks/data/test/without_mask/*'): #assuming gif
    im2=io.imread(filename)
    im2 = resize(im2, (size, size), preserve_range=True, mode='constant')
    im2 /= 255.
    #im2 = np.asarray(im2)
    test_nomask_list.append(im2)


#Saving the lists in arrays
"""Might have to concatenate train and test, and split it up into a bigger testset"""

X_train_mask = np.asarray(train_mask_list)
X_train_nomask = np.asarray(train_nomask_list)
X_test_mask = np.asarray(test_mask_list)
X_test_nomask = np.asarray(test_nomask_list)
#X_test_nomask=test_nomask_list
"""
#Setting up the training and test sets
X_train = np.concatenate((X_train_mask, X_train_nomask), axis=0)
X_test = np.concatenate((X_test_mask, X_test_nomask), axis=0)
y_train= np.concatenate((y_train_mask, y_train_nomask), axis=None)
y_test= np.concatenate((y_test_mask, y_test_nomask), axis=None)

"""

#print(y_test)
#print(X_test)
#Shuffling the images
#X_train, y_train = shuffle(X_train, y_train, random_state=0)
#X_test, y_test = shuffle(X_test, y_test, random_state=0)
#print(y_test)
#print(X_test)
# Reshape the data, and convert to grayscale
#X_test= rgb2gray(X_test)
#X_test_mask= rgb2gray(X_test_mask)

#X_train_shape= X_train.shape[1:]
#X_test_shape= X_test.shape[1:]
#X_test_shape= X_test_mask.shape[1:]

#Converting to greyscale?
#X_train_grey = rgb2gray(X_train)
# = rgb2gray(X_test)


X_train_mask= rgb2gray(X_train_mask)
X_train_nomask = rgb2gray(X_train_nomask)
X_test_mask = rgb2gray(X_test_mask)
X_test_nomask= rgb2gray(X_test_nomask)



#Ravels and reshaping
#X_train.ravel()
#X_test=np.ravel(X_test)
#X_test_mask=np.ravel(X_test_mask)

""" Ravel etterpå, concatenate, shuffle"""

#or is the colours allready in the array, so we dont need to multiply with 3??
#X_train_reshape=X_train.reshape(X_train.shape[0],X_train_shape[1]*X_train_shape[2])
X_train_mask= X_train_mask.reshape(X_train_mask.shape[0],size*size)
X_train_nomask= X_train_nomask.reshape(X_train_nomask.shape[0],size*size)
X_test_mask= X_test_mask.reshape(X_test_mask.shape[0],size*size)
X_test_nomask= X_test_nomask.reshape(X_test_nomask.shape[0],size*size)

#X_test_reshape= X_test.reshape(X_test.shape[0],size*size)

np.save('X_train_mask.npy', X_test_mask)
np.save('X_train_nomask.npy', X_train_nomask)
np.save('X_test_mask.npy', X_test_mask)
np.save('X_test_nomask.npy', X_test_nomask)
