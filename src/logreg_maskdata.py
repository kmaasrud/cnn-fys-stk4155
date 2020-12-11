#Import modules
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
import glob
import numpy as np

size=224

#Defining lists
train_mask_list = []
train_nomask_list = []
test_mask_list = []
test_nomask_list = []

#Extract the images from the folders and makes them into 128x128sizes for faster
#handling of the images.

for filename in glob.glob('facemasks/data/train/with_mask/*.jpg'): #assuming gif
    im1=io.imread(filename)
    im1 = np.asarray(im1)
    im1 = resize(im1, (size, size), preserve_range=True, mode='constant')
    im1 /= 255.
    train_mask_list.append(im1)

for filename in glob.glob('facemasks/data/train/without_mask/*'): #assuming gif
    im2=io.imread(filename)
    im2 = resize(im2, (size, size), preserve_range=True, mode='constant')
    im2 /= 255.
    train_nomask_list.append(im2)

for filename in glob.glob('facemasks/data/test/with_mask/*'): #assuming gif
    im3=io.imread(filename)
    im3 = resize(im3, (size, size), preserve_range=True, mode='constant')
    im3 /= 255.
    im3 = np.asarray(im3)
    test_mask_list.append(im3)

for filename in glob.glob('facemasks/data/test/without_mask/*'): #assuming gif
    im4=io.imread(filename)
    im4 = resize(im4, (size, size), preserve_range=True, mode='constant')
    im4 /= 255.
    im4 = np.asarray(im4)
    test_nomask_list.append(im4)

#Saving the lists in arrays
"""Might have to concatenate train and test, and split it up into a bigger testset"""
X_train_mask_original = np.asarray(train_mask_list)
X_train_nomask_original = np.asarray(train_nomask_list)
X_test_mask_original = np.asarray(test_mask_list)
X_test_nomask_original = np.asarray(test_nomask_list)

"""
X_train_mask= rgb2gray(X_train_mask)
X_train_nomask = rgb2gray(X_train_nomask)
X_test_mask = rgb2gray(X_test_mask)
X_test_nomask= rgb2gray(X_test_nomask)
"""

"""
#Resahping the data
X_train_mask= X_train_mask.reshape(X_train_mask.shape[0],size*size)
X_train_nomask= X_train_nomask.reshape(X_train_nomask.shape[0],size*size)
X_test_mask= X_test_mask.reshape(X_test_mask.shape[0],size*size)
X_test_nomask= X_test_nomask.reshape(X_test_nomask.shape[0],size*size)
"""

#print(f"Shape of image array: {X_test_mask.shape}")


np.save('X_train_mask.npy', X_train_mask_original)
np.save('X_train_nomask.npy', X_train_nomask_original)
np.save('X_test_mask.npy', X_test_mask_original)
np.save('X_test_nomask.npy', X_test_nomask_original)
