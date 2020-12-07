
"""
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

# load the image
#image = Image.open('00034.jpg')
# summarize some details about the image
#print(image.format)
#print(image.mode)
#print(image.size)
# show the image
#image.show()

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



#Saving the lists in arrays
#Might have to concatenate train and test, and split it up into a bigger testset

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
"""

#Load the data

X_train=np.load('X_train.npy')
X_test=np.load('X_test.npy')
y_train=np.load('y_train.npy')
y_train=np.load('y_test.npy')


#Just a function to get some overview of the dataset
def info_about_the_data(y_train=y_train, y_test=y_test):
    articles=['tops', 'trousers', 'pullovers', 'dresss', 'coats', 'sandals',
    'shirts', 'sneakers', 'bags', 'ankle_boots']
    article_values=np.zeros(len(articles))
    y = np.concatenate((y_train, y_test))
    #Defining the values
    for i in range(0,len(articles)):
        article_values[i]=np.count_nonzero(y==i)

    total=np.sum(article_values)

    #If all elements are the same:
    z=np.count_nonzero(article_values==article_values[1])
    if z==10:
        print(f"The dataset includes {article_values[0]} images of each article type, which is {np.around(article_values[0]*100/(total),1)} % of the dataset each")
    else:
        for k in range(0,len(articles)):
            print(f"The dataset includes {article_values[k]} {articles[k]}, which is {np.around(article_values[k]*100/(total),1)} % of the dataset")

    return

#Function that performs logisitc regression using scikit learn

def log_reg_scikit_learn(X_train=X_train, X_test=
        X_test, y_test=y_test, y_train=y_train):

    #Measuring the the time as in project 2
    #Using sklearns logisitc regression class
    start = time.time()
    log_reg_scikit= lm.LogisticRegression(max_iter=100)
    log_reg_scikit.fit(X_train, y_train)
    y_pred=log_reg_scikit.predict(X_test)
    accuracy_scikit=format(log_reg_scikit.score(X_test,y_test))
    end = time.time()

    #Making a heatmap of the confusion matrix by distributing all the samples
    #and calculating the accuracy
    heatmap(y_test, y_pred, 'mask')

    print(f" Accuracy: logistic regression using the scikit: {accuracy_scikit}")
    print(f" The scikit function used {end-start} seconds to run")

    return accuracy_scikit

#Remember to print some missclassified images


#Calling the functions
#info_about_the_data()
#log_reg_scikit_learn()
