#Import modules
import sklearn.linear_model as lm
from sklearn.utils import shuffle

import numpy as np
import time

from utils import heatmap

# load and show an image with Pillow
from PIL import Image
import glob
from matplotlib import image, pyplot
import random

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
    im2=Image.open(filename)
    im=im2.resize((100,100))
    train_mask_list.append(im)
    im.close()

for filename in glob.glob('facemasks/data/train/without_mask/*'): #assuming gif
    im2=Image.open(filename)
    im=im2.resize((100,100))
    train_nomask_list.append(im)
    im.close()

for filename in glob.glob('facemasks/data/test/with_mask/*'): #assuming gif
    im2=Image.open(filename)
    im=im2.resize((100,100))
    test_mask_list.append(im)
    im.close()

for filename in glob.glob('facemasks/data/test/without_mask/*'): #assuming gif
    im2=Image.open(filename)
    im=im2.resize((100,100))
    test_nomask_list.append(im)
    im.close()


#Saving the lists in arrays
"""Might have to concatenate train and test, and split it up into a bigger testset"""
X_train_mask = np.asarray(train_mask_list)
X_train_nomask = np.asarray(train_nomask_list)
X_test_mask = np.asarray(test_mask_list)
X_test_nomask = np.asarray(test_nomask_list)

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

#Ravels and reshaping
X_train.ravel()
X_test.ravel()

colours=1
#or is the colours allready in the array, so we dont need to multiply with 3??
X_train_reshape=X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*colours)
X_test_reshape= X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*colours)

#Scaling the sets
X_train_scaled=X_train_reshape/np.max(X_train)
X_test_scaled=X_test_reshape/np.max(X_test)

#Shuffling the images
X_train_scaled, y_train = shuffle(X_train_scaled, y_train, random_state=0)
X_test_scaled, y_test = shuffle(X_test_scaled, y_test, random_state=0)



"""
image = Image.open('00034.jpg')
# convert image to numpy array
data = np.asarray(image)
# summarize shape
print(data.shape)
# create Pillow image
image2 = Image.fromarray(train_mask[8])
# summarize image details
print(image2.format)
print(image2.mode)
print(image2.size)

# load image as pixel array
#data = image.imread('00034.jpg')
# summarize shape of the pixel array
#print(data.dtype)
#print(data.shape)
# display the array of pixels as an image
"""


#Just a function to get some overview of the dataset
def info_about_the_data(y_train=y_train, y_test=y_test):
    articles=['tops', 'trousers', 'pullovers', 'dresss', 'coats', 'sandals', 'shirts', 'sneakers', 'bags', 'ankle_boots']
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

#Gives 84% sucess rate, might give higher with a different solver for example saga
def log_reg_scikit_learn(X_train=X_train_scaled, X_test=
        X_test_scaled, y_test=y_test, y_train=y_train):

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
log_reg_scikit_learn()
