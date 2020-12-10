#Import modules
import sklearn.linear_model as lm
from sklearn.utils import shuffle
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import scale, StandardScaler

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


#Load the data
X_train_mask=np.load('X_train_mask.npy')
X_train_nomask=np.load('X_train_nomask.npy')
X_test_mask=np.load('X_test_mask.npy')
X_test_nomask=np.load('X_test_nomask.npy')

print(len(X_train_mask))
print(len(X_train_nomask))
print(len(X_test_mask))
print(len(X_test_nomask))

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

X_train, y_train = shuffle(X_train, y_train, random_state=0)
X_test, y_test = shuffle(X_test, y_test, random_state=0)

"""Increase size of test set"""
X=np.concatenate((X_train,X_test), axis=0)
y=np.concatenate((y_train,y_test), axis=None)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

y_train=np.ravel(y_train)
y_test=np.ravel(y_test)

# Scaling the data using the scikit learn modules
scaler = StandardScaler();  scaler.fit(X_train)
X_train= scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Raveling
#X_train=np.ravel(X_train)
#X_test=np.ravel(X_test)

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
log_reg_scikit_learn()
