#Import modules
import sklearn.linear_model as lm
from sklearn.utils import shuffle
import random
from sklearn.model_selection import  train_test_split
from skimage import io
import numpy as np
import time

from utils import heatmap

random.seed(4155)

#Load the data
X_train_mask=np.load('X_train_mask.npy')
X_train_nomask=np.load('X_train_nomask.npy')
X_test_mask=np.load('X_test_mask.npy')
X_test_nomask=np.load('X_test_nomask.npy')

#Making y values: 1= mask, 0=no masks
y_train_mask=np.ones(len(X_train_mask))
y_train_nomask=np.zeros(len(X_train_nomask))
y_test_mask=np.ones(len(X_test_mask))
y_test_nomask=np.zeros(len(X_test_nomask))

#Making a larger testset by concatenating the train and testsets
X_train = np.concatenate((X_train_mask, X_train_nomask), axis=0)
X_test = np.concatenate((X_test_mask, X_test_nomask), axis=0)
y_train= np.concatenate((y_train_mask, y_train_nomask), axis=None)
y_test= np.concatenate((y_test_mask, y_test_nomask), axis=None)

#Making the arrays ready for a full shuffle
X=np.concatenate((X_train,X_test), axis=0)
y=np.concatenate((y_train,y_test), axis=None)

#Shuffling the data
X_and_y =list(zip(X, y))
random.shuffle(X_and_y)
X, y = zip(*X_and_y)

#Splitting up into a test and training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Ravels the values
y_train=np.ravel(y_train)
y_test=np.ravel(y_test)

#Just a function to get some overview of the dataset
def info_about_the_data(y_train=y_train, y_test=y_test):
    articles=['masks', 'no masks']
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
""" Accuracy: logistic regression using the scikit, train accuracy: 0.9996406755300036
 Accuracy: logistic regression using the scikit, test accuracy 0.9547738693467337"""
def log_reg_scikit_learn(X_train=X_train, X_test=
        X_test, y_test=y_test, y_train=y_train):

    #Measuring the time as in project 2
    #Using sklearns logisitc regression class
    start = time.time()
    log_reg_scikit= lm.LogisticRegression(solver='liblinear', max_iter=500)
    log_reg_scikit.fit(X_train, y_train)

    y_pred_train=log_reg_scikit.predict(X_train)
    y_pred_test=log_reg_scikit.predict(X_test)

    accuracy_scikit_test=format(log_reg_scikit.score(X_test,y_test))
    accuracy_scikit_train=format(log_reg_scikit.score(X_train,y_train))
    end = time.time()

    #Making a heatmap of the confusion matrix by distributing all the samples
    #and calculating the accuracy
    heatmap(y_test, y_pred_test, 'mask')

    print(f" Accuracy: logistic regression using the scikit, train accuracy: {accuracy_scikit_train}")
    print(f" Accuracy: logistic regression using the scikit, test accuracy {accuracy_scikit_test}")
    print(f" The scikit function used {end-start} seconds to run")

    return


#Calling the functions
#info_about_the_data()
#log_reg_scikit_learn()
