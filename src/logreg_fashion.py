#Import modules
from tensorflow.keras.datasets import fashion_mnist
import sklearn.linear_model as lm
import numpy as np
import time

from utils import heatmap

#Setting up the data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X=np.concatenate((X_train,X_test))
y=np.concatenate((y_train,y_test))

#Raveling the X-values
X_train.ravel()
X_test.ravel()

#Scaling the 3D matrix, by splitting it up into multiple 2D matrices
# Reshaping the data by putting (28,28) together as 28x28 and scaling the data
#X_train.shape[2]*3 with color?
X_train_reshape=X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test_reshape= X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])

X_train_scaled=X_train_reshape/np.max(X_train)
X_test_scaled=X_test_reshape/np.max(X_test)

"""
# Note to self, when working with colours reshape dataset to have a single channel
width, height, channels = X_train.shape[1], X_train.shape[2], 1
X_train = X_train.reshape((X_train.shape[0], width, height, channels))
X_test = X_test.reshape((X_test.shape[0], width, height, channels))
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
    log_reg_scikit= lm.LogisticRegression(max_iter=500)
    log_reg_scikit.fit(X_train, y_train)
    y_pred=log_reg_scikit.predict(X_test)
    accuracy_scikit=format(log_reg_scikit.score(X_test,y_test))
    end = time.time()

    #Making a heatmap of the confusion matrix by distributing all the samples
    #and calculating the accuracy
    heatmap(y_test, y_pred, 'fashion')

    print(f" Accuracy: logistic regression using the scikit: {accuracy_scikit}")
    print(f" The scikit function used {end-start} seconds to run")

    return accuracy_scikit




#Calling the functions
info_about_the_data()
log_reg_scikit_learn()
