#Import modules
import sklearn.linear_model as lm
from sklearn.utils import shuffle
from sklearn.model_selection import  train_test_split
import numpy as np
import time
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

from utils import heatmap, accuracy

#Original test set=0, new larger testset=1
original_test_set=0

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
X_train_orig = np.concatenate((X_train_mask, X_train_nomask), axis=0)
X_test_orig = np.concatenate((X_test_mask, X_test_nomask), axis=0)

y_train= np.concatenate((y_train_mask, y_train_nomask), axis=None)
y_test= np.concatenate((y_test_mask, y_test_nomask), axis=None)

#If you only want to use the train set and split it set to 1, if you want to
#use the offered test data as the test set set to 0
if original_test_set==1:
    #Making the arrays ready for a full shuffling
    X=X_train_orig
    y=y_train
    X=rgb2gray(X)

    #Shuffling the data
    X, y = shuffle(X, y)

    #Splitting up into a test and training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
else:
    X_train_grey= rgb2gray(X_train_orig)
    X_test_grey = rgb2gray(X_test_orig)

    #Shuffling the data
    X_train, y_train = shuffle(X_train_grey, y_train)
    X_test, y_test = shuffle(X_test_grey, y_test)

#Resahping the data
X_train= X_train.reshape(X_train.shape[0],224*224)
X_test= X_test.reshape(X_test.shape[0],224*224)

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

    #Using sklearns logisitc regression class
    log_reg_scikit= lm.LogisticRegression(solver='newton-cg', multi_class='multinomial',max_iter=500, penalty='none')
    log_reg_scikit.fit(X_train, y_train)
    #Predicting
    y_pred_train=log_reg_scikit.predict(X_train)
    y_pred_test=log_reg_scikit.predict(X_test)

    #Calculating the accuracy score
    accuracy_scikit_test=format(log_reg_scikit.score(X_test,y_test))
    accuracy_scikit_train=format(log_reg_scikit.score(X_train,y_train))

    #Making a heatmap of the confusion matrix by distributing all the samples
    #and calculating the accuracy
    heatmap(y_test, y_pred_test, 'mask')

    print(f" Accuracy: logistic regression using the scikit, train accuracy: {accuracy_scikit_train}")
    print(f" Accuracy: logistic regression using the scikit, test accuracy {accuracy_scikit_test}")

    return y_pred_test

#Calling the functions
#info_about_the_data()
y_pred=log_reg_scikit_learn()

"""
#Finding the misclassified images
wrong_pred_mask=[]
wrong_pred_nomask=[]
k=0
while k<59:
    if y_pred[k]!=y_test[k]:
        if y_test[k]==1:
            wrong_pred_mask.append(k)
        else:
            wrong_pred_nomask.append(k)
    k+=1

plt.rcParams["axes.grid"] = False
fig, axes = plt.subplots(1, 4, figsize=(6, 6))
ax = axes.ravel()
print(wrong_pred_nomask)
print(wrong_pred_mask)
#Printing some of the wrongly predicted samples
ax[0].imshow(X_test_mask[wrong_pred_mask[2]])
ax[0].title.set_text(f'Solution={y_test_mask[wrong_pred_mask[2]]}\nPredicted={y_pred[wrong_pred_mask[2]]}')
ax[1].imshow(X_test_mask[wrong_pred_mask[3]])
ax[1].title.set_text(f'Solution={y_test_mask[wrong_pred_mask[3]]}\nPredicted={y_pred[wrong_pred_mask[3]]}')
ax[2].imshow(X_test_nomask[wrong_pred_nomask[0]-len(y_test_mask)])
ax[2].title.set_text(f'Solution={y_test_nomask[wrong_pred_nomask[0]-len(wrong_pred_mask)]}\nPredicted={y_pred[wrong_pred_nomask[0]-len(wrong_pred_mask)]}')
ax[3].imshow(X_test_mask[wrong_pred_mask[20]])
ax[3].title.set_text(f'Solution={y_test_mask[wrong_pred_mask[20]]}\nPredicted={y_pred[wrong_pred_mask[20]]}')

fig.tight_layout()
plt.show()
"""
