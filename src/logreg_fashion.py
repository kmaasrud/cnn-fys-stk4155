#Import modules
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import scale, StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
import time

np.random.seed(2020)

#Setting up the data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X=np.concatenate((X_train,X_test))
y=np.concatenate((y_train,y_test))

#Raveling the X-values
X_train.ravel()
X_test.ravel()

"""
#Scaling the 3D matrix, by splitting it up into multiple 2D matrices
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test_scaled= scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
"""
# Reshaping the data by putting (28,28) together as 28x28 and scaling the data
#X_train.shape[2]*3 with color?
X_train_reshape=X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test_reshape= X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])

X_train_scaled=X_train_reshape/np.max(X_train)
X_test_scaled=X_test_reshape/np.max(X_test)

"""
# reshape dataset to have a single channel
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
    start = time.time()
    log_reg_scikit= LogisticRegression(multi_class='multinomial', max_iter=100)
    y_pred=log_reg_scikit.fit(X_train, y_train)
    accuracy_scikit=format(log_reg_scikit.score(X_test,y_test))

    print(f" Accuracy: logistic regression using the scikit: {accuracy_scikit}")

    end = time.time()

    print(f" The scikit function used {end-start} seconds to run")

    return accuracy_scikit


#Not sure if I will use the functions below, with our own SGD, might take too long time to process
#large images

#Finding the best number of mini_batches with a set amount og epochs
def log_reg_best_mini_batch(epochs = 110, X=X, y=y):
    #Making a figure to plot the functions in
    plt.figure()

    n=10
    #Defining empty lists
    accuracy_test=[]
    accuracy_train=[]
    mini_batches_amount=[]

    #Iterating over the batches
    for i in range(n):
        print(f"{i*10} %")

        #Clearing the lists before calculating a new mini batch
        accuracy_test.clear()
        accuracy_train.clear()
        mini_batches_amount.clear()

        #looping over the mini batches
        for j in np.arange(1,150, n):
            #mini_batches_amount.clear()
            mini_batches_amount.append(j)
            test_accuracy_temp, train_accuracy_temp = CV_log_reg(X, y, epochs=epochs, mini_batches=j)
            accuracy_test.append(test_accuracy_temp)
            accuracy_train.append(train_accuracy_temp)

        #Plotting the accuracy scores for the train and test sets
        plt.plot(mini_batches_amount, accuracy_test, 'tab:red')
        plt.plot(mini_batches_amount, accuracy_train, 'tab:green')

    #Standard ploting commands
    plt.xlabel("Number of minibatches")
    plt.ylabel("Accuracy")
    plt.legend(['Test set', 'Train set'])
    #save_fig("LogRegcancer_accuracy_vs_mini_batches")
    plt.show()

    #Finding the best parameters
    best_index=np.where(accuracy_test==np.nanmax(accuracy_test))
    best_mini_batches=(best_index[0][0])*10+1

    print(f" Best amount of minibatches to use: {best_mini_batches}")

    return best_mini_batches

#Finding the best number of epochs with a set amount og mini_batches
def log_reg_best_epochs(mini_batches = 40, X=X, y=y):

    #Making a figure to plot the functions in
    plt.figure()

    n=10
    #Defining empty lists
    accuracy_test=[]
    accuracy_train=[]
    epochs_amount=[]

    #Iterating over the batches
    for i in range(n):
        print(f"{i*10} %")

        #Clearing the lists before calculating a new mini batch
        accuracy_test.clear()
        accuracy_train.clear()
        epochs_amount.clear()

        #looping over the mini batches
        for j in np.arange(1,200, n):
            #epochs_amount.clear()
            epochs_amount.append(j)
            test_accuracy_temp, train_accuracy_temp = CV_log_reg(X, y, epochs=j, mini_batches=mini_batches)
            accuracy_test.append(test_accuracy_temp)
            accuracy_train.append(train_accuracy_temp)

        #Plotting the accuracy scores for the train and test sets
        plt.plot(epochs_amount, accuracy_test, 'tab:red')
        plt.plot(epochs_amount, accuracy_train, 'tab:green')

    #Standard ploting commands
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.legend(['Test set', 'Train set'])
    #save_fig("LogRegcancer_accuracy_vs_mini_batches")
    plt.show()

    #Finding the best parameters
    best_index=np.where(accuracy_test==np.nanmax(accuracy_test))
    best_epochs=(best_index[0][0])*10+1

    print(f" Best amount of minibatches to use: {best_epochs}")

    return best_epochs

#Function that performs logisitc regression using using the code
def logistic_reg(epochs=140, mini_batches=35, X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled, y_test=y_test, y_train=y_train):
    start = time.time()
    # Performs logistic regression
    log_reg_code = LogReg(X_train_scaled, y_train)
    log_reg_code.SGD_logreg(epochs=epochs, mini_batches=mini_batches)
    pred = log_reg_code.predict(X_test_scaled)
    accuracy_code = accuracy(y_test, pred)

    print(f" Accuracy: logistic regression using the code: {accuracy_code}")

    end = time.time()

    print(f" The self writtten function used {end-start} seconds to run")
    return accuracy_code


#Finding the best number of mini_batches and epochs without the L2 parametrization.
#The chosen epochs and minibatches i only optimal for this specific case
def log_reg_best_mini_batch_epoch(X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled, y_test=y_test, y_train=y_train):
    #Defining empty lists
    accuracy_list=[]
    mini_batch_list=[]
    epochs_list=[]

    #Iterating over the batches
    for e in range(40,201, 1):
        print(f"{(e-40)/1.6} %")
        #looping over the mini batches
        for mini in range(1,151, 1):
            #mini_batches_amount.clear()

            log_reg_code = LogReg(X_train_scaled, y_train)
            log_reg_code.SGD_logreg(epochs=e, mini_batches=mini)
            pred = log_reg_code.predict(X_test_scaled)
            accuracy_list.append(accuracy(y_test, pred))
            mini_batch_list.append(mini)
            epochs_list.append(e)

    max_accuracy = max(accuracy_list)
    max_index = accuracy_list.index(max_accuracy)
    best_mini_batch=mini_batch_list[max_index]
    best_epoch=epochs_list[max_index]


    print(f" Best amount of minibatches to use: {best_mini_batch}")
    print(f" Best amount of epochs to use: {best_epoch}")

    print(max_accuracy)
    return


#Calling the functions- Log reg with best parameters is run by running logisitc_reg()
info_about_the_data()
log_reg_scikit_learn()
#log_reg_best_mini_batch()
#log_reg_best_epochs()
#logistic_reg()

#Best parameters in this specific case
#log_reg_best_mini_batch_epoch()
