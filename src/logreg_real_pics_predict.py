#Import modules
from skimage.transform import resize
from skimage.color import rgb2gray
import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.utils import shuffle
import skimage.io

from utils import heatmap, accuracy

size=224

#Defining lists
real_pic_mask_list = []
real_pic_nomask_list = []

#Extract the images from the folders and makes them into 224x224 sizes for faster
#handling of the images.
for filename in glob.glob('real_pictures/mask/*.jpg'): #assuming gif
    im1=skimage.io.imread(filename)
    im1 = np.asarray(im1)
    im1 = resize(im1, (size, size), preserve_range=True, mode='constant')
    im1 /= 255.
    real_pic_mask_list.append(im1)

for filename in glob.glob('real_pictures/without_mask/*.jpg'): #assuming gif
    im1=skimage.io.imread(filename)
    im1 = np.asarray(im1)
    im1 = resize(im1, (size, size), preserve_range=True, mode='constant')
    im1 /= 255.
    real_pic_nomask_list.append(im1)

#Saving the lists in arrays
X_real_pic_mask = np.asarray(real_pic_mask_list)
X_real_pic_nomask = np.asarray(real_pic_nomask_list)

#Load the data
X_train_mask=np.load('X_train_mask.npy')
X_train_nomask=np.load('X_train_nomask.npy')

#Making y values: 1= mask, 0=no masks
y_train_mask=np.ones(len(X_train_mask))
y_train_nomask=np.zeros(len(X_train_nomask))
y_real_pic_mask=np.ones(len(X_real_pic_mask))
y_real_pic_nomask=np.zeros(len(X_real_pic_nomask))

#Making a larger testset by concatenating the train and testsets
X_train = np.concatenate((X_train_mask, X_train_nomask), axis=0)
X_test_pics = np.concatenate((X_real_pic_mask, X_real_pic_nomask), axis=0)
y_train= np.concatenate((y_train_mask, y_train_nomask), axis=None)
y_test_pics= np.concatenate((y_real_pic_mask, y_real_pic_nomask), axis=None)

#Removing the three colour channels
X_train= rgb2gray(X_train)
X_test_pics = rgb2gray(X_test_pics)

#Shuffling the data
X_train, y_train = shuffle(X_train, y_train)

#Resahping the data
X_train= X_train.reshape(X_train.shape[0],224*224)
X_test_pics= X_test_pics.reshape(X_test_pics.shape[0],224*224)

def log_reg_scikit_learn(X_train, X_test, y_test, y_train):
    #Using sklearns logisitc regression class
    log_reg_scikit= lm.LogisticRegression(solver='newton-cg', multi_class='multinomial',max_iter=500, penalty='none')
    log_reg_scikit.fit(X_train, y_train)

    y_pred_train=log_reg_scikit.predict(X_train)
    y_pred_test=log_reg_scikit.predict(X_test)

    return y_pred_test

y_pred=log_reg_scikit_learn(X_train, X_test_pics, y_test_pics,y_train)


#plt.rc('title', labelsize=13)
plt.rcParams["axes.grid"] = False
fig, axes = plt.subplots(1, 4, figsize=(10, 10))
ax = axes.ravel()
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large'}
plt.rcParams.update(params)
ax[0].imshow(real_pic_mask_list[0])
ax[0].title.set_text(f'Solution=1\nPredicted={y_pred[0]}')
ax[1].imshow(real_pic_mask_list[1])
ax[1].title.set_text(f'Solution=1\nPredicted={y_pred[1]}')
ax[2].imshow(real_pic_nomask_list[1])
ax[2].title.set_text(f'Solution=0\nPredicted={y_pred[2]}')
ax[3].imshow(real_pic_nomask_list[0])
ax[3].title.set_text(f'Solution=0\nPredicted={y_pred[3]}')

fig.tight_layout()
plt.show()
