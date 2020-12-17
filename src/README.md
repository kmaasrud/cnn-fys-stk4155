This directory does not have a completely unified structure because we've programmed in parallell. Refer to these descriptions to find your way around it:

- Everything related to logistic regression is prefixed with `logreg`. They borrow some functions from `utils` and use data located in their respective subdirectories (`fashion_mnist` and `facemasks`). Before running any of the logistic regression scripts related to facemask detection, `logregmaskdata` needs to be run first to generate the data.
- `cnn.py` contains the CNN class, which creates and interacts with the Keras convolutional neural network. It acts as a sort of unified wrapper.
- `fashion_mnist` contains the Fashion-MNIST dataset and code related to doing CNN training, assessments and predictions on it.
- `facemasks` contains the facemask dataset and code related to doing CNN training, assessments and predictions on it.

## About the datasets

### Fashion-MNIST

The Fashion-MNIST dataset is downloaded from [Kaggle](https://www.kaggle.com/zalando-research/fashionmnist) and is an alternative to the original MNIST dataset. It contains simple pictures of different clothing from Zalando, and its output is the category the clothing belongs to. The categories are assigned the following labels

- `0` - T-shirt/top
- `1` - Trouser
- `2` - Pullover
- `3` - Dress
- `4` - Coat
- `5` - Sandal
- `6` - Shirt
- `7` - Sneaker
- `8` - Bag
- `9` - Ankle boot
