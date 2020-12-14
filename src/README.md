This directory does not have a completely unified structure because we've programmed in parallell. Refer to these descriptions to find your way around it:

- Everything related to logistic regression is prepended with `logreg`. They borrow some functions from `utils` and use data located in their respective subdirectories (`fashion_mnist` and `facemasks`)
- `cnn.py` contains the CNN class, which creates and interacts with the Keras convolutional neural network. It acts as a sort of unified wrapper.
- `fashion_mnist` contains the Fashion-MNIST dataset and code related to doing CNN training, assessments and predictions on it.
- `facemasks` contains the facemask dataset and code related to doing CNN training, assessments and predictions on it.