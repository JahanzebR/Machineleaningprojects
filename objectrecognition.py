# In this project, we will be deploying a convolutional neural network (CNN) for object recognition.
# More specifically, we will be using the All-CNN network published in the 2015 ICLR paper,
# "Striving For Simplicity: The All Convolutional Net".
# This paper can be found at the following link:

# https://arxiv.org/pdf/1412.6806.pdf

# This convolutional neural network obtained state-of-the-art performance at object recognition on the CIFAR-10 image dataset in 2015.
# We will build this model using Keras,
# a high-level neural network application programming interface (API) that supports both Theano and Tensorflow backends.
# You can use either backend; however, I will be using Theano.

# In this project, we will learn to:

# Import datasets from Keras
# Use one-hot vectors for categorical labels
# Addlayers to a Keras model
# Load pre-trained weights
# Make predictions using a trained Keras model
# The dataset we will be using is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6000 images per class.
# There are 50000 training images and 10000 test images.

# 1. Loading the Data
# Let's dive right in! In these first few cells, we will import necessary packages, load the dataset, and plot some example images.

from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import sys
import keras
# print('Python:{}'.format(sys.version))
# print('Keras:{}'.format(keras.__version__))


# load the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# Lets determine the dataset characteristics
# print('Training Images:{}'.format(X_train.shape))
# print('Testing Images:{}'.format(X_test.shape))

# A single image
# print(X_train[0].shape)


# create a grid of 3x3 images
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    img = X_train[50 + i].transpose([1, 2, 0])
    plt.imshow(img)


# show the plot
# plt.show()


# Preprocessing the dataset

# fix random seed for reproductibility
seed = 6
np.random.seed(seed)

# load the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize the inputs from 0-255 to 0.0-1.0
X_train = X_test.astype('float32')
X_test = X_test.astype('float32')

X_train = X_test / 255.0
X_test = X_test / 255.0

# print(X_train[0])

# class labels shape
# print(y_train.shape)
# print(y_train[0])


# 6 = [0,0,0,0,0,0,1,0,0,0] one-hot vector

# hot encode outputs
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)
num_class = Y_test.shape[1]

# print(num_class)
# print(Y_train.shape)
# print(Y_train[0])

# Model-C
# Input 32 × 32 RGB image
# 3 × 3 conv. 96 ReLU
# 3 × 3 conv. 96 ReLU
# 3 × 3 max-pooling stride 2
# 3 × 3 conv. 192 ReLU
# 3 × 3 conv. 192 ReLU
# 3 × 3 max-pooling stride 2
# 3 × 3 conv. 192 ReLU
# 1 × 1 conv. 192 ReLU
# 1 × 1 conv. 10 ReLU
# global averaging over 6 × 6 spatial dimensions
# 10 or 100-way softmax

# Start by importing necessary layers
from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D
from keras.optimizers import SGD


# Define the model function
def allcnn(weights=None):

    # define model type-Sequential
    model = Sequential()

    # add model layers
    model.add(Conv2D(96, (3, 3), padding='same', input_shape=(3, 32, 32)))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding='same', strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding='same', strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), padding='valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding='valid'))

    # add Global Averate Pooling Layer with Softmax activation
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    # load the weights
    if weights:
        model.load_weights(weights)

    # return the model
    return model


# define hyper parameters
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

# define weights and build model
weights = 'all_cnn_weights_0.9088_0.4994.hdf5'
model = allcnn(weights)

# define optimizer and compile model
sgd = SGD(lr=learning_rate, decay=weight_decay, momentum=momentum, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# print model summary
# print(model.summary())

# test the model with pretrained weights
scores = model.evaluate(X_test, Y_test, verbose=1)
# print("Accuracy: %.2f%%" % (scores[1] * 100))

# make dictionary of class labels and names
classes = range(0, 10)

names = ['airplane',
         'automobile',
         'bird',
         'cat',
         'deer',
         'dog',
         'frog',
         'horse',
         'ship',
         'truck']

# zip the names and classes to make a dictionary of class_labels
class_labels = dict(zip(classes, names))

# generate batch of 9 images to predict
batch = X_test[100:109]
labels = np.argmax(Y_test[100:109], axis=-1)

# make predictions
predictions = model.predict(batch, verbose=1)

# print our predictions
# print (predictions)

# these are individual class probabilities, should sum to 1.0 (100%)
# for image in predictions:
#     print(np.sum(image))

# create a grid of 3x3 images
fig, axs = plt.subplots(3, 3, figsize = (15, 6))
fig.subplots_adjust(hspace = 1)
axs = axs.flatten()

for i, img in enumerate(batch):

    # determine label for each prediction, set title
    for key, value in class_labels.items():
        if class_result[i] == key:
            title = 'Prediction: {}\nActual: {}'.format(class_labels[key], class_labels[labels[i]])
            axs[i].set_title(title)
            axs[i].axes.get_xaxis().set_visible(False)
            axs[i].axes.get_yaxis().set_visible(False)

    # plot the image
    axs[i].imshow(img.transpose([1,2,0]))

# show the plot
# plt.show()
