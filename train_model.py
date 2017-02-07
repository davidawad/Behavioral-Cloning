# @Author David Awad
# code to create our neural network and save it as a model.
# from __future__ import unicode_literals
import os
import csv
import cv2
import math
import json
import pickle
import random
import collections

from os.path import normpath, join

import scipy
from scipy import ndimage
from scipy.misc import imresize

from sklearn.model_selection import train_test_split

import pydot


# keras contents
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import model_from_json, Sequential
from keras.utils import np_utils
from keras.layers.advanced_activations import ELU as elu
from keras.layers import Flatten, ZeroPadding2D, MaxPooling2D, Activation, Dropout, Convolution2D
from keras.layers import Dense, Input, Activation, BatchNormalization, Lambda, ELU
from keras.optimizers import Adam
from keras.backend import ndim

# from keras.utils.visualize_util import plot


from tensorflow.python.framework.ops import convert_to_tensor

import numpy as np
# Fix obscure error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

# user level imports
from ops import *
from config import *
from networks import *

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# create testing and validation sets out of the training data
train_data, val_data, test_data = split_data_sets(csv_driving_data='/data/driving_log.csv')

# create our generators (training, validation, test) for keras.

# training set, used for training the neural network
training_set = create_generator(train_data)
# validation set, used during training to monitor progress
validation_set = create_valid_generator(val_data)
# testing set, used only once after the training is finished
testing_set = create_valid_generator(test_data)

# select model
model = experimental_cnn()

# model.summary()
model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

model.fit_generator(training_set,
                    samples_per_epoch=SAMPLES_PER_EPOCH,
                    nb_epoch=NB_EPOCHS,
                    nb_val_samples=128,
                    validation_data=validation_set)

# save plot of our model to memory
# plot(model, to_file='model.png')

# evaluate model on the testing set
score = model.evaluate_generator(testing_set, val_samples=256)

# POST PROCESSING, SAVE MODEL TO DISK
with open('model.json', 'w') as json_file:
    json_file.write(model.to_json())

# save weights as model.h5
model.save_weights('model.h5')

print('Test score:', score)
print('Saved model to disk successfully')
