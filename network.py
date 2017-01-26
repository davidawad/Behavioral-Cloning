# @Author David Awad
# code to create our neural network and save it as a model.
import os
import csv
import cv2
import sys
import math
import pickle
import collections

from os.path import normpath, join

import scipy
from scipy import ndimage
from scipy.misc import imresize

# sklearn
from sklearn.model_selection import train_test_split

# keras contents
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation, Dropout, Convolution2D
from keras.layers import Dense, Input, Activation, BatchNormalization, Lambda
from keras.models import model_from_json, Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.backend import ndim

from tensorflow.python.framework.ops import convert_to_tensor


import numpy as np
# Fix obscure error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


# NOTE: the quality of the following code is highly suspect. have mercy.

# some constant stuff
RESIZE_DIMENSIONS = (60, 220)
batch_size = 128
nb_classes = 10
nb_epochs = 6

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# DATA PREPROCESSING
def split_train_val(csv_driving_data, test_size=0.2):
    """
    Splits the csv containing driving data into training and validation
    :param csv_driving_data: file path of Udacity csv driving data
    :return: train_split, validation_split
    """
    csv_driving_data = normpath(os.getcwd() + csv_driving_data)
    with open(csv_driving_data, 'r') as f:
        reader = csv.reader(f)
        driving_data = [row for row in reader][1:]

    train_data, val_data = train_test_split(driving_data, test_size=test_size, random_state=1)

    return np.array(train_data), np.array(val_data)


def image_filter(fpath, resize_dimensions=RESIZE_DIMENSIONS):
    # return imresize(scipy.ndimage.imread(normpath(fpath).replace(" ", ""), mode='RGB'), resize_dimensions)
    return cv2.resize(scipy.ndimage.imread(normpath(fpath).replace(" ", ""), mode='RGB'), resize_dimensions)


def create_generator(data_points, resize_dimensions=RESIZE_DIMENSIONS):
    """ data_point:  [ 'C:\\Users\\david\\Desktop\\sel_driving\\data\\IMG\\center_2017_01_26_01_28_23_901.jpg'
 ' C:\\Users\\david\\Desktop\\sel_driving\\data\\IMG\\left_2017_01_26_01_28_23_901.jpg'
 ' C:\\Users\\david\\Desktop\\sel_driving\\data\\IMG\\right_2017_01_26_01_28_23_901.jpg'
 ' -0.3452184' ' 0.1864116' ' 0' ' 27.59044']
    """
    for row in data_points:
        temp_item = np.empty((3,) + resize_dimensions)

        # set temp_item numpy arrays for each camera
        for i in range(3):
            temp_item[i] = image_filter(row[i])

        yield convert_to_tensor(temp_item)


def create_labels(data_points):
    for row in data_points:
        yield float(row[3])


train_data, val_data = split_train_val(csv_driving_data='/data/driving_log.csv')

train_samples, train_labels = create_generator(train_data), create_labels(train_data)

val_samples, val_labels = create_generator(val_data), create_labels(val_data)

keep_prob = 0.5

model = Sequential()

# TODO try lambda normalization
# model.add(Lambda((lambda z: z / 127.5 - 1.), input_shape=(66, 200, 3)))
model.add(BatchNormalization(input_shape=(3,) + RESIZE_DIMENSIONS, axis=1))
# model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), input_shape=(66, 200, 3), name='conv1'))
model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), name='conv1'))
model.add(Activation('relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), name='conv2'))
model.add(Activation('relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), name='conv3'))
model.add(Activation('relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), name='conv4'))
model.add(Activation('relu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), name='conv5'))
model.add(Activation('relu'))
model.add(Dropout(keep_prob))

model.add(Flatten())

model.add(Dense(1164))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dense(1))
model.summary()

# Compile and train the model here.
model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=0.001),
              metrics=['mean_squared_error'])

model.fit_generator(train_samples,
                    train_labels,
                    batch_size=batch_size,
                    nb_epoch=nb_epochs,
                    verbose=1)

# print(len(x_train), len(y_train), len(x_validation), len(y_validation))

# evaluate model on training set
score = model.evaluate_generator(val_samples, val_labels, verbose=0)


# POST PROCESSING, SAVE MODEL TO DISK

# save the model to disk
with open('model.json', 'w') as json_file:
    json_file.write(model.to_json())

# save weights as model.h5
model.save_weights('model.h5')

print('Test score:', score[0])
print('Test accuracy:', score[1])
print('Saved model to disk.')
