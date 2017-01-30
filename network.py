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

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# NOTE: the quality of the following code is highly suspect. have mercy.

CROPPED_WIDTH = 320
CROPPED_HEIGHT = 108
COLOR_CHANNELS = 3

RESIZE_DIMENSIONS = (CROPPED_WIDTH, CROPPED_HEIGHT)
INPUT_DIMENSIONS = (1,) + RESIZE_DIMENSIONS + (COLOR_CHANNELS,)

CAMERA_INPUTS = 1

if CAMERA_INPUTS > 1:
    INPUT_DIMENSIONS = (CAMERA_INPUTS,) + RESIZE_DIMENSIONS

# hyperparameters to tune
BATCH_SIZE = 128
SAMPLES_PER_EPOCH = 10
NB_EPOCHS = 10
KEEP_PROB = 0.25
LEARNING_RATE = 0.001


# DATA PREPROCESSING
def split_train_val(csv_driving_data, test_size=0.2):
    csv_driving_data = normpath(os.getcwd() + csv_driving_data)
    with open(csv_driving_data, 'r') as f:
        reader = csv.reader(f)
        driving_data = [row for row in reader][1:]

    train_data, val_data = train_test_split(driving_data, test_size=test_size, random_state=1)
    return np.array(train_data), np.array(val_data)


def image_filter(fpath):
    a = np.float32(imresize(scipy.ndimage.imread(normpath(fpath).replace(" ", ""), mode='RGB'), RESIZE_DIMENSIONS))
    return a


def create_generator(data_points):
    """ data_point:
    [ 'C:\\Users\\david\\Desktop\\sel_driving\\data\\IMG\\center_2017_01_26_01_28_23_901.jpg'
      'C:\\Users\\david\\Desktop\\sel_driving\\data\\IMG\\left_2017_01_26_01_28_23_901.jpg'
      'C:\\Users\\david\\Desktop\\sel_driving\\data\\IMG\\right_2017_01_26_01_28_23_901.jpg'
     '-0.3452184' '0.1864116' '0' '27.59044']
    """
    for row in data_points:
        # set temp_item numpy arrays for each camera
        # for i in range(CAMERA_INPUTS):
            # fpath to the image file
            # temp_item[i] = image_filter(row[i])
        if row[0] is None or row[3] is None: continue
        temp_item = image_filter(row[0])

    yield temp_item, np.float32(row[3])


# read training data
train_data, val_data = split_train_val(csv_driving_data='/data/driving_log.csv')

# create testing and validation sets out of the training data
train_samples = create_generator(train_data)
val_samples = create_generator(val_data)

# NOTE debugging code for checking training item
# yolo = next(train_samples)
# print(yolo, yolo[0].shape, ndim(convert_to_tensor(yolo[0])))

# MODEL
model = Sequential()

# TODO try ELU
model.add(Lambda(lambda x: x / 255. - .5, input_shape=(320, 108, 3)))
# model.add(BatchNormalization(input_shape=(66,200, 3), axis=1))
# model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), input_shape=INPUT_DIMENSIONS, name='conv1'))
model.add(Convolution2D(24, 5, 5, border_mode='same', init='he_normal', subsample=(2, 2), name='conv1', activation="relu"))
# model.add(Activation('relu'))
model.add(Dropout(KEEP_PROB))
model.add(Convolution2D(36, 5, 5, init='he_normal', subsample=(2, 2), name='conv2', activation="relu"))
# model.add(Activation('relu'))
model.add(Dropout(KEEP_PROB))
model.add(Convolution2D(48, 5, 5, init='he_normal', subsample=(2, 2), name='conv3', activation="relu"))
# model.add(Activation('relu'))
model.add(Dropout(KEEP_PROB))
model.add(Convolution2D(64, 3, 3, init='he_normal', subsample=(1, 1), name='conv4', activation="relu"))
# model.add(Activation('relu'))
model.add(Dropout(KEEP_PROB))
model.add(Convolution2D(64, 3, 3, init='he_normal', subsample=(1, 1), name='conv5', activation="relu"))
# model.add(Activation('relu'))
model.add(Dropout(KEEP_PROB))

model.add(Flatten())

# TODO try removing 1164 layer
model.add(Dense(1164, init='he_normal', name="dense_1164", activation='relu'))
model.add(Dense(100, init='he_normal', name="dense_100", activation='relu'))
model.add(Dropout(KEEP_PROB))
model.add(Dense(50, init='he_normal', name="dense_50", activation='relu'))
model.add(Dropout(KEEP_PROB))
model.add(Dense(10, init='he_normal', name="dense_10", activation='relu'))
model.add(Dropout(KEEP_PROB))
model.add(Dense(1, init='he_normal', name="dense_1"))
# model.summary()

# Compile and train the model here.
model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=LEARNING_RATE),
              metrics=['mean_squared_error'])

model.fit_generator(train_samples,
                    samples_per_epoch=SAMPLES_PER_EPOCH,
                    nb_epoch=NB_EPOCHS,
                    verbose=1,
                    nb_val_samples=128,
                    validation_data=val_samples)

# evaluate model on training set
score = model.evaluate_generator(val_samples, verbose=1)

# POST PROCESSING, SAVE MODEL TO DISK
with open('model.json', 'w') as json_file:
    json_file.write(model.to_json())

# save weights as model.h5
model.save_weights('model.h5')

print('Test score:', score[0])
print('Test accuracy:', score[1])
print('Saved model to disk.')
