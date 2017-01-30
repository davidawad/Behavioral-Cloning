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

import config
from config import *

# NOTE: the quality of the following code is highly suspect. have mercy.

# read training data
train_data, val_data = split_train_val(csv_driving_data='/data/driving_log.csv')

# create testing and validation sets out of the training data
train_samples = create_generator(train_data)
val_samples = create_generator(val_data)

# TODO debugging code for checking training item
yolo = next(train_samples)
print(yolo, yolo[0].shape, ndim(convert_to_tensor(yolo[0])))

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

# TODO remove 1164 layer
model.add(Dense(1164, init='he_normal', name="dense_1164", activation='relu'))
model.add(Dense(100, init='he_normal', name="dense_100", activation='relu'))
model.add(Dropout(KEEP_PROB))
model.add(Dense(50, init='he_normal', name="dense_50", activation='relu'))
model.add(Dropout(KEEP_PROB))
model.add(Dense(10, init='he_normal', name="dense_10", activation='relu'))
model.add(Dropout(KEEP_PROB))
model.add(Dense(1, init='he_normal', name="dense_1"))
model.summary()

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
# score = model.evaluate_generator(val_samples, verbose=1)

# POST PROCESSING, SAVE MODEL TO DISK
with open('model.json', 'w') as json_file:
    json_file.write(model.to_json())

# save weights as model.h5
model.save_weights('model.h5')

print('Test score:', score[0])
print('Test accuracy:', score[1])
print('Saved model to disk.')
