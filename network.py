# @Author David Awad
# code to create our neural network and save it as a model.
from __future__ import unicode_literals
import os
import csv
import cv2
import sys
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


# keras contents
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.layers.advanced_activations import ELU as elu
from keras.layers import Flatten, ZeroPadding2D, MaxPooling2D, Activation, Dropout, Convolution2D
from keras.layers import Dense, Input, Activation, BatchNormalization, Lambda, ELU
from keras.models import model_from_json, Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.backend import ndim

from image_ops import *

from tensorflow.python.framework.ops import convert_to_tensor

import numpy as np
# Fix obscure error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# NOTE: the quality of the following code is highly suspect. have mercy.

# Image dimensions to resize to
CROPPED_WIDTH = 200
CROPPED_HEIGHT = 66
COLOR_CHANNELS = 3

RESIZE_DIMENSIONS = (CROPPED_HEIGHT, CROPPED_WIDTH)
INPUT_DIMENSIONS = RESIZE_DIMENSIONS + (COLOR_CHANNELS,)

CAMERA_INPUTS = 1

# hyperparameters to tune
BATCH_SIZE = 64          # number of samples in a batch of data
SAMPLES_PER_EPOCH = 512  # number of times the generator will yield
NB_EPOCHS = 5            # number of epochs to train for
KEEP_PROB = 0.25         # keep probability for hinton's dropout
LEARNING_RATE = 0.0001   # learning rate for convnet
ALPHA = 1.0              # ELU alpha param


def split_train_val(csv_driving_data, test_size=0.2):
    csv_driving_data = normpath(os.getcwd() + csv_driving_data)
    with open(csv_driving_data, 'r') as f:
        reader = csv.reader(f)
        driving_data = [row for row in reader][1:]

        train_data, val_data = train_test_split(driving_data, test_size=test_size, random_state=1)
        return train_data, val_data


def create_generator(data_points, batch_size=BATCH_SIZE):
    # propagate batch_images and batch_steering
    batch_images = np.zeros((batch_size, CROPPED_HEIGHT, CROPPED_WIDTH, 3))
    batch_steering = np.zeros(batch_size)

    while True:
        batch_filled = 0
        # TODO change to .shape
        while batch_filled < batch_size:

            # grab a random training example
            row = data_points[np.random.randint(len(data_points))]

            # select a random camera image and set path to one of our 3 camera images
            camera_selection = np.random.randint(3)
            impath = row[camera_selection]  # set image path
            angle = float(row[3])           # read steering angle

            # TODO make threshold a constant
            # ignore low angles
            min_ang_threshold = 0.2
            if abs(angle) < .1:
                # for every small angle, flip a coin to see if we use it.
                rand = np.random.uniform()
                if rand > min_ang_threshold: continue

            # center cam
            if (camera_selection == 0):
                shift_ang = 0.

            # left cam
            if (camera_selection == 1):
                shift_ang = .30

            # right cam
            if (camera_selection == 2):
                shift_ang = -.30

            # read our image from the camera of choice
            # impath = os.path.normpath(os.getcwd() + "/data/" + impath).replace(" ", "")
            impath = os.path.normpath(impath).replace(" ", "")
            image = cv2.imread(impath)
            angle = angle + shift_ang

            # translate the image randomly to better simulate road conditions
            image, angle = trans_image(image, angle, 100)

            # add random shadow
            image = add_random_shadows(image)

            # augment brightness
            image = augment_brightness_camera_images(image)

            # do the actual image preprocessing and cropping
            image = preprocess_image(image)

            # flip half the images
            flip_prob = np.random.randint(2)
            if flip_prob > 0:
                image = cv2.flip(image, 1)
                angle = -angle

            # fill batch of data
            batch_images[batch_filled] = image
            batch_steering[batch_filled] = angle
            batch_filled += 1
        yield batch_images, batch_steering


# create testing and validation sets out of the training data
train_data, val_data = split_train_val(csv_driving_data='/data/driving_log.csv')

# create our generators for keras.
train_samples = create_generator(train_data)

val_samples = create_generator(val_data)

# NOTE debugging code for checking training item
# yolo = next(val_samples)
# print(yolo, yolo[0], yolo[1], yolo[0].dtype, yolo[1].dtype)
# exit()

# NVIDIA MODEL
def nvidia_model():
    model = Sequential()
    # model.add(Lambda(lambda x: x / 255. - .5, input_shape=INPUT_DIMENSIONS))
#     model.add(BatchNormalization(input_shape=INPUT_DIMENSIONS, axis=1))
    model.add(Convolution2D(24, 5, 5, input_shape=INPUT_DIMENSIONS, border_mode='valid', init='he_normal', subsample=(2, 2), name='conv1'))

    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, border_mode='valid', init='he_normal', subsample=(2, 2), name='conv2'))

    model.add(ELU())
    # model.add(Dropout(KEEP_PROB))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', init='he_normal', subsample=(2, 2), name='conv3'))

    model.add(ELU())
    # model.add(Dropout(KEEP_PROB))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='he_normal', subsample=(1, 1), name='conv4'))

    model.add(ELU())
    # model.add(Dropout(KEEP_PROB))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='he_normal', subsample=(1, 1), name='conv5'))

    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(1164, init='he_normal', name="dense_1164"))

    model.add(ELU())
    model.add(Dense(100, init='he_normal', name="dense_100"))

    model.add(ELU())
    model.add(Dense(50, init='he_normal', name="dense_50"))

    model.add(ELU())
    model.add(Dense(10, init='he_normal', name="dense_10"))

    model.add(ELU())
    model.add(Dense(1, init='he_normal', name="dense_1"))
    return model


def comma_model():
    print('Comma Model...')
    model = Sequential()
    # Color conversion
    model.add(BatchNormalization(input_shape=INPUT_DIMENSIONS, axis=1))
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", activation='elu'))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu'))
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    return model


# select model
model = comma_model()

# model.summary()
model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))

model.fit_generator(train_samples,
                    samples_per_epoch=len(train_data),
                    nb_epoch=NB_EPOCHS,
                    nb_val_samples=128,
                    validation_data=val_samples)

# evaluate model on training set, here "val_samples" specifies the number of elements to test on.
score = model.evaluate_generator(val_samples, val_samples=200)

# POST PROCESSING, SAVE MODEL TO DISK
with open('model.json', 'w') as json_file:
    json_file.write(model.to_json())

# save weights as model.h5
model.save_weights('model.h5')

print('Test score:', score)
print('Saved model to disk successfully')
