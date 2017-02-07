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


# NVIDIA MODEL
def nvidia_model():
    print('Nvidia Model...')
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


def experimental_cnn():
    model = Sequential()
    # 2 CNNs blocks comprised of 32 filters of size 3x3.
    model.add(ZeroPadding2D((1, 1), input_shape=INPUT_DIMENSIONS))
    model.add(Convolution2D(32, 3, 3, activation='elu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='elu'))
    # Maxpooling
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # 2 CNNs blocks comprised of 64 filters of size 3x3.
    model.add(ZeroPadding2D((1, 1), input_shape=INPUT_DIMENSIONS))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    # Maxpooling + Dropout to avoid overfitting
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 2 CNNs blocks comprised of 128 filters of size 3x3.
    model.add(ZeroPadding2D((1, 1), input_shape=INPUT_DIMENSIONS))
    model.add(Convolution2D(128, 3, 3, activation='elu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='elu'))
    # Last Maxpooling. We went from an image (64, 64, 3), to an array of shape (8, 8, 128)
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Fully connected layers part.
    model.add(Flatten(input_shape=(3, CROPPED_WIDTH, CROPPED_HEIGHT)))
    model.add(Dense(256, activation='elu'))
    # Dropout here to avoid overfitting
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='elu'))
    # Last Dropout to avoid overfitting
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(1))

    return model
