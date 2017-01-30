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
        # temp_item = np.zeros((1,) + INPUT_DIMENSIONS)

        # set temp_item numpy arrays for each camera
        # for i in range(CAMERA_INPUTS):
            # fpath to the image file
            # temp_item[i] = image_filter(row[i])
        temp_item = image_filter(row[0])
        yield temp_item, np.float32(row[3])
