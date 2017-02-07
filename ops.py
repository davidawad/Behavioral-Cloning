##
# Contains various utility functions used in drive and train_model
import cv2
import math
import numpy as np

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

# keras contents
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.layers.advanced_activations import ELU as elu
from keras.layers import Flatten, ZeroPadding2D, MaxPooling2D, Activation, Dropout, Convolution2D
from keras.layers import Dense, Input, Activation, BatchNormalization, Lambda, ELU
from keras.models import model_from_json, Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.backend import ndim

from tensorflow.python.framework.ops import convert_to_tensor

import numpy as np
# Fix obscure error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

from config import *


# our image preprocessing scheme
def preprocess_image(image):
    """
    takes an image array and crops and normalizes it for training on the neural network
    """
    image_shape = image.shape

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    norm_image = np.zeros(image.shape)
    # normalize image colors for faster training
    image = cv2.normalize(image, norm_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # crop out the top 1/3 with horizon line & bottom 25px containing hood of the car
    image = image[math.floor(image_shape[0] / 5):image_shape[0] - 25, 0:image_shape[1]]
    # resize the image to our desired output dimensions
    image = cv2.resize(image, (CROPPED_WIDTH, CROPPED_HEIGHT), interpolation=cv2.INTER_AREA)
    return np.float32(image)


# randomly augment brightness
def augment_brightness_camera_images(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    bright_rand = .25 + np.random.uniform()

    image[:, :, 2] = image[:, :, 2] * bright_rand
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image


# translate an image randomly within a certain range
def trans_image(image, angle, trans_range):
    # Translation
    x_shift = trans_range * np.random.uniform() - trans_range / 2
    shifted_angle = angle + x_shift / trans_range * 2 * .2
    y_shift = 40 * np.random.uniform() - 40 / 2

    modifier = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    translated_image = cv2.warpAffine(image, modifier, (CROPPED_WIDTH, CROPPED_HEIGHT))

    return translated_image, shifted_angle


# add random shadows to the image
def add_random_shadows(image):
    top_y = CROPPED_HEIGHT * np.random.uniform()
    top_x = 0
    bot_x = CROPPED_WIDTH
    bot_y = CROPPED_HEIGHT * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    mask = 0 * image_hls[:, :, 1]
    mask_x = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    mask_y = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

    mask[((mask_x - top_x) * (bot_y - top_y) - (bot_x - top_x) * (mask_y - top_y) >= 0)] = 1

    if np.random.randint(2) == 1:
        bright_rand = .5
        cond1 = mask == 1
        cond0 = mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * bright_rand
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * bright_rand
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image


def split_data_sets(csv_driving_data, test_size=0.2):
    csv_driving_data = normpath(os.getcwd() + csv_driving_data)
    with open(csv_driving_data, 'r') as f:
        reader = csv.reader(f)
        driving_data = [row for row in reader][1:]

        train_data, val_data = train_test_split(driving_data, test_size=test_size, random_state=1)
        val_data, test_data = train_test_split(val_data, test_size=0.1, random_state=1)
        return train_data, val_data, test_data


def create_valid_generator(data_points, batch_size=BATCH_SIZE):
    """
    generator for purely vavlidation data
    """
    # propagate batch_images and batch_steering
    batch_images = np.zeros((batch_size, CROPPED_HEIGHT, CROPPED_WIDTH, 3))
    batch_steering = np.zeros(batch_size)

    while True:
        batch_filled = 0
        # TODO change to .shape
        while batch_filled < batch_size:

            # grab a random training example
            row = data_points[np.random.randint(len(data_points))]

            impath = row[0]  # set image path
            angle = float(row[3])           # read steering angle

            # read our image from the camera of choice
            impath = os.path.normpath(os.getcwd() + "/data/" + impath).replace(" ", "")
            # impath = os.path.normpath(impath).replace(" ", "")
            image = cv2.imread(impath)

            if image is None or angle == 0.0: continue

            image = preprocess_image(image)

            # fill batch of data
            batch_images[batch_filled] = image
            batch_steering[batch_filled] = angle
            batch_filled += 1
        yield batch_images, batch_steering


def create_generator(data_points, batch_size=BATCH_SIZE):
    """
    generator for training data
    """
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
            min_ang_threshold = 0.8
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
            impath = os.path.normpath(os.getcwd() + "/data/" + impath).replace(" ", "")
            # impath = os.path.normpath(impath).replace(" ", "")
            # NOTE: cv2.imread takes images in BGR format, NOT RGB
            image = cv2.imread(impath)

            if image is None or angle == 0.0: continue

            angle = angle + shift_ang

            # do the actual image preprocessing and cropping
            image = preprocess_image(image)

            # translate the image randomly to better simulate road conditions
            image, angle = trans_image(image, angle, 100)

            # add random shadow
            image = add_random_shadows(image)

            # augment brightness
            image = augment_brightness_camera_images(image)

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
