import argparse
import base64
import json
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
import math
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import scipy
from scipy import ndimage
from scipy.misc import imresize


# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

# TODO share this in config file
CROPPED_WIDTH = 200
CROPPED_HEIGHT = 66

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


from image_ops import *


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    # Preprocessing
    image_array = preprocess_image(image_array)
    # adjust dim for keras
    image_array = image_array[None, :, :, :]

    # import pdb; pdb.set_trace()

    # the model assumes that the features are just the image arrays.
    steering_angle = float(model.predict(image_array, batch_size=1))

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    # throttle = 0.2

    throttle = max(0.1, -0.15 / 0.05 * abs(steering_angle) + 0.35)
    # slow down for turns
    # if abs(steering_angle) > .07:
    #     throttle = .05
    print('Angle: {0:02f}, Throttle: {1:02f}'.format(steering_angle, throttle))
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
