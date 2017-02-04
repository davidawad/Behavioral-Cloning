import cv2
import math
import numpy as np

# TODO read this from a config file

# Image dimensions of input
CROPPED_WIDTH = 200
CROPPED_HEIGHT = 66
COLOR_CHANNELS = 3


# our image preprocessing scheme
def preprocess_image(image):
    """
    takes an image array and crops and normalizes it for training on the neural network
    """
    # image is a numpy array
    image_shape = image.shape
    # img = img[60:img_shape[0] - 25, 0:img_shape[1]] # STOCK CROP

    # don't remove this!
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # TODO color normalization between 0 and 1  ??
    # image = image / 255. - .5
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
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    # print(random_bright)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1

# translate an image randomly within a certain range
def trans_image(image, angle, trans_range):
    # Translation
    x_shift = trans_range * np.random.uniform() - trans_range / 2
    shifted_angle = angle + x_shift / trans_range * 2 * .2
    y_shift = 40 * np.random.uniform() - 40 / 2
    # y_shift = 0
    Trans_M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    image_tr = cv2.warpAffine(image, Trans_M, (CROPPED_WIDTH, CROPPED_HEIGHT))

    return image_tr, shifted_angle

# add random shadows to the image
def add_random_shadow(image):
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    shadow_mask = 0 * image_hls[:, :, 1]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    # random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2) == 1:
            image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
        else:
            image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
    image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
    return image
