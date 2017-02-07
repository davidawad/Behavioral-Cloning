##
# A bunch of constants for use in other files


# Image dimensions to resize to
CROPPED_WIDTH  = 200
CROPPED_HEIGHT = 66
COLOR_CHANNELS = 3

RESIZE_DIMENSIONS = (CROPPED_HEIGHT, CROPPED_WIDTH)
INPUT_DIMENSIONS = RESIZE_DIMENSIONS + (COLOR_CHANNELS,)

# hyperparameters to tune
BATCH_SIZE = 128           # number of samples in a batch of data in an epoch
SAMPLES_PER_EPOCH = 2048   # number of times the generator will yield per epoch
NB_EPOCHS = 100             # number of epochs to train for
KEEP_PROB = 0.25           # keep probability for hinton's dropout
LEARNING_RATE = 0.0001     # learning rate for convnet
ALPHA = 1.0                # ELU alpha param
