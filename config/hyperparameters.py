
import numpy as np
import tensorflow as tf
import random
from keras import backend as K

SEED = 1337
np.random.seed(SEED)
random.seed(SEED)
DEVICE= 'Desktop'
GPU = 0
VERSION = 1

BATCH = 128
TRAIN_FREQ = 4
EXP_MEMORY = 500000
HARD_UPDATE_FREQUENCY = 2000
LEARNING_RATE = 0.0001

maxStepsPerEpisode = 500

goal_to_train = [0,1,2,3,4,5,6]


recordFolder = "tfevents_v"+str(VERSION)
recordFileName = "result_v"+str(VERSION)


STOP_TRAINING_THRESHOLD = 0.90

HIDDEN_NODES = 512


#### Things that likely won't change
defaultGamma = 0.99
nb_Action = 8
nb_Option = 7
TRAIN_HIST_SIZE = 10000
EPISODE_LIMIT = 50000
STEPS_LIMIT = 5000000
