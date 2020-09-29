import os

os.environ['PYTHONHASHSEED'] = '0'

import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def reset_seeds():
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    print("KERAS AND TENSORFLOW GRAPHS RESET")  # optional

    np.random.seed(1)
    random.seed(2)
    tf.compat.v1.set_random_seed(3)
    print("RANDOM SEEDS RESET")  # optional
