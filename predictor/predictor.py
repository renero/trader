# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from cs_api import save_predictions, reorder_predictions, display_predictions
from cs_core import CSCore
from cs_logger import CSLogger
from params import Params
from ticks import Ticks

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(1)

params = Params(args=sys.argv)
log = CSLogger(params._log_level)
ticks = Ticks()
data = ticks.read_ohlc()
cs_model = CSCore()

if params.do_train is True:
    cs_model.train(data)
else:
    nn, encoder = cs_model.prepare_predict()
    if params._predict_training:
        predictions = cs_model.predict_training(data, nn, encoder, ticks)
    else:
        predictions = cs_model.predict_newdata(data, nn, encoder, ticks)

    predictions = reorder_predictions(predictions, params)
    save_predictions(predictions, params, log)
    display_predictions(predictions)

#
# EOF
#
