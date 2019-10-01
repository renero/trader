# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from cs_core import CSCore
from logger import Logger
from params import Params
from ticks import Ticks

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(1)

params = Params(args=sys.argv)
log = Logger(params._log_level)
ticks = Ticks()
data = ticks.read_ohlc()
predictor = CSCore()

if params.do_train is True:
    predictor.train(data)
else:
    nn, encoder = predictor.prepare_predict()
    if params._predict_training:
        predictions = predictor.predict_training(data, nn, encoder, ticks)
    else:
        predictions = predictor.predict_newdata(data, nn, encoder, ticks)

    predictions = predictor.reorder_predictions(predictions, params)
    predictor.save_predictions(predictions, params, log)
    predictor.display_predictions(predictions)

#
# EOF
#
