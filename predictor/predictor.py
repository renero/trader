# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from cs_api import split_datasets, train_nn, load_nn, load_encoders, \
    single_prediction, add_supervised_info, save_predictions
from cs_encoder import CSEncoder
from cs_logger import CSLogger
from params import Params
from ticks import Ticks

tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(1)
params = Params(args=sys.argv)
log = CSLogger(params._log_level)
ticks = Ticks()
ohlc_data = ticks.read_ohlc()

if params.do_train is True:

    encoder = CSEncoder().fit(ohlc_data)
    cse = encoder.ticks2cse(ohlc_data)
    dataset = split_datasets(encoder, cse, params.subtypes)
    nn = train_nn(dataset, params.subtypes)
    encoder.save()
else:
    nn = load_nn(params.model_names, params.subtypes)
    encoder = load_encoders(params.model_names)
    predictions = pd.DataFrame([])

    if params._predict_training:
        # TODO: I'm loading the first encoder, but I do need to use
        #       all of them, in case I'm loading more than one network
        cse = encoder[next(iter(params._model_names))].ticks2cse(ohlc_data)

        # TODO: I'm producing only 50 predictions here, to speed up testing.
        # for from_idx in range(0, ticks.shape[0] - params._window_size + 1):
        for from_idx in range(0, 50 - params._window_size + 1):
            tick_group = ohlc_data.iloc[from_idx:from_idx + params._window_size]
            prediction = single_prediction(tick_group, nn, encoder, params)
            prediction = add_supervised_info(
                prediction,
                ohlc_data.iloc[from_idx + params._window_size]['c'],
                params)
            predictions = predictions.append(prediction)
        predictions = ticks.scale_back(predictions)
    else:
        # TODO: take only the 10 last elements, leaving the last as the actual value
        tick_group = ohlc_data.tail(params._window_size)
        prediction = single_prediction(tick_group, nn, encoder, params)
        # TODO: Take the last element as the supervised information.
        prediction = add_supervised_info(prediction, tick_group['c'], params)
        predictions = predictions.append(prediction)
        predictions = ticks.scale_back(predictions)

    save_predictions(predictions, params, log)
#
# EOF
#
