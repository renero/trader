#!/Users/renero/anaconda3/envs/py36/bin/python

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from cs_api import prepare_datasets, train_nn, load_nn, load_encoders, \
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
    dataset = prepare_datasets(encoder, cse, params.subtypes)
    nn = train_nn(dataset, params.subtypes)
    encoder.save()
else:
    nn = load_nn(params.model_names, params.subtypes)
    encoder = load_encoders(params.model_names)
    predictions = pd.DataFrame([])

    if params._predict_training:
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
        tick_group = ohlc_data.tail(params._window_size)
        prediction = single_prediction(tick_group, nn, encoder, params)
        predictions = predictions.append(prediction)

    save_predictions(predictions, params, log)
#
# EOF
#
