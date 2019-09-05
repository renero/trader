#!/usr/bin/env pythonw

import os
import sys

import tensorflow as tf

from cs_api import *
from cs_encoder import CSEncoder
from cs_utils import random_tick_group
from params import Params
from ticks import Ticks

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

params = Params(args=sys.argv)
ticks = Ticks().read_ohlc()

if params.do_train is True:
    encoder = CSEncoder().fit(ticks)
    cse = encoder.ticks2cse(ticks)
    dataset = prepare_datasets(encoder, cse, params.subtypes)
    nn = train_nn(dataset, params.subtypes)
    encoder.save()
else:
    nn = load_nn(params.model_names, params.subtypes)
    encoder = load_encoders(params.model_names)
    predictions = pd.DataFrame([])

    for i in range(10):
        tick_group = random_tick_group(ticks, params.max_tick_series_length + 1)
        prediction = single_prediction(tick_group[:-1], nn, encoder, params)
        prediction = add_supervised_info(prediction, tick_group['c'][-1],
                                         params)
        predictions = predictions.append(prediction)

    if params._save_predictions is True:
        predictions.to_csv(params._predictions_path, index=False)
    print(predictions)

#
# EOF
#
