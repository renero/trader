# -*- coding: utf-8 -*-

import sys

import numpy as np

from cs_core import CSCore
from logger import Logger
from cs_dictionary import CSDictionary
from ticks import Ticks

if __name__ == "__main__":
    np.random.seed(1)
    params = CSDictionary(args=sys.argv)
    log = params.log
    ticks = Ticks(params)
    data = ticks.read_ohlc()
    predictor = CSCore(params)

    if params.train is True:
        predictor.train(data)
    else:
        nn, encoder = predictor.prepare_predict()
        if params.predict_training:
            predictions = predictor.predict_training(data, nn, encoder, ticks)
        else:
            predictions = predictor.predict_newdata(data, nn, encoder, ticks)

        predictions = predictor.reorder_predictions(predictions, params)
        predictor.save_predictions(predictions, params, log)
        predictor.display_predictions(predictions)
