# -*- coding: utf-8 -*-

import sys

import numpy as np

from ensemble import Ensemble as ensemble
from logger import Logger

if __name__ == "__main__":
    from cs_dictionary import CSDictionary

    params = CSDictionary(args=sys.argv)
    np.random.seed(params.seed)
    log: Logger = params.log

    from cs_core import CSCore
    from ticks import Ticks

    if params.ensemble:
        ensemble(params)
    else:
        ticks = Ticks(params)
        data = ticks.read_ohlc()
        predictor = CSCore(params)
        if params.train:
            predictor.train(data)
        else:
            predictions = None
            nn, encoder = predictor.prepare_predict()
            if params.predict_training:
                predictions = predictor.predict_training(
                    data, nn, encoder, ticks)
            elif params.predict:
                predictions = predictor.predict_newdata(
                    data, nn, encoder, ticks)

            predictions = predictor.reorder_predictions(predictions, params)
            predictor.save_predictions(predictions, params, log)
            predictor.display_predictions(predictions)
