# -*- coding: utf-8 -*-

import sys

import numpy as np

if __name__ == "__main__":
    from cs_dictionary import CSDictionary

    params = CSDictionary(args=sys.argv)
    np.random.seed(params.seed)
    log = params.log

    from cs_core import CSCore
    from ticks import Ticks

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
