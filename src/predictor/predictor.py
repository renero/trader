# -*- coding: utf-8 -*-

import sys

import numpy as np

from predictor.ensemble import Ensemble as ensemble
from utils.logger import Logger


def main(argv):
    from predictor.cs_dictionary import CSDictionary

    params = CSDictionary(args=argv)
    np.random.seed(params.seed)
    log: Logger = params.log

    from predictor.cs_core import CSCore
    from predictor.ticks import Ticks

    if params.ensemble_predictions or params.ensemble:
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
            if params.save_predictions is True:
                predictor.save_predictions(predictions, params, log)
            else:
                predictor.display_predictions(predictions)


if __name__ == "__main__":
    main(sys.argv)
