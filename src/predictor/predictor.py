# -*- coding: utf-8 -*-

import sys

import numpy as np

from utils.logger import Logger
from ensemble import Ensemble as ensemble


def main(argv):
    from cs_dictionary import CSDictionary

    params = CSDictionary(args=argv)
    log: Logger = params.log

    from cs_core import CSCore
    from ticks_reader import TicksReader

    if params.ensemble_predictions or params.ensemble:
        ensemble(params)
    else:
        ticks_reader = TicksReader(params)
        data = ticks_reader.read_ohlc()
        predictor = CSCore(params)
        if params.train:
            predictor.train(data)
        else:
            predictions = None
            nn, encoder = predictor.prepare_predict()
            if params.predict_training:
                predictions = predictor.predict_training(
                    data, nn, encoder, ticks_reader)
            elif params.predict:
                predictions = predictor.predict_newdata(
                    data, nn, encoder, ticks_reader)

            predictions = predictor.reorder_predictions(predictions, params)
            if params.save_predictions is True:
                predictor.save_predictions(predictions, params, log)
            else:
                predictor.display_predictions(predictions)


if __name__ == "__main__":
    main(sys.argv)
