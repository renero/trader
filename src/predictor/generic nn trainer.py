#!/usr/bin/env python
# coding: utf-8

import sys

import numpy as np

from logger import Logger
from lstm import lstm_1layer
from sequences import sequences
from ticks import Ticks


def main(argv):
    from cs_dictionary import CSDictionary

    params = CSDictionary(args=argv)
    ticks = Ticks(params, params.input_file).scale()
    ticks.append_indicator(["trend", "median_filter"])
    ticks.data.head()

    X_train, y_train, X_test, y_test = ticks.prepare_for_training(
        predict="trend", train_columns=["close", "trend"]
    )

    nn = lstm_1layer(params)
    nn.start_training(X_train, y_train)
    nn.evaluate(X_test, y_test)
    nn.end_experiment()


if __name__ == "__main__":
    main(sys.argv)
