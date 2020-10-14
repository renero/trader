#!/usr/bin/env python
# coding: utf-8

import sys

from dictionary import Dictionary
from networks import lstm
from ticks import Ticks
from utils.utils import print_bin_predictions_match

params: Dictionary = None


def read_ticks():
    global params

    ticks = Ticks(params, params.input_file).scale()
    ticks.append_indicator(["trend", "median_filter", "change"])
    ticks.append_indicator("gmf", monotonic_window=7, mf_window=3, sigma=5)
    ticks.append_indicator("gmf_mono", monotonic_window=7, mf_window=3, sigma=5)
    ticks.append_indicator("trend", column_name="gmf")
    return ticks


def main(argv):
    from dictionary import Dictionary
    global params

    params = Dictionary(args=argv)
    ticks = read_ticks()
    params.epochs = 100

    X_trainC, y_trainC, X_testC, y_testC = ticks.prepare_for_training(
        predict="close_trend",
        train_columns=["returns", "gmf", "gmf_mono", "gmf_trend"]
    )

    nn1 = lstm(params, n_layers=1, binary=True)
    nn1.start_training(X_trainC, y_trainC, name=None)
    yhatC_trend, acc = nn1.evaluate(X_testC, y_testC)

    print(nn1)
    print_bin_predictions_match(y_testC, yhatC_trend)


if __name__ == "__main__":
    main(sys.argv)
