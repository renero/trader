#!/usr/bin/env python
# coding: utf-8

import sys

from dictionary import Dictionary
from lstm import lstm
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
    params.window_size = 28
    params.batch_size = 16
    params.units = 42
    params.dropout = 0.1

    XT, yT, Xt, yt = ticks.prepare_for_training(
        predict_column="gmf_mono",
        train_columns=["gmf", "gmf_trend"])
    yT += 1
    yt += 1

    nn1 = lstm(params).build()
    nn1.start_training(XT, yT, name=None)
    yhat, acc = nn1.evaluate(Xt, yt)

    print(nn1)
    print_bin_predictions_match(yt, yhat, nn1.metadata['binary'])
    nn1.save()


if __name__ == "__main__":
    main(sys.argv)
