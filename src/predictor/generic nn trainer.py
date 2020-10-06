#!/usr/bin/env python
# coding: utf-8

import sys

import networks
from ticks import Ticks

params = None


def read_ticks():
    global params

    ticks = Ticks(params, params.input_file).scale()
    ticks.append_indicator(["trend", "median_filter", "change"])
    ticks.append_indicator("gmf", monotonic_window=7, mf_window=3, sigma=5)
    ticks.append_indicator("gmf_mono", monotonic_window=7, mf_window=3, sigma=5)
    ticks.append_indicator("trend", column_name="gmf")
    return ticks


def run(X_train, y_train, X_test, y_test, name, network, layers, binary=False):
    global params

    method = getattr(networks, network)
    nn = method(params, n_layers=layers, binary=binary)
    nn.start_training(X_train, y_train, name)
    yhat, acc = nn.evaluate(X_test, y_test)
    nn.end_experiment()
    return nn, yhat, acc


def split_and_run(window_size, epochs, network, layers, binary):
    global params

    ticks = read_ticks()
    params.window_size = window_size
    params.epochs = epochs
    X_train, y_train, X_test, y_test = ticks.prepare_for_training(
        predict="gmf_trend", train_columns=["gmf"]
    )
    nn, yhat, acc = run(X_train, y_train, X_test, y_test,
                        "Reproducibility",
                        network,
                        layers,
                        binary)
    del (X_train, y_train, X_test, y_test, ticks)
    return nn, yhat, acc


def main(argv):
    from cs_dictionary import CSDictionary
    global params

    params = CSDictionary(args=argv)
    nn, yhat, acc = split_and_run(window_size=14,
                                  epochs=20,
                                  network="lstm",
                                  layers=1,
                                  binary=True)


if __name__ == "__main__":
    main(sys.argv)
