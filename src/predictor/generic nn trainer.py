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
    np.random.seed(params.seed)
    log: Logger = params.log

    params = CSDictionary(args=argv)
    ticks = Ticks(params, params.input_file, scale=True)

    X_train, y_train, X_test, y_test = sequences.prepare(
        ticks.data, timesteps=params.window_size, test_size=params.test_size
    )

    params.num_features = sequences.get_num_features(X_train)
    params.num_target_labels = sequences.get_num_target_labels(y_train)
    nn = lstm_1layer(params)

    print(nn.metadata)

    exp_id = nn.start_experiment(X_train, y_train)
    nn.evaluate_experiment(X_test, y_test)
    nn.end_experiment()


if __name__ == "__main__":
    main(sys.argv)
