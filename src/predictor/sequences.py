from math import ceil, floor
from typing import Tuple, List

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

TrainTestVectors = Tuple[ndarray, ndarray, ndarray, ndarray]
TrainVectors = [TrainTestVectors, Tuple[ndarray, ndarray]]


class sequences:
    """
    From a dataframe with four categories (OHLC) in a multivariate timeseries,
    aggrupate it in timesteps windows, split it in training and testing subsets,
    and finally, aggrupate X and y values together.

    Consider a given multi-variate sequence (num_categories = 3):

        [[ 10  15  25]
         [ 20  25  45]
         [ 30  35  65]
         [ 40  45  85]
         [ 50  55 105]
         ...
        ]

    We can divide the sequence into multiple input/output patterns called
    samples, where three time steps are used as input and one time step is
    used as output for the one-step prediction that is being learned.

         X,             y_3
        --------------------
        [10, 15, 25
         20, 25, 45
         30, 35, 65]    85
        [20, 25, 45
         30, 35, 65
         40, 45, 85]    105
        ...

    In this example we consider timesteps=3, so X is composed of
    [ timesteps x num_categories ] samples, and y is the third column of
    the next sample.

    """

    def __init__(self):
        pass

    @classmethod
    def to_time_windows(
            cls,
            df: DataFrame,
            train_columns: List[str],
            y_label: str,
            timesteps: int,
            test_size: float = 0.1
    ) -> TrainVectors:
        """
        Prepare the input dataframe (OHLC) converting it into an ndarray
        of (num_samples x timesteps x num_categories), also splitting it
        into training and test sets.
        """
        X_indices, y_index = cls._get_indices(df, train_columns, y_label)
        df = cls._aggrupate_in_timesteps(df.values, timesteps)
        if test_size != 0.0:
            train, test = train_test_split(
                df,
                test_size=test_size,
                shuffle=False)
            X_train, y_train = cls._split(train, X_indices, y_index, timesteps)
            X_test, y_test = cls._split(test, X_indices, y_index, timesteps)
            return X_train, y_train, X_test, y_test

        return cls._split(df.values, X_indices, y_index, timesteps)

    @classmethod
    def _aggrupate_in_timesteps(cls, data: ndarray,
                                timesteps: int) -> DataFrame:
        """
        Given a dataframe, divide the sequence into multiple input/output
        patterns called samples, where a number of time steps are used as
        input and one time step is used as output for the one-step prediction
        that is being learned.
        """
        df = pd.DataFrame(data)
        series = df.copy(deep=True)
        series_s = series.copy(deep=True)
        for i in range(timesteps):
            series = pd.concat([series, series_s.shift(-(i + 1))], axis=1)
        series.dropna(axis=0, inplace=True)
        return series

    @classmethod
    def _get_indices(
            cls,
            df: DataFrame,
            train_columns: List[str],
            y_label: str) -> Tuple[List[int], int]:

        X_indices = [df.columns.get_loc(col_name) for col_name in train_columns]
        y_index = df.columns.get_loc(y_label)
        return X_indices, y_index

    @classmethod
    def _split(
            cls,
            data: ndarray,
            X_indices: List[int],
            y_index: int,
            timesteps: int) -> TrainVectors:
        """
        Take num_samples from data, and separate X and y from it into two
        new tensors that will be used to feed the LSTM.
        """
        num_samples = data.shape[0]
        num_categories = int(data.shape[1] / (timesteps + 1))
        subset = np.array(data).reshape(
            (num_samples, timesteps + 1, num_categories))

        X = subset[:, 0:timesteps, X_indices]
        y = subset[:, -1, y_index]

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        return X, y

    @classmethod
    def get_num_features(cls, X_train):
        assert len(X_train.shape) == 3, "Training set must be a 3D tensor"
        return X_train.shape[2]

    @classmethod
    def get_num_target_labels(cls, y_train):
        assert len(y_train.shape) == 2, "Training labels must be a 2D tensor"
        return y_train.shape[1]

    @classmethod
    def _last_index_in_training(
            cls, df: DataFrame, timesteps: int, test_size: float = 0.1
    ) -> int:
        """Returns the last value in the training set, from the original
        dataframe"""
        train_size = 1.0 - test_size
        n_samples = df.shape[0]
        n_test = ceil(test_size * n_samples)
        n_train = floor(train_size * n_samples)
        last_index_in_train = n_train - timesteps - 1
        return last_index_in_train

    @classmethod
    def last_in_training(
            cls, df: DataFrame, timesteps: int, test_size: float = 0.1
    ) -> Series:
        return df.iloc[
            cls._last_index_in_training(df, timesteps, test_size)]

    @classmethod
    def _last_date_in_training(
            cls, df: DataFrame, timesteps: int, test_size: float = 0.1
    ) -> str:
        """ Returns the date (no time) index of the last event in training """
        last_one = df.iloc[
            cls._last_index_in_training(df, timesteps, test_size)]
        return str(last_one.name).split(' ')[0]

    @classmethod
    def first_in_test(
            cls, df: DataFrame, timesteps: int, test_size: float = 0.1
    ) -> Series:
        return df.iloc[
            cls._last_index_in_training(df, timesteps, test_size) + 1
            ]

    @classmethod
    def previous_to_first_prediction(cls, df: DataFrame, timesteps: int
                                     ) -> Series:
        return df.iloc[timesteps - 1]
