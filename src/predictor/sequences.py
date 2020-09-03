import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from pandas import DataFrame, Series
from math import ceil, floor
from numpy import ndarray
from typing import Tuple, Union


class sequences:
    """
    From a dataframe with four categories (OHLC) in a multivariate timeseries,
    aggrupate it in timesteps windows, split it in training and testing subsets, and
    finally, aggrupate X and y values together.

    Consider a given multi-variate sequence (num_categories = 3):

        [[ 10  15  25]
         [ 20  25  45]
         [ 30  35  65]
         [ 40  45  85]
         [ 50  55 105]
         ...
        ]

    We can divide the sequence into multiple input/output patterns called samples,
    where three time steps are used as input and one time step is used as output for
    the one-step prediction that is being learned.

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
    def prepare(
        cls, df: DataFrame, timesteps: int, test_size: float = 0.1
    ) -> Union[Tuple[ndarray, ndarray, ndarray, ndarray], Tuple[ndarray, ndarray]]:
        """
        Prepare the input dataframe (OHLC) converting it into an ndarray
        of (num_samples x timesteps x num_categories), also splitting it
        into training and test sets.
        """
        num_categories = df.shape[1]

        df = cls.aggrupate_in_timesteps(df, timesteps)
        if test_size != 0.0:
            train, test = train_test_split(df, test_size=test_size, shuffle=False)
            X_train, y_train = cls.split_Xy(train, timesteps)
            X_test, y_test = cls.split_Xy(test, timesteps)
            return X_train, y_train, X_test, y_test

        return cls.split_Xy(df, timesteps)

    @classmethod
    def aggrupate_in_timesteps(cls, data: ndarray, timesteps: int) -> DataFrame:
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
    def split_Xy(cls, data: ndarray, timesteps) -> Tuple[ndarray, ndarray]:
        """
        Take num_samples from data, and separate X and y from it into two
        new tensors that will be used to feed the LSTM.
        """
        num_samples = data.shape[0]
        num_categories = int(data.shape[1] / (timesteps + 1))
        subset = np.array(data).reshape((num_samples, timesteps + 1, num_categories))

        X = subset[:, 0:timesteps, :]

        # maybe, data is encoded. In that case I must keep, NOT the last column
        # but the last $ num_categories / 4 $, which corresponds to the encoded
        # values of the last column, corresponding to the "close" value.
        from_column_index = int(num_categories - (num_categories / 4))
        to_column_index = num_categories
        y = subset[:, -1, from_column_index:to_column_index]

        return X, y

    @classmethod
    def get_num_features(cls, X_train):
        assert len(X_train.shape) == 3, "Training set must be a 3D tensor"
        return X_train.shape[2]

    @classmethod
    def get_num_target_labels(cls, y_train):
        assert len(y_train.shape)==2, "Training labels must be a 2D tensor"
        return y_train.shape[1]

    @classmethod
    def _last_index_in_training(
        cls, df: DataFrame, timesteps: int, test_size: float = 0.1
    ) -> int:
        """Returns the last value in the training set, from the original dataframe"""
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
        return ticks.data.iloc[cls._last_index_in_training(df, timesteps, test_size)]

    @classmethod
    def first_in_test(
        cls, df: DataFrame, timesteps: int, test_size: float = 0.1
    ) -> Series:
        return ticks.data.iloc[
            cls._last_index_in_training(df, timesteps, test_size) + 1
        ]
