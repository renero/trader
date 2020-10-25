from math import ceil
from typing import Tuple, List, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split

from predictor import TrainVectors, TrainTestVectors


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
            timesteps: int,
            train_columns: List[str],
            y_column: str = None,
            test_size: float = 0.0) -> TrainVectors:
        """
        Prepare the input dataframe (OHLC) converting it into an ndarray
        of (num_samples x timesteps x num_categories), also splitting it
        into training and test sets.

        Returns
        -------
        TrainVectors
            Depending on the request for test set (test_size != 0.), returns
            a 4-tuple with

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe with the data we want to prepare
        timesteps : int
            The width of the rolling window
        train_columns : List[str]
            The names of the columns to be used for training
        y_column : str (Default: None)
            The name of the column with the target variable.
        test_size : float (Default: 0.0)
            The proportion of the data to be used for training.

        Examples
        --------
        >>> df = pd.DataFrame({"A": [10, 20, 30, 40, 50], \
            "B": [-10, -20, -30, -40, -50], "C": [0, 0, 1, 1, 1]})
        >>> df
            A   B  C
        0  10 -10  0
        1  20 -20  0
        2  30 -30  1
        3  40 -40  1
        4  50 -50  1

        >>> sequences.to_time_windows(df, 2, ['A', 'B'], 'C', test_size=0.)
        (array([[[ 10., -10.],
                 [ 20., -20.]],

                [[ 20., -20.],
                 [ 30., -30.]],

                [[ 30., -30.],
                 [ 40., -40.]]]),
         array([[1.],
                [1.],

        >>> sequences.to_time_windows(df, 2, ['A', 'B'])
        array([[[ 10., -10.],
                [ 20., -20.]],
               [[ 20., -20.],
                [ 30., -30.]],
               [[ 30., -30.],
                [ 40., -40.]],
               [[ 40., -40.],
                [ 50., -50.]]])
        """
        if y_column is None:
            X_indices = cls._get_indices(df, train_columns)
            y_index = -1
        else:
            X_indices, y_index = cls._get_indices(df, train_columns, y_column)
        df = cls._aggregate_in_timesteps(
            df.values,
            timesteps,
            no_prediction=(y_column is None))

        if test_size == 0.:
            return cls._split(df.values, timesteps, X_indices, y_index)
        return cls._train_test_split(
            df, test_size, timesteps, X_indices, y_index)

    @classmethod
    def _train_test_split(
            cls,
            df: DataFrame,
            test_size: float,
            timesteps: int,
            X_indices: List[int],
            y_index: int) -> TrainTestVectors:
        train, test = train_test_split(
            df,
            test_size=test_size,
            shuffle=False)
        X_train, y_train = cls._split(train, timesteps, X_indices, y_index)
        X_test, y_test = cls._split(test, timesteps, X_indices, y_index)
        return X_train, y_train, X_test, y_test

    @classmethod
    def _aggregate_in_timesteps(
            cls,
            data: ndarray,
            timesteps: int,
            no_prediction: bool = False) -> DataFrame:
        """
        Given a dataframe, divide the sequence into multiple input/output
        patterns called samples, where a number of time steps are used as
        input and one time step is used as output for the one-step prediction
        that is being learned.

        Parameters
        ----------

        data : ndarray
            The array to aggregate in rolling windows
        timesteps : int
            The timesteps of each rolling window
        no_prediction : bool
            Default FALSE, indicates whether adding an additional column
            to each window with the expected value. When expecting to use
            the resulting data for training, this parameter should be False.
            Set to TRUE only if you're using this method to accomodate a
            set of values to make a prediction.

        Example
        -------

        >>> df = pd.DataFrame({"Col1": [10, 20, 15, 30, 45],\
                               "Col2": [17, 27, 22, 37, 52]},\
                               index=pd.date_range("2020-01-01", "2020-01-05"))
        >>> df
                    Col1  Col2
        2020-01-01    10    17
        2020-01-02    20    27
        2020-01-03    15    22
        2020-01-04    30    37
        2020-01-05    45    52

        >>> sequences._aggregate_in_timesteps(df, timesteps=3)
                    Col1  Col2  Col1  Col2  Col1  Col2  Col1  Col2
        2020-01-01    10    17    20    27    15    22    30    37
        2020-01-02    20    27    15    22    30    37    45    52

        >>> sequences._aggregate_in_timesteps(df, timesteps=3, \
                                              no_prediction=True)
                    Col1  Col2  Col1  Col2  Col1  Col2
        2020-01-01    10    17    20    27    15    22
        2020-01-02    20    27    15    22    30    37
        2020-01-03    15    22    30    37    45    52

        """
        df = pd.DataFrame(data)
        series = df.copy(deep=True)
        series_s = series.copy(deep=True)
        for i in range(timesteps - int(no_prediction)):
            series = pd.concat([series, series_s.shift(-(i + 1))], axis=1)
        series.dropna(axis=0, inplace=True)
        return series

    @classmethod
    def _get_indices(
            cls,
            df: DataFrame,
            train_columns: List[str],
            y_label: str = None) -> Union[Tuple[list, int], list]:
        """
        Get the column indices of the names passed for X and Y

        Example
        -------

        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> sequences._get_indices(df, ['A', 'B'], 'C')
        ([0, 1], 2)

        >>> sequences._get_indices(df, ['A', 'B'])
        [0, 1]

        """
        X_indices = [df.columns.get_loc(col_name) for col_name in train_columns]
        if y_label is not None:
            y_index = df.columns.get_loc(y_label)
            return X_indices, y_index
        else:
            return X_indices

    @classmethod
    def _split(cls, data: ndarray, timesteps: int, X_indices: List[int],
               y_index: int = -1) -> TrainVectors:
        """
        Take num_samples from data, and separate X and y from it into two
        new tensors that will be used to feed the LSTM.

        Example
        -------

        >>> df = pd.DataFrame({"Col1": [10, 20, 15, 30, 45],
                               "Col2": [17, 27, 22, 37, 52]},
                               index=pd.date_range("2020-01-01", "2020-01-05"))
        >>> df
                    Col1  Col2
        2020-01-01    10    17
        2020-01-02    20    27
        2020-01-03    15    22
        2020-01-04    30    37
        2020-01-05    45    52

        >>> X_indices, y_index = sequences._get_indices(\
                df, train_columns=['Col1'], y_label='Col2')
        >>> X_indices, y_index
        ([0], 1)
        >>> df = sequences._aggregate_in_timesteps(df, timesteps=3)
        >>> df
                    Col1  Col2  Col1  Col2  Col1  Col2  Col1  Col2
        2020-01-01    10    17    20    27    15    22    30    37
        2020-01-02    20    27    15    22    30    37    45    52

        >>> sequences._split(df.values,3, X_indices, y_index)
        array(
           array([
                 [[10.],[20.],[15.]],
                 [[20.],[15.],[30.]]]
            ),
           array(
                [[37.],
                 [52.]]
            )
        )
        """
        num_samples = data.shape[0]
        prediction_column = 0 if y_index == -1 else 1
        num_categories = int(data.shape[1] / (timesteps + prediction_column))
        subset = np.array(data).reshape(
            (num_samples, timesteps + prediction_column, num_categories))

        X = subset[:, 0:timesteps, X_indices]
        if y_index != -1:
            y = subset[:, -1, y_index]
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            return X, y
        else:
            return X

    @classmethod
    def get_num_features(cls, X_train):
        assert len(X_train.shape) == 3, "Training set must be a 3D tensor"
        return X_train.shape[2]

    @classmethod
    def get_num_target_labels(cls, y_train):
        assert len(y_train.shape) == 2, "Training labels must be a 2D tensor"
        return y_train.shape[1]

    @classmethod
    def get_num_target_values(cls, y_train):
        assert len(y_train.shape) == 2, "Training labels must be a 2D tensor"
        return len(np.unique(y_train))

    @classmethod
    def _last_index_in_training(
            cls, df: DataFrame, timesteps: int, test_size: float = 0.1
    ) -> int:
        """Returns the last value in the training set, from the original
        dataframe"""
        train_size = 1.0 - test_size
        n_samples = df.shape[0]
        # n_test = ceil(test_size * n_samples)
        # n_train = floor(train_size * n_samples)
        n_train = ceil(train_size * n_samples)
        # last_index_in_train = n_train - timesteps - 1
        last_index_in_train = n_train - timesteps + 1
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
