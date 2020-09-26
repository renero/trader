from importlib import import_module
from typing import Union, List, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import RobustScaler

from cs_dictionary import CSDictionary
from sequences import sequences

TrainTestVectors = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
TrainVectors = Union[TrainTestVectors, Tuple[np.ndarray, np.ndarray]]


class Ticks:
    """Ticks data read from file with all the relevant parameters and metadata
from it. The way it should be:

    # to predict close and use OHLC values

>>> ticks.prepare_for_training(predict='close')
>>> X, y, Xt, yt = ticks.split(test_size=0.1)

    # To generate X and y without test set
>>> ticks.prepare_for_training(predict='close')
>>> X, y = ticks.split(test_size=0.0)

    # if we want to predict "trend"
>>> ticks.append_indicator('trend')
>>> ticks.prepare_for_training(predict='trend')
>>> X, y , Xt, yt = split(test_size=0.1)

    # if we want to use less variables in prediction
>>> ticks.append_indicator(['moving_average', 'trend'])
>>> ticks.prepare_for_training(predict='trend',
        train_columns=['close', 'moving_average'])

    So, let's work on it.
    """

    data = None
    _scaler = None
    _scaler_file = None
    _encoder = None
    _volatility = 0.0

    def __init__(self, params: CSDictionary, url: str):
        self.params = params
        self.data = pd.read_csv(url).round(self.params.precision)
        self.data = self._fix_datetime_index(self.data)

        self.raw = self.data.copy(deep=True)
        self._volatility = self.data.close.std()

    def _fix_datetime_index(self, data: DataFrame):
        """Set the index in the proper `datetime` type """
        format = "%Y-%m-%d %H:%M:%S"
        date_column = self.params.csv_dict['d']
        data["Datetime"] = pd.to_datetime(data[date_column] + " 00:00:00",
                                          format=format)
        data = data.set_index(pd.DatetimeIndex(data["Datetime"]))
        data = self._drop_unused_columns(data)
        data.columns = self.params.ohlc
        return data

    def _drop_unused_columns(self, data):
        cols_to_drop = ["Datetime", self.params.csv_dict['d']]
        if self.params.csv_dict['v'] in data.columns:
            cols_to_drop.append(self.params.csv_dict['v'])
        data = data.drop(cols_to_drop, axis=1)
        return data

    def scale(self) -> DataFrame:
        """Scales the OHLC ticks from the dataframe read"""
        self._scaler = RobustScaler().fit(self.data[self.params.ohlc])
        self.data = pd.DataFrame(
            data=self._scaler.transform(self.data[self.params.ohlc]),
            columns=self.params.ohlc,
            index=self.data.index,
        ).round(self.params.precision)
        self._update_internal_attributes()
        return self

    def _update_internal_attributes(self) -> None:
        self._volatility = self.data.close.std()

    def scale_back(self, scaled_df: DataFrame) -> DataFrame:
        """Scales back a dataframe with the scaler stored in filename"""
        assert self._scaler is not None, "Scaler must be set first (scale())"
        return pd.DataFrame(
            data=self._scaler.inverse_transform(scaled_df[self.params.ohlc]),
            columns=self.params.ohlc,
            index=scaled_df.index,
        ).round(self.params.precision)

    def prepare_for_training(
            self,
            predict: str,
            train_columns: List[str] = None) -> TrainVectors:
        """
        Prepare the input dataframe (OHLC) converting it in a 3D tensor
        and update internal parameters.
        """
        data_vectors = sequences.to_time_windows(
            self.data,
            self._training_columns(train_columns),
            predict,
            timesteps=self.params.window_size,
            test_size=self.params.test_size
        )
        # First and second elements in tuple are the training vectors
        # Third and fourth are the test set (if any).
        # I use the first two to update parameters
        self.params.num_features = sequences.get_num_features(data_vectors[0])
        self.params.num_target_labels = sequences.get_num_target_labels(
            data_vectors[1])
        return data_vectors

    def _training_columns(self, train_columns):
        """Return the list of columns to be used for training.
        Depends on whether the value of train_columns is None."""
        if train_columns is None:
            training_columns = list(self.data.columns)
        else:
            training_columns = train_columns
        return training_columns

    def compute_indicator(
            self,
            indicators: Union[str, List[str]],
            *args,
            **kwargs) -> DataFrame:
        """Compute the listed indicators and return them as a dataframe"""
        if isinstance(indicators, str):
            indicators = [indicators]
        for indicator in indicators:
            return self._try_indicator(indicator, *args, **kwargs)

    def append_indicator(
            self,
            indicators: Union[str, List[str]],
            *args,
            **kwargs) -> DataFrame:
        """Append the listed indicators to the back of the data dataframe"""
        if isinstance(indicators, str):
            indicators = [indicators]
        for indicator in indicators:
            self._try_indicator(indicator, append=True, *args, **kwargs)

    def _try_indicator(
            self,
            indicator,
            append=False,
            *args,
            **kwargs) -> DataFrame:
        try:
            module = import_module(indicator, package='indicators')
            ix = getattr(module, indicator)(
                self.data,
                self.params,
                *args,
                **kwargs)
            if append:
                for ix_column in ix.values.columns:
                    self.data[ix_column] = ix.values[ix_column].values
                return self.data
            return ix.values
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"Indicator {indicator} does not exist")

    def save_scaler(self, filename: str) -> None:
        """
        Saves the scaler with the filename specified

        :param filename: The complete path and filename where the scaler
            will be saved
        :returns: None
        :raises AssertionError: raises if scaler is not set before trying to
            save it
        """
        assert (
                self._scaler is not None
        ), "Scaler has not yet been created. Use scale() method first."
        self._scaler_file = filename
        joblib.dump(self._scaler, self._scaler_file)
        print(f"RobustScaler saved at: {self._scaler_file}")

    def load_scaler(self, filename: str):
        """Loads the scaler from the filename specified"""
        self._scaler_file = filename
        self._scaler = joblib.load(self._scaler_file)
        return self._scaler

    @property
    def scaler_file(self) -> str:
        if hasattr(self, "_scaler_file"):
            return self._scaler_file.as_posix()
        return None
