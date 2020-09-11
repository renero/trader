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
    from it"""

    data = None
    _scaler = None
    _scaler_file = None
    _encoder = None
    _volatility = 0.0

    def __init__(self, params: CSDictionary, url: str):
        self.params = params
        self.data = pd.read_csv(url).round(2)
        self.data = self._fix_datetime_index(self.data)

        self.raw = self.data.copy(deep=True)
        self._volatility = self.data.close.std()

    def _fix_datetime_index(self, data: DataFrame):
        """Set the index in the proper `datetime` type """
        format = "%Y-%m-%d %H:%M:%S"
        data["Datetime"] = pd.to_datetime(data["Date"] + " 00:00:00",
                                          format=format)
        data = data.set_index(pd.DatetimeIndex(data["Datetime"]))
        data = data.drop(
            ["Datetime", self.params.csv_dict['d'], self.params.csv_dict['v']],
            axis=1)
        data.columns = self.params.ohlc
        return data

    def scale(self) -> DataFrame:
        """Scales the OHLC ticks from the dataframe read"""
        self._scaler = RobustScaler().fit(self.data[self.params.ohlc])
        self.data = pd.DataFrame(
            data=self._scaler.transform(self.data[self.params.ohlc]),
            columns=self.params.ohlc,
            index=self.data.index,
        ).round(2)
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
        ).round(2)

    def to_timewindows(
            self,
            predict: str,
            train_columns: List[str] = None) -> TrainVectors:
        return sequences.prepare(
            self.data,
            self._training_columns(train_columns),
            predict,
            timesteps=self.params.window_size,
            test_size=self.params.test_size
        )

    def _training_columns(self, train_columns):
        """Return the list of columns to be used for training.
        Depends on whether the value of train_columns is None."""
        if train_columns is None:
            training_columns = list(self.data.columns)
        else:
            training_columns = train_columns
        return training_columns

    def append_indicator(self, indicators: Union[str, List[str]]) -> None:
        """Append the listed indicators to the back of the data dataframe"""
        if isinstance(indicators, str):
            indicators = [indicators]
        for indicator in indicators:
            method_name = f'_append_{indicator}'
            try:
                getattr(self, method_name)()
            except AttributeError:
                raise AttributeError

    def _get_trend_sign(self) -> pd.Series:
        """
        Computes the trend as positive or negative, and return it as a Series
        :param values: default values for negative and positive trend
        :return: A Series with the trend values.
        """
        y = self.data.close.values
        y_trend = np.sign(y[1:] - y[:-1])
        y_trend = np.insert(y_trend, 0, 1., axis=0)
        return pd.Series(
            map(lambda x: 0. if x == -1 else 1., y_trend))

    def _append_trend(self):
        """
        Append the column trend to the end of the dataframe
        :param values: default values for negative and positive trend
        :return: None
        """
        trend = self._get_trend_sign()
        self.data = self.data.assign(trend=pd.Series(trend).values)

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
