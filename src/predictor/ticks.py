from pathlib import Path

import joblib
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import RobustScaler

from cs_dictionary import CSDictionary


class Ticks:
    """Ticks data read from file with all the relevant parameters and metadata
    from it"""

    data = None
    _scaler = None
    scaler_file = None
    _encoder = None
    _volatility = 0.0

    def __init__(self, params: CSDictionary, url: str, scale: bool = False):

        self.params = params
        data = pd.read_csv(url).round(2)

        format = "%Y-%m-%d %H:%M:%S"
        data["Datetime"] = pd.to_datetime(data["Date"] + " 00:00:00",
                                          format=format)
        data = data.set_index(pd.DatetimeIndex(data["Datetime"]))
        data = data.drop(["Date", "Datetime", "Volume"], axis=1)
        data.columns =self.params.ohlc

        self.data = data
        self.raw = self.data.copy(deep=True)
        if scale is True:
            self.data = self.transform()
        self._volatility = self.data[self.params.csv_dict['c']].std()

    def transform(self, inline: bool = False) -> DataFrame:
        """Scales the OHLC ticks from the dataframe read"""
        self._scaler = RobustScaler().fit(self.data[self.params.ohlc])
        scaled_df = pd.DataFrame(
            data=self._scaler.transform(self.data[self.params.ohlc]),
            columns=self.params.ohlc,
            index=self.data.index,
        ).round(2)
        if not inline:
            return scaled_df
        self.raw = self.data.copy(deep=True)
        self.data = scaled_df.copy(deep=True)
        self._volatility = self.data[self.params.csv_dict['c']].std()
        return self.data

    def inverse_transform(self, scaled_df: DataFrame) -> DataFrame:
        """Scales back a dataframe with the scaler stored in filename"""
        assert self._scaler is not None, "Scaler must be set first (scale())"
        return pd.DataFrame(
            data=self._scaler.inverse_transform(scaled_df[self.params.ohlc]),
            columns=self.params.ohlc,
            index=scaled_df.index,
        ).round(2)

    def save_scaler(self, filename: str) -> None:
        """
        Saves the scaler with the filename specified

        :param filename: The complete path and filename where the scaler will be saved
        :returns: None
        :raises AssertionError: raises if scaler is not set before trying to save it
        """
        assert (
                self._scaler is not None
        ), "Scaler has not yet been created. Use scale() method first."
        self.scaler_file = filename
        joblib.dump(self._scaler, self._scaler_file)
        print(f"RobustScaler saved at: {self._scaler_file}")

    def load_scaler(self, filename: str):
        """Loads the scaler from the filename specified"""
        self.scaler_file = filename
        self._scaler = joblib.load(self._scaler_file)
        return self._scaler

    @property
    def scaler_file(self) -> str:
        if hasattr(self, "_scaler_file"):
            return self._scaler_file.as_posix()
        return None

    @scaler_file.setter
    def scaler_file(self, filename):
        file = Path(filename)
        if not file.is_file():
            raise ValueError("Filename for scaler does not exist")
        self._scaler_file = file
