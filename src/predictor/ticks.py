from importlib import import_module
from os.path import basename, splitext
from typing import Union, List

import joblib
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import RobustScaler

from dictionary import Dictionary
from predictor import TrainVectors
from sequences import sequences
from utils.file_utils import valid_output_name
from utils.utils import reset_seeds


class Ticks:
    """Ticks data read from file with all the relevant parameters and metadata
from it. The way it should be:

    # to predict close and use OHLC values

    >>> params = Dictionary(args=argv)
    >>> df = pd.DataFrame({"Col1": [10, 20, 15, 30, 45],\
                           "Col2": [17, 27, 22, 37, 52]},\
                           index=pd.date_range("2020-01-01", "2020-01-05"))
    >>> ticks = Ticks(params, df=df)
    >>> X, y, Xt, yt = ticks.prepare_for_training( \
            predict_column='Col2', train_columns='Col1')

    # To generate X and y without test set, ensure params.test_size=0.
    >>> X, y = ticks.prepare_for_training(predict_column='close')

    # if we want to predict "trend"
    >>> ticks.append_indicator('trend')
    >>> ticks.prepare_for_training(predict_column='trend')
    >>> X, y , Xt, yt = split(test_size=0.1)

    # if we want to use less variables in prediction
    >>> ticks.append_indicator(['moving_average', 'trend'])
    >>> ticks.prepare_for_training(predict_column='trend', \
            train_columns=['close', 'moving_average'])

    So, let's work on it.
    """

    data: pd.DataFrame = None
    _scaler = None
    _scaler_file = None
    _encoder = None
    _volatility = 0.0

    def __init__(self,
                 params: Dictionary,
                 csv_file: str = None,
                 df: pd.DataFrame = None):
        """
        Initialize a Ticks object by passing a CSV file URL * or * a dataframe
        """
        assert csv_file is not None or df is not None, \
            "Either a CSV file path or an existing dataframe must be specified"
        self.params = params
        self.log = params.log
        reset_seeds()
        self.data = self._from_csv(
            url=csv_file) if csv_file is not None else self._from_dataframe(df)
        self._volatility = self.data.close.std()

    def _from_csv(self, url: str) -> pd.DataFrame:
        return self._fix_datetime_index(
            pd.read_csv(url).round(self.params.precision)
        )

    def _from_dataframe(self, df: pd.DataFrame) -> "Ticks":
        return self._fix_datetime_index(
            df.round(self.params.precision)
        )

    def _fix_datetime_index(self, data: DataFrame):
        """Set the index in the proper `datetime` type """
        if isinstance(data.index, pd.core.indexes.datetimes.DatetimeIndex):
            return data
        format = "%Y-%m-%d %H:%M:%S"
        date_column = self.params.csv_dict['d']
        data["Datetime"] = pd.to_datetime(data[date_column] + " 00:00:00",
                                          format=format)
        data = data.set_index(pd.DatetimeIndex(data["Datetime"]))
        data = self._drop_unused_columns(data)
        data.columns = self.params.ohlc
        return data

    def _unused_columns(self, data: DataFrame) -> List[str]:
        columns_in_data = list(
            map(lambda t: t.lower(), list(data)))  # - set(cols_to_drop))))
        columns_to_keep = list(map(lambda t: t.lower(), self.params.ohlc))
        return list(set(columns_in_data) - set(columns_to_keep))

    def _drop_unused_columns(self, data):
        cols_to_drop = self._unused_columns(data)

        def find_col_name(name):
            col_list = list(data)
            try:
                # this uses a generator to find the index if it matches,
                # will raise an exception if not found
                return col_list[next(
                    i for i, v in enumerate(col_list) if v.lower() == name)]
            except:
                return ''

        for column_to_drop in cols_to_drop:
            data = data.drop(find_col_name(column_to_drop), axis=1)
        return data

    def scale(self, scaler_path: str = None) -> "Ticks":
        """Scales the OHLC ticks from the dataframe read"""
        if scaler_path is None:
            self._scaler = RobustScaler().fit(self.data[self.params.ohlc])
        else:
            self.load_scaler(scaler_path)
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

    def prepare_for_training(self, predict_column: str,
                             train_columns: List[str] = None) -> TrainVectors:
        """
        Prepare the input dataframe (OHLC) converting it in a 3D tensor
        and update internal parameters.
        """
        data_vectors = sequences.to_time_windows(
            self.data,
            timesteps=self.params.window_size,
            train_columns=self._training_columns(
                train_columns),
            y_column=predict_column,
            test_size=self.params.test_size)
        # First and second elements in tuple are the training vectors
        # Third and fourth are the test set (if any).
        # I use the first two to update parameters
        self.params.num_features = sequences.get_num_features(data_vectors[0])
        self.params.num_target_labels = sequences.get_num_target_labels(
            data_vectors[1])
        return data_vectors

    def prepare_for_predict(self, train_columns: List[str]) -> TrainVectors:
        """
        Prepare a dataframe to be used as input to a trained network,
        to obtain a prediction.
        """
        return sequences.to_time_windows(
            df=self.data, timesteps=self.params.window_size,
            train_columns=train_columns)

    def last_date_in_training(self) -> str:
        """Returns the date (string) of the last event in the training set"""
        return sequences._last_date_in_training(self.data,
                                                self.params.window_size,
                                                self.params.test_size)

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
            module = import_module(f'.{indicator}', package='indicators')
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

    def save_scaler(self, filename: str = None) -> None:
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
        dataset_name = splitext(basename(self.params.input_file))[0]
        scaler_filename = filename if filename is not None else \
            f"scaler_{dataset_name}"
        self._scaler_file = valid_output_name(
            filename=scaler_filename,
            path=self.params.models_dir,
            extension='bin'
        )
        joblib.dump(self._scaler, self._scaler_file)
        self.log.info(f"Scaler saved at: {self._scaler_file}")

    def load_scaler(self, filename: str):
        """Loads the scaler from the filename specified"""
        self._scaler_file = filename
        self._scaler = joblib.load(self._scaler_file)
        self.log.info(f"Scaler loaded from {filename}")
        return self._scaler

    @property
    def scaler_file(self) -> str:
        if hasattr(self, "_scaler_file"):
            return self._scaler_file.as_posix()
        return None
