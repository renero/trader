import numpy as np
import pandas as pd
from params import Params


class Ticks(Params):
    min_value = 0.
    max_value = 0.

    def __init__(self):
        super(Ticks, self).__init__()

    def normalize(self, x):
        return (x - self.min_value) / (self.max_value - self.min_value)

    def denormalize(self, x):
        return (x * (self.max_value - self.min_value)) + self.min_value

    def scale_back(self, df):
        if len(list(self.params['model_names'].keys())) == 1:
            return df.applymap(np.vectorize(self.denormalize))
        else:
            return df.loc[:, df.columns != 'winner'].applymap(
                np.vectorize(self.denormalize))

    def read_ohlc(self,
                  filepath=None,
                  columns=None,
                  ohlc_tags=None,
                  do_normalize=True):
        _filepath = self._ticks_file if filepath is None else filepath
        _columns = self._columns if columns is None else columns
        _ohlc_tags = self._ohlc_tags if ohlc_tags is None else ohlc_tags

        cols_mapper = dict(zip(_columns, _ohlc_tags))
        df = pd.read_csv(_filepath, delimiter=self._delimiter)
        df = df[list(cols_mapper.keys())].rename(
            index=str, columns=cols_mapper)
        self.max_value = df.values.max()
        self.min_value = df.values.min()

        if do_normalize is True:
            df = df.applymap(np.vectorize(self.normalize))

        info_msg = 'Read ticksfile: {}, output DF dim{}'
        self.log.info(info_msg.format(self._ticks_file, df.shape))
        return df

    @staticmethod
    def new_ohlc(values):  # , columns):
        df = pd.Series([values])  # , columns=columns)
        return df
