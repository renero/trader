import numpy as np
import pandas as pd
from params import Params


class Ticks(Params):

    def __init__(self):
        super(Ticks, self).__init__()

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
        max_value = df.values.max()
        min_value = df.values.min()

        def normalize(x):
            return (x - min_value) / (max_value - min_value)

        if do_normalize is True:
            df = df.applymap(np.vectorize(normalize))

        info_msg = 'Read ticksfile: {}, output DF dim{}'
        self.log.info(info_msg.format(self._ticks_file, df.shape))
        return df

    @staticmethod
    def new_ohlc(values):  # , columns):
        df = pd.Series([values])  # , columns=columns)
        return df
