from os.path import dirname, realpath

import numpy as np
import pandas as pd
from pandas import DataFrame

from utils.file_io import file_exists


class Ticks:
    min_value = 0.
    max_value = 0.

    def __init__(self, params):
        self.params = params
        self.log = params.log

    def normalize(self, x):
        return (x - self.min_value) / (self.max_value - self.min_value)

    def denormalize(self, x):
        return (x * (self.max_value - self.min_value)) + self.min_value

    def scale_back(self, df):
        num_models = len(list(self.params['model_names'].keys()))
        if num_models == 1:
            return df.applymap(np.vectorize(self.denormalize))
        else:
            scaled = df.loc[:, df.columns != 'winner'].applymap(
                np.vectorize(self.denormalize))
            return pd.concat([scaled, df['winner']], axis=1)

    def read_ohlc(self,
                  filepath=None,
                  do_normalize=True) -> DataFrame:
        # Set the input filename as either the argument
        # passed or the one found in the YAML file.
        _filepath = self.params.input_file if filepath is None else filepath
        filepath = file_exists(_filepath, dirname(realpath(__file__)))
        # Read the OHLC file.
        df = pd.read_csv(filepath, delimiter=self.params.delimiter)
        date_column = df[self.params.csv_dict['d']]
        # Reorder and rename
        df = df[[self.params.csv_dict['o'], self.params.csv_dict['h'],
                 self.params.csv_dict['l'], self.params.csv_dict['c']]]
        df.columns = ['o', 'h', 'l', 'c']

        # Set the max and min for the values found (all)
        self.max_value = df.values.max()
        self.min_value = df.values.min()
        if do_normalize is True:
            df = df.applymap(np.vectorize(self.normalize))
        # Log info about what happened
        info_msg = 'Read ticks file: {}, output DF dim{}'
        self.log.info(info_msg.format(self.params.input_file, df.shape))

        return pd.concat((date_column, df), axis=1)

    @staticmethod
    def new_ohlc(values):
        df = pd.Series([values])
        return df
