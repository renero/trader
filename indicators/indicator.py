"""
Generic class from which each indicator must inherit
Indicator Class computes an indicator and keeps all meta-info in its
internal structure

(C) 2019 J. Renero
"""
from os.path import splitext, basename

import pandas as pd
from pandas import DataFrame

from file_io import read_ohlc, save_dataframe


class Indicator:
    data = None
    name = None
    columns = None
    ix_columns = None
    # All the columns relevant to be saved
    final_columns = None
    values: DataFrame = None

    def __init__(self, params):
        self.params = params
        self.log = params.log

        # Read the input file specified in arguments (params)
        self.data = read_ohlc(params.input_file, params.separator,
                              params.csv_dict)
        self.log.info('Read file: {}, {} rows, {} cols.'.format(
            params.input_file, self.data.shape[0], self.data.shape[1]))

        self.final_columns = list(self.data.columns) + self.ix_columns

        # Initialize result, index name, and column names for this indicator
        self.values = None

    def save(self, today):
        output_values = self.values[self.final_columns]
        output = save_dataframe(
            '{}_{}'.format(self.name,
                           splitext(basename(self.params.input_data))[0]),
            output_values,
            self.params.output_path,
            cols_to_scale=self.ix_columns)
        self.log.info('Saved index to file: {}'.format(output))

    def merge(self, today):
        self.log.info('Merge mode')
        mergeable_data = pd.read_csv(self.params.merge_file,
                                     delimiter=self.params.separator)
        indicator_data = pd.DataFrame()
        indicator_data[self.columns] = self.values[self.columns].copy(deep=True)
        indicator_data = indicator_data.reset_index(drop=True)

        df = pd.concat([mergeable_data, indicator_data], axis=1)
        fused = save_dataframe(
            '{}_{}'.format(
                splitext(basename(self.params.forecast_file))[0], self.name),
            df,
            self.params.output_path,
            cols_to_scale=self.columns)
        self.log.info('Saved forecast and index FUSED: {}'.format(fused))
