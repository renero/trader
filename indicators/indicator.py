"""
Generic class from which each indicator must inherit
Indicator Class computes an indicator and keeps all meta-info in its
internal structure

(C) 2019 J. Renero
"""
from os.path import splitext, basename

import joblib
import numpy as np
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

    def save(self):
        output_values = self.values[self.final_columns]
        idx_basename = '{}_{}'.format(
            self.name,
            splitext(basename(self.params.input_file))[0])
        scaler_name = 'scaler_{}_{}'.format(
            self.name,
            splitext(basename(self.params.input_file))[0])
        output, scaler_saved = save_dataframe(
            idx_basename,
            output_values,
            self.params.output_path,
            cols_to_scale=self.ix_columns,
            scaler_name=scaler_name)
        self.log.info('Saved scaler to file: {}'.format(scaler_saved))
        self.log.info('Saved index to file: {}'.format(output))

    def register(self):
        """
        Scales the indicator values and save them to the specified filename
        :return: N/A
        """
        # ix_scaled = self.values.copy(deep=True)
        scaler = joblib.load(self.params.scaler_name)
        ix_scaled = scaler.transform(self.values[self.ix_columns])
        last_row = np.array([[ix_scaled[-1, 0]], [ix_scaled[-1, 1]]])
        ix_row = pd.DataFrame(data=last_row.T,
                              columns=self.ix_columns,
                              index=None)
        ix_row.iloc[0].to_json(self.params.json_indicator.format(self.name))
        self.log.info(
            'Saved indicators to: {}'.format(
                self.params.json_indicator.format(self.name)))
