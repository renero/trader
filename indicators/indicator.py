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
        """
        Save the indicator data to a CSV file, whose name matches the
        input file, trailed by the indicator name.
        :return: None
        """
        data = self.values[self.final_columns]
        ix_saved, scaler_saved = self.save_data(data)
        self.log.info('Saved scaler to file: {}'.format(scaler_saved))
        self.log.info('Saved index to file: {}'.format(ix_saved))

    def merge(self):
        """
        Merges the indicator columns at the end of each row of a file
        which is specified in command line arguments as `merge_file`.
        The result file has the same name as the file to be merged, but
        is never overwritten.
        :return: None
        """
        mergeable_data = pd.read_csv(self.params.merge_file,
                                     delimiter=self.params.separator)
        ix_data = pd.DataFrame()
        ix_data[self.ix_columns] = self.values[self.ix_columns].copy(deep=True)
        ix_data = ix_data.reset_index(drop=True)
        data = pd.concat([mergeable_data, ix_data], axis=1)

        merged_saved, scaler_saved = self.save_data(data)
        self.log.info('Saved scaler to file: {}'.format(scaler_saved))
        self.log.info('Index merged into: {}'.format(merged_saved))

    def save_data(self, df):
        """
        Saves the dataframe specified, as built by the `save` or `merge`
        methods, and also saves the scaler used within the indicator.
        :param df: The data frame to be saved.
        :return: the names of the files saved.
        """
        idx_basename = '{}_{}'.format(
            self.name,
            splitext(basename(self.params.input_file))[0])
        scaler_name = 'scaler_{}_{}'.format(
            self.name,
            splitext(basename(self.params.input_file))[0])
        fused, scaler_saved = save_dataframe(
            idx_basename,
            df,
            self.params.output_path,
            cols_to_scale=self.ix_columns,
            scaler_name=scaler_name,
            index=False)
        return fused, scaler_saved

    def register(self):
        """
        Scales the indicator values and save them to the specified filename
        :return: N/A
        """
        # ix_scaled = self.values.copy(deep=True)
        scaler = joblib.load(self.params.scaler_name)
        self.log.info('Scaler loaded: {}'.format(self.params.scaler_name))
        ix_scaled = scaler.transform(self.values[self.ix_columns])
        last_row = np.array([[ix_scaled[-1, 0]], [ix_scaled[-1, 1]]])
        ix_row = pd.DataFrame(data=last_row.T,
                              columns=self.ix_columns,
                              index=None)
        ix_row.iloc[0].to_json(self.params.json_indicator.format(self.name))
        self.log.info(
            'Saved indicators to: {}'.format(
                self.params.json_indicator.format(self.name)))
