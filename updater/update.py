import json
import os
import sys

import numpy as np
import pandas as pd


class Update:

    def __init__(self, configuration, action):
        self.params = configuration
        self.log = self.params.log
        getattr(self, action)()

    def predictions(self):
        """
        Take yesterday's OHLC Closing and predictions to update the preds file
        - Opens the file,
        - checks that date is last working day
        - opens latest forecast file,
        - if there's no file, returns
        - check that latest date doesn't match the one from last working day
        - append a row with the predictions from each network, and the stats
        """
        self.log.info('Updating predictions file')
        preds = self.read_json(self.params.tmp_predictions)
        ohlc = self.read_json(self.params.tmp_ohlc)
        if preds is None or ohlc is None:
            self.log.info('No temporary files to be used to update preds.')
            return False

        # Read the preds file to check if last line has already been updated
        existing_preds = pd.read_csv(self.params.file, self.params.delimiter)
        last_ohlc_date = ohlc[self.params.tmp_dictionary.date]
        if self.last_date_is(existing_preds, last_ohlc_date):
            self.log.info(
                'Predictions file already contains entry for date: {}'.format(
                    last_ohlc_date))
            return False

        # Method to compute element-wise difference
        def diff_with(array, value):
            return [abs(array[i] - value) for i in range(len(array))]

        # Method to compute the closest value in array to a given value
        def find_nearest(array, value: float):
            array = np.asarray(array).astype(np.float32)
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        # Build the csv row to be added
        close_colname = self.params.tmp_dictionary['close']
        pred_values = np.around(np.array([preds[k] for k in preds.keys()]),
                                decimals=2)
        csv_row = '{},{}'.format(
            last_ohlc_date, ','.join(map(str, pred_values)))
        csv_row = csv_row + ',{},{},{},{},{}'.format(
            np.mean(pred_values),
            np.mean(diff_with(pred_values, np.mean(pred_values))),
            np.median(pred_values),
            np.mean(diff_with(pred_values, np.median(pred_values))),
            preds.keys()[find_nearest(pred_values, ohlc[close_colname])]
        )

        # Append the row at the end of the file
        with open(self.params.file, 'a') as predictions_file:
            predictions_file.write(csv_row)
            self.log.info(
                'Predictions file UPDATED for date: {}'.format(last_ohlc_date))
        return True

    def forecast(self):
        self.log.info('Updating forecast file')
        # Read the temporary files
        ohlc = self.read_json(self.params.tmp_ohlc)
        ensemble = self.read_json(self.params.tmp_forecast)
        indicator = self.read_json(self.params.tmp_indicator)
        self.log.info('Read temporary files')

        # flesh out the CSV row
        forecast_items = [
            ohlc[self.params.tmp_dictionary.date],
            float(ohlc[self.params.tmp_dictionary.close]),
            ensemble[self.params.tmp_dictionary.ensemble],
            indicator[self.params.tmp_dictionary.green],
            indicator[self.params.tmp_dictionary.blue]
        ]
        csv_row = ','.join(map(Update.round2two, forecast_items)) + '\n'

        # Read the forecast file to check if last line has already been updated
        forecast_data = pd.read_csv(self.params.file,
                                    delimiter=self.params.delimiter)
        date_column = self.params.forecast_column_names[0]
        last_date_in_file = forecast_data.iloc[-1][date_column]
        if last_date_in_file == ohlc[self.params.tmp_dictionary.date]:
            self.log.info(
                'Forecast file already contains entry for date: {}'.format(
                    ohlc[self.params.tmp_dictionary.date]))
            return False

        # Append the last row, normally
        with open(self.params.file, 'a') as forecast_file:
            forecast_file.write(csv_row)
            self.log.info('Forecast file UPDATED for date: {}'.format(
                ohlc[self.params.tmp_dictionary.date]))

        return True

    def last_date_is(self, df, this_date):
        """ Checks if last row's date matches the one passed. """
        date_column = self.params.forecast_column_names[0]
        last_date_in_file = df.iloc[-1][date_column]
        return last_date_in_file == this_date

    def read_json(self, filename):
        """ Reads a JSON file, returns a dict. If file does not exists
        returns None """
        if os.path.exists(filename):
            with open(filename) as json_file:
                try:
                    data = json.load(json_file)
                except:
                    self.log.error(
                        'Unexpected error, reading JSON File ({}): {}'.format(
                            filename, sys.exc_info()[0]))
                    raise
            return data
        else:
            return None

    @staticmethod
    def round2two(x):
        if type(x) == str:
            return x
        return '{:.2f}'.format(x)
