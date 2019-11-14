import json

import pandas as pd


class Update:

    def __init__(self, configuration):
        self.params = configuration
        self.log = self.params.log
        self.update()

    def update(self):
        # Read the temporary files
        ohlc = Update.read_json(self.params.tmp_ohlc)
        ensemble = Update.read_json(self.params.tmp_forecast)
        indicator = Update.read_json(self.params.tmp_indicator)
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
            return

        # Append the last row, normally
        with open(self.params.file, 'a') as forecast_file:
            forecast_file.write(csv_row)
            self.log.info('Forecast file UPDATED for date: {}'.format(
                ohlc[self.params.tmp_dictionary.date]))

    @staticmethod
    def read_json(filename):
        with open(filename) as json_file:
            data = json.load(json_file)
        return data

    @staticmethod
    def round2two(x):
        if type(x) == str:
            return x
        return '{:.2f}'.format(x)
