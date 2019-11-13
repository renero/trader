import pandas as pd
import requests

from last import last
from logger import Logger


class closing:
    """
    Get the last closing values from the specified stock provider
    """

    @staticmethod
    def alpha_vantage(url='https://www.alphavantage.co',
                      api_entry='/query?',
                      api_key=None,
                      **kwargs) -> dict:
        endpoint = url + api_entry
        arguments = '&'.join(
            '{}={}'.format(key, value) for key, value in kwargs.items())
        arguments += '&apikey={}'.format(api_key)
        endpoint = endpoint + arguments
        response = requests.get(endpoint).json()
        stock_closing = dict()
        for old_key in response['Global Quote'].keys():
            stock_closing[old_key[4:]] = response['Global Quote'][old_key]
        return stock_closing

    @staticmethod
    def csv_row(stock_data: dict,
                json_columns: list,
                ohlc_columns: list,
                json_file: str,
                log: Logger) -> str:
        """
        Append the OHLCV data from provider to the OHLCV file used in the
        project.

        :param stock_data:      the dictionary with the data retrieved from
                                provider (Alpha Vantage)
        :param json_columns:    the columns names that contain the info I want
        :param ohlc_columns:    the column names ordered as in the OHLCV file
        :param json_file:       the name of the file that will contain the data
                                in json format
        :param log:             the logger used to report.
        :return:                the csv row that is inserted in the OHLCV file
        """
        sd = stock_data.copy()
        for v in sd.keys():
            sd[v] = [sd[v]]

        # Create a data frame from it
        my_columns = json_columns
        latest_ohlcv = pd.DataFrame.from_dict(sd)
        latest_ohlcv = latest_ohlcv[my_columns].copy(deep=True)
        # rename columns
        latest_ohlcv.columns = ['Date', 'Open', 'High', 'Low', 'Close',
                                'Volume']
        # Reorder columns
        latest_ohlcv = latest_ohlcv[ohlc_columns]
        latest_ohlcv.columns = ohlc_columns

        # record the OHLCV values to a temporary json file.
        latest_ohlcv.iloc[-1].to_json(json_file)
        log.info('Json file saved: {}'.format(json_file))

        # reduce the precision to two decimals, only.
        def f(x):
            if '.' in x:
                return x[0:x.index('.') + 3]
            else:
                return x

        row = list(map(f, list(latest_ohlcv.values[0])))
        row = ','.join(map(str, row)) + '\n'
        return row

    @staticmethod
    def append_to_file(row, file, date_to_append, log):
        if last.row_date_is(date_to_append, file):
            log.info('Data already appended. Doing nothing here.')
            return
        with open(file, 'a') as fd:
            fd.write(row)
        fd.close()
        log.info('Appended as CSV row to file {}'.format(file))
