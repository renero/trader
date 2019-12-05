import pandas as pd
import requests

from last import last
from logger import Logger


class closing:

    @staticmethod
    def retrieve_stock_data(params) -> (dict, str):
        endpoint = params.url + params.api_entry
        endpoint = endpoint + '&symbol={}'.format(params.symbol)
        endpoint = endpoint + '&{}={}'.format(params.api_key_name, params.api_key)
        params.log.debug('Request: <{}>'.format(endpoint))

        response = requests.get(endpoint).json()
        params.log.debug('Response: \n{}'.format(response))

        stock_closing = dict()
        data_chunk = response[params.json_chunk_key]
        if params.chunk_is_array:
            data_chunk = data_chunk[0]
        for json_key in params.json_columns.keys():
            stock_closing[json_key] = data_chunk[
                params.json_columns[json_key]]
        if ' ' in stock_closing['date']:
            stock_closing['date'] = stock_closing['date'].split(' ')[0]
        return stock_closing, stock_closing['date']

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
            log.warn('Data already appended. Doing nothing here.')
            return
        with open(file, 'a') as fd:
            fd.write(row)
        fd.close()
        log.info('Appended as CSV row to file {}'.format(file))
