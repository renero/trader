import pandas as pd
import requests


class closing:
    """
    Get the last closing values from the specified stock provider
    """

    @staticmethod
    def alpha_vantage(url='https://www.alphavantage.co',
                      api_entry='/query?',
                      api_key='HF9S3IZSBKSKHPV3',
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
    def csv_row(stock_data: dict, json_columns: list) -> str:
        """
        Copy the original response from the provider to preserve it

        :param stock_data:
        :param json_columns:
        :return:
        """
        sd = stock_data.copy()
        for v in sd.keys():
            sd[v] = [sd[v]]

        # Create a data frame from it
        # ['latest trading day', 'open', 'high', 'low', 'price', 'volume']
        my_columns = json_columns
        latest_ohlcv = pd.DataFrame.from_dict(sd)
        latest_ohlcv = latest_ohlcv[my_columns].copy(deep=True)
        latest_ohlcv.columns = ['Date', 'Open', 'High', 'Low', 'Close',
                                'Volume']

        # reduce the precision to two decimals, only.
        def f(x):
            if '.' in x:
                return x[0:x.index('.') + 3]
            else:
                return x

        row = list(map(f, list(latest_ohlcv.values[0])))
        row = ','.join(map(str, row)) + '\n'
        return row
