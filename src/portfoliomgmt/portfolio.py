import pandas as pd
from .stock import Stock


class Portfolio(object):
    _stocks: pd.Series

    def __init__(self):
        _stocks = pd.Series(dtype='object', name="Portfolio")

    def add(self, stock_name, stock_package):
        if self._stocks.get(stock_name) is None:  # stock_name is not in the portfolio
            stock = Stock(stock_name, stock_package)
            self._stocks.add(pd.Series(stock, index=stock_name))
        else:  # stock_name already exists
            stock = self._stocks.get(stock_name)
            stock.add(stock_package)

    def getStock(self, stock_name):
         return  self._stocks.get(stock_name)

    def __str__(self):
        return self._stocks.__str__()
