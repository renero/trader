import pandas as pd
from .stock import Stock


class Portfolio:

    _stocks: pd.Series

    def __init__(self):
        self._stocks = pd.Series(dtype='object', name="Portfolio")

    def add(self, stock_name, stock_package):
        if self._stocks.get(stock_name) is None:  # stock_name is not in the portfolio
            stock = Stock(stock_name, stock_package)
            self._stocks[stock_name] = stock
        else:  # stock_name already exists
            stock = self._stocks.get(stock_name)
            stock.add(stock_package)

    def getstock(self, stock_name):
        return self._stocks.get(stock_name)

    def __str__(self):
        out: str = ""

        out += "Portfolio\n"
        for value in self._stocks:
            out += "\t" + str(value) + "\n"

        return out

    def save(self):
        for value in self._stocks:
            print(value.save()+"\n")
