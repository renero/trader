from datetime import datetime

from .toperation import TOperation


class StockPackage(object):
    _buy_date: datetime
    _buy_price: float = 0.  # price by share
    _num: int = 0  # num of shares, decremented in sell operations
    _original_num: int = 0  # num of shares, it stay immutable for historical purpose
    # _total_price: float = 0.
    _mode: TOperation
    _closed: bool = False
    # _current_price: float = 0.
    _profit: float = 0.  # Accumulated profit  by all shares in the package

    def __init__(self, date, price, num, mode=TOperation.bull):
        self._buy_date = date
        self._buy_price = price
        self._num = num
        self._original_num = num
        self._mode = mode
        self._closed = False
        # self._current_price = 0
        self._profit = 0

    def __str__(self):
        return "buy date: " + str(self._buy_date) + ", share price: " + "{:.4f}".format(self._buy_price) + ", number of shares: " + str(self._num) + ", total: " + "{:.4f}".format((self._buy_price * self._num)) + ", " + str(self._mode) + ", closed: " + str(self._closed)

    @property  # when you do Stock.closed, it will call this function
    def buy_date(self):
        return self._buy_date

    @property  # when you do Stock.closed, it will call this function
    def buy_price(self):
        return self._buy_price

    @property  # when you do Stock.num, it will call this function
    def num(self):
        return self._num

    @property  # when you do Stock.num, it will call this function
    def original_num(self):
        return self._original_num

    @property  # when you do Stock.num, it will call this function
    def mode(self):
        return self._mode

    @property  # when you do Stock.closed, it will call this function
    def closed(self):
        return self._closed

    @property  # when you do Stock.closed, it will call this function
    def profit(self):
        return self.profit

    @num.setter
    def num(self, num):
        if num <= 0:
            raise Exception("Number of shares must be a positive integer")

        self._num = num

    @buy_price.setter
    def buy_price(self, buy_price):
        if buy_price <= 0:
            raise Exception("Buy Price must be a positive number")

        self._buy_price = buy_price

    def sell(self, num_shares, sell_price_share, simulation):
        sold_shares: int
        sell_profit: float = 0.

        if num_shares >= self._num:  # all shares are sold
            sold_shares = self._num
            if not simulation:
                self._closed = True
                self._num = 0
        else:  # num_shares are sold
            sold_shares = num_shares
            if not simulation:
                self._num -= num_shares  # num shares are sold

        if self._mode is TOperation.bull:
            sell_profit += (sold_shares * sell_price_share) - (sold_shares * self._buy_price)
        else:
            sell_profit += (sold_shares * self._buy_price) - (sold_shares * sell_price_share)

        if not simulation:
            self._profit += sell_profit

        return sold_shares, sell_profit
