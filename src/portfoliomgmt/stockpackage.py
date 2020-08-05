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
        self.buy_price = price
        self.num = num
        self._original_num = num
        self._mode = mode
        self._closed = False
        #self._current_price = 0
        self._profit = 0

    def __str__(self):
        return self._buy_date + "," + "{:.4f}".format(self._buy_price) + "," + str(self._num) + "," + "{:.4f}".format(
            (self._buy_price * self._num)) + "," + str(self._mode)

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
        return self.original_num

    @property  # when you do Stock.num, it will call this function
    def mode(self):
        return self.mode

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

    def sell(self, num, sell_price_share):
        sold_shares: int
        sell_profit: float = 0.

        if num >= self._num:  # all shares are sold
            self._closed = True
            sold_shares = self._num
        else:
            self._num -= num  # num shares are sold
            sold_shares = num

        if self._mode is TOperation.bull:
            sell_profit += (sold_shares * sell_price_share) - (sold_shares * self._buy_price)
        else:
            sell_profit += (sold_shares * self._buy_price) - (sold_shares * sell_price_share)

        self._profit += sell_profit

        return num, sell_profit

    def update(self, new_price=None):
        """
        Updates position according to the new price.

        :param new_price:   the new price at which the position is re-evaluated.
        :return:            value and profit
        """
        if new_price is None:
            new_price = self.current_price_
        self.current_price_ = new_price
        self.value_ = self.num_ * new_price
        if self.mode_ == 'bull':
            self.performance_ = (self.current_price_ - self.buy_price_) / \
                                self.buy_price_
        else:
            self.performance_ = (self.buy_price_ - self.current_price_) / \
                                self.buy_price_
        self.profit_ = self.performance_ * self.cost_
        return self.value_, self.profit_

    def valuate(self, num=None):
        """
        Returns the value and profit represented by a given number of shares
        on this current position. If no `num` is given, the current number
        of shares in the position is considered. If no `price` is given, the
        current price is considered.
        """
        if num is None:
            num = self.num_

        value = num * self.buy_price_
        profit = self.performance_ * num * self.buy_price_
        return value, profit
