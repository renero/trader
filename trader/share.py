class share:
    buy_price_: float = 0.
    current_price_: float = 0.
    num_: float = 0.
    cost_: float = 0.
    value_: float = 0.
    profit_: float = 0.
    performance_: float = 0.
    mode_: str = 'bull'

    def __init__(self, price, num, mode='bull'):
        self.buy_price_ = price
        self.current_price_ = price
        self.num_ = num
        self.cost_ = self.buy_price_ * self.num_
        self.value_ = self.cost_
        self.profit_ = 0.
        self.performance_ = (self.current_price_ - self.buy_price_) / \
                             self.buy_price_
        self.mode_ = mode

    def sell(self, num, sell_price):
        """
        Sell a portion of this position, or totally. In case there're no
        more positions

        :param num:        number of shares to sell from current position
        :param sell_price: the price at which the portion is sold
        :return:           the value of the portion of shares sold, and profit
        """
        assert num <= self.num_, 'Attempt to sell more shares than owned!'
        # Captures the value of the portion sold.-
        value, profit = self.valuate(num, sell_price)
        # Updates the remaining part, without changing price
        self.num_ -= num
        self.update()
        return value, profit

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
        self.profit_ = self.performance_ * self.value_
        return self.value_, self.profit_

    def valuate(self, num=None, price=None):
        """
        Returns the value and profit represented by a given number of shares
        on this current position. If no `num` is given, the current number
        of shares in the position is considered. If no `price` is given, the
        current price is considered.
        """
        if num is None:
            num = self.num_
        if price is None:
            price = self.current_price_

        value = num * price
        profit = self.performance_ * value
        return value, profit
