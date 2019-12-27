from typing import List

from dictionary import Dictionary
from share import share


class Positions:
    configuration: Dictionary
    book: List[share] = []

    def __init__(self, configuration):
        self.params = configuration
        self.book = []

    def reset(self):
        del self.book[:]
        self.book = []

    def buy_position(self, num, price, mode='bull'):
        """ Buy a number of positions, at a given price """
        self.book.append(share(price, num, mode))

    def sell_all(self, price):
        """ Sell all the options we have. Clear the book """
        self.update(price)
        sell_value = self.value()
        sell_profit = self.profit()
        del self.book[:]
        return sell_value, sell_profit

    def sell_positions(self, num_shares_to_sell, sell_price):
        """
        Sell `num` positions from our portfolio. The shares must be already
        updated with the proper current price, before executing the sell
        operation.

        - If num is greater than the number of shares, then sell all.
        - If no positions, return 0, 0.
        - If trying to sell more shares than owned, return 0, 0.

        :param num_shares_to_sell:  ditto
        :param sell_price:          the selling price

        :return:                    the amount at which the positions were sold,
                                    and the benefit.
        """
        num_shares_i_have = self.num_shares()
        if num_shares_i_have == 0. or num_shares_to_sell > num_shares_i_have:
            return 0., 0.
        if num_shares_i_have == num_shares_to_sell:
            return self.sell_all(sell_price)

        # Sort positions by profit to sell first those with max prof.
        sell_book: list[share] = sorted(
            self.book, key=lambda x: x.performance_, reverse=True)
        num_shares_sold = 0.
        total_income = 0.
        total_profit = 0.
        idx = 0
        while num_shares_sold != num_shares_to_sell:
            num = num_shares_to_sell - num_shares_sold
            sold, income, profit = self.sell(sell_book[idx], num, sell_price)
            total_income += income
            total_profit += profit
            num_shares_sold += sold
            idx += 1
        return total_income, total_profit

    def sell(self, position: share, num_shares: float, sell_price: float):
        """
        Sell the number of shares specified from position.
        - If the nr of shares to sell is lower than the nr of shares
          available, then those are removed from the position.
        - If as a result the position has zero shares, the position is removed
        - If the nr. of shares to sell is greater than the nr of shares
          available the position is also removed

        :param position:   the position to be sold
        :param num_shares: the nr of shares to sell
        :param sell_price: the selling price
        :return:           the number of shares sold.
        """
        # Get the reference in the actual book, not the selling book.
        position = self.book[self.book.index(position)]

        # Adjust how many shares can I sell.
        if num_shares > position.num_:
            num_shares = position.num_

        # Sell the adjusted amount from the position
        value, profit = position.sell(num_shares, sell_price)

        # Check if position is empty, to remove it from the book.
        if position.num_ == 0.:
            self.book.remove(position)

        return num_shares, value, profit

    def num_positions(self):
        """ Returns the total number of positions currently in portfolio """
        if len(self.book) == 0:
            return 0.
        return len(self.book)

    def num_shares(self):
        """ Returns the total number of shares purchased """
        num_shares = 0.
        for s in self.book:
            num_shares += s.num_
        return num_shares

    def update(self, current_price):
        """ Updates each position's value according to the new price. """
        for s in self.book:
            s.update(current_price)

    def value(self):
        """ Computes the value of the positions with the price passed """
        total_value = 0.
        for s in self.book:
            total_value += s.value_
        return total_value

    def cost(self):
        """ Computes the cost of the positions with the price passed """
        total_cost = 0.
        for s in self.book:
            total_cost += s.cost_
        return total_cost

    def profit(self):
        """ Returns the total profit accumulated by each share """
        total_profit = 0.
        for s in self.book:
            total_profit += s.profit_
        return total_profit

    def debug(self):
        if not self.book:
            return
        debug = self.params.log.debug
        debug('  Book ----------------------------------------------+')
        for s in self.book:
            dm = '  | {:<4.1f}({:>7.2f}) | val.{:>8.2f} |   profit={:>8.2f} |'
            debug(dm.format(s.num_, s.buy_price_, s.value_, s.profit_))

        debug('  |---------------+--------------+-------------------|')
        dm =  '  | {:<4.1f}({:>7.2f}) |     {:>8.2f} |          {:>8.2f} |'

        debug(dm.format(
            self.num_shares(), self.cost(), self.value(), self.profit()
        ))
        debug('  +---------------+--------------+-------------------+')
