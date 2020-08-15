from typing import List

from utils.dictionary import Dictionary
from trader.share import share


class Positions:
    configuration: Dictionary
    book: List[share] = []
    initial_budget: float = 0.
    budget = 0.

    def __init__(self, configuration, initial_budget):
        self.params = configuration
        self.book = []
        self.initial_budget = initial_budget
        self.budget = self.initial_budget

    def reset(self):
        del self.book[:]
        self.book = []

    def buy(self, num, price, mode='bull'):
        """
        Buy a number of positions, at a given price.
        Returns the cost of the buy operation.
        """
        self.book.append(share(price, num, mode))
        self.budget -= self.book[-1].cost_
        return self.book[-1].cost_

    def sell(self, num_shares_to_sell, sell_price):
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
        # if num_shares_i_have == num_shares_to_sell:
        #     return self.sell_all(sell_price)

        # Sort positions by profit to sell first those with max prof.
        sell_book: list[share] = sorted(
            self.book, key=lambda x: x.performance_, reverse=True)
        num_shares_sold = 0.
        total_income = 0.
        total_profit = 0.
        idx = 0
        while num_shares_sold != num_shares_to_sell:
            num = num_shares_to_sell - num_shares_sold
            sold, income, profit = self.sell_position(sell_book[idx], num,
                                                      sell_price)
            total_income += income
            total_profit += profit
            num_shares_sold += sold
            idx += 1
        self.budget += (total_income + total_profit)
        return total_income, total_profit

    # def sell_all(self, price):
    #     """ Sell all the options we have. Clear the book """
    #     self.update(price)
    #     sell_value = self.cost()
    #     sell_profit = self.profit()
    #     del self.book[:]
    #     return sell_value, sell_profit

    def sell_position(self, position: share, num_shares: float,
                      sell_price: float):
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
        :return:           the number of shares sold, income and profit
        """
        # Get the reference in the actual book, not the selling book.
        position = self.book[self.book.index(position)]

        # Adjust how many shares can I sell.
        if num_shares > position.num_:
            num_shares = position.num_

        # Sell the adjusted amount from the position
        value, profit = position.sell(num_shares) # , sell_price)

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

    def profit(self) -> float:
        """ Returns the total profit accumulated by each share """
        total_profit = 0.
        for s in self.book:
            total_profit += s.profit_
        return total_profit

    def debug(self):
        if not self.book:
            return
        debug = self.params.log.debug
        ht = '  | num.( b.price) | cost      | value     | perf  | proft | m |'
        hl = '  +----------------+-----------+-----------+-------+-------+---+'
        debug(hl)
        debug(ht)
        debug(hl)
        # '  | 12.4(123456.8) | 123456.89 | 123456.89 | 12.45 | 12.45 | c |')
        fm = '  | {:<4.1f}({:>8.1f}) | {:>9.2f} | {:>9.2f} | {:>5.2f} '
        fm += '| {:>5.2f} | {} |'
        for s in self.book:
            debug(fm.format(s.num_, s.buy_price_, s.cost_, s.value_,
                            s.performance_, s.profit_,
                            'B' if s.mode_ == 'bull' else 'b'))
        debug(hl)
        fm = '  | {:<14.1f} | {:>9.2f} | {:>9.2f} | ----- | {:>5.2f} | - |'
        debug(fm.format(
            self.num_shares(), self.cost(), self.value(), self.profit()))
        debug(hl)
