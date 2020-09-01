import pandas as pd

from .tstrategy import TStrategy


class Stock:

    _name: str
    _operations: pd.Series

    def __init__(self, name, stock_package=None):
        self._name = name
        if stock_package is None:
            self._operations = pd.Series(dtype='object', name=self._name)
        else:
            self._operations = pd.Series(stock_package, dtype='object', name=self._name)

    def add(self, stock_package):
        self._operations[self._operations.size] = stock_package

    def get_total_packages(self):
        return self._operations.size

    def get_active_packages(self):
        i: int = 0

        for value in self._operations:
            if not value.closed:
                i += 1

        return i

    def get_total_shares(self):
        total_shares: int = 0

        for value in self._operations:
            if not value.closed:
                total_shares += value.num

        return total_shares

    def sell(self, num, sell_price_share, strategy=TStrategy.fifo, simulation=False):
        total_profit: float = 0.

        if self.get_num_active_operations() < num:
            raise Exception("There aren't enough stocks of "+self._name+" to sell, currently there are "+str(self.get_num_active_operations()))

        if strategy is TStrategy.fifo:
            step = 1
        else:
            step = -1

        # for index, value in self._operations.items()[::step]:
        for value in self._operations[::step]:
            if not value.closed:
                package_num, package_profit = value.sell(num, sell_price_share, simulation)
                total_profit += package_profit
                num -= package_num
                if num == 0:
                    return total_profit

    def sell_all(self, sell_price_share, strategy=TStrategy.fifo, simulation=False):
        return self.sell(self.get_num_active_operations(), sell_price_share, strategy, simulation)

    def get_num_active_operations(self):
        actives: int = 0

        for value in self._operations:
            if not value.closed:
                actives += value.num

        return actives

    def __str__(self):
        out: str = self._name+"\n"

        for value in self._operations:
            out += "\t\t"+str(value)+"\n"

        return out

    def save_to(self):
        return self._operations.to_json(orient="split")
