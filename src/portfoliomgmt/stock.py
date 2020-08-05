import pandas as pd


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
        i : int = 0

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

    def sell(self, num, sell_price_share):
        package_num: int = 0
        package_profit: float = 0.
        total_profit: float = 0.

        if self.get_num_active_operations () < num:
            raise Exception("There aren't enough stocks of "+self._name+" to sell, currently there are "+str(self.get_num_active_operations()))

        # for index, value in self._operations.items()[::-1]:
        for value in self._operations:
            if not value.closed:
                package_num, package_profit = value.sell(num, sell_price_share)
                total_profit += package_profit
                num -= package_num
                if num == 0:
                    return total_profit

    def get_num_active_operations (self):
        actives: int=0

        for value in self._operations:
            if not value.closed:
                actives += value.num

        return actives

    def __str__(self):
        return self._operations.__str__()








