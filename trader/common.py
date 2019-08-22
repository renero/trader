"""
I will place here common functions used across all modules.
"""
from termcolor import colored

import pandas as pd
from pandas import DataFrame


class Common:

    def log(self, *args, **kwargs):
        if self.configuration._debug is True:
            print(*args, **kwargs)

    def append(self, table: object, kvp: dict) -> DataFrame:
        for key in kvp:
            table[key] = kvp[key]
        return table

    def add(self, table, key, value):
        table[key] = value
        return table

    def green(self, string):
        return colored('{}'.format(string), 'green')

    def red(self, string):
        return colored('{}'.format(string), 'red')

    def white(self, string):
        return colored('{}'.format(string), 'white')

    def color(self, number):
        string = '{:+.1f}'.format(number)
        if number > 0.0:
            return self.green(string)
        elif number < 0.0:
            return self.red(string)
        else:
            number = 0.0
            string = '{:.1f}'.format(number)
            return self.white(string)

    def cond_color(self, number, ref):
        string = '{:.1f}'.format(number)
        if number > ref:
            return self.green(string)
        elif number < ref:
            return self.red(string)
        else:
            number = 0.0
            string = '{:.1f}'.format(number)
            return self.white(string)
