"""
I will place here common functions used across all modules.
"""
from termcolor import colored

import pandas as pd
from pandas import DataFrame


class Common:

    Reset = '\033[0m'
    Green = '\033[32m'
    White = '\033[37m'
    Red = '\033[31m'

    def log(self, *args, **kwargs):
        if self.configuration._debug is True:
            print(*args, **kwargs)

    def green(self, string):
        return colored(string, 'green')

    def red(self, string):
        return colored(string, 'red')

    def white(self, string):
        return colored(string, 'white')

    def no_color(self, number: float):
        string = '{:.1f}'.format(number)
        return self.white(string)

    def color(self, number: float):
        string = '{:.1f}'.format(number)
        if number > 0.0:
            return self.green(string)
        elif number < 0.0:
            return self.red(string)
        else:
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

    def recolor(self, df, column_name):
        df[column_name] = df[column_name].apply(
            lambda x: '{}'.format(self.color(x)))

    def reformat(self, df, column_name):
        df[column_name] = df[column_name].apply(
            lambda x: '{}'.format(self.no_color(x)))

    def recolor_ref(self, df: DataFrame, col1: str, col2: str):
        df[col1] = df.apply(
            lambda row: self.cond_color(row[col1], row[col2]),
            axis=1
        )
