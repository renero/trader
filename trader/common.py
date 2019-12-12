"""
I will place here common functions used across all modules.
"""
from pandas import DataFrame
from termcolor import colored


class Common:
    Reset = '\033[0m'
    Green = '\033[32m'
    White = '\033[37m'
    Red = '\033[31m'

    def green(self, string):
        return colored(string, 'green')

    def red(self, string):
        return colored(string, 'red')

    def white(self, string):
        return colored(string, 'white')

    def no_color(self, number: float):
        string = '{:.2f}'.format(number)
        return self.white(string)

    def color(self, number: float):
        string = '{:.2f}'.format(number)
        if number > 0.0:
            return self.green(string)
        elif number < 0.0:
            return self.red(string)
        else:
            return self.white(string)

    def cond_color(self, number: float, ref: float):
        string = '{:.2f}'.format(number)
        if number > ref:
            return self.green(string)
        elif number < ref:
            return self.red(string)
        else:
            number = 0.0
            string = '{:.2f}'.format(number)
            return self.white(string)

    def color_action(self, action):
        if action[:2] == 'f.':
            return self.red(action)
        else:
            return self.white(action)

    def recolor(self, df, column_name):
        if column_name not in df.columns:
            return
        df[column_name] = df[column_name].apply(
            lambda x: '{}'.format(self.color(x)))

    def reformat(self, df, column_name):
        df[column_name] = df[column_name].apply(
            lambda x: '{}'.format(self.no_color(x)))

    def recolor_ref(self, df: DataFrame, col1: str, col2: str):
        df[col1] = df.apply(
            lambda row: self.cond_color(row[col1], row[col2]),
            axis=1)

    def recolor_action(self, df: DataFrame, action_column: str = 'action'):
        df[action_column] = df[action_column].apply(
            lambda x: '{}'.format(self.color_action(x)))

    def recolor_pref(self, df: DataFrame, col: str):
        new = [df[col].values[0]]
        for i in range(1, len(df[col])):
            new.append(self.cond_color(df.iloc[i][col], df.iloc[i - 1][col]))
        df[col] = new
