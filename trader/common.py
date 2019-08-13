"""
I will place here common functions used across all modules.
"""
from termcolor import colored


class Common:

    def log(self, *args, **kwargs):
        if self.configuration._debug is True:
            print(*args, **kwargs)

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
