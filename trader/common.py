"""
I will place here common functions used across all modules.
"""


class Common:

    def log(self, *args, **kwargs):
        if self.configuration._debug is True:
            print(*args, **kwargs)

    def green(self, string):
        return '\033[92m{}\033[0m'.format(string)

    def red(self, string):
        return '\033[91m{}\033[0m'.format(string)

    def color(self, number):
        string = '{:.1f}'.format(number)
        if number >= 0.0:
            return self.green(string)
        else:
            return self.red(string)
