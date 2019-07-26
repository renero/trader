"""
I will place here common functions used across all modules.
"""


class Common:

    def log(self, *args, **kwargs):
        if self.configuration._debug is True:
            print(*args, **kwargs)
