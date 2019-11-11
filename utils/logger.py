import datetime
import inspect
import sys


def who(n):
    """
    Get the class name and the method name of the caller to the log

    :param n: For current func name, specify 0 or no argument, for name of
    caller of current func, specify 1. For name of caller of caller of
    current func, specify 2. etc.
    :return: A string with the class name (if applicable, '<module>' otherwise
    and the calling function name.
    """
    max_stack_len = len(inspect.stack())
    depth = n + 1
    if (n + 1) > max_stack_len:
        depth = n
    if 'self' not in sys._getframe(depth).f_locals:
        class_name = 'NA'
    else:
        class_name = sys._getframe(depth).f_locals["self"].__class__.__name__
    calling_function = sys._getframe(depth).f_code.co_name

    return '{}:{}'.format(class_name, calling_function)


class Logger:
    _DEBUG = 4
    _INFO = 3
    _WARN = 2
    _ERROR = 1

    # https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    INFOGREY = '\033[30m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    _level = 3

    def __init__(self, level=0):
        self._level = level
        if self._level > 3:
            print('Log level:', self._level)

    def set_level(self, level):
        self._level = level
        self.info('Setting new log level to: {}'.format(level))

    def debug(self, what, **kwargs):
        if self._level < self._DEBUG:
            return
        when = '{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())
        print('{} - {}DEBUG - {:<30} - {}{}'.format(
            when, self.OKBLUE, who(1), what, self.ENDC, **kwargs))

    def highlight(self, what):
        now = '{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())
        print('{} - INFO  - {:<30} - {}{}{}'.format(
            now,
            who(1),
            self.INFOGREY, what, self.ENDC))

    def info(self, what, **kwargs):
        if self._level < self._INFO:
            return
        when = '{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())
        print('{} - INFO  - {:<30} - {}'.format(
            when, who(1), what, **kwargs))

    def warn(self, what):
        if self._level < self._WARN:
            return
        print('{}WARN: {}{}'.format(self.WARNING, what, self.ENDC))

    def error(self, what):
        if self._level < self._ERROR:
            return
        print('{}ERROR: {}{}'.format(self.FAIL, what, self.ENDC))
