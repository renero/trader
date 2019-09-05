import datetime
import inspect
import sys


def caller_name(n):
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
    if (n+1) > max_stack_len:
        depth = n
    if 'self' not in sys._getframe(depth).f_locals:
        class_name = 'NA'
    else:
        class_name = sys._getframe(depth).f_locals["self"].__class__.__name__
    calling_function = sys._getframe(depth).f_code.co_name

    return '{}:{}'.format(class_name, calling_function)


class CSLogger:
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

    _level = 0

    def __init__(self, level=0):
        self._level = level

    def debug(self, msg):
        if self._level < self._DEBUG:
            return
        print('DEBUG: {}'.format(msg))

    def highlight(self, msg):
        now = '{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())
        print('{} - INFO - {:<30} - {}{}{}'.format(
            now,
            caller_name(1),
            self.INFOGREY, msg, self.ENDC))

    def info(self, msg):
        if self._level < self._INFO:
            return
        now = '{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())
        print('{} - INFO - {:<30} - {}'.format(now, caller_name(1), msg))

    def warn(self, msg):
        if self._level < self._WARN:
            return
        print('{}WARN: {}{}'.format(self.WARNING, msg, self.ENDC))

    def error(self, msg):
        if self._level < self._ERROR:
            return
        print('{}ERROR: {}{}'.format(self.FAIL, msg, self.ENDC))
