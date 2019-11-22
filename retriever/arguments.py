import argparse
from argparse import ArgumentParser


class Arguments(object):
    args = None
    parser: ArgumentParser = None

    def __init__(self, *args):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument(
            '-c', '--config-file', nargs=1, type=str,
            help='Relative path to configuration file to be used (YAML).')
        self.parser.add_argument(
            '-d', '--debug', nargs=1, type=int,
            help='Debug level (0..4), default 3.')
        self.parser.add_argument(
            '-f', '--file', nargs=1, type=str,
            required=True, help='OHLCV file to append')
        self.parser.add_argument(
            '-s', '--symbol', nargs=1, type=str,
            required=True, help='Stock symbol to retrieve')

        self.args = self.parser.parse_args()
