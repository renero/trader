import argparse
from argparse import ArgumentParser


class Arguments(object):
    args = None
    possible_actions = ['forecast', 'predictions']
    parser: ArgumentParser = None

    def __init__(self, *args):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument(
            'action', nargs='?', choices=self.possible_actions,
            help='Update the forecast file or the predictions file.')
        self.parser.add_argument(
            '-c', '--config-file', nargs=1, type=str,
            help='Relative path to configuration file to be used (YAML).')
        self.parser.add_argument(
            '-f', '--file', nargs=1, type=str,
            required=True, help='Forecast or prediction file to update')
        self.parser.add_argument(
            '-d', '--debug', nargs=1, type=int,
            help='Debug level (0..4), default 0.')

        self.args = self.parser.parse_args()
