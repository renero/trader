import argparse
from argparse import ArgumentParser


class Arguments(object):
    args = None
    possible_actions = ['simulate', 'train', 'retrain', 'predict']
    parser: ArgumentParser = None

    def __init__(self, *args):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument(
            'action', nargs='?', default='train', choices=self.possible_actions,
            help='What to do with the trader')
        self.parser.add_argument(
            '-c', '--config-file', nargs=1, type=str,
            help='Relative path to configuration file to be used (YAML).')
        self.parser.add_argument(
            '-d', '--debug', nargs=1, type=int,
            help='Debug level (0..4), default 3.')
        self.parser.add_argument(
            '-e', '--epochs', nargs='?', type=int, const=20, default=20,
            help='Number of epochs in training')
        self.parser.add_argument(
            '-f', '--forecast', nargs=1, type=str, required=True,
            help='Forecast file to process')
        self.parser.add_argument(
            '--init-portfolio', nargs=1, type=str,
            help='Create a new portfolio')
        self.parser.add_argument(
            '--portfolio', nargs=1, type=str,
            help='Use an existing portfolio in predict and simulate')
        self.parser.add_argument(
            '-m', '--model', nargs=1, type=str,
            help='RL Model file to be loaded (without the extension)')
        self.parser.add_argument(
            '--short', action='store_true',
            help='Display a short version of summary table when simulating')
        self.parser.add_argument(
            '--no-dump', action='store_true',
            help='Do not dump temp JSON portfolio file (ONLY) after predict')
        self.parser.add_argument(
            '-o', '--output', nargs=1, type=str,
            help='Output filename to be used to save results (w/out extension)')
        self.parser.add_argument(
            '-p', '--plot', action='store_true',
            help='Plot summary charts, default OFF')
        self.parser.add_argument(
            '-s', '--save', action='store_true',
            help='Save ON, default OFF')
        self.parser.add_argument(
            '-T', '--totals', action='store_true',
            help='When in simulate mode, display only the totals summary.')
        self.parser.add_argument(
            '--stepwise', action='store_true',
            help='Pauses every step, waiting for user to hit Enter')
        self.parser.add_argument(
            '-t', '--trading-mode', nargs=1, type=str,
            help='Trading mode: either \'bull\' (default) or \'bear\'.')
        self.parser.add_argument(
            '--decay-factor', nargs=1, type=float,
            help='Decay factor applied to epsilon after each iteration.')
        self.parser.add_argument(
            '--initial-budget', nargs=1, type=float,
            help='Initial budget to start with either training or simulation.')

        self.args = self.parser.parse_args()
        action_name = 'arg_{}'.format(self.args.action)
        setattr(self, action_name, True)
        for action in set(self.possible_actions) - {action_name[1:]}:
            setattr(self, 'arg_{}'.format(action), False)
