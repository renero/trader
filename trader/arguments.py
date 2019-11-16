import argparse
from argparse import ArgumentParser


class Arguments(object):
    args = None
    possible_actions = ['simulate', 'train', 'retrain', 'predict']
    parser: ArgumentParser = None

    def __init__(self, *args):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('action', nargs='?', default='train',
                                 help='What to do with the trader',
                                 choices=self.possible_actions)
        self.parser.add_argument('-d', '--debug', nargs=1, type=int,
                                 help='Debug level (0..4), default 0.')
        self.parser.add_argument('-e', '--epochs', nargs='?', type=int,
                                 const=20, default=20,
                                 help='Number of epochs in training')
        self.parser.add_argument('-f', '--forecast', nargs=1, type=str,
                                 required=True, help='Forecast file to process')
        self.parser.add_argument('--init-portfolio', nargs=1, type=str,
                                 help='Create a new portfolio')
        self.parser.add_argument('--portfolio', nargs=1, type=str,
                                 help='Use an existing portfolio in predict '
                                      'and simulate')
        self.parser.add_argument('-m', '--model', nargs=1, type=str,
                                 help='NN Model file to be loaded')
        self.parser.add_argument('-p', '--plot', action='store_true',
                                 help='Plot summary charts, default OFF')
        self.parser.add_argument('-s', '--save', action='store_true',
                                 help='Save ON, default OFF')

        self.args = self.parser.parse_args()
        action_name = 'arg_{}'.format(self.args.action)
        setattr(self, action_name, True)
        for action in set(self.possible_actions) - {action_name[1:]}:
            setattr(self, 'arg_{}'.format(action), False)
