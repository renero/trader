import argparse


class Arguments(object):
    args = None
    possible_actions = ['simulate', 'learn', 'retrain', 'predict']

    def __init__(self, *args):
        parser = argparse.ArgumentParser()

        parser.add_argument('action', nargs='?', default='train',
                            help='What to do with the trader',
                            choices=self.possible_actions)
        parser.add_argument('-d', '--debug', nargs=1, type=int,
                            help='Debug level (0..4), default 0.')
        parser.add_argument('-s', '--save', action='store_true',
                            help='Save ON, default OFF')
        parser.add_argument('-e', '--epochs', nargs=1, type=int,
                            help='Number of epochs in training')
        parser.add_argument('-f', '--file', nargs=1, type=str,
                            help='Forecast file to process')

        self.args = parser.parse_args()
        action_name = 'arg_{}'.format(self.args.action)
        setattr(self, action_name, True)
        for action in set(self.possible_actions) - {action_name[1:]}:
            setattr(self, 'arg_{}'.format(action), False)
