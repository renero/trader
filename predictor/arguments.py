import argparse


class Arguments(object):
    args = None
    possible_actions = ['train', 'predict', 'ensemble']

    def __init__(self, *args):
        parser = argparse.ArgumentParser()

        parser.add_argument('action', default='predict',
                            choices=self.possible_actions)
        parser.add_argument('-t', '--ticks', nargs=1, type=str,
                            help='Ticks file to process')
        parser.add_argument('-w', '--window', nargs=1, type=int,
                            help='Window size for the LSTM')

        self.args = parser.parse_args()
        action_name = '_{}'.format(self.args.action)
        setattr(self, action_name, True)
        for action in set(self.possible_actions) - {action_name[1:]}:
            setattr(self, '_{}'.format(action), False)

    @property
    def arg_ticks_file(self):
        if self.args.ticks is not None:
            return self.args.ticks[0]
        else:
            return None

    @property
    def arg_window_size(self):
        if self.args.window is not None:
            return self.args.window[0]
        else:
            return None
