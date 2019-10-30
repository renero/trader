import numpy as np

from arguments import Arguments
from dictionary import Dictionary
from logger import Logger


class IDXDictionary(Dictionary):

    def __init__(self, default_params_filename='params.yaml', *args, **kwargs):
        super().__init__(default_params_filename, **kwargs)

        #
        # Extend the dictionary with the values passed in arguments.
        #
        arguments = Arguments(args, kwargs)

        mask = [False] * len(arguments.possible_actions)
        for possible_action in arguments.possible_actions:
            setattr(self, possible_action, False)
        setattr(self, arguments.args.action, True)
        mask[arguments.possible_actions.index(arguments.args.action)] = True
        setattr(self,
                'indicator_name',
                np.array(arguments.possible_actions)[mask][0])

        setattr(self, 'input_file', arguments.args.input[0])
        setattr(self, 'append', arguments.args.append)
        if arguments.args.merge is not None:
            setattr(self, 'merge_file', arguments.args.merge[0])
            setattr(self, 'merge', True)
        else:
            setattr(self, 'merge_file', None)
            setattr(self, 'merge', False)

        if self.append and self.merge_file is not None:
            arguments.parser.error(
                'Append option is not compatible with merge option')

        #
        # Set log_level and start the logger
        #
        setattr(self, 'log_level',
                arguments.args.debug[0] if arguments.args.debug is not None \
                    else 0)
        if 'log_level' not in self:
            self.log_level = 2  # default value = WARNING
        self.log = Logger(self.log_level)
