import numpy as np

from arguments import Arguments
from dictionaries import customdict_trait
from logger import Logger


class indicators_dict(customdict_trait):

    def __init__(self, default_params_filename='params.yaml', *args, **kwargs):

        # Extend the dictionary with the values passed in arguments.
        # Call the Dictionary constructor once the parameters file is set.
        arguments = Arguments(args, kwargs)
        if arguments.args.config_file is not None:
            parameters_file = arguments.args.config_file[0]
        else:
            parameters_file = default_params_filename
        super().__init__(parameters_file, **kwargs)

        mask = [False] * len(arguments.possible_actions)
        for possible_action in arguments.possible_actions:
            setattr(self, possible_action, False)
        setattr(self, arguments.args.action, True)

        # This is to get the name of the indicator to compute in a param
        mask[arguments.possible_actions.index(arguments.args.action)] = True
        setattr(self,
                'indicator_name',
                np.array(arguments.possible_actions)[mask][0])

        # The name of the class implementing the indicator
        setattr(self, 'indicator_class',
                '{}{}'.format(
                    self.indicator_name[0].upper(), self.indicator_name[1:])
                )

        setattr(self, 'input_file', arguments.args.file[0])
        setattr(self, 'save', arguments.args.save)

        # Do I merge files indicator to original OHLC file?
        if arguments.args.merge_to_file is not None:
            setattr(self, 'merge_file', arguments.args.merge_to_file[0])
            setattr(self, 'merge', True)
        else:
            setattr(self, 'merge_file', None)
            setattr(self, 'merge', False)

        # This one controls whether to compute/display/append only the
        # value for today (last in the series). Used for daily invocation
        if arguments.args.today is not None:
            setattr(self, 'today', arguments.args.today)
        else:
            setattr(self, 'today', False)

        # Extract the scaler name, if any.
        if self.today and 'scaler_file' not in arguments.args:
            arguments.parser.error(
                'When generating index for a single day, a scaler must be set')
        if arguments.args.today:
            setattr(self, 'scaler_name', arguments.args.scaler_file[0])

        # Check that all parameters needed are in place.
        if self.save and self.merge_file is not None:
            arguments.parser.error(
                'Save option is not compatible with merge option')

        # Output filename specified
        if arguments.args.output is not None:
            setattr(self, 'output', arguments.args.output[0])
        else:
            setattr(self, 'output', None)

        #
        # Set log_level and start the logger
        #
        setattr(self, 'log_level',
                arguments.args.debug[0] if arguments.args.debug is not None \
                    else 3)
        if 'log_level' not in self:
            self.log_level = 3  # default value = WARNING
        self.log = Logger(self.log_level)

        self.log.info(
            'Using configuration parameters from: {}'.format(parameters_file))
