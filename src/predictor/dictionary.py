from arguments import Arguments
from common.dictionary_trait import DictionaryTrait
from common.logger import Logger


class Dictionary(DictionaryTrait):

    def __init__(self, default_params_filename='params.yaml', *args, **kwargs):

        # Extend the dictionary with the values passed in arguments.
        # Call the Dictionary constructor once the parameters file is set.
        arguments = Arguments(args, kwargs)
        if arguments.args.config_file is not None:
            parameters_file = arguments.args.config_file[0]
        else:
            parameters_file = default_params_filename
        super().__init__(parameters_file, **kwargs)

        for possible_action in arguments.possible_actions:
            setattr(self, possible_action, False)
        setattr(self, arguments.args.action, True)

        setattr(self, 'do_plot', arguments.args.plot)
        setattr(self, 'save_predictions', arguments.args.save)
        setattr(self, 'input_file', arguments.args.file[0])
        if arguments.args.window is not None:
            setattr(self, 'window_size', arguments.args.window[0])
        else:
            setattr(self, 'window_size', 10)
        if arguments.args.epochs is not None:
            setattr(self, 'epochs', arguments.args.epochs[0])
        else:
            setattr(self, 'epochs', 1)

        # Output filename specified
        if arguments.args.output is not None:
            setattr(self, 'output', arguments.args.output[0])
        else:
            setattr(self, 'output', None)

        #
        # Extend the dictionary with custom meta-parameters
        #
        setattr(self, 'ohlc_tags', list(list(self.csv_dict.keys())[1:]))

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
