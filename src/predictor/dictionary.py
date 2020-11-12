from arguments import Arguments
from common.dictionaries import customdict_trait
from common.logger import Logger


class Dictionary(customdict_trait):

    def __init__(self, default_params_filename='params.yaml', *args, **kwargs):

        # Extend the dictionary with the values passed in arguments.
        # Call the Dictionary constructor once the parameters file is set.
        arguments = Arguments(args, kwargs)
        if arguments.args.config_file is not None:
            parameters_file = arguments.args.config_file[0]
        else:
            parameters_file = default_params_filename
        super().__init__(parameters_file, *args)

        for possible_action in arguments.possible_actions:
            setattr(self, possible_action, False)
        setattr(self, arguments.args.action, True)

        # setattr(self, 'do_plot', arguments.args.plot)
        self.do_plot = arguments.args.plot
        self.save_predictions = arguments.args.save
        self.input_file = arguments.args.file[0]
        if arguments.args.window is not None:
            self.window_size = arguments.args.window[0]
        else:
            self.window_size = 10
        if arguments.args.epochs is not None:
            self.epochs = arguments.args.epochs[0]
        else:
            self.epochs = 1

        # Output filename specified
        if arguments.args.output is not None:
            self.output = arguments.args.output[0]
        else:
            self.output = None

        #
        # Extend the dictionary with custom meta-parameters
        #
        self.ohlc_tags = list(list(self.csv_dict.keys())[1:])

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
