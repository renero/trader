from arguments import Arguments
from dictionary import Dictionary
from logger import Logger


class CSDictionary(Dictionary):

    def __init__(self, default_params_filename='params.yaml', *args, **kwargs):
        super().__init__(default_params_filename, **kwargs)

        #
        # Extend the dictionary with the values passed in arguments.
        #
        arguments = Arguments(args, kwargs)
        for possible_action in arguments.possible_actions:
            setattr(self, possible_action, False)
        setattr(self, arguments.args.action, True)
        setattr(self, 'save_predictions', arguments.args.save)
        setattr(self, 'input_file', arguments.args.file[0])
        if arguments.args.window is not None:
            setattr(self, 'window_size', arguments.args.window[0])

        #
        # Extend the dictionary with custom meta-parameters
        #
        setattr(self, 'num_models', len(self.model_names.keys()))

        #
        # Set log_level and start the logger
        #
        setattr(self, 'log_level',
                arguments.args.debug[0] if arguments.args.debug is not None \
                    else 0)
        if 'log_level' not in self:
            self.log_level = 2  # default value = WARNING
        self.log = Logger(self.log_level)
