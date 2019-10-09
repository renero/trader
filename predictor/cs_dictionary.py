from arguments import Arguments
from dictionary import Dictionary
from logger import Logger


class CSDictionary(Dictionary):

    def __init__(self, default_params_filename='params.yaml', *args, **kwargs):
        super().__init__(default_params_filename, **kwargs)

        arguments = Arguments(args, kwargs)
        setattr(self, 'train', False)
        setattr(self, 'predict', False)
        setattr(self, arguments.args.action, True)
        if arguments.args.ticks is not None:
            setattr(self, 'ticks_file', arguments.args.ticks)
        if arguments.args.window is not None:
            setattr(self, 'window_size', arguments.args.window[0])

        #
        # Extensions to the base YAML dictionary of parameters
        #

        # Build a self with a sequential number associated to each action
        setattr(self, 'num_models', len(self.model_names.keys()))

        # Start the logger
        self.log = Logger(self.log_level)
