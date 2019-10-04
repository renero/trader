from arguments import Arguments
from dictionary import Dictionary
from logger import Logger
from my_dict import MyDict


class PDictionary(Dictionary):

    def __init__(self, default_params_filename='params.yaml', *args, **kwargs):
        super().__init__(default_params_filename, **kwargs)

        arguments = Arguments(args, kwargs)
        print(arguments)
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

        #     self.action_id[tup[0]] = tup[1]
        #
        # # Overwrite attributes specified by argument.
        # if self.arg_window_size is not None:
        #     self.window_size = self.arg_window_size
        # if self.arg_ticks_file is not None:
        #     self.ticks_file = self.arg_ticks_file

        # Start the logger
        self.log = Logger(self.log_level)
