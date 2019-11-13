from arguments import Arguments
from dictionary import Dictionary
from logger import Logger


class RTDictionary(Dictionary):

    def __init__(self, default_params_filename='params.yaml', *args, **kwargs):
        super().__init__(default_params_filename, **kwargs)

        #
        # Extend the dictionary with the values passed in arguments.
        #
        arguments = Arguments(args, kwargs)

        setattr(self, 'file', arguments.args.file[0])
        setattr(self, 'symbol', arguments.args.symbol[0])

        #
        # Set log_level and start the logger
        #
        setattr(self, 'log_level',
                arguments.args.debug[0] if arguments.args.debug is not None \
                    else 3)
        if 'log_level' not in self:
            self.log_level = 3  # default value = WARNING
        self.log = Logger(self.log_level)
