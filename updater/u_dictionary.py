from arguments import Arguments
from dictionary import Dictionary
from logger import Logger


class UDictionary(Dictionary):

    def __init__(self, default_params_filename='params.yaml', *args, **kwargs):

        # Extend the dictionary with the values passed in arguments.
        # Call the Dictionary constructor once the parameters file is set.
        arguments = Arguments(args, kwargs)
        if arguments.args.config_file is not None:
            parameters_file = arguments.args.config_file[0]
        else:
            parameters_file = default_params_filename
        super().__init__(parameters_file, **kwargs)

        setattr(self, 'file', arguments.args.file[0])

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
