import sys

import yaml

from arguments import Arguments
from cs_logger import CSLogger
from utils.file_io import file_exists
from os.path import dirname, realpath


class Params(Arguments):

    def __init__(self, filepath='params.yaml', args=[]):
        """
        Init a class with all the parameters in the default YAML file.
        For each of them, create a new class attribute, with the same name
        but preceded by '_' character.
        This class inherits from Arguments which reads any possible argument
        in the command line, resulting in overwriting those in the YAML file.
        """
        super(Params, self).__init__(*args)

        # Check if file exists
        filepath = file_exists(filepath, dirname(realpath(__file__)))
        with open(filepath, 'r') as stream:
            try:
                self.params = yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)

        for param_name in self.params.keys():
            attribute_name = '_{}'.format(param_name)
            if not hasattr(self, attribute_name):
                setattr(self, attribute_name, self.params[param_name])

        # Overwrite attributes specified by argument.
        if self.arg_window_size is not None:
            self._window_size = self.arg_window_size
        if self.arg_ticks_file is not None:
            self._ticks_file = self.arg_ticks_file

        # Start the logger
        self.log = CSLogger(self._log_level)

    @property
    def do_train(self):
        return self._train is True

    @property
    def model_names(self):
        return self._model_names

    @property
    def subtypes(self):
        return self._subtypes

    @property
    def max_tick_series_length(self):
        return self._max_tick_series_length