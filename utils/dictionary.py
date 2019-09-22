"""
This class reads params from a YAML file and creates an object that
contains attributes named as the params in the file, accessible through
getters:

  object.parameter

in addition to classical dictionary access method

  object[parameter]

The structure of attributes is built recursively if they contain a dictionary.

  object.attr1.attr2.attr3

"""

from os import getcwd
from os.path import basename
from pathlib import Path
from pandas import DataFrame

from utils.my_dict import MyDict
from display import Display
from yaml import safe_load, YAMLError


class Dictionary(MyDict):

    def __init__(self, default_params_filename='params.yaml', **kwargs):
        """
        Read the parameters from a default filename
        :return:
        """
        super().__init__(**kwargs)
        params = {}
        cwd = Path(getcwd())
        params_path: str = str(cwd.joinpath(default_params_filename))

        with open(params_path, 'r') as stream:
            try:
                params = safe_load(stream)
            except YAMLError as exc:
                print(exc)

        self.add_dict(self, params)

    @property
    def save_model(self):
        return self._save_model

    @property
    def state(self):
        return self._state

    @property
    def states_list(self):
        return self._states_list
