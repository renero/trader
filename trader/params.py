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

from pathlib import Path
from yaml import safe_load, YAMLError
from os import getcwd


class MyDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self.get(key)
        raise KeyError('Key <{}> not present in dictionary'.format(key))

    def __setattr__(self, key, value):
        self[key] = value

    def add_dict(self, this_object, param_dictionary):
        for param_name in param_dictionary.keys():
            print('ATTR: <{}> type is {}'.format(
                param_name, type(param_dictionary[param_name])))
            attribute_name = '_{}'.format(param_name)
            if type(param_dictionary[param_name]) is not dict:
                print(' - Setting attr name {} to {}'.format(
                    attribute_name, param_dictionary[param_name]))
                setattr(this_object, attribute_name, param_dictionary[param_name])
            else:
                print(' x Dictionary Found!')

                print('   > Creating new dict() with name <{}>'.format(
                    attribute_name))
                setattr(this_object, attribute_name, MyDict())

                print('     > New Attribute <{}> type is: {}'.format(
                    attribute_name, type(getattr(this_object, attribute_name))
                ))
                new_object = getattr(this_object, attribute_name)

                print('   > Calling recursively with dict')
                print('     {}'.format(param_dictionary[param_name]))
                this_object.add_dict(new_object, param_dictionary[param_name])


class Params(MyDict):

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

