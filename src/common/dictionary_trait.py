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
from pathlib import Path

from yaml import safe_load, YAMLError

debug = False

class MyCustomDictionary(dict):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, key):
        """
        Check out https://stackoverflow.com/a/42272450
        """
        if key in self:
            return self.get(key)
        raise AttributeError('Key <{}> not present in dictionary'.format(key))

    def __setattr__(self, key, value):
        self[key] = value

    @staticmethod
    def logdebug(*args, **kwargs):
        if debug is True:
            print(*args, **kwargs)

    def add_dict(self, this_object, param_dictionary, add_underscore=True):
        for param_name in param_dictionary.keys():
            self.logdebug('ATTR: <{}> type is {}'.format(
                param_name, type(param_dictionary[param_name])))

            if add_underscore is True:
                attribute_name = '{}'.format(param_name)
            else:
                attribute_name = param_name

            if type(param_dictionary[param_name]) is not dict:
                self.logdebug(' - Setting attr name {} to {}'.format(
                    attribute_name, param_dictionary[param_name]))
                setattr(this_object, attribute_name,
                        param_dictionary[param_name])
            else:
                self.logdebug(' x Dictionary Found!')

                self.logdebug('   > Creating new dict() with name <{}>'.format(
                    attribute_name))
                setattr(this_object, attribute_name, MyCustomDictionary())

                self.logdebug('     > New Attribute <{}> type is: {}'.format(
                    attribute_name, type(getattr(this_object, attribute_name))
                ))
                new_object = getattr(this_object, attribute_name)

                self.logdebug('   > Calling recursively with dict')
                self.logdebug('     {}'.format(param_dictionary[param_name]))
                this_object.add_dict(new_object, param_dictionary[param_name])


class DictionaryTrait(MyCustomDictionary):

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
