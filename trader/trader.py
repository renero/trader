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


class MyDict(dict):
    debug = False

    def __init__(self, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.debug = debug

    def __getattr__(self, key):
        if key in self:
            return self.get(key)
        raise KeyError('Key <{}> not present in dictionary'.format(key))

    def __setattr__(self, key, value):
        self[key] = value

    def log(self, *args, **kwargs):
        if self.debug is True:
            print(*args, **kwargs)

    def add_dict(self, this_object, param_dictionary):
        for param_name in param_dictionary.keys():
            self.log('ATTR: <{}> type is {}'.format(
                param_name, type(param_dictionary[param_name])))
            attribute_name = '_{}'.format(param_name)
            if type(param_dictionary[param_name]) is not dict:
                self.log(' - Setting attr name {} to {}'.format(
                    attribute_name, param_dictionary[param_name]))
                setattr(this_object, attribute_name,
                        param_dictionary[param_name])
            else:
                self.log(' x Dictionary Found!')

                self.log('   > Creating new dict() with name <{}>'.format(
                    attribute_name))
                setattr(this_object, attribute_name, MyDict())

                self.log('     > New Attribute <{}> type is: {}'.format(
                    attribute_name, type(getattr(this_object, attribute_name))
                ))
                new_object = getattr(this_object, attribute_name)

                self.log('   > Calling recursively with dict')
                self.log('     {}'.format(param_dictionary[param_name]))
                this_object.add_dict(new_object, param_dictionary[param_name])


class Trader(MyDict):

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

        # Check that I've states and actions to start playing with.
        if not self._action or not self._state:
            raise AssertionError('No states or actions defined in config file.')

        # Build the reverse dictionary for the actions dictionary
        setattr(self, '_action_id', dict())
        for tup in zip(range(len(self._action)), self._action):
            self._action_id[tup[0]] = tup[1]
