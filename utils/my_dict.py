debug = False


class MyDict(dict):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, key):
        if key in self:
            return self.get(key)
        raise KeyError('Key <{}> not present in dictionary'.format(key))

    def __setattr__(self, key, value):
        self[key] = value

    @staticmethod
    def debug(*args, **kwargs):
        if debug is True:
            print(*args, **kwargs)

    def add_dict(self, this_object, param_dictionary, add_underscore=True):
        for param_name in param_dictionary.keys():
            self.debug('ATTR: <{}> type is {}'.format(
                param_name, type(param_dictionary[param_name])))

            if add_underscore is True:
                attribute_name = '{}'.format(param_name)
            else:
                attribute_name = param_name

            if type(param_dictionary[param_name]) is not dict:
                self.debug(' - Setting attr name {} to {}'.format(
                    attribute_name, param_dictionary[param_name]))
                setattr(this_object, attribute_name,
                        param_dictionary[param_name])
            else:
                self.debug(' x Dictionary Found!')

                self.debug('   > Creating new dict() with name <{}>'.format(
                    attribute_name))
                setattr(this_object, attribute_name, MyDict())

                self.debug('     > New Attribute <{}> type is: {}'.format(
                    attribute_name, type(getattr(this_object, attribute_name))
                ))
                new_object = getattr(this_object, attribute_name)

                self.debug('   > Calling recursively with dict')
                self.debug('     {}'.format(param_dictionary[param_name]))
                this_object.add_dict(new_object, param_dictionary[param_name])
