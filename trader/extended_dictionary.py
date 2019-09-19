from pandas import DataFrame

from utils.dictionary import Dictionary
from display import Display
from utils.my_dict import MyDict


class ExtendedDictionary(Dictionary):

    def __init__(self, default_params_filename='params.yaml', **kwargs):

        super().__init__(default_params_filename, **kwargs)
        # Check that I've states and actions to start playing with.
        if not (self._action and self._state):
            raise AssertionError('No states or actions defined in config file.')

        # Build a self with a sequential number associated to each action
        setattr(self, '_action_id', MyDict())
        for tup in zip(self._action, range(len(self._action))):
            self._action_id[tup[0]] = tup[1]

        # Build the reverse self for the actions self
        setattr(self, '_action_name', MyDict())
        for tup in zip(range(len(self._action)), self._action):
            self._action_name[tup[0]] = tup[1]

        # Specific attributes to store number of actions and states.
        setattr(self, '_num_actions', len(self._action))

        # Build a list of lists with the names of all possible states.
        setattr(self, '_states_list', list())
        for state in self._state.keys():
            if state[0] == '_':
                self._states_list.append(self._state[state]._names)

        # Compute the total number of states as the multiplication of the
        # number of substates in eachs posible state-stack
        setattr(self, '_num_states', int)
        self._num_states = 1
        for state in self._state.keys():
            self._num_states = self._num_states * len(
                self._state[state]._names)

        # Create a display property to centralize all reporting activity into
        # a single function. That way I can store it all in a single dataframe
        # for later analysis.
        setattr(self, 'display', Display)
        self.display = Display(self)

        # Create a DataFrame within the configuration to store all the values
        # that are relevant to later perform data analysis.
        # The YAML file contains the column names in a parameter called
        # table_headers.
        setattr(self, 'results', DataFrame)
        self.results = DataFrame(columns=self._table_headers)
