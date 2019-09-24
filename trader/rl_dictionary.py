from pandas import DataFrame

from utils.dictionary import Dictionary
from display import Display
from utils.my_dict import MyDict


class RLDictionary(Dictionary):

    def __init__(self, default_params_filename='params.yaml', **kwargs):

        super().__init__(default_params_filename, **kwargs)
        # Check that I've states and actions to start playing with.
        if not (self.action and self.state):
            raise AssertionError('No states or actions defined in config file.')

        # Build a self with a sequential number associated to each action
        setattr(self, 'action_id', MyDict())
        for tup in zip(self.action, range(len(self.action))):
            self.action_id[tup[0]] = tup[1]

        # Build the reverse self for the actions self
        setattr(self, 'action_name', MyDict())
        for tup in zip(range(len(self.action)), self.action):
            self.action_name[tup[0]] = tup[1]

        # Specific attributes to store number of actions and states.
        setattr(self, 'num_actions', len(self.action))

        # Build a list of lists with the names of all possible states.
        setattr(self, 'states_list', list())
        for state in self.state.keys():
            # if state[0] == '_':
            #   self.states_list.append(self.state[state].names)
            self.states_list.append(self.state[state].names)

        # Compute the total number of states as the multiplication of the
        # number of substates in eachs posible state-stack
        setattr(self, 'num_states', int)
        self.num_states = 1
        for state in self.state.keys():
            self.num_states = self.num_states * len(
                self.state[state].names)

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
        self.results = DataFrame(columns=self.table_headers)

    # @property
    # def states_list(self):
    #     return self.states_list
