from arguments import Arguments
from display import Display
from logger import Logger
from utils.dictionary import Dictionary
from utils.my_dict import MyDict


class RLDictionary(Dictionary):

    def __init__(self, default_params_filename='params.yaml', *args, **kwargs):

        super().__init__(default_params_filename, **kwargs)
        # Check that I've states and actions to start playing with.
        if not (self.action and self.state):
            raise AssertionError('No states or actions defined in config file.')

        # Read the arguments passed in CLI to override parameters
        arguments = Arguments(args, kwargs)

        # Override other potential parameters specified in command line.
        setattr(self, 'debug', arguments.args.debug is not None)
        setattr(self, 'log_level',
                arguments.args.debug[0] if arguments.args.debug is not None \
                    else 0)

        # Start the logger
        if 'log_level' not in self:
            self.log_level = 3  # default value = INFO
        self.log = Logger(self.log_level)

        # Define what to do
        setattr(self, 'possible_actions', arguments.possible_actions)
        setattr(self, 'what_to_do', arguments.args.action)
        self.log.info(self.what_to_do)

        setattr(self, 'save_model', arguments.args.save)
        if arguments.args.epochs is not None:
            setattr(self, 'num_episodes', int(arguments.args.epochs[0]))
        if arguments.args.file is not None:
            setattr(self, 'data_path', arguments.args.forecast[0])

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
            self.states_list.append(self.state[state].names)

        # Compute the total number of states as the multiplication of the
        # number of substates in eachs posible state-stack
        setattr(self, 'num_states', int)
        self.num_states = 1
        for state in self.state.keys():
            self.num_states = self.num_states * len(
                self.state[state].names)
        self.log.debug('{} possible states'.format(self.num_states))

        # Create a display property to centralize all reporting activity into
        # a single function. That way I can store it all in a single dataframe
        # for later analysis.
        setattr(self, 'display', Display)
        self.display: Display = Display(self)
