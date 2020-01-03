import math

from arguments import Arguments
from display import Display
from logger import Logger
from utils.dictionary import Dictionary
from utils.my_dict import MyDict


class RLDictionary(Dictionary):

    def __init__(self,
                 default_params_filename='params.yaml',
                 *args,
                 **kwargs):

        # Extend the dictionary with the values passed in arguments.
        # Call the Dictionary constructor once the parameters file is set.
        arguments = Arguments(args, kwargs)
        if arguments.args.config_file is not None:
            parameters_file = arguments.args.config_file[0]
        else:
            parameters_file = default_params_filename
        super().__init__(parameters_file, **kwargs)

        # Check that I've states and actions to start playing with.
        if not (self.action and self.state):
            raise AssertionError('No states or actions defined in config file.')

        # Override other potential parameters specified in command line.
        setattr(self, 'debug', arguments.args.debug is not None)
        setattr(self, 'log_level',
                arguments.args.debug[0] if arguments.args.debug else 3)

        # Interactive or Stepwise mode, overrides debug and log_level
        setattr(self, 'stepwise', arguments.args.stepwise)
        if self.stepwise is True:
            self.debug = True
            self.log_level = 4

        # Start the logger
        if 'log_level' not in self:
            self.log_level = 3  # default value = INFO
        self.log = Logger(self.log_level)

        self.log.info(
            'Using configuration parameters from: {}'.format(parameters_file))

        # Define if acting in BULL or BEAR mode
        if arguments.args.trading_mode is None:
            setattr(self, 'mode', 'bull')
        else:
            setattr(self, 'mode', arguments.args.trading_mode[0])
        self.log.info('Trading in {} mode'.format(self.mode))

        # Define what to do
        setattr(self, 'possible_actions', arguments.possible_actions)
        setattr(self, 'what_to_do', arguments.args.action)
        self.log.info('{} mode'.format(self.what_to_do))

        setattr(self, 'forecast_file', arguments.args.forecast[0])

        # Load the NN model file, only if the action is not "train"
        if arguments.args.action != 'train' and arguments.args.model:
            setattr(self, 'model_file', arguments.args.model[0])
        elif arguments.args.action != 'train':
            self.log.error('Model file must be specified with -m argument')
            raise ValueError('Model file must be specified with -m argument')

        setattr(self, 'no_dump', arguments.args.no_dump)
        setattr(self, 'do_plot', arguments.args.plot)
        setattr(self, 'save_model', arguments.args.save)
        setattr(self, 'totals', arguments.args.totals)
        setattr(self, 'short', arguments.args.short)
        if arguments.args.epochs is not None:
            setattr(self, 'num_episodes', int(arguments.args.epochs))
        else:
            setattr(self, 'num_episodes', 1)

        if arguments.args.decay_factor is not None:
            setattr(self, 'decay_factor', float(arguments.args.decay_factor[0]))
        else:
            setattr(self, 'decay_factor', 0.996)
        self.log.info('Decay factor: {:.4f}'.format(self.decay_factor))

        if arguments.args.initial_budget is not None:
            setattr(self, 'initial_budget',
                    float(arguments.args.initial_budget[0]))
        else:
            setattr(self, 'initial_budget', self.initial_budget)
        self.log.info('Initial Budget: {:.0f}'.format(self.initial_budget))

        # Init portfolio
        if arguments.args.init_portfolio is not None:
            setattr(self, 'init_portfolio', True)
            setattr(self, 'portfolio_name', arguments.args.init_portfolio[0])
        else:
            setattr(self, 'init_portfolio', False)
        # Use portfolio
        if arguments.args.portfolio is not None:
            setattr(self, 'use_portfolio', True)
            setattr(self, 'portfolio_name', arguments.args.portfolio[0])
        else:
            setattr(self, 'use_portfolio', False)

        # Check that if I want to predict, portfolio needs to be specified
        if self.what_to_do == 'predict' and 'portfolio_name' not in self:
            self.log.error(
                'When calling `predict`, provide a portfolio filename.')
            self.log.error(
                'To generate a portfolio, `simulate` with `--init-portfolio`')
            raise ValueError('wrong parameters')

        # Output filename specified
        if arguments.args.output is not None:
            setattr(self, 'output', arguments.args.output[0])
        else:
            setattr(self, 'output', None)

        #
        # Extensions to the dictionary
        #

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
        # number of substates in each possible state-stack
        setattr(self, 'num_states', int)
        self.num_states = 1
        for state in self.state.keys():
            self.num_states = self.num_states * len(
                self.state[state].names)
        self.log.info('{} possible states'.format(self.num_states))

        # Set the number of substates (values inside each state)
        setattr(self, 'num_substates', int)
        self.num_substates = 0
        for _, v in self.state.items():
            if isinstance(v.names, list):
                self.num_substates += len(v.names)
            else:
                self.num_substates += 1
        self.log.info('{} substates in all states'.format(self.num_substates))

        # Create a display property to centralize all reporting activity into
        # a single function. That way I can store it all in a single dataframe
        # for later analysis.
        setattr(self, 'display', Display)
        self.display: Display = Display(self)
