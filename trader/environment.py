import importlib

import pandas as pd

from common import Common
from display import Display
from portfolio import Portfolio
from scombiner import SCombiner


class Environment(Common):
    configuration = None
    max_states_ = 0
    data_ = None
    current_state_ = 0
    t = 0
    portfolio_ = None
    price_ = 0.
    forecast_ = 0.
    max_actions_ = 0
    done_ = False
    reward_ = 0
    new_state_: int = 0

    def __init__(self, configuration):

        self.configuration = configuration
        self.display = Display(configuration)
        self.states = SCombiner(self.configuration.states_list)
        self.read_market_data(self.configuration._data_path)
        self.init_environment()

    def init_environment(self):
        """
        Initialize the portfolio by updating market price according to the
        current timeslot 't', creating a new object, and updating the
        internal state of the environment accordingly.
        :return: The initial state.
        """
        self.update_market_price()
        self.portfolio_ = Portfolio(self.configuration,
                                    self.price_,
                                    self.forecast_)
        return self.update_state()

    def reset(self):
        """
        Reset all internal states
        :return:
        """
        self.done_ = False
        self.t = 0
        del self.portfolio_
        return self.init_environment()

    def read_market_data(self, path):
        """
        Reads the simulation data.
        :param path:
        :return:
        """
        self.data_ = pd.read_csv(path)
        self.max_states_ = self.data_.shape[0]

    def update_market_price(self):
        """
        Set the price to the current time slot,
        reading column 0 from DF
        """
        assert self.data_ is not None, 'Price series data has not been read yet'
        self.price_ = self.data_.iloc[self.t, 0]
        self.forecast_ = self.data_.iloc[self.t, 1]

    @staticmethod
    def decide_next_action(state, strategy):
        return strategy[state]

    def update_state(self):
        """
        Determine the state of my portfolio value
        :return: New state
        """
        # Iterate through the list of states defined in the parameters file
        # and call the update_state() static method in them.
        new_substates = []
        for module_param_name in self.configuration._state.keys():
            # The extended classes are defined in the params file and must
            # start with the 'state_' string.
            # The '[1:]' serves to remove the leading underscore.
            module_name = 'state_' + module_param_name[1:]
            module = importlib.import_module(module_name)
            state_class = getattr(module, module_name)
            new_substate = state_class.update_state(self.portfolio_)
            new_substates.append(new_substate)

        # Get the ID resulting from the combination of the sub-states
        self.current_state_ = self.states.get_id(*new_substates)
        return self.current_state_

    def step(self, action):
        """
        Send an action to my Environment.
        :param action: the action.
        :return: state, reward, done and iter count.
        """
        assert action < self.configuration._num_actions, \
            'Action ID must be between 0 and {}'.format(
                self.configuration._num_actions)

        # Call to the proper portfolio method, based on the action number
        # passed to this argument.
        self.reward_ = getattr(self.portfolio_,
                               self.configuration._action_name[action])()
        self.log(' | {:>15} | {:s}'.format(
            self.color(self.reward_), self.states.name(self.current_state_)))

        self.t += 1
        if self.t >= self.max_states_:
            self.done_ = True
            self.display.report(self.portfolio_,
                                self.t - 1,
                                disp_footer=True)
            self.portfolio_.reset_history()
            self.log("")
            return self.new_state_, self.reward_, self.done_, self.t

        self.update_market_price()
        self.portfolio_.update(self.price_, self.forecast_)
        self.new_state_ = self.update_state()
        self.display.report(self.portfolio_, self.t)
        self.portfolio_.append_to_history(self)

        return self.new_state_, self.reward_, self.done_, self.t
