import importlib

import numpy as np
import pandas as pd

from common import Common
from memory import Memory
from portfolio import Portfolio
from states_combiner import StatesCombiner


class Environment(Common):
    configuration = None
    max_states_ = 0
    data_ = None
    current_state_ = 0
    t = 0
    portfolio = None
    price_ = 0.
    forecast_ = 0.
    green_ = 0.
    blue_ = 0.
    konkorde_ = 0.
    max_actions_ = 0
    done_ = False
    reward_ = 0
    new_state_: int = 0
    have_konkorde = False

    def __init__(self, configuration):
        self.params = configuration
        self.log = self.params.log
        self.display = self.params.display
        self.memory = Memory(self.params)
        self.results = self.memory.results
        self.log.info('Creating Environment')

        if 'seed' in self.params:
            np.random.seed(self.params.seed)
        else:
            np.random.seed(1)

        self.states = StatesCombiner(self.params)
        self.read_market_data(self.params.data_path)
        self.portfolio = Portfolio(self.params,
                                   self.price_, self.forecast_, self.memory)
        self.init_environment(creation_time=True)
        self.log.info('Environment created')

    def init_environment(self, creation_time):
        """
        Initialize the portfolio by updating market price according to the
        current timeslot 't', creating a new object, and updating the
        internal state of the environment accordingly.
        :return: The initial state.
        """
        self.update_market_price()
        # self.portfolio = Portfolio(self.params,
        #                            self.price_,
        #                            self.forecast_,
        #                            self.memory)
        self.portfolio.reset(self.price_,
                             self.forecast_,
                             self.memory)
        if creation_time is not True:
            self.memory.record_values(self.portfolio, t=0)
        return self.update_state()

    def reset(self):
        """
        Reset all internal states
        :return:
        """
        self.log.debug('Resetting environment')
        self.done_ = False
        self.t = 0
        # del self.portfolio
        self.memory.reset()
        return self.init_environment(creation_time=False)

    def read_market_data(self, path):
        """
        Reads the simulation data.
        :param path:
        :return:
        """
        if 'delimiter' not in self.params:
            delimiter = ','
        else:
            delimiter = self.params.delimiter
        self.data_ = pd.read_csv(path, delimiter)
        self.max_states_ = self.data_.shape[0]
        self.log.info('Read trader file: {}'.format(path))

        # Do i have konkorde?
        setattr(self.params, 'have_konkorde', bool)
        self.params.have_konkorde = False
        if self.params.column_name['green'] in self.data_.columns and \
                self.params.column_name['blue'] in self.data_.columns:
            self.params.have_konkorde = True
            self.log.info('Konkorde index present!')

    def update_market_price(self):
        """
        Set the price to the current time slot,
        reading column 0 from DF
        """
        assert self.data_ is not None, 'Price series data has not been read yet'
        col_names = list(self.data_.columns)

        self.price_ = self.data_.iloc[
            self.t, col_names.index(self.params.column_name['price'])]
        self.forecast_ = self.data_.iloc[
            self.t, col_names.index(self.params.column_name['forecast'])]

        # If I do have konkorde indicators, I also read them.
        if self.params.have_konkorde:
            self.green_ = self.data_.iloc[
                self.t, col_names.index(self.params.column_name['green'])]
            self.blue_ = self.data_.iloc[
                self.t, col_names.index(self.params.column_name['blue'])]
            self.konkorde_ = self.green_ + self.blue_

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
        for module_param_name in self.params.state.keys():
            # The extended classes are defined in the params file and must
            # start with the 'state_' string.
            module_name = 'State' + module_param_name
            module = importlib.import_module('state_classes')  # module_name)
            state_class = getattr(module, module_name)
            new_substate = state_class.update_state(self.portfolio)
            new_substates.append(new_substate)

        # Get the ID resulting from the combination of the sub-states
        self.current_state_ = self.states.get_id(*new_substates)
        self.log.debug('t={}, state updated to: {}'.format(
            self.t, self.states.name(self.current_state_)))
        return self.current_state_

    def step(self, action):
        """
        Send an action to my Environment.
        :param action: the action.
        :return: state, reward, done and iter count.
        """
        assert action < self.params.num_actions, \
            'Action ID must be between 0 and {}'.format(
                self.params.num_actions)

        # Call to the proper portfolio method, based on the action number
        # passed to this argument.
        self.log.debug('t={}, price={}, action decided={}'.format(
            self.t, self.price_, action))
        self.reward_ = getattr(self.portfolio,
                               self.params.action_name[action])()
        self.memory.record_reward(self.reward_,
                                  self.states.name(self.current_state_))
        self.log.debug(
            't={}, reward ({:.2f}), recorded to action \'{}\''.format(
                self.t, self.reward_, self.states.name(self.current_state_)
            ))

        self.t += 1
        if self.t >= self.max_states_:
            self.done_ = True
            self.memory.record_values(self.portfolio, self.t - 1)
            self.portfolio.reset_history()
            return self.new_state_, self.reward_, self.done_, self.t

        self.update_market_price()
        if self.params.have_konkorde:
            self.portfolio.update_after_step(self.price_, self.forecast_,
                                             self.konkorde_)
        else:
            self.portfolio.update_after_step(self.price_, self.forecast_)
        self.new_state_ = self.update_state()
        self.memory.record_values(self.portfolio, self.t)
        self.portfolio.append_to_history(self)

        return self.new_state_, self.reward_, self.done_, self.t
