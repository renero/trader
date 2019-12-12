import importlib
import json

import numpy as np
import pandas as pd

from common import Common
from file_io import valid_output_name
from last import last
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
    ts_ = None
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
        self.read_market_data(self.params.forecast_file)
        self.portfolio = Portfolio(self.params,
                                   self.price_, self.forecast_, self.memory)
        self.init_environment(creation_time=True)
        self.log.info('Environment created')

    def init_environment(self, creation_time):
        """
        Initialize the portfolio by updating market price according to the
        current timeslot 't', resetting the object, and updating the
        internal state of the environment accordingly.
        :return: The initial state.
        """
        self.update_mkt_price()
        self.portfolio.reset(self.price_,
                             self.forecast_,
                             self.memory)
        if creation_time is not True:
            self.memory.record_values(self.portfolio, t=0, ts=self.ts_)
        return self.update_state()

    def reset(self):
        """
        Reset all internal states
        :return:
        """
        self.log.debug('Resetting environment')
        self.done_ = False
        self.t = 0
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
        self.log.info('Read trader forecast file: {}'.format(path))

        # Do i have konkorde?
        setattr(self.params, 'have_konkorde', bool)
        self.params.have_konkorde = False
        if self.params.column_name['green'] in self.data_.columns and \
                self.params.column_name['blue'] in self.data_.columns:
            self.params.have_konkorde = True
            self.log.info('Konkorde index present!')

    def update_mkt_price(self):
        """
        Set the price to the current time slot,
        reading column 0 from DF
        """
        assert self.data_ is not None, 'Price series data has not been read yet'
        col_names = list(self.data_.columns)

        self.ts_ = self.data_.iloc[
            self.t, col_names.index(self.params.column_name['date'])]
        self.price_ = self.data_.iloc[
            self.t, col_names.index(self.params.column_name['price'])]
        self.forecast_ = self.data_.iloc[
            self.t, col_names.index(self.params.column_name['forecast'])]

        self.log.debug('  t={}, updated market price/forecast ({}/{})'.format(
            self.t, self.price_, self.forecast_))

        # If I do have konkorde indicators, I also read them.
        if self.params.have_konkorde:
            self.green_ = self.data_.iloc[
                self.t, col_names.index(self.params.column_name['green'])]
            self.blue_ = self.data_.iloc[
                self.t, col_names.index(self.params.column_name['blue'])]
            self.konkorde_ = self.green_ + self.blue_
            self.log.debug('  konkorde ({}/{})'.format(
                self.green_, self.blue_))

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
        self.log.debug('t={}, state updated to: {} ({})'.format(
            self.t, self.current_state_, self.states.name(self.current_state_)))
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
        self.log.debug('STEP ITERATION - t={} -'.format(self.t))
        self.log.debug('t={}, price={}, action decided={} ({})'.format(
            self.t, self.price_, action, self.params.action_name[action]))

        # Compute reward by calling action and record experience.
        action_name = self.params.action_name[action]
        self.reward_ = getattr(self.portfolio, action_name)()
        self.memory.record_reward(self.reward_,
                                  self.current_state_,
                                  self.states.name(self.current_state_))
        self.log.debug(
            't={}, reward ({:.2f}), recorded to action {} in state {}'.format(
                self.t, self.reward_, action, self.current_state_))

        # Increase `time pointer`
        self.t += 1
        if self.t >= self.max_states_:
            self.log.debug('End of step (t({}) >= max_states({}))'.format(
                self.t, self.max_states_))
            self.done_ = True
            return self.new_state_, self.reward_, self.done_, self.t

        self.log.debug('Updating environment, after step.')
        self.update_mkt_price()
        self.portfolio.update(self.price_, self.forecast_, self.konkorde_)
        self.new_state_ = self.update_state()
        self.memory.record_values(self.portfolio, self.t, self.ts_)

        return self.new_state_, self.reward_, self.done_, self.t

    def save_portfolio(self, init=False):
        """
        Save all relevant information about the environment to later
        recover the state during a simulation. This implies saving Portfolio,
        and Memory.
        :return: None
        """
        portfolio_state = dict()
        for local_state_var in self.portfolio.state_variables:
            portfolio_state[local_state_var] = self.portfolio.__dict__[
                local_state_var]
        memory_state = self.memory.results.to_dict(orient='index')

        # Combine the two dicts
        merged_dict = {'portfolio': portfolio_state,
                       'memory': memory_state}

        # Get a valid filename and save the JSON into it, if it is being
        # initialized, otherwise it will be replaced.
        if init is True:
            json_filename = valid_output_name(self.params.portfolio_name,
                                              path=self.params.models_dir,
                                              extension='json')
        else:
            json_filename = self.params.portfolio_name

        with open(json_filename, "w") as outfile:
            json.dump(merged_dict, outfile)
        self.log.info('Saved portfolio to: {}'.format(json_filename))

    def resume(self) -> int:
        """
        Retrieve the internal state of portfolio and memory from a JSON file,
        an perform all actions that would have been performed if the last
        step saved would not have been the last.
        :return: The latest known state retrieved from memory.
        """
        with open(self.params.portfolio_name) as data_file:
            merged_dict = json.load(data_file)

        # Reconstruct internal portfolio variables
        for portfolio_local in merged_dict['portfolio'].keys():
            self.portfolio.__dict__[portfolio_local] = merged_dict['portfolio'][
                portfolio_local]
        self.memory.results = pd.DataFrame.from_dict(merged_dict['memory'],
                                                     orient='index')
        self.log.info('Retrieved portfolio and memory from: {}'.format(
            self.params.portfolio_name))

        # Set the time pointer to the last event in internal memory retrieved
        self.t = self.memory.len

        # If portfolio matches data, then we cannot run. Data must always
        # be ahead of portfolio.
        if self.t >= self.data_.shape[0]:
            return -1

        self.update_mkt_price()
        self.portfolio.latest_price = self.price_
        self.portfolio.forecast = self.forecast_
        self.portfolio.memory = self.memory
        self.portfolio.update(self.price_, self.forecast_, self.konkorde_)
        self.update_state()
        self.memory.record_values(self.portfolio, self.t, self.ts_)

        # latest state is the previous to the last int the table.
        return self.current_state_
