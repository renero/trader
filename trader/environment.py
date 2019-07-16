import pandas as pd

from portfolio import Portfolio
from rlstates import RLStates


class Environment(object):
    num_states_ = 0
    max_states_ = 0
    num_actions_ = 0
    data_ = None
    current_state_ = 0
    t = 0
    portfolio_ = None
    price_ = 0.
    forecast_ = 0.
    max_actions_ = 0
    done_ = False
    reward_ = 0.
    new_state_: int = 0
    debug = False

    def __init__(self, context_dictionary):
        self.__dict__.update(context_dictionary)
        self.initialize()

    def initialize(self,
                   num_actions=3,
                   debug=False):
        self.debug = debug
        self.num_actions_ = num_actions
        self.read_data(self._data_path)
        self.set_price()
        self.portfolio_ = Portfolio(self._environment._initial_budget,
                                    self.price_,
                                    self.forecast_,
                                    debug)
        self.states = RLStates(self._states_list)
        self.num_states_ = self.states.max_id
        self.set_state()
        return self

    def log(self, *args, **kwargs):
        if self.debug is True:
            print(*args, **kwargs)

    def reset(self, debug=False):
        self.debug = debug
        del self.portfolio_
        self.done_ = False
        self.t = 0
        self.set_price()
        self.portfolio_ = Portfolio(self._environment._initial_budget,
                                    self.price_,
                                    self.forecast_,
                                    self.debug)
        return self.set_state()

    def set_state(self):
        # state of my portfolio value
        if self.portfolio_.budget == self.portfolio_.initial_budget:
            value = 'EVEN'
        elif self.portfolio_.budget > self.portfolio_.initial_budget:
            value = 'WIN'
        else:
            value = 'LOSE'
        # guess what the state, given the forecast
        if self.portfolio_.forecast == self.portfolio_.latest_price:
            forecast = 'EVEN'
        elif self.portfolio_.forecast > self.portfolio_.latest_price:
            forecast = 'WIN'
        else:
            forecast = 'LOSE'
        # Do I have shares in my portfolio?
        if self.portfolio_.shares > 0.:
            shares_state = 'HAVE'
        else:
            shares_state = 'DONTHAVE'

        self.current_state_ = self.states.get_id(value, forecast, shares_state)
        return self.current_state_

    def read_data(self, path):
        self.data_ = pd.read_csv(path)
        self.max_states_ = self.data_.shape[0]

    def set_price(self):
        """ Set the price to the current time slot, reading column 0 from DF """
        assert self.data_ is not None, 'Price series data has not been read yet'
        self.price_ = self.data_.iloc[self.t, 0]
        self.forecast_ = self.data_.iloc[self.t, 1]

    def step(self, action):
        assert action < self.num_actions_, \
            'Action ID must be between 0 and {}'.format(
                self.num_actions_)

        if action == self._action_name.do_nothing:
            self.portfolio_.do_nothing()
            self.reward_ = 0.
        if action == self._action_name.buy:
            self.reward_ = self.portfolio_.buy()
        if action == self._action_name.sell:
            self.reward_ = self.portfolio_.sell()
        self.log(' | R: {:>+5.1f} | {:s}'.format(
            self.reward_, self.states.name(self.current_state_)))

        self.t += 1
        if self.t >= self.max_states_:
            self.done_ = True
            self.portfolio_.report(self.t - 1, disp_footer=True)
            self.log("")
            return self.new_state_, self.reward_, self.done_, self.t

        self.set_price()
        self.portfolio_.update(self.price_, self.forecast_)
        self.new_state_ = self.set_state()
        self.portfolio_.report(self.t)

        return self.new_state_, self.reward_, self.done_, self.t

    @staticmethod
    def decide(state, strategy):
        return strategy[state]
