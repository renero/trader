import math

from common import Common
from utils.dictionary import Dictionary


class Portfolio(Common):
    configuration: Dictionary
    initial_budget = 0.
    budget = 0.
    investment = 0
    portfolio_value: float = 0.
    net_value: float = 0
    shares: float = 0.
    latest_price: float = 0.
    forecast: float = 0.
    konkorde = 0.
    reward = 0.
    movements = []
    history = []

    # Constants
    BUY = +1
    SELL = -1

    def __init__(self,
                 configuration,
                 initial_price,
                 forecast,
                 env_memory):
        # copy the contents of the dictionary passed as argument. This dict
        # contains the parameters read in the initialization.
        self.params = configuration
        self.display = self.params.display
        self.log = self.params.log
        self.environment = self.params.environment
        self.memory = env_memory

        self.budget = self.environment.initial_budget
        self.initial_budget = self.environment.initial_budget
        self.latest_price = initial_price
        self.forecast = forecast

    def update_after_step(self, price, forecast, konkorde=None):
        """
        Updates portfolio after an interation step.
        :param price: new price registered
        :param forecast: new forecast registered
        :param konkorde: the konkorde value (computed from green & blue read
            the data file, if applicable)
        :return: the portfolio object
        """
        self.portfolio_value = self.shares * price
        self.latest_price = price
        self.forecast = forecast
        if konkorde is not None:
            self.konkorde = konkorde
        return self

    def wait(self):
        action_name = 'wait'
        self.memory.record_action(action_name)
        self.reward = self.decide_reward(action_name, num_shares=0)
        return self.reward

    def buy(self, num_shares: float = 1.0) -> object:
        buy_price = num_shares * self.latest_price
        if buy_price > self.budget:
            action_name = 'f.buy'
            self.memory.record_action(action_name)
            self.reward = self.decide_reward(action_name, num_shares)
            return self.reward  # self.reward

        action_name = 'buy'
        self.update_after_buy(num_shares, buy_price)
        self.reward = self.decide_reward(action_name, num_shares)

        self.memory.record_action(action_name)
        return self.reward

    def update_after_buy(self, num_shares, buy_price):
        """
        Update portfolio parameters after buying shares
        :param num_shares: the nr. shares bought
        :param buy_price: the price at which the purchase takes place
        :return:
        """
        self.budget -= buy_price
        self.investment += buy_price
        self.shares += num_shares
        self.portfolio_value += buy_price
        self.movements.append((self.BUY, num_shares, self.latest_price))
        # what is the value of my investment after selling?
        self.net_value = self.portfolio_value - self.investment

    def sell(self, num_shares=1.0):
        """
        SELL Operation
        :param num_shares: number of shares. Deault to 1.
        :return: The reward obtained by the operation.
        """
        sell_price = num_shares * self.latest_price
        if num_shares > self.shares:
            action_name = 'f.sell'
            self.reward = self.decide_reward(action_name, num_shares)
            self.memory.record_action(action_name)
            return self.reward

        action_name = 'sell'
        self.update_after_sell(num_shares, sell_price)
        self.reward = self.decide_reward(action_name, num_shares)
        self.memory.record_action(action_name)
        return self.reward

    def update_after_sell(self, num_shares, sell_price):
        """
        Update specific portfolio parameters after selling
        :param num_shares: the nr. of shares sold
        :param sell_price: the price at which the selling takes place
        :return:
        """
        self.budget += sell_price
        self.investment -= sell_price
        self.shares -= num_shares
        self.portfolio_value -= sell_price
        self.movements.append((self.SELL, num_shares, self.latest_price))

        # what is the value of my investment after selling?
        self.net_value = self.portfolio_value - self.investment

    def decide_reward(self, action_name, num_shares):
        def sigmoid(x: float):
            return 1. / math.sqrt(1. + math.pow(x, 2.))

        # Are we working with direct reward?
        if self.params.environment.direct_reward is True:
            if action_name == 'buy':
                return 0.0
            else:
                if action_name == 'wait' and self.shares == 0.:
                    return -0.05
                return sigmoid(self.portfolio_value - self.investment)

        # From this point, we're in preset reward
        # Reward, in case of sell, can be proportional to gain/loss, if not
        # set that multiplier to 1.0
        reward = 0.
        if action_name == 'wait':
            reward = self.environment.reward_do_nothing
        elif action_name == 'buy':
            reward = self.environment.reward_success_buy
        elif action_name == 'sell':
            gain_loss = 1.0
            if self.environment.proportional_reward is True:
                gain_loss = abs(self.net_value) + 1.0
            if self.net_value >= 0:
                reward = self.environment.reward_positive_sell * gain_loss
            else:
                reward = self.environment.reward_negative_sell * gain_loss
        elif action_name == 'f.buy':
            reward = self.environment.reward_failed_buy
        elif action_name == 'f.sell':
            reward = self.environment.reward_failed_sell
        return reward

    def reset_history(self):
        del self.history[:]

    def append_to_history(self, environment):
        # Stack the current state into the history
        self.history.append({'price_': environment.price_,
                             'forecast_': environment.forecast_})
        if len(self.history) > self.params.stack_size:
            self.history.pop(0)

    def values_to_record(self):
        net_value = self.portfolio_value - self.investment
        values = [
                self.latest_price,
                self.forecast,
                self.budget,
                self.investment,
                self.portfolio_value,
                net_value,
                self.shares
        ]
        if self.params.have_konkorde:
            return values + [self.konkorde]
        return values

    @property
    def gain(self):
        return (self.portfolio_value - self.investment) >= 0

    @property
    def have_shares(self):
        return self.shares > 0

    @property
    def can_buy(self) -> bool:
        return self.budget >= self.latest_price

    @property
    def can_sell(self) -> bool:
        return self.shares > 0.

    @property
    def prediction_upward(self):
        return self.latest_price <= self.last_forecast

    @property
    def last_forecast(self):
        if len(self.history) > 0:
            return self.history[-1]['forecast_']
        return 0.

    @property
    def prevlast_forecast(self):
        return self.history[-2]['forecast_']

    @property
    def last_price(self):
        return self.history[-1]['price_']

    @property
    def prevlast_price(self):
        return self.history[-2]['price_']

    @property
    def can_buy(self) -> bool:
        return self.budget >= self.latest_price
