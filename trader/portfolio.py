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
    failed_actions = ['f.buy', 'f.sell']
    # These are the variables that MUST be saved by the `dump()` method
    # in `environment`, in order to be able to resume the state.
    state_variables = ['initial_budget', 'budget', 'latest_price',
                       'forecast', 'investment',
                       'portfolio_value', 'net_value',
                       'shares', 'konkorde']

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

        self.budget = self.environment.initial_budget
        self.initial_budget = self.environment.initial_budget
        self.latest_price = initial_price
        self.forecast = forecast
        self.memory = env_memory
        self.log.info('Portfolio created with initial budget: {:.1f}'.format(
            self.initial_budget))

    def reset(self, initial_price, forecast, env_memory):
        self.initial_budget = self.environment.initial_budget
        self.budget = self.environment.initial_budget
        self.latest_price = initial_price
        self.forecast = forecast
        self.memory = env_memory
        self.investment = 0
        self.portfolio_value: float = 0.
        self.net_value: float = 0
        self.shares: float = 0.
        self.konkorde = 0.
        self.reward = 0.

    def update(self, price, forecast, konkorde=None):
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
        self.log.debug('Updating portfolio after STEP.')
        self.log.debug(
            '  > portfolio_value={}, latest_price={}, forecast={}'.format(
                self.portfolio_value, self.latest_price, self.forecast))
        return self

    def wait(self):
        action_name = 'wait'
        self.reward = self.decide_reward(action_name, num_shares=0)
        self.memory.record_action(action_name)
        self.log.debug('  {} action recorded. Reward={:.2f}'.format(
            action_name, self.reward))
        return self.reward

    def buy(self, num_shares: float = 1.0) -> object:
        buy_price = num_shares * self.latest_price
        if buy_price > self.budget:
            action_name = 'f.buy'
            self.memory.record_action(action_name)
            self.reward = self.decide_reward(action_name, num_shares)
            self.log.debug('  {} action recorded. Reward={:.2f}'.format(
                action_name, self.reward))
            return self.reward

        action_name = 'buy'
        self.update_after_buy(num_shares, buy_price)
        self.reward = self.decide_reward(action_name, num_shares)
        self.memory.record_action(action_name)
        self.log.debug('  {} action recorded. Reward={:.2f}'.format(
            action_name, self.reward))
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
            self.log.debug('  {} action recorded. Reward={:.2f}'.format(
                action_name, self.reward))
            return self.reward

        action_name = 'sell'
        self.update_after_sell(num_shares, sell_price)
        self.reward = self.decide_reward(action_name, num_shares)
        self.memory.record_action(action_name)
        self.log.debug('  {} action recorded. Reward={:.2f}'.format(
            action_name, self.reward))
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

        # what is the value of my investment after selling?
        self.net_value = self.portfolio_value - self.investment

    def decide_reward(self, action_name, num_shares):
        """ Decide what is the reward for this action """
        if self.params.environment.direct_reward is True:
            return self.direct_reward(action_name, num_shares)
        else:
            return self.preset_reward(action_name, num_shares)

    def direct_reward(self, action_name, num_shares):
        """ Direct reward is directly related to portfolio value """

        def sigmoid(x: float):
            return x / math.sqrt(1. + math.pow(x, 2.))

        if action_name == 'buy':
            self.log.debug('  direct reward: buy = 0.0')
            return 0.0
        else:
            if action_name == 'wait' and self.shares == 0.:
                self.log.debug('  direct reward: wait & shares>0 => -.05')
                return -0.05
            net_value = self.portfolio_value - self.investment
            # Check if this is a failed situation 'f.buy' or 'f.sell',
            # to reverse the reward sign to negative.
            if action_name in self.failed_actions:
                net_value = -1. * abs(net_value)
                # There is a corner case when it is a failed action but there
                # are no shares, and value is 0. In that case, we must punish.
                if net_value == 0.0:
                    net_value = -1.0
            self.log.debug(
                '  direct reward: net value s({:.2f})={:.2f}'.format(
                    net_value, sigmoid(net_value)
                ))
            return sigmoid(net_value)

    def preset_reward(self, action_name, num_shares):
        """
        Reward, is preset in values stored in params.
        """
        self.log.debug('Preset reward mode')
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
        self.log.debug(
            '  Pred sign ({}) latest price({}) vs. last_forecast({})'.format(
               '↑' if self.latest_price <= self.forecast else '↓',
                self.latest_price, self.forecast))
        return self.latest_price <= self.forecast

    @property
    def last_forecast(self):
        self.log.debug('    Last forecast in MEM = {}'.format(
            self.memory.last('forecast')))
        return self.memory.last('forecast')

    @property
    def last_price(self):
        self.log.debug(
            '    Last price in MEM = {}'.format(self.memory.last('price')))
        return self.memory.last('price')

    @property
    def prevlast_forecast(self):
        self.log.debug(
            '    PrevLast forecast in MEM = {}'.format(
                self.memory.prevlast('forecast')))
        return self.memory.prevlast('forecast')

    @property
    def prevlast_price(self):
        self.log.debug(
            '    PrevLast price in MEM = {}'.format(
                self.memory.prevlast('price')))
        return self.memory.prevlast('price')

    @property
    def prevprevlast_forecast(self):
        self.log.debug(
            '    PrevPrevLast forecast in MEM = {}'.format(
                self.memory.prevprevlast('forecast')))
        return self.memory.prevprevlast('forecast')

    @property
    def prevprevlast_price(self):
        self.log.debug(
            '    PrevPrevLast price in MEM = {}'.format(
                self.memory.prevprevlast('price')))
        return self.memory.prevprevlast('price')
