import math

from common import Common
from utils.dictionary import Dictionary


class Portfolio:
    configuration: Dictionary
    env_scaler = None
    memory = None
    initial_budget = 0.
    budget = 0.
    cost = 0
    portfolio_value: float = 0.
    profit: float = 0
    num_shares: float = 0.
    latest_price: float = 0.
    forecast: float = 0.
    konkorde = 0.
    reward = 0.
    failed_actions = ['f.buy', 'f.sell']

    # These are the variables that MUST be saved by the `dump()` method
    # in `environment`, in order to be able to resume the state.
    state_variables = ['initial_budget', 'budget', 'latest_price', 'forecast',
                       'cost', 'portfolio_value', 'profit', 'shares',
                       'konkorde']

    def __init__(self,
                 configuration,
                 initial_price,
                 forecast,
                 env_memory,
                 env_scaler):

        self.params = configuration
        self.display = self.params.display
        self.log = self.params.log
        self.env_params = self.params.environment
        self.env_scaler = env_scaler
        self.reset(initial_price, forecast, env_memory, env_scaler)

    def reset(self, initial_price, forecast, env_memory, env_scaler):
        """
        Initializes portfolio to the initial state.
        """
        self.initial_budget = self.scale_budget(self.env_params.initial_budget)
        self.budget = self.initial_budget
        self.latest_price = initial_price
        self.forecast = forecast
        self.memory = env_memory
        self.cost = 0
        self.portfolio_value: float = 0.
        self.profit: float = 0
        self.num_shares: float = 0.
        self.konkorde = 0.
        self.reward = 0.
        self.log.debug('Portfolio reset. Initial budget: {:.1f}'.format(
            self.initial_budget))

    def scale_budget(self, budget):
        mn = self.params.fcast_file.min_support
        ptp = self.params.fcast_file.max_support - mn
        if ptp == 0.0:
            ptp = 0.000001
        return (budget - mn) / ptp

    def wait(self):
        """
        WAIT Operation
        """
        action_name = 'wait'
        self.reward = self.decide_reward(action_name, num_shares=0)
        msg = '  WAIT: ' + \
              'prc({:.2f})|bdg({:.2f})|val({:.2f})|prf({:.2f})|inv({:.2f})'
        self.log.debug(msg.format(
                self.latest_price, self.budget, self.portfolio_value,
                self.profit, self.cost))

        return action_name, self.reward

    def buy(self, num_shares: float = 1.0) -> object:
        """
        BUY Operation
        :param num_shares: number of shares. Default to 1.
        :return: The action executed and the reward obtained by the operation.
        """
        action_name = 'buy'
        buy_price = num_shares * self.latest_price
        if buy_price > self.budget:
            action_name = 'f.buy'
            self.log.debug('  FAILED buy')
        self.reward = self.decide_reward(action_name, num_shares)
        self.update_after_buy(action_name, num_shares, buy_price)

        return action_name, self.reward

    def update_after_buy(self, action_name, num_shares, buy_price):
        """
        Update portfolio parameters after buying shares

        :param action_name: The name of the action taken
        :param num_shares: the nr. shares bought
        :param buy_price: the price at which the purchase takes place
        :return:
        """
        if action_name in self.failed_actions:
            return
        self.budget -= buy_price
        self.cost += buy_price
        self.num_shares += num_shares
        self.portfolio_value += buy_price

        # what is the value of my investment after selling?
        self.profit = self.compute_portfolio_value()
        msg = '  BUY: ' + \
              'prc({:.2f})|bdg({:.2f})|val({:.2f})|prf({:.2f})|inv({:.2f})'
        self.log.debug(msg.format(
                self.latest_price, self.budget, self.portfolio_value,
                self.profit, self.cost))

    def sell(self, num_shares=1.0):
        """
        SELL Operation
        :param num_shares: number of shares. Default to 1.
        :return: The action executed and the reward obtained by the operation.
        """
        action_name = 'sell'
        sell_price = num_shares * self.latest_price
        if num_shares > self.num_shares:
            action_name = 'f.sell'
            self.log.debug('  FAILED sell')
        self.reward = self.decide_reward(action_name, num_shares)
        self.update_after_sell(action_name, num_shares, sell_price)

        return action_name, self.reward

    def update_after_sell(self, action_name, num_shares, sell_price):
        """
        Update specific portfolio parameters after selling
        :param action_name: the name of the action taken
        :param num_shares: the nr. of shares sold
        :param sell_price: the price at which the selling takes place
        :return:
        """
        if action_name in self.failed_actions:
            return
        if self.params.mode == 'bull':
            self.budget += sell_price
        else:
            self.budget += self.memory.last('cost')
            self.budget += self.compute_portfolio_value()
        self.cost -= sell_price
        self.num_shares -= num_shares
        self.portfolio_value -= sell_price

        # what is the value of my investment after selling?
        self.profit = self.compute_portfolio_value()
        msg = '  SELL: ' + \
              'prc({:.2f})|bdg({:.2f})|val({:.2f})|prf({:.2f})|inv({:.2f})'
        self.log.debug(msg.format(
                self.latest_price, self.budget, self.portfolio_value,
                self.profit, self.cost))

    def decide_reward(self, action_name, num_shares):
        """ Decide what is the reward for this action """
        if self.params.environment.direct_reward is True:
            return self.direct_reward(action_name, num_shares)
        else:
            return self.preset_reward(action_name, num_shares)

    def direct_reward(self, action_name, num_shares):
        """ Direct reward is directly related to portfolio value """

        # Check if this is a failed situation 'f.buy' or 'f.sell',
        # to reverse the reward sign to negative.
        if action_name in self.failed_actions:
            return -1.0

        def sigmoid(x: float):
            return x / math.sqrt(1. + math.pow(x, 2.))

        if action_name == 'buy':
            self.log.debug('  direct reward: buy = 0.0')
            return 0.0
        else:
            if action_name == 'wait' and self.num_shares == 0.:
                self.log.debug('  direct reward: wait & shares=0 => -.05')
                return -0.05
            net_value = self.compute_portfolio_value()
            self.log.debug(
                '  direct reward: net value s({:.2f})={:.2f}'.format(
                    net_value, sigmoid(net_value)))
            return sigmoid(net_value)

    def preset_reward(self, action_name, num_shares):
        """
        Reward, is preset in values stored in params.
        """
        self.log.debug('Preset reward mode')
        reward = 0.
        if action_name == 'wait':
            reward = self.env_params.reward_do_nothing
        elif action_name == 'buy':
            reward = self.env_params.reward_success_buy
        elif action_name == 'sell':
            gain_loss = 1.0
            if self.env_params.proportional_reward is True:
                gain_loss = abs(self.profit) + 1.0
            if self.profit >= 0:
                reward = self.env_params.reward_positive_sell * gain_loss
            else:
                reward = self.env_params.reward_negative_sell * gain_loss
        elif action_name == 'f.buy':
            reward = self.env_params.reward_failed_buy
        elif action_name == 'f.sell':
            reward = self.env_params.reward_failed_sell
        return reward

    def update(self, price, forecast, konkorde=None):
        """
        Updates portfolio after an iteration step.

        :param price:    new price registered
        :param forecast: new forecast registered
        :param konkorde: the konkorde value (computed from green & blue read
                         the data file, if applicable)
        :return:         the portfolio object
        """
        self.portfolio_value = self.num_shares * price
        self.latest_price = price
        self.forecast = forecast
        if konkorde is not None:
            self.konkorde = konkorde
        self.log.debug('  Updating portfolio after STEP.')
        msg = '  > portfolio_value={:.2f}, latest_price={:.2f}, forecast={:.2f}'
        self.log.debug(msg.format(
            self.portfolio_value, self.latest_price, self.forecast))
        return self

    def values_to_record(self):
        profit = self.compute_portfolio_value() * self.num_shares
        values = [
            self.latest_price,
            self.forecast,
            self.budget,
            self.cost,
            self.portfolio_value,
            profit,
            self.num_shares
        ]
        if self.params.have_konkorde:
            values += [self.konkorde]
        values += ['', 0., 0., '']
        return values

    def compute_portfolio_value(self):
        """
        Compute the value of the portfolio depending on whether it is bear
        or bull, as the difference between the value of the shares acquired
        at the moment, and the investment made to purchase them.
        """
        if self.params.mode == 'bear':
            return self.cost - self.portfolio_value
        else:
            return self.portfolio_value - self.cost

    def failed_action(self, action, price):
        """
        Determines if the action can be done or will be a failed operation
        """
        if action == self.params.action.index('buy'):
            if price > self.budget:
                return True
            else:
                return False
        elif action == self.params.action.index('sell'):
            if self.num_shares == 0.:
                return True
            else:
                return False
        else:
            return False

    @property
    def gain(self):
        return (self.portfolio_value - self.cost) >= 0

    @property
    def have_shares(self):
        return self.num_shares > 0

    @property
    def can_buy(self) -> bool:
        return self.budget >= self.latest_price

    @property
    def can_sell(self) -> bool:
        return self.num_shares > 0.

    @property
    def prediction_upward(self):
        msg = '  Pred sign ({}) latest price({:.2f}) vs. last_forecast({:.2f})'
        self.log.debug(
            msg.format(
                '↑' if self.latest_price <= self.forecast else '↓',
                self.latest_price, self.forecast))
        return self.latest_price <= self.forecast

    @property
    def last_forecast(self):
        self.log.debug('    Last forecast in MEM = {:.2f}'.format(
            self.memory.last('forecast')))
        return self.memory.last('forecast')

    @property
    def last_price(self):
        # self.log.debug(
        #     '    Last price in MEM = {:.2f}'.format(self.memory.last('price')))
        return self.memory.last('price')

    @property
    def prevlast_forecast(self):
        # self.log.debug(
        #     '    PrevLast forecast in MEM = {:.2f}'.format(
        #         self.memory.prevlast('forecast')))
        return self.memory.prevlast('forecast')

    @property
    def prevlast_price(self):
        # self.log.debug(
        #     '    PrevLast price in MEM = {:.2f}'.format(
        #         self.memory.prevlast('price')))
        return self.memory.prevlast('price')

    @property
    def prevprevlast_forecast(self):
        # self.log.debug(
        #     '    PrevPrevLast forecast in MEM = {:.2f}'.format(
        #         self.memory.prevprevlast('forecast')))
        return self.memory.prevprevlast('forecast')

    @property
    def prevprevlast_price(self):
        # self.log.debug(
        #     '    PrevPrevLast price in MEM = {:.2f}'.format(
        #         self.memory.prevprevlast('price')))
        return self.memory.prevprevlast('price')
