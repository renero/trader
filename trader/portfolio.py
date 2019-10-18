import math

from common import Common
from utils.dictionary import Dictionary


class Portfolio(Common):
    configuration: Dictionary
    initial_budget = 0.
    budget = 0.
    investment = 0
    portfolio_value: float = 0.
    shares: float = 0.
    latest_price: float = 0.
    forecast: float = 0.
    reward = 0.
    movements = []
    history = []

    # Constants
    BUY = +1
    SELL = -1

    def __init__(self, configuration, initial_price=0., forecast=0.):

        # copy the contents of the dictionary passed as argument. This dict
        # contains the parameters read in the initialization.
        self.params = configuration
        self.display = self.params.display
        self.environment = self.params.environment

        self.budget = self.environment.initial_budget
        self.initial_budget = self.environment.initial_budget
        self.latest_price = initial_price
        self.forecast = forecast

    def wait(self):
        self.display.report_action('none')
        self.reward = self.portfolio_value - self.investment  # self.environment.reward_do_nothing
        return self.reward  # self.reward

    def buy(self, num_shares: float = 1.0) -> object:
        purchase_amount = num_shares * self.latest_price
        if purchase_amount > self.budget:
            self.display.report_action('f.buy')
            self.reward = 0.  # self.environment.reward_failed_buy
            return self.reward  # self.reward

        self.budget -= purchase_amount
        self.investment += purchase_amount
        self.shares += num_shares
        self.portfolio_value += purchase_amount
        self.movements.append((self.BUY, num_shares, self.latest_price))
        self.reward = 0.  # self.environment.reward_success_buy

        self.display.report_action('buy')
        return self.reward  # self.reward

    def sell(self, num_shares=1.0):
        """
        SELL Operation
        :param num_shares: number of shares. Deault to 1.
        :return: The reward obtained by the operation.
        """
        sell_price = num_shares * self.latest_price
        if num_shares > self.shares:
            self.display.report_action('f.sell')
            self.reward = self.portfolio_value - self.investment  # self.environment.reward_failed_sell
            return self.reward

        net_value_after = self.update_after_sell(num_shares, sell_price)
        self.reward = net_value_after

        # Reward, in case of sell, can be proportional to gain/loss, if not
        # set that multiplier to 1.0
        # gain_loss = 1.0
        # if self.environment.proportional_reward is True:
        #     gain_loss = abs(net_value_after) + 1.0
        #
        # if net_value_after >= 0:
        #     self.reward = self.environment.reward_positive_sell * gain_loss
        # else:
        #     self.reward = self.environment.reward_negative_sell * gain_loss

        self.display.report_action('sell')
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
        # net_value_after = self.portfolio_value - sell_price
        net_value_after = self.portfolio_value - self.investment
        return net_value_after

    def update(self, price, forecast):
        self.portfolio_value = self.shares * price
        self.latest_price = price
        self.forecast = forecast
        return self

    def reset_history(self):
        del self.history[:]

    def append_to_history(self, environment):
        # Stack the current state into the history
        self.history.append({'price_': environment.price_,
                             'forecast_': environment.forecast_})
        if len(self.history) > self.params.stack_size:
            self.history.pop(0)

    def values_to_report(self):
        net_value = self.portfolio_value - self.investment
        return [
            self.latest_price,
            self.forecast,
            self.budget,
            self.investment,
            self.portfolio_value,
            net_value,
            self.shares]

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
