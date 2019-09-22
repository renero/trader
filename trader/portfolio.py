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

    def __init__(self,
                 configuration,
                 initial_price=0.,
                 forecast=0.):

        # copy the contents of the dictionary passed as argument. This dict
        # contains the parameters read in the initialization.
        self.configuration = configuration
        self.display = self.configuration.display

        self.budget = self.configuration._environment._initial_budget
        self.initial_budget = self.configuration._environment._initial_budget
        self.latest_price = initial_price
        self.forecast = forecast

    def wait(self):
        self.display.report_action('none')
        self.reward = self.configuration._environment._reward_do_nothing
        return self.reward

    def buy(self, num_shares: float = 1.0) -> object:
        purchase_amount = num_shares * self.latest_price
        if purchase_amount > self.budget:
            self.display.report_action('n/a')
            self.reward = self.configuration._environment._reward_failed_buy
            return self.reward

        self.budget -= purchase_amount
        self.investment += purchase_amount
        self.shares += num_shares
        self.portfolio_value += purchase_amount
        self.movements.append((self.BUY, num_shares, self.latest_price))
        self.reward = self.configuration._environment._reward_success_buy

        self.display.report_action('buy')
        return self.reward

    def sell(self, num_shares=1.0):
        sell_price = num_shares * self.latest_price
        if num_shares > self.shares:
            self.display.report_action('n/a')
            self.reward = self.configuration._environment._reward_failed_sell
            return self.reward

        self.budget += sell_price
        self.investment -= sell_price
        self.shares -= num_shares
        self.portfolio_value -= sell_price
        self.movements.append((self.SELL, num_shares, self.latest_price))
        if self.budget > self.initial_budget:
            self.reward = self.configuration._environment._reward_positive_sell
        else:
            self.reward = self.configuration._environment._reward_negative_sell

        self.display.report_action('sell')
        return self.reward

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
        if len(self.history) > self.configuration._stack_size:
            self.history.pop(0)

    @property
    def last_forecast(self):
        return self.history[-1]['forecast_']

    @property
    def last_price(self):
        return self.history[-1]['price_']

    @property
    def prevlast_forecast(self):
        return self.history[-2]['forecast_']

    @property
    def prevlast_price(self):
        return self.history[-2]['price_']

    def values_to_report(self):
        return [
            self.latest_price,
            self.forecast,
            self.budget,
            self.investment,
            self.portfolio_value,
            self.portfolio_value - self.investment,
            self.shares]
