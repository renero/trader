from common import Common
from dictionary import Dictionary

h1 = ' {:<3s} |{:>8s} |{:>9s} |{:>9s} |{:>9s} |{:>8s} |{:>7s} '
h2 = '| {:<7s} | {:<9s}| {:20s}'
h = h1 + h2
s = ' {:>03d} |{:>8.1f} |{:>9.1f} |{:>9.1f} |{:>+9.1f} |{:>8.1f} |{:>7.1f} '
f = '                           {:>9.1f} |{:>9.1f} |{:>8.1f} |{:>7.1f}'


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

        self.budget = self.configuration._environment._initial_budget
        self.initial_budget = self.configuration._environment._initial_budget
        self.latest_price = initial_price
        self.forecast = forecast
        self.report(t=0, disp_header=True)

    def do_nothing(self):
        self.log('| {:<7s}'.format('none'), end='')

        self.reward = self.configuration._environment._reward_do_nothing
        return self.reward

    def buy(self, num_shares: float = 1.0) -> object:
        purchase_amount = num_shares * self.latest_price
        if purchase_amount > self.budget:
            self.log('| {:<7s}'.format('n/a'), end='')
            self.reward = self.configuration._environment._reward_failed_buy
            return self.reward

        self.budget -= purchase_amount
        self.investment += purchase_amount
        self.shares += num_shares
        self.portfolio_value += purchase_amount
        self.movements.append((self.BUY, num_shares, self.latest_price))

        self.log('| +{:6.1f}'.format(self.latest_price), end='')

        self.reward = self.configuration._environment._reward_success_buy

        return self.reward

    def sell(self, num_shares=1.0):
        sell_price = num_shares * self.latest_price
        if num_shares > self.shares:
            self.log('| {:<7s}'.format('n/a'), end='')
            self.reward = self.configuration._environment._reward_failed_sell
            return self.reward

        self.budget += sell_price
        self.investment -= sell_price
        self.shares -= num_shares
        self.portfolio_value -= sell_price
        self.movements.append((self.SELL, num_shares, self.latest_price))

        self.log('| -{:6.1f}'.format(self.latest_price), end='')

        if self.budget > self.initial_budget:
            self.reward = self.configuration._environment._reward_positive_sell
        else:
            self.reward = self.configuration._environment._reward_negative_sell

        return self.reward

    def update(self, price, forecast):
        self.portfolio_value = self.shares * price
        self.latest_price = price
        self.forecast = forecast
        return self

    def report(self, t, disp_header=False, disp_footer=False):
        header = h.format('t', 'price', 'forecast', 'budget', 'net val.',
                          'value', 'shares', 'action', 'reward', 'state')
        if disp_header is True:
            self.log(header)
            self.log('{}'.format('-' * (len(header) + 8), sep=''))

        if disp_footer is True:
            footer = f.format(self.budget,
                              self.investment * -1.,
                              self.portfolio_value,
                              self.shares)
            self.log('{}'.format('-' * (len(header) + 8), sep=''))
            self.log(footer)

            # total outcome
            if self.portfolio_value != 0.0:
                total = self.budget + self.portfolio_value
            else:
                total = self.budget
            percentage = 100. * ((total / self.initial_budget) - 1.0)
            self.log('Final: €{:.2f} [{}%]'.format(
                total, self.color(percentage)))
            return

        self.log(s.format(
            t,
            self.latest_price,
            self.forecast,
            self.budget,
            self.investment * -1.,
            self.portfolio_value,
            self.shares), end='')

    def reset_history(self):
        # print('>> RESET HISTORY')
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
