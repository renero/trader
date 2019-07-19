BUY = +1
SELL = -1

h1 = ' {:<3s} |{:>8s} |{:>9s} |{:>9s} |{:>9s} |{:>8s} |{:>7s} '
h2 = '| {:<7s} | {:<9s}| {:20s}'
h = h1 + h2
s = ' {:>03d} |{:>8.1f} |{:>9.1f} |{:>9.1f} |{:>+9.1f} |{:>8.1f} |{:>7.1f} '
f = '                           {:>9.1f} |{:>9.1f} |{:>8.1f} |{:>7.1f}'


class Portfolio:
    initial_budget = 0.
    budget = 0.
    investment = 0
    portfolio_value: float = 0.
    shares: float = 0.
    latest_price: float = 0.
    forecast: float = 0.
    reward = 0.
    movements = []
    debug = False

    def __init__(self,
                 context_dictionary,
                 initial_price=0.,
                 forecast=0.,
                 debug=False):
        self.__dict__.update(context_dictionary)
        self.debug = debug
        self.budget = self._environment._initial_budget
        self.initial_budget = self._environment._initial_budget
        self.latest_price = initial_price
        self.forecast = forecast
        self.report(t=0, disp_header=True)
        return

    def log(self, *args, **kwargs):
        if self.debug is True:
            print(*args, **kwargs)

    def do_nothing(self):
        self.log('| {:<7s}'.format('none'), end='')

        self.reward = self._environment._reward_do_nothing
        return self.reward

    def buy(self, num_shares: float = 1.0) -> object:
        purchase_amount = num_shares * self.latest_price
        if purchase_amount > self.budget:
            self.log('| {:<7s}'.format('n/a'), end='')
            self.reward = self._environment._reward_failed_buy
            return self.reward

        self.budget -= purchase_amount
        self.investment += purchase_amount
        self.shares += num_shares
        self.portfolio_value += purchase_amount
        self.movements.append((BUY, num_shares, self.latest_price))

        self.log('| +{:6.1f}'.format(self.latest_price), end='')

        self.reward = self._environment._reward_success_buy

        return self.reward

    def sell(self, num_shares=1.0):
        sell_price = num_shares * self.latest_price
        if num_shares > self.shares:
            self.log('| {:<7s}'.format('n/a'), end='')
            self.reward = self._environment._reward_failed_sell
            return self.reward

        self.budget += sell_price
        self.investment -= sell_price
        self.shares -= num_shares
        self.portfolio_value -= sell_price
        self.movements.append((SELL, num_shares, self.latest_price))

        self.log('| -{:6.1f}'.format(self.latest_price), end='')

        if self.budget > self.initial_budget:
            self.reward = self._environment._reward_positive_sell
        else:
            self.reward = self._environment._reward_negative_sell

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
            return

        self.log(s.format(
            t,
            self.latest_price,
            self.forecast,
            self.budget,
            self.investment * -1.,
            self.portfolio_value,
            self.shares), end='')
