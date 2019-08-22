import numpy as np

from common import Common

h1 = ' {:<3s} |{:>8s} |{:>9s} |{:>9s} |{:>9s} |{:>9s} |{:>9s} |{:>7s} '
h2 = '| {:<7}| {:<7s}| {:20s}'
h = h1 + h2

s1 = ' {:>03d} |'
s2 = '{:>8.1f} |{:>18} |{:>9.1f} |{:>18} |{:>9.1f} |{:>18} |{:>7.1f} '
s = s1 + s2

f = '                           {:>9.1f} |{:>18} |{:>9.1f} |{:>18} |{:>7.1f}'
act_h = '| {:<15}'


class Display(Common):

    def __init__(self, configuration):
        self.configuration = configuration

    @staticmethod
    def strategy(trader, env, model, num_states, strategy):
        print('\nStrategy learned')
        strategy_string = "State {{:<{}s}} -> {{:<10s}} {{}}".format(
            env.states.max_len)
        for i in range(num_states):
            print(strategy_string.format(
                env.states.name(i),
                trader.configuration._action_name[strategy[i]],
                model.predict(np.identity(num_states)[i:i + 1])))
        print()

    def report(self, portfolio, t, disp_header=False, disp_footer=False):
        header = h.format('t', 'price', 'forecast', 'budget', '€.flow',
                          'value', 'net.Val', 'shares',
                          'action', 'reward', 'state')
        if disp_header is True:
            self.log()
            self.log(header)
            self.log('{}'.format('-' * (len(header) + 8), sep=''))

        if disp_footer is True:
            footer = f.format(
                portfolio.budget,
                self.color(portfolio.investment * -1.),
                portfolio.portfolio_value,
                self.color(
                    portfolio.portfolio_value - portfolio.investment),
                portfolio.shares)
            self.log('{}'.format('-' * (len(header) + 8), sep=''))
            self.log(footer)

            # total outcome
            if portfolio.portfolio_value != 0.0:
                total = portfolio.budget + portfolio.portfolio_value
            else:
                total = portfolio.budget
            percentage = 100. * ((total / portfolio.initial_budget) - 1.0)
            self.log('Final: €{:.2f} [{} %]'.format(
                total, self.color(percentage)))
            return

        self.log(s.format(
            t,
            portfolio.latest_price,
            self.cond_color(portfolio.forecast, portfolio.latest_price),
            portfolio.budget,
            self.color(portfolio.investment * -1.),
            portfolio.portfolio_value,
            self.color(portfolio.portfolio_value - portfolio.investment),
            portfolio.shares), end='')

    def report_action(self, action_name):
        if action_name == 'sell':
            self.log(act_h.format(self.green('sell')), end='')
        elif action_name == 'buy':
            self.log(act_h.format(self.red('buy')), end='')
        else:
            self.log(act_h.format(self.white('none')), end='')

    def report_reward(self, reward, current_state):
        self.log(' | {:>15} | {:s}'.format(
            self.color(reward), current_state))

    def progress(self, i, last_avg, start, end):
        percentage = (i / self.configuration._num_episodes) * 100.0
        print(
            "Episode {:>5}/{:<5} [{:>5.1f}%] Avg reward: {:+.3f}".format(
                i,
                self.configuration._num_episodes,
                percentage,
                last_avg), end='')
        if percentage == 0.0:
            print(' Est.time: UNKNOWN')
            return
        elapsed = end - start
        remaining = ((100. - percentage) * elapsed) / percentage
        print(' Est.time: {}'.format(self.timer(remaining)))

    @staticmethod
    def timer(elapsed):
        """
        Returns a string with a time lapse duration passed in seconds as
        a combination of hours, minutes and seconds.
        :param elapsed: the period of time to express in hh:mm:ss
        :return: a string
        """
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        if int(seconds) == 0:
            return 'UNKNOWN'
        else:
            return '{:0>2}:{:0>2}:{:0>2}'.format(
                int(hours), int(minutes), int(seconds))
