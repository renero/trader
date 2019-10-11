import time
from math import log10, pow

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
from tabulate import tabulate

from common import Common


class Display(Common):

    def __init__(self, configuration):
        self.params = configuration

    @staticmethod
    def strategy(trader, env, model, num_states, strategy):
        """
        Displays the strategy resulting from the learning process.
        :param trader:
        :param env:
        :param model:
        :param num_states:
        :param strategy:
        :return:
        """
        print('\nStrategy learned')
        strategy_string = "State {{:<{}s}} -> {{:<10s}} {{}}".format(
            env.states.max_len)
        for i in range(num_states):
            print(strategy_string.format(
                env.states.name(i),
                trader.configuration.action_name[strategy[i]],
                model.predict(np.identity(num_states)[i:i + 1])))
        print()

    def report(self, portfolio, t: int, disp_header=False, disp_footer=False):
        """
        Displays a simple report of tha main variables in the QLearning algo
        :param portfolio:
        :param t:
        :param disp_header:
        :param disp_footer:
        :return:
        """
        values = [t] + portfolio.values_to_report()
        self.add_to_table(values, self.params.table_headers)

    def add_to_table(self, values_to_report, table_headers):
        """
        Add the report values to the results table.
        :param values_to_report: the list of values
        :param table_headers:
        :return:
        """
        row = Series(dict(zip(
            table_headers,
            values_to_report
        )))
        self.params.results = self.params.results.append(
            row, ignore_index=True)

    def summary(self, portfolio, do_plot=False):
        df = self.params.results.copy()
        self.recolor_ref(df, 'forecast', 'price')
        self.reformat(df, 'price')
        self.reformat(df, 'value')
        self.reformat(df, 'shares')
        self.recolor(df, 'budget')
        self.recolor(df, 'netValue')
        self.recolor(df, 'investment')
        self.recolor(df, 'reward')
        print(tabulate(df,
                       headers='keys',
                       tablefmt='psql',
                       showindex=False,
                       floatfmt=['.0f'] + ['.1f' for i in range(6)]))
        self.report_totals(portfolio)
        if do_plot is True:
            # self.plot_value()
            self.plot_results(self.params.results)

    def report_totals(self, portfolio):
        # total outcome and final metrics.
        if portfolio.portfolio_value != 0.0:
            total = portfolio.budget + portfolio.portfolio_value
        else:
            total = portfolio.budget
        percentage = 100. * ((total / portfolio.initial_budget) - 1.0)
        self.log('Final....: € {:.2f} [{} %]'.format(
            total, self.color(percentage)))
        self.log('Budget...: € {:.1f} [{} %]'.format(
            portfolio.budget,
            self.color((portfolio.budget / portfolio.initial_budget) * 100.)))
        self.log('Cash Flow: {}'.format(
            self.color(portfolio.investment * -1.)))
        self.log(
            'Shares...: {:d}'.format(int(portfolio.shares)))
        self.log(
            'Sh.Value.: {:.1f}'.format(portfolio.portfolio_value))
        self.log('P/L......: € {}'.format(
            self.color(portfolio.portfolio_value - portfolio.investment)))

    def report_action(self, action_name):
        """
        Display only what action was selected.
        :param action_name:
        :return:
        """
        last_index = self.params.results.shape[0] - 1
        self.params.results.loc[last_index, 'action'] = action_name

    def report_reward(self, reward, current_state):
        """
        Display only what is the reward resulting from the action selected.
        :param reward:
        :param current_state:
        :return:
        """
        last_index = self.params.results.shape[0] - 1
        self.params.results.loc[last_index, 'reward'] = reward
        self.params.results.loc[last_index, 'state'] = current_state

    def progress(self, i, num_episodes, last_avg, start, end):
        """
        Report the progress during learning
        :return:
        :param i:
        :param num_episodes:
        :param last_avg:
        :param start:
        :param end:
        :return:
        """
        percentage = (i / num_episodes) * 100.0
        self.log(
            "Epoch {:>5}/{:<5} [{:>5.1f}%] Avg reward: {:+.3f}".format(
                i,
                num_episodes,
                percentage,
                last_avg), end='')
        if percentage == 0.0:
            self.log(' Est.time: UNKNOWN')
            return
        elapsed = end - start
        remaining = ((100. - percentage) * elapsed) / percentage
        self.log(
            ' Est.time: {}'.format(self.timer(remaining)))

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

    def states_list(self, states):
        """
        Simply print the list of states that have been read from configuration
        file.
        :return: None
        """
        self.log('List of states: [{}]'.format(
            ' | '.join([(lambda x: x[1:])(s) for s in
                        states.keys()])))
        return

    @staticmethod
    def plot_results(results):
        data = results.copy(deep=True)
        data = data.dropna()

        def color_action(a):
            actions = ['buy', 'sell', 'f.buy', 'f.sell', 'n/a']
            return actions.index(a)

        data['action_id'] = data.action.apply(lambda a: color_action(a))
        data.head()
        colors = {0: 'green', 1: 'red', 2: '#E8D842', 3: '#BE5B11',
                  4: '#BBBBBB'}
        fig, (ax1, ax2) = plt.subplots(2,
                                       sharex=True,
                                       figsize=(14, 10),
                                       gridspec_kw={'height_ratios': [1, 3]})
        fig.suptitle('Portfolio Value and Shares price')
        ax1.axhline(y=0, color='red', alpha=0.4)
        ax1.plot(data.netValue)
        ax1.xaxis.set_ticks_position('none')
        ax2.scatter(range(len(data.price)), data.price,
                    c=data.action_id.apply(lambda x: colors[x]),
                    marker='.')
        ax2.plot(data.price, c='black', linewidth=0.5)
        plt.show()

    @staticmethod
    def chart(array,
              metric_name='',
              chart_type='line',
              ma: bool = False):
        """
        Plots a chart of the array passed
        :param array:
        :param metric_name: label to be displayed on Y axis
        :param chart_type: either 'line' or 'scatter'
        :param ma: moving average?
        :return:
        """
        if ma is True:
            magnitude = int(log10(len(array))) - 1
            period = int(pow(10, magnitude))
            if period == 1:
                period = 10
            data = np.convolve(array, np.ones((period,)) / period, mode='valid')
        else:
            data = array
        if chart_type == 'scatter':
            plt.scatter(range(len(data)), data)
        else:
            plt.plot(data)
        plt.ylabel(metric_name)
        plt.xlabel('Number of games')
        plt.show()

    def plot_metrics(self, avg_loss, avg_mae, avg_rewards):
        self.chart(avg_rewards, 'Average reward per game', 'line',
                   ma=True)
        self.chart(avg_loss, 'Avg loss', 'line', ma=True)
        self.chart(avg_mae, 'Avg MAE', 'line', ma=True)

    def rl_train_report(self, index, avg_rewards, last_avg, start):
        """
        Displays report periodically
        :param index:
        :param avg_rewards:
        :param last_avg:
        :param start:
        :return:
        """
        if (index % self.params.num_episodes_update == 0) or \
                (index == (self.params.num_episodes - 1)):
            end = time.time()
            if avg_rewards:
                last_avg = avg_rewards[-1]
            self.progress(index, self.params.num_episodes,
                          last_avg, start, end)
