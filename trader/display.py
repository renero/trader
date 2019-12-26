import time
from datetime import datetime
from math import log10, pow, floor

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from tabulate import tabulate

from common import Common
from logger import Logger


def ts() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class Display(Common):
    """
    Consider this: https://stackoverflow.com/a/56982302
    to make plots independent processes.
    """

    def __init__(self, configuration):
        self.params = configuration
        self.log: Logger = self.params.log

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

    def summary(self, results: DataFrame, totals=True, do_plot=False) -> None:
        """
        Displays a nice colored table with the summary of results from current
        simulation.
        :param results: The table with the results from each iteration
        :param totals:  Do I want totals at the end?
        :param do_plot: Do I want to also display a nice plot?
        :return: None
        """
        # First, guess what do I need to show.
        if self.params.short:
            to_remove = {'t', 'budget', 'cost', 'value', 'reward',
                         'state', 'state_desc'}
            to_display = list(
                set(results.columns) - to_remove)
        else:
            to_display = results.columns
        df = results[to_display].copy()

        # Recolor some columns
        self.recolor_ref(df, 'forecast', 'price')
        # self.reformat(df, 'price')
        self.recolor_pref(df, 'price')

        if 'value' in df.columns:
            self.reformat(df, 'value')
        if 'shares' in df.columns:
            self.reformat(df, 'shares')
        if 'budget' in df.columns:
            self.recolor(df, 'budget')
        if 'profit' in df.columns:
            self.recolor(df, 'profit')
        if 'cost' in df.columns:
            self.recolor(df, 'cost')
        if 'reward' in df.columns:
            self.recolor(df, 'reward')
        if 'konkorde' in df.columns:
            self.recolor(df, 'konkorde')
        if 'action' in df.columns:
            self.recolor_action(df, 'action')

        # Reorder columns
        df_cols = list(df.columns)
        predef_cols = self.params.table_headers.copy()
        reordered_cols = list(
            filter(lambda x: x is not None,
                   map(lambda x: x if x in df_cols else None, predef_cols)))
        df = df[reordered_cols]
        print(tabulate(df,
                       headers='keys',
                       tablefmt='psql',
                       showindex=False,
                       floatfmt=['.0f'] + ['.2f' for i in range(6)]))
        if totals:
            self.report_totals(results, self.params.mode)
        if do_plot is True:
            self.plot_results(results, self.params.have_konkorde)

    def report_totals(self, results, mode):
        # Extract Portfolio valuation from the table
        initial_budget = results.iloc[0].budget
        budget = results.iloc[-1].budget
        value = results.iloc[-1].value
        shares = results.iloc[-1].shares
        profit = (budget + value) - initial_budget
        no_action_perf = results.iloc[-1].price - results.iloc[0].price
        if mode == 'bull':
            performance = (profit - no_action_perf) / no_action_perf
        else:
            performance = (no_action_perf - profit) / no_action_perf

        print('P/L........: €{} [{:.1f}% over nop ({:.2f})]'.format(
            self.color(profit), performance*100., no_action_perf))
        if value != 0.0:
            balance = budget + value
        else:
            balance = budget
        percentage = 100. * ((balance / initial_budget) - 1.0)
        print('Balance....: €{} [{} %]'.format(
            self.cond_color(balance, initial_budget), self.color(percentage)))
        print('Sh.Value...: {} shares = €{:.1f}'.format(int(shares), value))

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

        def magnitude(x):
            return int(floor(log10(x)))

        percentage = (i / num_episodes) * 100.0
        msg = 'Epoch {:0{m}}/{:<{m}} [{:>5.1f}%] Avg.reward: {:+.3f}'.format(
            i,
            num_episodes,
            percentage,
            last_avg,
            m=magnitude(num_episodes) + 1)
        if percentage == 0.0:
            self.log.info('{} ETA: UNKNOWN'.format(msg))
            return
        elapsed = end - start
        remaining = ((100. - percentage) * elapsed) / percentage
        self.log.info('{} ETA: {}'.format(msg, self.timer(remaining)))

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
        self.log.info('List of states: [{}]'.format(
            ' | '.join([(lambda x: x[1:])(s) for s in
                        states.keys()])))
        return

    @staticmethod
    def plot_results(results, have_konkorde):
        data = results.copy(deep=True)
        data = data.dropna()
        ini_budget = data.iloc[0].budget

        def color_action(a):
            actions = ['buy', 'sell', 'f.buy', 'f.sell', 'n/a', 'wait']
            return actions.index(a)

        data['action_id'] = data.action.apply(lambda a: color_action(a))
        data.head()
        colors = {0: 'green', 1: 'red', 2: '#E8D842', 3: '#BE5B11',
                  4: 'white', 5: 'white'}
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(14, 10),
                                       gridspec_kw={'height_ratios': [1, 3]})
        fig.suptitle(
            'Portfolio Value and Shares price ({})'.format(ts()))
        #
        # Portfolio Value
        #
        ax1.axhline(y=0, color='red', alpha=0.4, linewidth=0.4)
        ax1.scatter(range(len(data.price)),
                    (data.budget + data.investment + data.profit) - ini_budget,
                    c=data.action_id.apply(lambda x: colors[x]),
                    marker='.')
        ax1.plot((data.budget + data.investment + data.profit) - ini_budget,
                 linewidth=0.6)
        ax1.xaxis.set_ticks_position('none')
        #
        # Price, forecast and operations
        #
        ax2.set_xticks(ax2.get_xticks()[::10])
        ax2.plot(data.price, c='black', linewidth=0.6)
        ax2.scatter(range(len(data.price)), data.price,
                    c=data.action_id.apply(lambda x: colors[x]),
                    marker='.')
        ax2.plot(data.forecast, 'k--', linewidth=0.5)
        ax2.grid(True, which='major', axis='x')
        #
        # Konkorde ?
        #
        if have_konkorde and 'konkorde' in data.columns:
            ax3 = ax2.twinx()
            ax3.set_ylim(-2., +10.)
            ax3.axhline(0, color='black', alpha=0.3)
            ax3.fill_between(range(data.konkorde.shape[0]), 0, data.konkorde,
                             color='green', alpha=0.3)
        plt.show()

    @staticmethod
    def chart(arrays,
              metric_name=None,
              chart_type='line',
              ma: bool = False):
        """
        Plots a chart of the array passed
        :param arrays:
        :param metric_name: label to be displayed on Y axis
        :param chart_type: either 'line' or 'scatter'
        :param ma: moving average?
        :return:
        """
        if metric_name is None:
            metric_name = ['']
        if type(arrays[0]) is not list:
            arrays = [arrays]
        if type(metric_name) is not list:
            metric_name = [metric_name]
        assert len(arrays) == len(metric_name), \
            '{} arrays passed, but only {} metric names'.format(
                len(arrays), len(metric_name))

        # Plot every array passed
        timestamp = ts()
        color = ['blue', 'red', 'orange', 'green']
        fig, ax = plt.subplots()
        for index, array in enumerate(arrays):
            ax.axhline(0, color='grey', linewidth=0.5, alpha=0.4)
            data = Display.smooth(array) if ma is True else array
            if index > 0:
                ax = ax.twinx()
            if chart_type == 'scatter':
                ax.scatter(range(len(data)), data,
                           color=color[index], alpha=0.5)
            else:
                ax.plot(data, label=metric_name[index],
                        color=color[index], alpha=0.5, linewidth=0.8)
            ax.set_ylabel(metric_name[index], color=color[index])
        plt.xlabel('Number of games')
        plt.legend()
        plt.title(timestamp)
        fig.tight_layout()
        plt.show()

    @staticmethod
    def smooth(array):
        """Compute the moving average over the array"""
        if len(array) < 10:
            return array
        magnitude = int(log10(len(array))) - 1
        period = int(pow(10, magnitude))
        if period == 1:
            period = 10
        data = np.convolve(array, np.ones((period,)) / period, mode='valid')
        return data

    def plot_metrics(self, avg_loss, avg_mae, avg_rewards, avg_value):
        self.chart([avg_rewards, avg_value],
                   ['Average reward', 'Average net value'],
                   ma=True)
        self.chart(avg_loss, 'Avg loss', 'line', ma=True)
        self.chart(avg_mae, 'Avg MAE', 'line', ma=True)

    def rl_train_report(self,
                        episode,
                        episode_step,
                        avg_rewards,
                        last_avg,
                        start):
        """
        Displays report periodically
        :param episode:
        :param episode_step:
        :param avg_rewards:
        :param last_avg:
        :param start:
        :return:
        """
        if (episode % self.params.num_episodes_update == 0) or \
                (episode == (self.params.num_episodes - 1)):
            end = time.time()
            if avg_rewards:
                last_avg = avg_rewards[-1]
            self.progress(episode, self.params.num_episodes,
                          last_avg, start, end)
        if (episode % self.params.num_episodes_update == 0) or \
                (episode == (self.params.num_episodes - 1)):
            self.log.debug(
                'Finished episode {} after {} steps]'.format(
                    episode, episode_step))
