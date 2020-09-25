import time
from collections import deque

import numpy as np
import pandas as pd

from rl_nn import RL_NN
from rl_stats import RLStats
from spring import spring


class Agent:
    configuration = None
    tensorboard = None
    experience = deque(maxlen=20000)

    def __init__(self, configuration, env):
        self.params = configuration
        self.env = env
        self.display = self.params.display
        self.log = self.params.log

        self.log.info('Creating agent')
        self.nn = RL_NN(self.params, env)
        self.model = None

        self.info_learning_mode()

    def q_learn(self,
                fresh_model: bool = True,
                display_strategy: bool = False) -> list:
        """
        Learns an strategy to follow over a given environment, using RL.
        :param fresh_model: if False, it does not create the NN from scratch,
                                 but, it uses the one previously loaded.
        :param display_strategy:
        """
        start = time.time()
        # create the Keras model and learn, or load it from disk.
        if fresh_model is True:
            self.model = self.nn.create_model()

        avg_rewards, avg_loss, avg_mae, avg_value = self.reinforce_learn()
        self.log.info('Time elapsed: {}'.format(
            self.params.display.timer(time.time() - start)))

        # Save the model?
        if self.params.save_model is True:
            self.nn.save_model(self.model, self.env.memory.results)

        # Extract the strategy matrix from the model.
        strategy = self.nn.infer_strategy()

        # Simulate what has been learnt with the data.
        self.simulate(strategy)

        # display anything?
        if display_strategy:
            self.display.strategy(self,
                                  self.env,
                                  self.model,
                                  self.params.num_states,
                                  strategy)
        # Plot metrics?
        plot_metrics = self.params.what_to_do == 'train' or \
                       self.params.what_to_do == 'retrain'
        if self.params.do_plot is True and plot_metrics is True:
            self.display.plot_metrics(avg_loss, avg_mae, avg_rewards, avg_value)

        return strategy

    def reinforce_learn(self):
        """
        Implements the learning loop over the states, actions and strategies
        to learn what is the sequence of actions that maximize reward.
        :return: avg_rewards, avg_loss, avg_mae, last_profit
        """
        rl_stats = RLStats()
        epsilon = self.params.epsilon
        stop_drop = spring(self.params, self.env.price_)

        # Loop over 'num_episodes'
        self.log.debug('Loop over {} episodes'.format(self.params.num_episodes))
        for episode in range(self.params.num_episodes):
            state = self.env.reset()
            done = False
            rl_stats.reset()
            episode_step = 0
            while not done:
                self.log.debug('----- t={}, ts={}, price={:.2f} ------'.format(
                    self.env.t, self.env.ts_, self.env.price_))

                # Decide whether generating random action or predict most
                # likely from the give state.
                action = self.epsilon_greedy(epsilon, state)

                # Experimental: Insert the behavior defined by stop_drop
                # overriding random choice or predictions.
                action = stop_drop.correction(action, self.env)

                # Send the action to the environment and get new state,
                # reward and information on whether we've finish.
                new_state, reward, done, _ = self.env.step(action)

                self.experience.append((state, action, reward, new_state, done))
                if self.time_to_learn(episode, episode_step):
                    loss, mae = self.nn.do_learn(episode, episode_step,
                                                 self.experience)
                    rl_stats.step(loss, mae, reward)

                # Update states and metrics
                state = new_state
                episode_step += 1

            self.display.rl_train_report(
                episode, episode_step, rl_stats.avg_rewards,
                rl_stats.last_avg, rl_stats.start)

            #  Update average metrics
            budget = self.env.memory.results.iloc[-1].budget
            cost = self.env.memory.results.iloc[-1].cost
            last_profit = self.env.memory.results.iloc[-1].profit
            balance = budget + cost + last_profit
            rl_stats.update(self.params.num_episodes, balance)

            # Epsilon decays here
            if epsilon >= self.params.epsilon_min:
                epsilon *= self.params.decay_factor
                self.log.debug('Updated epsilon: {:.2f}'.format(epsilon))

        return rl_stats.avg_rewards, rl_stats.avg_loss, \
               rl_stats.avg_mae, rl_stats.avg_balance

    def epsilon_greedy(self, epsilon, state):
        """
        Epsilon greedy routine
        :param epsilon: current value of epsilon after applying decaying f.
        :param state: current state
        :return: action predicted by the network
        """
        rn = np.random.random()
        if rn < epsilon:
            action = np.random.randint(0, self.params.num_actions)
            self.log.debug('Action (random): {}({}) - ε={:.2f}<{:.2f}'.format(
                self.params.action_name[action], action, rn, epsilon))
        else:
            action = self.nn._predict(state)
            self.log.debug('Action (predicted): {}({}) - {:.2f}>{:.2f}'.format(
                self.params.action_name[action], action, rn, epsilon))
        return action

    def simulate(self, strategy: list):
        """
        Simulate over a dataset, given a strategy and an environment.
        :param strategy: strategy data structure to be used in the simulation
        :return:
        """
        done = False
        total_reward = 0.
        self.params.debug = True
        state = self.env.reset()
        stop_drop = spring(self.params, self.env.price_)
        self.log.debug('STARTING Simulation')
        while not done:
            action = self.env.decide_next_action(state, strategy)
            action = stop_drop.correction(action, self.env)

            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state

        # Do I need to init a portfolio, after a simulation
        if self.params.init_portfolio:
            self.env.save_portfolio(init=True)

        # display the result of the simulation (maybe only the totals summ)
        if self.params.totals is True:
            self.params.display.report_totals(self.env.memory.results,
                                              unscale=True)
        else:
            self.params.display.summary(self.env.memory.results,
                                        do_plot=self.params.do_plot)
        return 0

    def single_step(self, strategy):
        """
        Simulate a single step, given a strategy and an environment.
        :param strategy: strategy data structure to be used in the simulation
        :return: None
        """
        state = self.env.resume()
        if state == -1:
            self.log.warn(
                'Portfolio({}) and forecast({}) are in same state(len)'.format(
                    self.params.portfolio_name, self.params.forecast_file))
            self.log.warn('No action/recommendation produced by environment')
            return -1

        action = self.env.decide_next_action(state, strategy)
        self.log.info('Decided action is: {}'.format(action))
        if self.params.stop_drop is True:
            # is this a failed action?
            is_failed_action = self.env.portfolio.failed_action(
                action, self.env.price_)
            action = spring(self.params, self.env.price_).check(
                action, self.env.price_, is_failed_action)

        next_state, reward, done, _ = self.env.step(action)

        # Save the action to the tmp file.
        last_action = self.env.memory.results.iloc[-1]['action']
        self.log.info('Last action is: {}'.format(last_action))
        pd.Series({'action': last_action}).to_json(self.params.json_action)
        self.log.info('Saved action to: {}'.format(self.params.json_action))

        # Save the updated portfolio, overwriting the file.
        if self.params.no_dump is not True:
            self.env.save_portfolio()
        return 0

    def q_load(self,
               retrain: bool = False,
               display_strategy: bool = False) -> list:
        """
        Load an strategy to follow over a given environment, using RL,
        and acts following the strategy defined on it.

        :param retrain:          If this flag is set to true, then compile
                                 the loaded model to continue learning over it.
        :param display_strategy: flag indicating if I want to display the
                                 strategy on stdout.
        """
        # create the Keras model and learn, or load it from disk.
        self.model = self.nn.load_model(self.params.model_file)
        if retrain is True:
            self.model = self.nn.compile_model()

        # Extract the strategy matrix from the model.
        strategy = self.nn.infer_strategy()
        if display_strategy:
            self.display.strategy(self,
                                  self.env,
                                  self.model,
                                  self.params.num_states,
                                  strategy)
        return strategy

    def info_learning_mode(self):
        # display some helpful info
        if self.params.experience_replay is True:
            self.log.info(
                'Experience replay mode {}'.format(self.params.exp_batch_size))
        else:
            self.log.info(
                'Minibatch learning mode {}'.format(self.params.batch_size))

    def time_to_learn(self, episode, episode_step):
        self.log.debug(
            'TTL [ {} AND {} ] (ep:{}, step:{}, st_ep:{}, tr_ss:{})'.format(
                episode_step % self.params.train_steps == 0,
                episode >= self.params.start_episodes,
                episode, episode_step,
                self.params.start_episodes,
                self.params.train_steps
            ))
        return episode_step % self.params.train_steps == 0 and \
               episode >= self.params.start_episodes
