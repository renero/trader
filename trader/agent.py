import time
from collections import deque

import numpy as np
import pandas as pd

from common import Common
from environment import Environment
from rl_nn import RL_NN
from rl_stats import RLStats
from spring import Spring


class Agent(Common):
    configuration = None
    tensorboard = None
    experience = deque(maxlen=20000)

    def __init__(self, configuration):
        self.params = configuration
        self.display = self.params.display
        self.log = self.params.log
        env_params = self.params.environment

        self.log.info('Creating agent')
        self.nn = RL_NN(self.params)
        self.model = None

        self.info_learning_mode(env_params)

    def q_learn(self,
                env: Environment,
                fresh_model: bool = True,
                display_strategy: bool = False,
                do_plot: bool = False) -> list:
        """
        Learns an strategy to follow over a given environment,
        using RL.
        :type env: Environment
        :param fresh_model: if False, it does not create the NN from scratch,
            but, it uses the one previously loaded.
        :param display_strategy:
        :type do_plot: bool
        """
        start = time.time()
        # create the Keras model and learn, or load it from disk.
        if fresh_model is True:
            self.model = self.nn.create_model()
        avg_rewards, avg_loss, avg_mae, avg_value = self.reinforce_learn(env)

        # display anything?
        plot_metrics = self.params.what_to_do == 'train' or \
                       self.params.what_to_do == 'retrain'
        if do_plot is True and plot_metrics is True:
            self.display.plot_metrics(avg_loss, avg_mae, avg_rewards, avg_value)

        # Extract the strategy matrix from the model.
        strategy = self.nn.infer_strategy()
        if display_strategy:
            self.display.strategy(self,
                                  env,
                                  self.model,
                                  self.params.num_states,
                                  strategy)

        self.log.info('Time elapsed: {}'.format(
            self.params.display.timer(time.time() - start)))
        return strategy

    def reinforce_learn(self, env: Environment):
        """
        Implements the learning loop over the states, actions and strategies
        to learn what is the sequence of actions that maximize reward.
        :param env: the environment
        :return: avg_rewards, avg_loss, avg_mae, last_profit
        """
        rl_stats = RLStats()
        epsilon = self.params.epsilon

        # Loop over 'num_episodes'
        self.log.debug('Loop over {} episodes'.format(self.params.num_episodes))
        for episode in range(self.params.num_episodes):
            state = env.reset()
            done = False
            rl_stats.reset()
            episode_step = 0
            while not done:
                # Decide whether generating random action or predict most
                # likely from the give state.
                action = self.epsilon_greedy(epsilon, state)

                # Send the action to the environment and get new state,
                # reward and information on whether we've finish.
                new_state, reward, done, _ = env.step(action)
                # Experimental BEARish mode
                if self.params.mode == 'bear':
                    reward *= -1.
                self.experience.append((state, action, reward, new_state, done))
                loss, mae = self.nn.do_learn(episode, episode_step,
                                             self.experience)

                # Update states and metrics
                rl_stats.step(loss, mae, reward)
                state = new_state
                episode_step += 1

            self.display.rl_train_report(
                episode, episode_step, rl_stats.avg_rewards,
                rl_stats.last_avg, rl_stats.start)

            #  Update average metrics
            rl_stats.update(self.params.num_episodes,
                            env.memory.results.profit.iloc[-1])

            # Epsilon decays here
            if epsilon >= self.params.epsilon_min:
                epsilon *= self.params.decay_factor

        return rl_stats.avg_rewards, rl_stats.avg_loss, \
               rl_stats.avg_mae, rl_stats.avg_profit

    def epsilon_greedy(self, epsilon, state):
        """
        Epsilon greedy routine
        :param epsilon: current value of epsilon after applying decaying f.
        :param state: current state
        :return: action predicted by the network
        """
        if np.random.random() < epsilon:
            action = np.random.randint(
                0, self.params.num_actions)
        else:
            action = self.nn.predict(state)
        return action

    def simulate(self, environment: Environment, strategy: list):
        """
        Simulate over a dataset, given a strategy and an environment.
        :param environment: the environment for the simulation
        :param strategy: strategy data structure to be used in the simulat.
        :return:
        """
        done = False
        total_reward = 0.
        self.params.debug = True
        state = environment.reset()
        stop_drop = Spring(self.params, environment.price_)
        self.log.debug('STARTING Simulation')
        while not done:
            action = environment.decide_next_action(state, strategy)
            if self.params.stop_drop is True:
                action = stop_drop.check(action, environment.price_)
            next_state, reward, done, _ = environment.step(action)
            total_reward += reward
            state = next_state
        # Do I need to init a portfolio, after a simulation
        if self.params.init_portfolio:
            environment.save_portfolio(init=True)
        # display the result of the simulation
        self.params.display.summary(environment.memory.results,
                                    environment.portfolio,
                                    do_plot=self.params.do_plot)

    def single_step(self, environment: Environment, strategy):
        """
        Simulate a single step, given a strategy and an environment.
        :param environment: the environment for the simulation
        :param strategy: strategy data structure to be used in the simulation
        :return: None
        """
        state = environment.resume()
        action = environment.decide_next_action(state, strategy)
        if self.params.stop_drop is True:
            action = Spring(self.params, environment.price_).check(
                action, environment.price_)

        next_state, reward, done, _ = environment.step(action)

        # Save the action to the tmp file.
        last_action = environment.memory.results.iloc[-1]['action']
        self.log.info('Last action is: {}'.format(last_action))
        pd.Series({'action': last_action}).to_json(self.params.json_action)
        self.log.info('Saved action to: {}'.format(self.params.json_action))

        # Save the updated portfolio, overwriting the file.
        if self.params.no_dump is not True:
            environment.save_portfolio()

    def q_load(self,
               env: Environment,
               retrain: bool = False,
               display_strategy: bool = False) -> list:
        """
        Load an strategy to follow over a given environment, using RL,
        and acts following the strategy defined on it.
        :param env: Environment
        :param retrain: If this flag is set to true, then compile the loaded
        model to continue learning over it.
        :param display_strategy:
        """
        # create the Keras model and learn, or load it from disk.
        self.model = self.nn.load_model(self.params.model_file)
        if retrain is True:
            self.model = self.nn.compile_model()

        # Extract the strategy matrix from the model.
        strategy = self.nn.infer_strategy()
        if display_strategy:
            self.display.strategy(self,
                                  env,
                                  self.model,
                                  self.params.num_states,
                                  strategy)
        return strategy

    def info_learning_mode(self, env_params):
        # display some helpful info
        if self.params.environment.direct_reward is True:
            self.log.info('Direct Reward mode')
        else:
            self.log.info('Preset reward mode {}'.format(
                '(proport.)' if env_params.proportional_reward is True else ''
            ))
        if self.params.experience_replay is True:
            self.log.info(
                'Experience replay mode {}'.format(self.params.exp_batch_size))
        else:
            self.log.info(
                'Minibatch learning mode {}'.format(self.params.batch_size))
