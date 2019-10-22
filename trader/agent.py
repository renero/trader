import random
import time
from collections import deque

import numpy as np
from keras.callbacks import TensorBoard

from common import Common
from environment import Environment
from rl_nn import RL_NN


class Agent(Common):
    configuration = None
    tensorboard = None
    memory = deque(maxlen=20000)

    def __init__(self, configuration):
        self.params = configuration
        self.display = self.params.display
        self.log = self.params.log
        env_params = self.params.environment

        self.log.info('Creating agent')
        self.nn = RL_NN(self.params)
        self.model = None

        # display some helpful info
        if self.params.environment.direct_reward is True:
            self.log.info('Direct Reward mode')
        else:
            self.log.info('Preset reward mode {}'.format(
                '(proport.)' if env_params.proportional_reward is True else ''
            ))

        self.callback_args = {}
        if self.params.tensorboard is True:
            self.log.info('Using TensorBoard')
            self.tensorboard = TensorBoard(
                log_dir=self.params.tbdir,
                histogram_freq=0, write_graph=True, write_images=False)
            self.callback_args = {'callbacks': self.tensorboard}

    def q_load(self,
               env: Environment,
               retrain: bool = False,
               display_strategy: bool = False) -> list:
        """
        Load an strategy to follow over a given environment, using RL,
        and acts following the strategy defined on it.
        :type env: Environment
        :param retrain: If this flag is set to true, then compile the loaded
        model to continue learning over it.
        :param display_strategy:
        """
        # create the Keras model and learn, or load it from disk.
        self.model = self.nn.load_model(self.params.model_file)
        if retrain is True:
            self.model = self.nn.compile_model(self.model)

        # Extract the strategy matrix from the model.
        strategy = self.get_strategy()
        if display_strategy:
            self.display.strategy(self,
                                  env,
                                  self.model,
                                  self.params.num_states,
                                  strategy)
        return strategy

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
        plot_metrics = self.params.what_to_do == 'learn' or \
                       self.params.what_to_do == 'retrain'
        if do_plot is True and plot_metrics is True:
            self.display.plot_metrics(avg_loss, avg_mae, avg_rewards, avg_value)

        # Extract the strategy matrix from the model.
        strategy = self.get_strategy()
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
        :return:
        """
        # now execute the q learning
        avg_rewards = []
        avg_loss = []
        avg_mae = []
        avg_netValue = []
        last_avg: float = 0.0
        start = time.time()
        epsilon = self.params.epsilon

        # Loop over 'num_episodes'
        self.log.debug('Loop over {} episodes'.format(self.params.num_episodes))
        for episode in range(self.params.num_episodes):
            state = env.reset()
            done = False
            sum_rewards = 0
            sum_loss = 0
            sum_mae = 0
            episode_step = 0
            num_calls_learn = 0
            sum_netvalue = 0.
            while not done:
                # Decide whether generating random action or predict most
                # likely from the give state.
                action = self.epsilon_greedy(epsilon, state)

                # Send the action to the environment and get new state,
                # reward and information on whether we've finish.
                new_state, reward, done, _ = env.step(action)
                self.memory.append((state, action, reward, new_state, done))

                # loss, mae = self.step_learn(state, action, reward, new_state)
                if episode_step % self.params.train_steps == 0 and \
                        episode > self.params.start_episodes:
                    loss, mae = self.minibatch_learn(self.params.batch_size)
                    # Update states and metrics
                    num_calls_learn += 1
                    state = new_state
                    sum_rewards += reward
                    sum_loss += loss
                    sum_mae += mae
                    sum_netvalue += env.memory.results.netValue.iloc[-1]

                episode_step += 1

            self.display.rl_train_report(episode, avg_rewards, last_avg, start)
            if (episode % self.params.num_episodes_update == 0) or \
                    (episode == (self.params.num_episodes - 1)):
                self.log.debug(
                    'Finished episode {} after {} steps [{} calls]'.format(
                        episode, episode_step, num_calls_learn))

            #  Update average metrics
            avg_rewards.append(sum_rewards / self.params.num_episodes)
            avg_loss.append(sum_loss / self.params.num_episodes)
            avg_mae.append(sum_mae / self.params.num_episodes)
            avg_netValue.append(sum_netvalue / self.params.num_episodes)

            # Batch Replay
            if self.params.experience_replay is True:
                if len(self.memory) > self.params.exp_batch_size:
                    self.experience_replay()

            # Epsilon decays here
            if epsilon >= self.params.epsilon_min:
                epsilon *= self.params.decay_factor

        return avg_rewards, avg_loss, avg_mae, avg_netValue

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
            action = self.predict(state)
        return action

    def minibatch_learn(self, batch_size):
        """
        MiniBatch Learning routine.
        :param batch_size:
        :return: loss and mae
        """
        mem_size = len(self.memory)
        if mem_size < batch_size:
            return 0.0, 0.0

        mini_batch = np.empty(shape=(0, 5), dtype=np.int32)
        for i in range(mem_size - batch_size - 1, mem_size - 1):
            mini_batch = np.append(
                mini_batch,
                np.asarray(self.memory[i]).astype(int).reshape(1, -1),
                axis=0)
        nn_input, nn_output = self.prepare_nn_data(mini_batch)
        h = self.model.fit(
            nn_input, nn_output,
            epochs=1, verbose=0, batch_size=batch_size,
            **self.callback_args)
        return h.history['loss'][0], h.history['mae'][0]

    def prepare_nn_data(self, mini_batch):
        nn_input = np.empty((0, self.params.num_states), dtype=np.int32)
        nn_output = np.empty((0, self.params.num_actions))
        for state, action, reward, next_state, done in mini_batch:
            y = self.predict_output(state, action, reward, next_state, done)
            nn_input = np.append(nn_input, self.onehot(state), axis=0)
            nn_output = np.append(nn_output, y, axis=0)
        return nn_input, nn_output

    def predict_output(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.params.gamma * self.predict_value(
                next_state)
        labeled_output = self.model.predict(self.onehot(state))[0]
        labeled_output[action] = target
        y = labeled_output.reshape(-1, self.params.num_actions)
        return y

    def experience_replay(self):
        """
        Primarily from: https://github.com/edwardhdlu/q-trader
        :return: None
        """
        mini_batch = random.sample(self.memory,
                                   self.params.exp_batch_size)

        nn_input, nn_output = self.prepare_nn_data(mini_batch)
        self.model.fit(
            nn_input, nn_output,
            epochs=1, verbose=0, batch_size=self.params.exp_batch_size,
            **self.callback_args)

    def onehot(self, state: int) -> np.ndarray:
        return np.identity(self.params.num_states)[state:state + 1]

    def predict(self, state) -> int:
        return int(
            np.argmax(
                self.model.predict(
                    self.onehot(state))))

    def predict_value(self, state):
        return np.max(self.model.predict(self.onehot(state)))

    def get_strategy(self) -> list:
        """
        Get the defined strategy from the weights of the model.
        :return: strategy matrix
        """
        strategy = [
            np.argmax(
                self.model.predict(self.onehot(i))[0])
            for i in range(self.params.num_states)
        ]
        return strategy

    def simulate(self, environment: Environment, strategy: list, do_plot=True):
        """
        Simulate over a dataset, given a strategy and an environment.
        :param environment:
        :param strategy:
        :param do_plot:
        :return:
        """
        done = False
        total_reward = 0.
        self.params.debug = True
        state = environment.reset()
        while not done:
            action = environment.decide_next_action(state, strategy)
            next_state, reward, done, _ = environment.step(action)
            total_reward += reward
            state = next_state
        self.params.display.summary(environment.memory.results,
                                    environment.portfolio,
                                    do_plot=do_plot)
