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
        self.configuration = configuration
        self.display = self.configuration.display
        self.nn = RL_NN(self.configuration)
        self.model = None
        self.callback_args = {}
        if self.configuration.tensorboard is True:
            self.tensorboard = TensorBoard(
                log_dir=self.configuration.tbdir,
                histogram_freq=0, write_graph=True, write_images=False)
            self.callback_args = {'callbacks': self.tensorboard}

    def q_learn(self,
                env: Environment,
                display_strategy: bool = False,
                do_plot: bool = False) -> list:
        """
        Learns or Load an strategy to follow over a given environment,
        using RL.
        :type env: Environment
        :param display_strategy:
        :type do_plot: bool
        """
        start = time.time()
        # create the Keras model and learn, or load it from disk.
        if self.configuration.load_model is True:
            self.model = self.nn.load_model(self.configuration.model_file,
                                            self.configuration.weights_file)
        else:
            self.model = self.nn.create_model()
            avg_rewards, avg_loss, avg_mae = self.reinforce_learn(env)

        # display anything?
        if do_plot is True and self.configuration.load_model is False:
            self.display.plot_metrics(avg_loss, avg_mae, avg_rewards)

        # Extract the strategy matrix from the model.
        strategy = self.get_strategy()
        if display_strategy:
            self.display.strategy(self,
                                  env,
                                  self.model,
                                  self.configuration.num_states,
                                  strategy)

        self.log('\nTime elapsed: {}'.format(
            self.configuration.display.timer(time.time() - start)))
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
        last_avg: float = 0.0
        start = time.time()
        epsilon = self.configuration.epsilon

        # Loop over 'num_episodes'
        for i in range(self.configuration.num_episodes):
            state = env.reset()
            self.display.rl_train_report(i, avg_rewards, last_avg, start)

            done = False
            sum_rewards = 0
            sum_loss = 0
            sum_mae = 0
            j = 0
            while not done:
                # Decide whether generating random action or predict most
                # likely from the give state.
                action = self.epsilon_greedy(epsilon, state)

                # Send the action to the environment and get new state,
                # reward and information on whether we've finish.
                new_state, reward, done, _ = env.step(action)
                self.memory.append((state, action, reward, new_state, done))
                # loss, mae = self.step_learn(state, action, reward, new_state)
                if (j > 0) and ((j % 64) == 0):
                    loss, mae = self.minibatch_learn(batch_size=64)
                    # Update states and metrics
                    state = new_state
                    sum_rewards += reward
                    sum_loss += loss
                    sum_mae += mae
                j += 1

            avg_rewards.append(sum_rewards / self.configuration.num_episodes)
            avg_loss.append(sum_loss / self.configuration.num_episodes)
            avg_mae.append(sum_mae / self.configuration.num_episodes)

            # Batch Replay
            if self.configuration.experience_replay is True:
                if len(self.memory) > self.configuration.exp_batch_size:
                    self.experience_replay()

            # Epsilon decays here
            if epsilon >= self.configuration.epsilon_min:
                epsilon *= self.configuration.decay_factor

        return avg_rewards, avg_loss, avg_mae

    def epsilon_greedy(self, epsilon, state):
        """
        Epsilon greedy routine
        :param epsilon: current value of epsilon after applying decaying f.
        :param state: current state
        :return: action predicted by the network
        """
        if np.random.random() < epsilon:
            action = np.random.randint(
                0, self.configuration.num_actions)
        else:
            action = self.predict(state)
        return action

    def step_learn(self, state, action, reward, new_state):
        """
        Fit the NN model to predict the action, given the action and
        current state.
        :param state:
        :param action:
        :param reward:
        :param new_state:
        :return: the loss and the metric resulting from the training.
        """
        target = reward + self.configuration.gamma * self.predict_value(
            new_state)
        target_vec = self.model.predict(self.onehot(state))[0]
        target_vec[action] = target

        history = self.model.fit(
            self.onehot(state),
            target_vec.reshape(-1, self.configuration.num_actions),
            epochs=1, verbose=0, **self.callback_args
        )
        return history.history['loss'][0], \
               history.history['mean_absolute_error'][0]

    def minibatch_learn(self, batch_size):
        """
        MiniBatch Learning routine.
        :param batch_size:
        :return:
        """
        mini_batch = np.empty(shape=(0, 5), dtype=np.int32)
        mem_size = len(self.memory)
        for i in range(mem_size - batch_size - 1, mem_size - 1):
            mini_batch = np.append(
                mini_batch,
                np.asarray(self.memory[i]).astype(int).reshape(1, -1),
                axis=0)

        nn_input = np.empty((0, self.configuration.num_states), dtype=np.int32)
        nn_output = np.empty((0, self.configuration.num_actions))
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.configuration.gamma * self.predict_value(
                    next_state)
            nn_input = np.append(nn_input, self.onehot(state), axis=0)
            labeled_output = self.model.predict(self.onehot(state))[0]
            labeled_output[action] = target
            y = labeled_output.reshape(-1, self.configuration.num_actions)
            nn_output = np.append(nn_output, y, axis=0)

        print('Fitting with {} samples, shaped {}'.format(nn_input.shape[0],
                                                          nn_input.shape))
        h = self.model.train_on_batch(
            nn_input, nn_output)#,
            #epochs=1, verbose=0, batch_size=batch_size,
            #**self.callback_args)

        #return h.history['loss'][0], h.history['mean_absolute_error'][0]
        return h[0], h[1]

    def experience_replay(self):
        """
        Primarily from: https://github.com/edwardhdlu/q-trader
        :return: None
        """
        mini_batch = random.sample(self.memory,
                                   self.configuration.exp_batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.configuration.gamma * self.predict_value(
                    next_state)
            target_vec = self.model.predict(self.onehot(state))[0]
            target_vec[action] = target
            self.model.fit(
                self.onehot(state),
                target_vec.reshape(-1, self.configuration.num_actions),
                epochs=1, verbose=0,
                **self.callback_args)

    def onehot(self, state: int) -> np.ndarray:
        return np.identity(self.configuration.num_states)[state:state + 1]

    def predict(self, state) -> int:
        return int(
            np.argmax(
                self.model.predict(
                    self.onehot(state))))

    def predict_value(self, state):
        return np.max(self.model.predict(self.onehot(state)))

    def get_strategy(self):
        """
        Get the defined strategy from the weights of the model.
        :return: strategy matrix
        """
        strategy = [
            np.argmax(
                self.model.predict(self.onehot(i))[0])
            for i in range(self.configuration.num_states)
        ]
        return strategy
