import numpy as np

from chart import Chart as plot
from display import Display as display
from environment import Environment
from nn import NN


class QLearning(object):

    def __init__(self, context_dictionary):
        self.__dict__.update(context_dictionary)
        self.nn = NN(context_dictionary)
        self.model = None

    def onehot(self, state: int) -> np.ndarray:
        return np.identity(self._num_states)[state:state + 1]

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
            for i in range(self._num_states)
        ]
        return strategy

    def learn(self, env):
        """
        Implements the learning loop over the states, actions and strategies
        to learn what is the sequence of actions that maximize reward.
        :param env:
        :return:
        """
        # now execute the q learning
        avg_rewards = []
        # Loop over 'num_episodes'
        for i in range(self._num_episodes):
            s = env.reset()
            self._eps *= self._decay_factor
            if i % self._num_episodes_update == 0:
                print("Episode {} of {}".format(i, self._num_episodes))
            done = False
            r_sum = 0
            while not done:
                if np.random.random() < self._eps:
                    a = np.random.randint(0, self._num_actions)
                else:
                    a = self.predict(s)
                new_s, r, done, _ = env.step(a)
                target = r + self._y * self.predict_value(new_s)
                target_vec = \
                    self.model.predict(self.onehot(s))[0]
                target_vec[a] = target
                self.model.fit(self.onehot(s),
                               target_vec.reshape(-1, self._num_actions),
                               epochs=1, verbose=0)
                s = new_s
                r_sum += r
            avg_rewards.append(r_sum / self._num_episodes)
        return avg_rewards

    def q_learn(self, env: Environment, do_plot: bool = False) -> list:
        """
        Learns or Load an strategy to follow over a given environment,
        using RL.

        :type env: Environment
        :type do_plot: bool
        """
        # create the Keras model and learn, or load it from disk.
        if self._load_model is True:
            self.model = self.nn.load_model(self._model_file,
                                            self._weights_file)
        else:
            self.model = self.nn.create_model()
            avg_rewards = self.learn(env)

        # Extract the strategy matrix from the model.
        strategy = self.get_strategy()

        # display anything?
        if do_plot is True and self._load_model is False:
            plot.reinforcement(avg_rewards, do_plot)
        display.strategy(self, env, self.model, self._num_states, strategy)

        return strategy
