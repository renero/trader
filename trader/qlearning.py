import numpy as np

from chart import Chart as plot
from display import Display as display
from environment import Environment
from nnmodel import NNModel


class QLearning(object):

    def __init__(self, context_dictionary):
        self.__dict__.update(context_dictionary)
        self.nn = NNModel(context_dictionary)
        self.model = None

    @staticmethod
    def onehot(num_states: int, state: int) -> np.ndarray:
        return np.identity(num_states)[state:state + 1]

    def predict(self, state) -> int:
        return int(
            np.argmax(self.model.predict(self.onehot(self._num_states, state))))

    def predict_value(self, state):
        return np.max(self.model.predict(self.onehot(self._num_states, state)))

    def q_learn(self, env: Environment, do_plot: bool = False) -> list:
        """
        Implements the RL learning loop over an environment.

        :type env: Environment
        :type num_episodes: int
        :type do_plot: bool
        """
        # create the Keras model
        self.model = self.nn.create_model(env)

        # now execute the q learning
        r_avg_list = []

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
                    self.model.predict(self.onehot(self._num_states, s))[0]
                target_vec[a] = target
                self.model.fit(self.onehot(self._num_states, s),
                               target_vec.reshape(-1, self._num_actions),
                               epochs=1, verbose=0)
                s = new_s
                r_sum += r
            r_avg_list.append(r_sum / self._num_episodes)

        if do_plot is True:
            plot.reinforcement(r_avg_list, do_plot)
        strategy = [
            np.argmax(
                self.model.predict(np.identity(self._num_states)[i:i + 1])[0])
            for i in range(self._num_states)
        ]
        display.strategy(self, env, self.model, self._num_states, strategy)

        return strategy
