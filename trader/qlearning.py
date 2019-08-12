import numpy as np

from chart import Chart as plot
from display import Display as display
from environment import Environment
from nn import NN


class QLearning(object):
    configuration = None

    def __init__(self, configuration):
        self.configuration = configuration
        self.nn = NN(self.configuration)
        self.model = None

    def onehot(self, state: int) -> np.ndarray:
        return np.identity(self.configuration._num_states)[state:state + 1]

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
            for i in range(self.configuration._num_states)
        ]
        return strategy

    def learn(self, env: Environment):
        """
        Implements the learning loop over the states, actions and strategies
        to learn what is the sequence of actions that maximize reward.
        :param env: the environment
        :return:
        """
        # now execute the q learning
        avg_rewards = []
        # Loop over 'num_episodes'
        for i in range(self.configuration._num_episodes):
            state = env.reset()
            self.configuration._eps *= self.configuration._decay_factor
            if i % self.configuration._num_episodes_update == 0:
                print(
                    "Episode {} of {}".format(
                        i, self.configuration._num_episodes))
            done = False
            sum_rewards = 0
            while not done:
                # Decide whether generating random action or predict most
                # likely from the give state.
                if np.random.random() < self.configuration._eps:
                    action = np.random.randint(0,
                                               self.configuration._num_actions)
                else:
                    action = self.predict(state)

                # Send the action to the environment and get new state,
                # reward and information on whether we've finish.
                new_state, reward, done, _ = env.step(action)
                self.learn_step(action, new_state, reward, state)
                state = new_state
                sum_rewards += reward

            avg_rewards.append(sum_rewards / self.configuration._num_episodes)
        return avg_rewards

    def learn_step(self, action, new_state, reward, state):
        """
        Fit the NN model to predict the action, given the action and
        current state.
        :param action:
        :param new_state:
        :param reward:
        :param state:
        :return:
        """
        target = reward + self.configuration._y * self.predict_value(
            new_state)
        target_vec = self.model.predict(self.onehot(state))[0]
        target_vec[action] = target
        self.model.fit(
            self.onehot(state),
            target_vec.reshape(-1, self.configuration._num_actions),
            epochs=1, verbose=0)

    def q_learn(self,
                env: Environment,
                display_strategy: bool = False,
                do_plot: bool = False) -> list:
        """
        Learns or Load an strategy to follow over a given environment,
        using RL.

        :type env: Environment
        :type do_plot: bool
        """
        # create the Keras model and learn, or load it from disk.
        if self.configuration._load_model is True:
            self.model = self.nn.load_model(self.configuration._model_file,
                                            self.configuration._weights_file)
        else:
            self.model = self.nn.create_model()
            avg_rewards = self.learn(env)

        # Extract the strategy matrix from the model.
        strategy = self.get_strategy()

        # display anything?
        if do_plot is True and self.configuration._load_model is False:
            plot.reinforcement(avg_rewards, do_plot)
        if display_strategy:
            display.strategy(self,
                             env,
                             self.model,
                             self.configuration._num_states,
                             strategy)

        return strategy
