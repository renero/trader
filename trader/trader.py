#
# Largely based on:
# https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
#
# Forecast for GoldPrice comes from the code in DeepLearninginFinance:
#   https://github.com/sonaam1234/DeepLearningInFinance
#

import os

import tensorflow as tf

from environment import Environment
from rl_dictionary import RLDictionary
from qlearning import QLearning

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    # Init
    configuration = RLDictionary()
    environment = Environment(configuration)
    learner = QLearning(configuration)

    # Learn
    configuration.debug = True
    strategy = learner.q_learn(environment, do_plot=True)
    configuration.debug = False

    # Test
    done = False
    total_reward = 0.
    configuration.debug = True
    state = environment.reset()
    while not done:
        action = environment.decide_next_action(state, strategy)
        state, reward, done, _ = environment.step(action)
        total_reward += reward

    configuration.display.results(environment.portfolio_, do_plot=True)

    # Save the model?
    if configuration.save_model is True:
        learner.nn.save_model(learner.model)


main()
