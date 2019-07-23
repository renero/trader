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
from qlearning import QLearning
from trader import Trader

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    trader = Trader()
    environment = Environment(trader)
    learner = QLearning(trader)
    strategy = learner.q_learn(environment)

    done = False
    total_reward = 0.
    environment.debug = True
    environment.portfolio_.debug = True
    state = environment.reset()
    while not done:
        action = environment.decide_next_action(state, strategy)
        state, reward, done, _ = environment.step(action)
        total_reward += reward

    # Save the model?
    if trader.save_model is True:
        learner.nn.save_model(learner.model)


main()
