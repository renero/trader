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
    strategy = learner.q_learn(environment, do_plot=True)

    done = False
    state = environment.reset(debug=True)
    total_reward = 0.
    while not done:
        a = environment.decide(state, strategy)
        new_state, reward, done, _ = environment.step(a)
        state = new_state
        total_reward += reward

    # Save the model?
    if trader._save_model is True:
        learner.nn.save_model(learner.model)


main()
