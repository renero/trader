#
# Largely based on:
# https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
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
    strategy = QLearning(trader).q_learn(environment)

    done = False
    state = environment.reset(debug=True)
    while not done:
        a = environment.decide(state, strategy)
        new_state, r, done, _ = environment.step(a)
        state = new_state


main()
