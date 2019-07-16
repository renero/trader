#
# Largely based on:
# https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
#

from environment import Environment
from qlearning import QLearning
from trader import Trader


def main():
    trader = Trader()
    environment = Environment(trader)
    strategy = QLearning(trader).q_learn(environment)

    done = False
    state = environment.reset(debug=True)
    while not done:
        a = environment.decide(state, strategy)
        state, r, done, _ = environment.step(a)


main()
