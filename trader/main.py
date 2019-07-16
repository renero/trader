#
# Largely based on:
# https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
#

from environment import *
from qlearning import QLearning
from trader import Trader

value_states = ['EVEN', 'WIN', 'LOSE']
forecast_states = ['EVEN', 'WIN', 'LOSE']
share_states = ['HAVE', 'DONTHAVE']


def main():
    trader = Trader()
    env = Environment(trader).initialize(
        [value_states, forecast_states, share_states], debug=True)
    strategy = QLearning(trader).q_learn(env)

    done = False
    state = env.reset(debug=True)
    while not done:
        a = env.decide(state, strategy)
        new_state, r, done, _ = env.step(a)


main()
