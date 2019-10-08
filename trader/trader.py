#
# Largely based on:
# https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
#
# Forecast for GoldPrice comes from the code in DeepLearninginFinance:
#   https://github.com/sonaam1234/DeepLearningInFinance
#

import warnings

from agent import Agent
from environment import Environment
from rl_dictionary import RLDictionary

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    # Init
    params = RLDictionary()
    environment = Environment(params)
    agent = Agent(params)

    # Learn
    params.debug = True
    strategy = agent.q_learn(environment, do_plot=True)
    params.debug = False

    # Test
    done = False
    total_reward = 0.
    params.debug = True
    state = environment.reset()
    while not done:
        action = environment.decide_next_action(state, strategy)
        next_state, reward, done, _ = environment.step(action)
        total_reward += reward
        state = next_state

    params.display.results(environment.portfolio, do_plot=True)

    # Save the model?
    if params.save_model is True:
        agent.nn.save_model(agent.model)
