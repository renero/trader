#
# Largely based on:
# https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/
#
# Forecast for GoldPrice comes from the code in DeepLearninginFinance:
#   https://github.com/sonaam1234/DeepLearningInFinance
#
import sys
import warnings

from agent import Agent
from environment import Environment
from rl_dictionary import RLDictionary

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    # Init
    params = RLDictionary(args=sys.argv)
    environment = Environment(params)
    agent = Agent(params)

    # Flags with the actions specified in arguments
    flag = {key: params.what_to_do == key for key in
            params.possible_actions}

    # Do something
    if flag['simulate'] is True:
        strategy = agent.q_load(environment)
        agent.simulate(environment, strategy)
    elif flag['learn'] or flag['retrain']:
        if flag['retrain']:
            agent.q_load(environment, retrain=flag['retrain'])
        strategy = agent.q_learn(environment,
                                 fresh_model=flag['learn'],
                                 do_plot=True)
        # Save the model?
        if params.save_model is True:
            agent.nn.save_model(agent.model)
        # Simulate what has been learnt with the data.
        agent.simulate(environment, strategy)
    else:  # predict
        print('Still do not know how to perform a single prediction!')
        exit(0)
