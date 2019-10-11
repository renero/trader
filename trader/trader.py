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

    # Learn
    if params.what_to_do == 'simulate':
        strategy = agent.q_load(environment)
        agent.simulate(environment,strategy )
    elif params.what_to_do == 'learn':
        strategy = agent.q_learn(environment, do_plot=True)
        # Save the model?
        if params.save_model is True:
            agent.nn.save_model(agent.model)
        # Simulate what has been learnt with the data.
        agent.simulate(environment, strategy)
    elif params.what_to_do == 'retrain':
        print('Still do not know how to retrain an existing model!')
        exit(0)
    else:  # predict
        print('Still do not know how to performa single prediction!')
        exit(0)

