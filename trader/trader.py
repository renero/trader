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


def main(argv):
    # Init
    params = RLDictionary(args=argv)
    environment = Environment(params)
    agent = Agent(params)

    # Flags with the actions specified in arguments
    flag = {key: params.what_to_do == key for key in
            params.possible_actions}

    # Do something
    if flag['train'] or flag['retrain']:
        if flag['retrain']:
            agent.q_load(environment, retrain=flag['retrain'])
            params.epsilon = params.epsilon_min
        strategy = agent.q_learn(environment,
                                 fresh_model=flag['train'],
                                 do_plot=params.do_plot)
        # Save the model?
        if params.save_model is True:
            agent.nn.save_model(agent.model, environment.memory.results)
        # Simulate what has been learnt with the data.
        agent.simulate(environment, strategy)
    else:
        # simulate or predict
        if flag['simulate']:
            last_prediction_only = (flag['predict'] == True)
            strategy = agent.q_load(environment)
            agent.simulate(environment, strategy)
        else:  # predict.
            strategy = agent.q_load(environment)
            agent.single_step(environment, strategy)


if __name__ == "__main__":
    main(sys.argv)
