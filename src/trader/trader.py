import sys
import warnings

from agent import Agent
from environment import Environment
from rl_dictionary import RLDictionary

warnings.simplefilter(action='ignore', category=FutureWarning)


def main(argv) -> int:
    # Read parameters and arguments
    return_value = 0
    params = RLDictionary(args=argv)

    # Set flag with the action specified in arguments
    flag = {key: params.what_to_do == key for key in
            params.possible_actions}

    # Initialize the environment
    environment = Environment(params, flag['train'])
    agent = Agent(params, environment)

    # Do something
    if flag['train'] or flag['retrain']:
        if flag['retrain']:
            agent.q_load(retrain=flag['retrain'])
            params.epsilon = params.epsilon_min
        agent.q_learn(fresh_model=flag['train'])
    else:
        # simulate or predict
        if flag['simulate']:
            strategy = agent.q_load()
            return_value = agent.simulate(strategy)
        else:  # predict.
            strategy = agent.q_load()
            return_value = agent.single_step(strategy)

    return return_value


if __name__ == "__main__":
    sys.exit(main(sys.argv))
