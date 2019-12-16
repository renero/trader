import sys
import warnings

from agent import Agent
from environment import Environment
from rl_dictionary import RLDictionary

warnings.simplefilter(action='ignore', category=FutureWarning)


def main(argv) -> int:
    # Init
    return_value = 0
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
        agent.q_learn(environment, fresh_model=flag['train'])
    else:
        # simulate or predict
        if flag['simulate']:
            strategy = agent.q_load(environment)
            return_value = agent.simulate(environment, strategy)
        else:  # predict.
            strategy = agent.q_load(environment)
            return_value = agent.single_step(environment, strategy)

    return return_value


if __name__ == "__main__":
    sys.exit(main(sys.argv))
