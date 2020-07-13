import sys

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from agent import Agent
from environment import Environment
from rl_dictionary import RLDictionary

if __name__ == "__main__":
    # Init
    params = RLDictionary(args=sys.argv)
    environment = Environment(params)
    agent = Agent(params)

    # define the space of hyperparameters to search
    search_space = [Real(-1., +1., name='reward_do_nothing'),
                    Real(-1., +1., name='reward_success_buy'),
                    Real(-1., +1., name='reward_positive_sell'),
                    Real(-1., +1., name='reward_negative_sell'),
                    Real(-1., +1., name='reward_failed_buy'),
                    Real(-1., +1., name='reward_failed_sell')]

    # define the function used to evaluate a given configuration
    @use_named_args(search_space)
    def evaluate_model(**new_params):
        # something
        for k in new_params:
            params.log.info('Setting {} to {:.2f}'.format(k, new_params[k]))
            params.environment[k] = new_params[k]

        # calculate the output metric from the model
        strategy = agent.q_learn(environment, fresh_model=True, do_plot=False)
        portfolio = environment.portfolio
        total = portfolio.budget + portfolio.portfolio_value
        estimate = total / (portfolio.budget * 1.5)
        params.log.info('Estimate got (not inverse): {:.3f}'.format(estimate))
        return 1.0 - estimate


    # perform optimization
    result = gp_minimize(evaluate_model, search_space)
    # summarizing finding:
    params.log.info('Best Performance: %.3f' % (1.0 - result.fun))
    params.log.info('Best Parameters:', result.x)
    for i in range(6):
        print(result.x[i])
