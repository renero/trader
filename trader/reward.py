import math


class reward:

    def __init__(self):
        pass

    @staticmethod
    def decide(action_name, positions, initial_budget, budget):
        """ Decide what is the reward for this action """
        # return reward.direct_reward(action_name, positions)
        return reward.balance_reward(action_name, positions, initial_budget,
                                     budget)

    @staticmethod
    def direct_reward(action_name, positions):
        """ Direct reward is directly related to portfolio value """

        def sigmoid(x: float):
            return x / math.sqrt(1. + math.pow(x, 2.))

        if action_name == 'buy':
            return 0.0
        else:
            if action_name == 'wait' and positions.num_shares() == 0.:
                return -0.05
            net_value = positions.profit()
            return sigmoid(net_value)

    @staticmethod
    def balance_reward(action_name, positions, initial_budget, budget):
        """ Direct reward directly related to balance of operations """
        if action_name == 'buy':
            return 0.0
        else:
            if action_name == 'wait' and positions.num_shares() == 0.:
                return -0.05
            if action_name == 'wait' and positions.num_shares() > 0.:
                return positions.profit()
            my_balance = (budget + positions.profit() + positions.cost())
            return (my_balance - initial_budget) * 10.

    @staticmethod
    def failed() -> float:
        return -1.0
