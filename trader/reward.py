import math


class reward:

    def __init__(self):
        pass

    @staticmethod
    def decide(action_name, positions):
        """ Decide what is the reward for this action """
        return reward.direct_reward(action_name, positions)
        # return reward.balance_reward(action_name, positions)

    @staticmethod
    def direct_reward(action_name, positions):
        """ Direct reward is directly related to portfolio value """

        if action_name == 'buy':
            return 0.0
        else:
            if action_name == 'wait' and positions.num_shares() == 0.:
                return -0.05
            if action_name == 'wait' and positions.num_shares() > 0.:
                return positions.profit() * 10.
            net_value = positions.profit()
            return reward.sigmoid(net_value * 10.)

    @staticmethod
    def balance_reward(action_name, positions):
        """ Direct reward directly related to balance of operations """
        if action_name == 'buy':
            return 0.0
        else:
            if action_name == 'wait' and positions.num_shares() == 0.:
                return -0.05
            if action_name == 'wait' and positions.num_shares() > 0.:
                return positions.profit() * 10.
            balance = (positions.budget + positions.profit() + positions.cost())
            return reward.sigmoid((balance - positions.initial_budget) * 10.)

    @staticmethod
    def failed() -> float:
        return -1.0

    @staticmethod
    def sigmoid(x: float):
        return x / math.sqrt(1. + math.pow(x, 2.))
