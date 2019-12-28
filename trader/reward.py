import math


class reward:

    def __init__(self):
        pass

    @staticmethod
    def decide(action_name, positions):
        """ Decide what is the reward for this action """
        return reward.direct_reward(action_name, positions)

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
    def failed(action_name: str) -> float:
        return -1.0
