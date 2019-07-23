from portfolio import Portfolio


class RL_State:
    """
    This class is a trait to be fulfilled by every internal sub-state we want
    to define in our problem. It must implement a method that determines,
    based on the Portfolio internal state, what is its new value.
    """
    def __init__(self):
        pass

    @staticmethod
    def update_state(portfolio: Portfolio):
        pass
