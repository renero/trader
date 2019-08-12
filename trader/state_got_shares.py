from portfolio import Portfolio
from rl_state import RL_State


class state_got_shares(RL_State):

    @staticmethod
    def update_state(portfolio: Portfolio):
        # Do I have shares in my portfolio?
        if portfolio.shares > 0.:
            shares_state = 'YESHAVE'
        else:
            shares_state = 'NOTHAVE'
        return shares_state
