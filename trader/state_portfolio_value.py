from portfolio import Portfolio
from rl_state import RL_State


class state_portfolio_value(RL_State):

    @staticmethod
    def update_state(portfolio: Portfolio):
        if portfolio.budget == portfolio.initial_budget:
            value = 'EVEN'
        elif portfolio.budget > portfolio.initial_budget:
            value = 'WIN'
        else:
            value = 'LOSE'
        return value
