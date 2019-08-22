from math import fabs

from portfolio import Portfolio
from rl_state import RL_State


class state_stop_loss(RL_State):

    @staticmethod
    def update_state(portfolio: Portfolio):
        net_value = portfolio.portfolio_value - portfolio.investment
        stop_loss = portfolio.configuration._environment._stop_loss
        if net_value == 0.:
            return 'NOSTPLOS'
        if stop_loss < 1.0:  # percentage of initial budget
            if (net_value / portfolio.initial_budget) < 0.0 and \
                    fabs(net_value / portfolio.initial_budget) >= stop_loss:
                value = 'STOPLOSS'
            else:
                value = 'NOSTPLOS'
        else:  # actual value
            if net_value < stop_loss:
                value = 'STOPLOSS'
            else:
                value = 'NOSTPLOS'
        return value
