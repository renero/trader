from portfolio import Portfolio
from rl_state import RL_State


class state_forecast(RL_State):

    @staticmethod
    def update_state(portfolio: Portfolio):
        # guess what the state, given the forecast
        if portfolio.forecast == portfolio.latest_price:
            forecast = 'EVEN'
        elif portfolio.forecast > portfolio.latest_price:
            forecast = 'WIN'
        else:
            forecast = 'LOSE'
        return forecast
