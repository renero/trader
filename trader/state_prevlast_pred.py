from math import copysign

from portfolio import Portfolio
from rl_state import RL_State


class state_prevlast_pred(RL_State):

    @staticmethod
    def update_state(portfolio: Portfolio):
        sign = lambda x: copysign(1, x)

        if len(portfolio.history) < 2:
            return 'PPUNKNW'

        prediction_sign = sign(
            portfolio.prevlast_forecast - portfolio.prevlast_price)
        actual_sign = sign(portfolio.last_price - portfolio.prevlast_price)

        # guess what the state, given the forecast
        if prediction_sign == actual_sign:
            return 'PPVALID'
        else:
            return 'PPWRONG'
