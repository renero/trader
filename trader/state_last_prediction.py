from math import copysign

from common import Common
from portfolio import Portfolio
from rl_state import RL_State


class state_last_prediction(RL_State, Common):

    @staticmethod
    def update_state(portfolio: Portfolio):
        sign = lambda x: copysign(1, x)

        if len(portfolio.history) < 1:
            return 'PUNKNW'
        prediction_sign = sign(portfolio.last_forecast - portfolio.last_price)
        actual_sign = sign(portfolio.latest_price - portfolio.last_price)

        # guess what the state, given the forecast
        if prediction_sign == actual_sign:
            return 'PVALID'
        else:
            return 'PWRONG'
