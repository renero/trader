from math import copysign

from common import Common
from portfolio import Portfolio
from rl_state import RL_State


class state_pplast_pred(RL_State):

    @staticmethod
    def update_state(portfolio: Portfolio):
        sign = lambda x: copysign(1, x)

        if len(portfolio.history) < 3:
            return 'PPPUNKNW'

        pplast_forecast = portfolio.history[-2]['forecast_']
        pplast_price = portfolio.history[-2]['price_']

        prediction_sign = sign(
            pplast_forecast - pplast_price)
        actual_sign = sign(portfolio.prevlast_price - pplast_price)

        # guess what the state, given the forecast
        if prediction_sign == actual_sign:
            return 'PPPVALID'
        else:
            return 'PPPWRONG'
