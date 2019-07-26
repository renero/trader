from math import copysign

from common import Common
from portfolio import Portfolio
from rl_state import RL_State


class state_last_prediction(RL_State, Common):

    @staticmethod
    def update_state(portfolio: Portfolio):
        sign = lambda x: copysign(1, x)

        if len(portfolio.history) < 1:
            return 'PRED_UNK'
        prediction_sign = sign(portfolio.last_forecast - portfolio.last_price)
        #    portfolio.history[-1]['forecast_'] - portfolio.history[-1][
        #        'price_'])
        actual_sign = sign(portfolio.latest_price - portfolio.last_price)
        #    portfolio.latest_price - portfolio.history[-1]['price_'])

        # print('latest Price:{}, pred:{}, actual price:{}, {}/{}'.format(
        #     portfolio.history[-1]['price_'],
        #     portfolio.history[-1]['forecast_'],
        #     portfolio.latest_price,
        #     prediction_sign,
        #     actual_sign
        # ))

        # guess what the state, given the forecast
        if prediction_sign == actual_sign:
            return 'PRED_OK'
        else:
            return 'PRED_NOK'
