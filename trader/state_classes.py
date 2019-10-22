from math import copysign

from portfolio import Portfolio
from rl_state import RL_State


class StateGain(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        return 'GAIN' if portfolio.gain else 'LOSE'


class StateHaveShares(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        return 'HAVE' if portfolio.have_shares else 'DONT'


class StateCanBuy(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        return 'BUY' if portfolio.can_buy else 'NOB'


class StateCanSell(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        return 'SELL' if portfolio.can_sell else 'NOS'


class StatePredUpward(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        return 'UPW' if portfolio.prediction_upward else 'DWN'


class StateKonkorde(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        return 'UPTREND' if portfolio.konkorde >= portfolio.params.k_threshold \
            else 'DOWNTREND'


class StateLastPredOk(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        sign = lambda x: copysign(1, x)
        if len(portfolio.history) < 1:
            return 'LNOK'
        prediction_sign = sign(portfolio.last_forecast - portfolio.last_price)
        actual_sign = sign(portfolio.latest_price - portfolio.last_price)
        # guess what the state, given the forecast
        if prediction_sign == actual_sign:
            return 'LOK'
        else:
            return 'LNOK'


class StatePrevLastPredOk(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        sign = lambda x: copysign(1, x)
        if len(portfolio.history) < 2:
            return 'PLNOK'
        prediction_sign = sign(
            portfolio.prevlast_forecast - portfolio.prevlast_price)
        actual_sign = sign(portfolio.last_price - portfolio.prevlast_price)
        # guess what the state, given the forecast
        if prediction_sign == actual_sign:
            return 'PLOK'
        else:
            return 'PLNOK'


class StatePrevPrevLastPredOk(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        sign = lambda x: copysign(1, x)
        if len(portfolio.history) < 3:
            return 'PPLNOK'
        pplast_forecast = portfolio.history[-2]['forecast_']
        pplast_price = portfolio.history[-2]['price_']
        prediction_sign = sign(
            pplast_forecast - pplast_price)
        actual_sign = sign(portfolio.prevlast_price - pplast_price)
        # guess what the state, given the forecast
        if prediction_sign == actual_sign:
            return 'PPLOK'
        else:
            return 'PPLNOK'
