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
        if portfolio.prediction_upward:
            portfolio.log.debug('  ↗︎ prediction trends price raise')
            return 'UPW'
        else:
            portfolio.log.debug('  ↘︎ prediction trends price down')
            return 'DWN'


class StateKonkorde(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        return 'UPTREND' if portfolio.konkorde >= portfolio.params.k_threshold \
            else 'DOWNTREND'


class StateLastPredOk(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        log = portfolio.log
        sign = lambda x: copysign(1, x)
        if portfolio.memory.len < 1:
            log.debug('  Not enough history to check last prediction')
            return 'LNOK'
        prediction_sign = sign(portfolio.last_forecast - portfolio.last_price)
        actual_sign = sign(portfolio.latest_price - portfolio.last_price)
        # guess what the state, given the forecast
        if prediction_sign == actual_sign:
            log.debug(' ✓ Last prediction was OK')
            return 'LOK'
        else:
            log.debug(' ✕ Last prediction was NOT OK')
            return 'LNOK'


class StatePrevLastPredOk(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        log = portfolio.log
        sign = lambda x: copysign(1, x)
        if portfolio.memory.len < 2:
            log.debug('  Not enough history to check prev. last prediction')
            return 'PLNOK'

        prediction_sign = sign(
            portfolio.prevlast_forecast - portfolio.prevlast_price)
        actual_sign = sign(portfolio.last_price - portfolio.prevlast_price)
        # guess what the state, given the forecast
        if prediction_sign == actual_sign:
            log.debug(' ✓ Prev. last prediction was OK')
            return 'PLOK'
        else:
            log.debug(' ✕ Prev. last prediction was NOT OK')
            return 'PLNOK'


class StatePrevPrevLastPredOk(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        log = portfolio.log
        sign = lambda x: copysign(1, x)
        if portfolio.memory.len < 3:
            log.debug('  Not enough history to check prev.prev.last prediction')
            return 'PPLNOK'

        prediction_sign = sign(
            portfolio.prevprevlast_forecast - portfolio.prevprevlast_price)
        actual_sign = sign(
            portfolio.prevlast_price - portfolio.prevprevlast_price)
        # guess what the state, given the forecast
        if prediction_sign == actual_sign:
            log.debug(' ✓ Prev. prev. last prediction was OK')
            return 'PPLOK'
        else:
            log.debug(' ✕ Prev. prev. last prediction was NOT OK')
            return 'PPLNOK'
