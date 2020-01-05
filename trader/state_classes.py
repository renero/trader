from math import copysign

from portfolio import Portfolio
from rl_state import RL_State


class StatePriceTrend(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        if portfolio.latest_price >= portfolio.last_price:
            portfolio.log.debug('  ↗︎ price trend up')
            return 'UP'
        portfolio.log.debug('  ↘︎ price trend down')
        return 'DW'


class StatePrevPriceTrend(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        if portfolio.last_price >= portfolio.prevlast_price:
            portfolio.log.debug('  ↗︎ prev. price trend up')
            return 'UP'
        portfolio.log.debug('  ↘︎ prev. price trend down')
        return 'DW'


class StatePrevPrevPriceTrend(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        if portfolio.prevlast_price >= portfolio.prevprevlast_price:
            portfolio.log.debug('  ↗︎ prev. prev. price trend up')
            return 'UP'
        portfolio.log.debug('  ↘︎ prev.prev. price trend down')
        return 'DW'


class StateGain(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        return 'GN' if portfolio.gain else 'LS'


class StateLastGain(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        return 'GN' if portfolio.last_gain else 'LS'


class StatePrevLastGain(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        return 'GN' if portfolio.prev_last_gain else 'LS'


class StateHaveShares(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        return 'HV' if portfolio.have_shares else 'DH'


class StateCanBuy(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        return 'BY' if portfolio.can_buy else 'NB'


class StateCanSell(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        return 'SL' if portfolio.can_sell else 'NS'


class StatePredUpward(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        if portfolio.prediction_upward:
            portfolio.log.debug('  ↗︎ prediction trends price raise')
            return 'UP'
        else:
            portfolio.log.debug('  ↘︎ prediction trends price down')
            return 'DW'


class StateLastPredOk(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        log = portfolio.log
        sign = lambda x: copysign(1, x)
        if portfolio.memory.len < 1:
            log.debug('  Not enough history to check last prediction')
            return 'PI'
        prediction_sign = sign(portfolio.last_forecast - portfolio.last_price)
        actual_sign = sign(portfolio.latest_price - portfolio.last_price)
        # guess what the state, given the forecast
        if prediction_sign == actual_sign:
            log.debug('  ✓ Last prediction was OK')
            return 'PC'
        else:
            log.debug('  ✕ Last prediction was NOT OK')
            return 'PI'


class StatePrevLastPredOk(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        log = portfolio.log
        sign = lambda x: copysign(1, x)
        if portfolio.memory.len < 2:
            log.debug('  Not enough history to check prev. last prediction')
            return 'PI'

        prediction_sign = sign(
            portfolio.prevlast_forecast - portfolio.prevlast_price)
        actual_sign = sign(portfolio.last_price - portfolio.prevlast_price)
        # guess what the state, given the forecast
        if prediction_sign == actual_sign:
            log.debug('  ✓ Prev. last prediction was OK')
            return 'PC'
        else:
            log.debug('  ✕ Prev. last prediction was NOT OK')
            return 'PI'


class StatePrevPrevLastPredOk(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        log = portfolio.log
        sign = lambda x: copysign(1, x)
        if portfolio.memory.len < 3:
            log.debug('  Not enough history to check prev.prev.last prediction')
            return 'PI'

        prediction_sign = sign(
            portfolio.prevprevlast_forecast - portfolio.prevprevlast_price)
        actual_sign = sign(
            portfolio.prevlast_price - portfolio.prevprevlast_price)
        # guess what the state, given the forecast
        if prediction_sign == actual_sign:
            log.debug('  ✓ Prev. prev. last prediction was OK')
            return 'PC'
        else:
            log.debug('  ✕ Prev. prev. last prediction was NOT OK')
            return 'PI'


class StateKonkorde(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        return 'KP' if portfolio.konkorde >= portfolio.params.k_threshold \
            else 'KN'


class StateKTrend(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        if portfolio.konkorde > portfolio.memory.last('konkorde'):
            return 'KU'
        else:
            return 'KD'


class StateLastKTrend(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        if portfolio.memory.last('konkorde') > portfolio.memory.prevlast(
                'konkorde'):
            return 'KU'
        else:
            return 'KD'


class StatePrevLastKTrend(RL_State):
    @staticmethod
    def update_state(portfolio: Portfolio):
        if portfolio.memory.prevlast(
                'konkorde') > portfolio.memory.prevprevlast('konkorde'):
            return 'KU'
        else:
            return 'KD'
