from math import copysign, fabs

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

# class state_portfolio_value(RL_State):
#     @staticmethod
#     def update_state(portfolio: Portfolio):
#         value = portfolio.gain
#         if value == 0:
#             value = 'EVEN'
#         elif value > 0:
#             value = 'WINN'
#         else:
#             value = 'LOSE'
#         return value
#
#
# class state_forecast(RL_State):
#     @staticmethod
#     def update_state(portfolio: Portfolio):
#         # guess what the state, given the forecast
#         if portfolio.forecast == portfolio.latest_price:
#             forecast = 'STAL'
#         elif portfolio.forecast > portfolio.latest_price:
#             forecast = 'GOUP'
#         else:
#             forecast = 'DOWN'
#         return forecast
#
#
# class state_got_shares(RL_State):
#     @staticmethod
#     def update_state(portfolio: Portfolio):
#         # Do I have shares in my portfolio?
#         if portfolio.shares > 0.:
#             shares_state = 'YESHAVE'
#         else:
#             shares_state = 'NOTHAVE'
#         return shares_state
#
#
# class state_last_prediction(RL_State):
#     @staticmethod
#     def update_state(portfolio: Portfolio):
#         sign = lambda x: copysign(1, x)
#
#         if len(portfolio.history) < 1:
#             return 'PUNKNW'
#         prediction_sign = sign(portfolio.last_forecast - portfolio.last_price)
#         actual_sign = sign(portfolio.latest_price - portfolio.last_price)
#
#         # guess what the state, given the forecast
#         if prediction_sign == actual_sign:
#             return 'PVALID'
#         else:
#             return 'PWRONG'
#
#
# class state_prevlast_pred(RL_State):
#     @staticmethod
#     def update_state(portfolio: Portfolio):
#         sign = lambda x: copysign(1, x)
#
#         if len(portfolio.history) < 2:
#             return 'PPUNKNW'
#
#         prediction_sign = sign(
#             portfolio.prevlast_forecast - portfolio.prevlast_price)
#         actual_sign = sign(portfolio.last_price - portfolio.prevlast_price)
#
#         # guess what the state, given the forecast
#         if prediction_sign == actual_sign:
#             return 'PPVALID'
#         else:
#             return 'PPWRONG'
#
#
# class state_pplast_pred(RL_State):
#     @staticmethod
#     def update_state(portfolio: Portfolio):
#         sign = lambda x: copysign(1, x)
#
#         if len(portfolio.history) < 3:
#             return 'PPPUNKNW'
#
#         pplast_forecast = portfolio.history[-2]['forecast_']
#         pplast_price = portfolio.history[-2]['price_']
#
#         prediction_sign = sign(
#             pplast_forecast - pplast_price)
#         actual_sign = sign(portfolio.prevlast_price - pplast_price)
#
#         # guess what the state, given the forecast
#         if prediction_sign == actual_sign:
#             return 'PPPVALID'
#         else:
#             return 'PPPWRONG'
#
#
# class state_stop_loss(RL_State):
#     @staticmethod
#     def update_state(portfolio: Portfolio):
#         net_value = portfolio.portfolio_value - portfolio.investment
#         stop_loss = portfolio.configuration.environment.stop_loss
#         if net_value == 0.:
#             return 'NOSTPLOS'
#         if stop_loss < 1.0:  # percentage of initial budget
#             if (net_value / portfolio.initial_budget) < 0.0 and \
#                     fabs(net_value / portfolio.initial_budget) >= stop_loss:
#                 value = 'STOPLOSS'
#             else:
#                 value = 'NOSTPLOS'
#         else:  # actual value
#             if net_value < stop_loss:
#                 value = 'STOPLOSS'
#             else:
#                 value = 'NOSTPLOS'
#         return value
#
#
# class state_what_can_do(RL_State):
#     @staticmethod
#     def update_state(portfolio: Portfolio):
#         can_buy = 'CB' if portfolio.can_buy else 'NB'
#         can_sell = 'CS' if portfolio.can_sell else 'NS'
#         return can_buy + can_sell
