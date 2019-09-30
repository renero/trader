from pandas import DataFrame

from base_indicators import *


class Konkorde(object):
    """
    Compute the actual konkorde indicator.
    FIRST actionable index value after 25 observations.

    pvi = PositiveVolumeIndex(close)
    pvim = ExponentialAverage[m](pvi)
    pvimax = highest[90](pvim)
    pvimin = lowest[90](pvim)
    oscp = (pvi - pvim) * 100/ (pvimax - pvimin)
    nvi = NegativeVolumeIndex(close)
    nvim = ExponentialAverage[m](nvi)
    nvimax = highest[90](nvim)
    nvimin = lowest[90](nvim)
    xmf = MoneyFlowIndex[14]
    OB1 = (BollingerUp[25](TotalPrice) + BollingerDown[25](TotalPrice)) / 2
    OB2 = BollingerUp[25](TotalPrice) - BollingerDown[25](TotalPrice)
    BollOsc = ((TotalPrice - OB1) / OB2 ) * 100
    xrsi = rsi [14](TotalPrice)
    STOC = Stochastic[21,3](TotalPrice)

    azul = (nvi - nvim) * 100/ (nvimax - nvimin)
    marron = (xrsi + xmf + BollOsc + (STOC / 3))/2
    verde = marron + oscp
    media = ExponentialAverage[m](marron)
    bandacero= 0

    return
    verde COLOURED(102,255,102) as "verde",
    marron COLOURED(255,204,153) as "marron",
    marron COLOURED(51,0,0) as "lmarron",
    azul COLOURED(0,255,255) as "azul",
    verde COLOURED(0,102,0) as "lineav",
    azul COLOURED(0,0,102) as "lazul",
    media COLOURED(255,0,0) as "media",
    bandacero COLOURED(0,0,0) as "cero"
    """

    def __init__(self, configuration):
        self.configuration = configuration

    @staticmethod
    def compute(input_data: DataFrame) -> DataFrame:
        data = input_data.copy(deep=True)

        data['close_m'] = data.close.rolling(10).mean()
        data['pvi'] = positive_volume_index(data.close, data.volume)
        data['pvim'] = ewma(data['pvi'], alpha=0.1)
        data['nvi'] = negative_volume_index(data.close, data.volume)
        data['nvim'] = ewma(data['nvi'], alpha=0.1)
        data['oscp'] = oscp(data['pvi'], data['pvim'])
        data['mfi'] = money_flow_index(data.high, data.low,
                                       data.close, data.volume)
        data['b_up'] = bollinger_band(data.close, 'up')
        data['b_down'] = bollinger_band(data.close, 'down')
        data['b_osc'] = b_osc(data.close, data['b_up'], data['b_down'])
        data['rsi'] = rsi(data.close)
        data['stoch'] = stoch_osc(data.high, data.low, data.close)

        data['marron'] = (data['rsi'] + data['mfi'] + data['b_osc'] + (
                data['stoch'] / 3.)) / 2.
        data['verde'] = data['marron'] + data['oscp']
        data['azul'] = (data['nvi'] - data['nvim']) * 100. / (
                data['nvi'].max() - data['nvi'].min())

        # data.date = pd.to_datetime(data.date)
        # data.set_index(data.iloc[:, 0].name)

        return data

    @staticmethod
    def cleanup(data):
        cols = ['close', 'marron', 'verde', 'azul']
        shift = data.apply(pd.Series.first_valid_index)
        r = data[cols]
        return r
