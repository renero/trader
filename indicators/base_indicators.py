"""
Copyright, J. Renero, 2019
"""

import numpy as np
import pandas as pd
from pandas import Series


def positive_volume_index(close: Series, volume: Series, start_pos: int = 6):
    """
    Positive Volume Index (PVI)
    Source: https://www.equities.com/news/the-secret-to-the-positive-volume-index
    """
    pvi = pd.Series(index=close.index, dtype='float64', name='pvi')
    pvi.iloc[:] = 1000.0
    price_chg = close.pct_change()
    vol_change = volume.pct_change()
    for i in range(start_pos, len(pvi)):
        if vol_change.iloc[i] > 0:
            pvi.iloc[i] = pvi.iloc[i - 1] + (
                    price_chg.iloc[i] * 100.)
        else:
            pvi.iloc[i] = pvi.iloc[i - 1]
    return pvi


def negative_volume_index(close: Series, volume: Series, start_pos: int = 1):
    """
    Negative Volume Index (PVI)
    Source: https://www.equities.com/news/the-secret-to-the-positive-volume-index
    """
    nvi = pd.Series(index=close.index, dtype='float64', name='nvi')
    nvi.iloc[:] = 1000.0
    price_chg = close.pct_change()
    vol_change = volume.pct_change()
    for i in range(start_pos, len(nvi)):
        if vol_change.iloc[i] < 0:
            nvi.iloc[i] = nvi.iloc[i - 1] + (
                    price_chg.iloc[i] * 100.)
        else:
            nvi.iloc[i] = nvi.iloc[i - 1]
    return nvi


def ewma(data, alpha, offset=None, dtype=None) -> Series:
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    out = np.empty_like(data, dtype=dtype)

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha,
                               np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    series_out = pd.Series(out)
    return series_out


def oscp(pvi: Series, pvim: Series) -> Series:
    min_pvi = pvi.min()
    max_pvi = pvi.max()
    return ((pvi - pvim) * 100.) / (max_pvi - min_pvi)


def money_flow_index(high, low, close, volume, n=14, fillna=False):
    """Money Flow Index (MFI)
    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """

    # 0 Prepare dataframe to work
    df = pd.DataFrame([high, low, close, volume]).T
    df.columns = ['High', 'Low', 'Close', 'Volume']

    # 1 typical price
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0

    # 2 up or down column
    df['Up_or_Down'] = 0
    df.loc[(tp > tp.shift(1)), 'Up_or_Down'] = 1
    df.loc[(tp < tp.shift(1)), 'Up_or_Down'] = -1

    # 3 money flow
    mf = tp * df['Volume'] * df['Up_or_Down']

    # 4 positive and negative money flow with n periods
    n_positive_mf = mf.rolling(n).apply(
        lambda x: np.sum(np.where(x >= 0.0, x, 0.0)),
        raw=True)
    n_negative_mf = abs(mf.rolling(n).apply(
        lambda x: np.sum(np.where(x < 0.0, x, 0.0)),
        raw=True))

    # 5 money flow index
    mr = n_positive_mf / n_negative_mf
    mr = (100 - (100 / (1 + mr)))

    if fillna:
        mr = mr.replace([np.inf, -np.inf], np.nan).fillna(50)

    return pd.Series(mr, name='mfi')


def bollinger_band(close, which, n=20):
    assert which is 'up' or which is 'down', \
        "Parameter which can only be 'up' or 'down'"
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n, min_periods=0).std()
    if which is 'up':
        band = mavg + (2. * mstd)
    else:
        band = mavg - (2. * mstd)
    return pd.Series(band, name='bollinger_{}'.format(which))


def b_osc(close: Series, bup, bdown, n=20) -> Series:
    """
    Bollinger Oscillator.
    OB1 = (BollingerUp[25](TotalPrice) + BollingerDown[25](TotalPrice)) / 2
    OB2 = BollingerUp[25](TotalPrice) - BollingerDown[25](TotalPrice)
    BollOsc = ((TotalPrice - OB1) / OB2 ) * 100
    """
    b = pd.DataFrame(columns=['close', 'b_avg, b_std'])
    b['close'] = close
    b['b_up'] = bup
    b['b_down'] = bdown
    b['b1'] = (b['b_up'] + b['b_down']) / 2.
    b['b2'] = b['b_up'] - b['b_down']
    b['b_osc'] = ((b['close'] - b['b1']) / b['b2']) * 100.

    return pd.Series(b['b_osc'], name='b_osc')


def rsi(close, window_size=14):
    """ rsi indicator """
    # Get the difference in price from previous step
    delta = close.diff()
    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    # delta = delta[1:]

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the SMA
    roll_up = up.rolling(window_size).mean()
    roll_down = down.abs().rolling(window_size).mean()

    # Calculate the RSI based on SMA
    rs = roll_up / roll_down
    return 100.0 - (100.0 / (1.0 + rs))


def stoch_osc(high, low, close, window_size=14, fillna=False):
    """
    Stochastic Oscillator Index.
    """
    smin = low.rolling(window_size, min_periods=window_size).min()
    smax = high.rolling(window_size, min_periods=window_size).max()
    stoch_k = 100 * (close - smin) / (smax - smin)

    if fillna:
        stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan).fillna(50)

    return pd.Series(stoch_k, name='stoch_k')
