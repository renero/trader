import pandas as pd
import numpy as np


def sample_ticks():
    data = pd.DataFrame({
        'o': [50.,  80.,  10.,  80.,  10., 100.],
        'h': [100, 100., 100., 100., 100., 100.],
        'l': [00.,  00.,  00.,  00.,  00.,  00.],
        'c': [50.5, 70.,  30.,  40.,  80.,  00.],
        'v': [100, 830, 230, 660, 500, 120.]
    })
    data = pd.concat(
        [pd.DataFrame({
            'Date': pd.date_range('2020-06-01', '2020-06-06', freq='D')
        }), data
        ], axis=1)
    data = data.set_index('Date')
    return data


def encoded_cs_to_df(cs, cols):
    df = pd.DataFrame([[cs.encoded_body,
                        cs.encoded_delta_open,
                        cs.encoded_delta_high,
                        cs.encoded_delta_low,
                        cs.encoded_delta_close]],
                      columns=cols)
    return df.iloc[0]


def cs_to_df(cse, cols):
    """Returns the body element of an array of encoded candlesticks"""
    ohlc = np.array([[
        cse[i].encoded_body,
        cse[i].encoded_delta_open, cse[i].encoded_delta_high,
        cse[i].encoded_delta_low, cse[i].encoded_delta_close
    ] for i in range(len(cse))])
    return pd.DataFrame(ohlc, columns=cols)

