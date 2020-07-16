import pandas as pd


def sample_ticks():
    data = pd.DataFrame({
        'o': [50.,   80.,  10.,  80.,  10.,  100.],
        'h': [100,  100., 100., 100., 100.,  100.],
        'l': [00.,   00.,  00.,  00.,  00.,   00.],
        'c': [50.5,  70.,  30.,  40.,  80.,   00.],
        'v': [100,   830,  230,  660,  500,  120.]
    })
    data = pd.concat(
        [pd.DataFrame({
            'Date': pd.date_range('2020-06-01', '2020-06-06', freq='D')
        }), data
        ], axis=1)
    data = data.set_index('Date')
    return data
