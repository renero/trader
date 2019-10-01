import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Series


def trend_lines(ax1, trend, **kwargs):
    # plot lines where trend changes
    for x in trend.index[trend.green == 1].values:
        ax1.axvline(x, color='green', linestyle='--', alpha=0.2, **kwargs)
    for x in trend.index[trend.blue == 1].values:
        ax1.axvline(x, color='blue', linestyle='-.', alpha=0.2, **kwargs)


def plot_konkorde(data_series):
    data = data_series.copy(deep=True)
    data = data.reset_index(drop=True)
    k_size = data['verde'].shape[0]

    def trends(data: Series) -> Series:
        s = data.rolling(2).apply(
            lambda x: 1 if np.sign(x.iloc[0]) != np.sign(x.iloc[1]) else 0,
            raw=False)
        s.iloc[0] = 0.
        return s

    trend = pd.DataFrame(
        {'green': trends(data.verde), 'blue': trends(data.azul)})

    # plot stock price
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(211)
    ax1.plot(data.close, 'k', linewidth=0.8)
    plt.setp(ax1.get_xticklabels(), visible=False)
    trend_lines(ax1, trend)

    # plot green and blue
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.axhline(0, color='lightgrey')
    ax2.fill_between(range(k_size), data.verde,
                     facecolor='green', alpha=0.2)
    ax2.fill_between(range(k_size), data.azul,
                     facecolor='blue', alpha=0.2)
    ax2.plot(data.verde, 'g--', linewidth=0.6)
    ax2.plot(data.azul, 'b-.', linewidth=0.6)
    trend_lines(ax2, trend, linewidth=0.5)

    # finish job and leave, quitely...
    plt.tight_layout()
    plt.show()