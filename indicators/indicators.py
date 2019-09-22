import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series

from dictionary import Dictionary
from konkorde import Konkorde


def ticks_to_ohlc(input_file):
    data = pd.read_csv('acciona_2018_modelos_CP_MP_LP_MLP.csv',
                       sep=',')
    data['Fecha'] = pd.to_datetime(
        data['Fecha'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    data = data.set_index('Fecha')
    return data.resample('D').agg('last').ohlc()


def plot_comp(data, column):
    fig, ax1 = plt.subplots()
    ax1.plot(data.Price)
    ax1.grid(b=True, which='major', axis='both',
             color='lightgrey', linestyle='-', linewidth=1)
    ax1.grid(b=True, which='minor', color='#999999', linestyle='-',
             alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(data[column], 'ro--', linewidth=0.5, markersize=1)
    ax2.axhline(0, color="red")
    fig.tight_layout()
    plt.show()


def read_data(file_path, separator=','):
    data = pd.read_csv(file_path, sep=separator)
    return data


def compare(data):
    k = pd.read_csv('../data/repsol_konkorde.csv', sep=';')
    r = data.loc[25:,
        ['Price', 'marron', 'verde', 'azul']].reset_index().drop('index',
                                                                 axis=1)
    plt.plot(k.azul, 'b--', alpha=0.4)
    plt.plot(r.azul, 'k', alpha=0.4)
    plt.show()
    plt.plot(k.verde, 'g--', alpha=0.4)
    plt.plot(r.verde, 'k', alpha=0.4)
    plt.show()
    plt.plot(k.marron, 'brown', linestyle='--', alpha=0.4)
    plt.plot(r.marron, 'k', alpha=0.4)
    plt.show()


def plot_result(data):
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
    plt.figure(figsize=(12,6))
    ax1 = plt.subplot(211)
    ax1.plot(data.Price, 'k', linewidth=0.8)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # plot lines where trend changes
    for x in trend.index[trend.green == 1].values:
        ax1.axvline(x, color='green', linestyle='--', alpha=0.2)
    for x in trend.index[trend.blue == 1].values:
        ax1.axvline(x, color='blue', linestyle='-.', alpha=0.2)

    # plot green and blue
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.axhline(0, color='lightgrey')
    ax2.fill_between(range(k_size), data.verde,
                     facecolor='green', alpha=0.2)
    ax2.fill_between(range(k_size), data.azul,
                     facecolor='blue', alpha=0.2)
    ax2.plot(data.verde, 'g--', linewidth=0.6)
    ax2.plot(data.azul, 'b-.', linewidth=0.6)

    # finish job and leave, quitely...
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    configuration = Dictionary()
    input_data = read_data(configuration._input_data, configuration._separator)
    konkorde = Konkorde(configuration)

    result = konkorde.compute(input_data)
    result = konkorde.cleanup(result, start_pos=20)

    print(result.iloc[:, [0, 1, -3, -2, -1]].head(20))
    plot_result(result)
