import matplotlib.pyplot as plt
import pandas as pd

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


def plot_result(result):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(result.Price, 'k')
    ax1.plot(result.close_m, 'k--')
    result.Day = pd.to_datetime(result.Day)
    result.set_index(result.Day)
    k_size = result['verde'].loc[19:].shape[0]
    ax2.axhline(0, color='lightgrey')
    ax2.plot(result.marron, 'brown')
    ax2.fill_between(range(k_size), result.verde.loc[19:], facecolor='green',
                     alpha=0.2)
    ax2.fill_between(range(k_size), result.azul.loc[19:], facecolor='blue',
                     alpha=0.2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    configuration = Dictionary()
    input_data = read_data(configuration._input_data, configuration._separator)
    konkorde = Konkorde(configuration)

    result = konkorde.compute(input_data)
    print(result.iloc[:, [0, 1, -3, -2, -1]].head(30))
    plot_result(result)
    # save_data(output)
