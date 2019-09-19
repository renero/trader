import pandas as pd
import matplotlib as plt

from dictionary import Dictionary


def ticks_to_ohlc(input_file):
    data = pd.read_csv('acciona_2018_modelos_CP_MP_LP_MLP.csv',
                       sep=',')
    data['Fecha'] = pd.to_datetime(
        data['Fecha'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    data = data.set_index('Fecha')
    return data.resample('D').agg('last')


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


def read_data(file_path):
    data = pd.read_csv(file_path, sep=';')
    return data


if __name__ == "__main__":
    configuration = Dictionary()
    input_data = read_data(configuration._input_data)
    konkorde = Konkorde(configuration)
    output = konkorde.compute(input_data)
    save_data(output)
