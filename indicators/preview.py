import pandas as pd
from matplotlib import pyplot as plt


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


