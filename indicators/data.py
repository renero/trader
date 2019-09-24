import pandas as pd


def ticks_to_ohlc(input_file):
    data = pd.read_csv('acciona_2018_modelos_CP_MP_LP_MLP.csv',
                       sep=',')
    data['Fecha'] = pd.to_datetime(
        data['Fecha'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    data = data.set_index('Fecha')
    return data.resample('D').agg('last').ohlc()


def read_data(file_path, separator=','):
    data = pd.read_csv(file_path, sep=separator)
    return data