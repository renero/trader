from os.path import dirname, realpath

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from file_io import file_exists


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


def read_ohlc(conf):
    filepath = file_exists(conf.input_data, dirname(realpath(__file__)))
    df = pd.read_csv(filepath, delimiter=conf.separator)
    # Reorder and rename
    df = df[[conf.csv_dict['date'],
             conf.csv_dict['open'], conf.csv_dict['high'],
             conf.csv_dict['low'], conf.csv_dict['close'],
             conf.csv_dict['volume']]]
    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    info_msg = 'Read file: {}, output DF dim{}'
    print(info_msg.format(filepath, df.shape))
    return df


def save_indicator(indicator, scale=True):
    """
    Save the index columns passed as argument.
    :param indicator:
    :param scale:
    :return:
    """
    scaler = MinMaxScaler(feature_range=(-1., 1.)).fit(indicator)
    print(indicator.head())
