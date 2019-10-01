import errno
import os
from os.path import dirname, realpath, join
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


def file_exists(given_filepath: str, my_dir: str) -> str:
    """
    Check if the file exists as specified in argument, or try to find
    it using the local path of the script
    :param given_filepath:
    :return: The path where the file is or None if it couldn't be found
    """
    if os.path.exists(given_filepath) is True:
        filepath = given_filepath
    else:
        new_filepath = os.path.join(my_dir, given_filepath)
        if os.path.exists(new_filepath) is True:
            filepath = new_filepath
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), new_filepath)
    return filepath


def valid_output_name(filename: str, path: str, extension=None) -> str:
    """
    Builds a valid name.
    Returns The filename if the name is valid and file does not exists,
            None otherwise.
    """
    path = file_exists(path, dirname(realpath(__file__)))
    if extension:
        base_filepath = join(path, filename) + '.{}'.format(extension)
    else:
        base_filepath = join(path, filename)
    output_filepath = base_filepath
    idx = 1
    while Path(output_filepath).is_file() is True:
        if extension:
            output_filepath = join(
                path, filename) + '_{:d}.{}'.format(
                    idx, extension)
        else:
            output_filepath = join(path, filename + '_{}'.format(idx))
        idx += 1

    return output_filepath


def save_dataframe(name: str,
                   df: DataFrame,
                   output_path: str,
                   cols_to_scale: list = None):
    """
    Save the data frame passed, with a valid output name in the output path
    scaling the columns specified, if applicable.
    :param name:
    :param df:
    :param output_path:
    :param cols_to_scale: array with the names of the columns to scale
    :return:
    """
    data = df.copy()
    file_name = valid_output_name(name, output_path, 'csv')
    if cols_to_scale is not None:
        scaler = MinMaxScaler(feature_range=(-1., 1.))
        data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
    data.to_csv(file_name)


def read_ohlc(filename: str, separator: str, csv_dict: dict) -> DataFrame:
    """
    Reads a filename passed as CSV, renaming columns according to the
    dictionary passed.
    :param filename:
    :param separator:
    :param csv_dict:
    :return:
    """
    filepath = file_exists(filename, dirname(realpath(__file__)))
    df = pd.read_csv(filepath, delimiter=separator)

    # Reorder and rename
    columns_order = [csv_dict[colname] for colname in csv_dict]
    df = df[columns_order]
    df.columns = csv_dict.keys()

    # Set index to date field
    df.date = pd.to_datetime(df.date)
    df = df.set_index(df.iloc[:, 0].name)

    info_msg = 'Read file: {}, output DF dim{}'
    print(info_msg.format(filepath, df.shape))

    return df