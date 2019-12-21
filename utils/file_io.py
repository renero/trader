import errno
import json
import os
from os.path import dirname, realpath, join, splitext, basename
from pathlib import Path
from typing import Optional

import joblib
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
    Builds a valid name. In case there's another file which already exists
    adds a number (1, 2, ...) until finds a valid filename which does not
    exist.
    Returns The filename if the name is valid and file does not exists,
            None otherwise.
    :param filename: The base filename to be set.
    :param path: The path where trying to set the filename
    :param extension: The extension of the file, without the dot '.'
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
                   cols_to_scale: list = None,
                   scaler_name: str = None,
                   index: bool = True) -> (str, str):
    """
    Save the data frame passed, with a valid output name in the output path
    scaling the columns specified, if applicable.

    :param name: the name to be used to save the df to file
    :param df: the dataframe
    :param output_path: the path where the df is to be saved
    :param cols_to_scale: array with the names of the columns to scale
    :param scaler_name: baseName of the file where saving the scaler used.
    :param index: save index in the csv

    :return: the full path of the file saved
    """
    data = df.copy()
    file_name = valid_output_name(name, output_path, 'csv')
    scaler_name = scale_columns(data, cols_to_scale, scaler_name, output_path)
    data.round(2).to_csv(file_name, index=index)

    return file_name, scaler_name


def scale_columns(df,
                  cols_to_scale,
                  train_mode,
                  fcast_filename,
                  path):
    """
    Scale columns from a dataframe, using a MinMaxScaler. If the parameter
    train_mode is set to True, then a new scaler is fit and saved, otherwise
    the specified path with and given base filename are used to load and use it

    :param df:              the dataframe
    :param cols_to_scale:   the columns in the DF to be scaled
    :param train_mode:      if true the scaler is created and saved, otherwise
                            the scaler is loaded using the path and name
                            contained in the following arguments.
    :param fcast_filename:  the basename used to store the serialized scaler
    :param path:            the path where the scaler is to be saved

    :return: The name of the scaler serialized with joblib and saved
    """
    if cols_to_scale is not None:
        if train_mode is True:
            # scale the columns and save the scaler
            scaler = MinMaxScaler(feature_range=(0., 1.))
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
            scaler_name = ''
        else:
            # Load the scaler and use it
            base_name = splitext(basename(fcast_filename))[0]
            scaler_name = join(path, 'scaler_{}.pickle'.format(base_name))
            scaler = joblib.load(scaler_name)
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        return scaler, scaler_name
    return None


def read_ohlc(filename: str, csv_dict: dict, **kwargs) -> DataFrame:
    """
    Reads a filename passed as CSV, renaming columns according to the
    dictionary passed.
    :param filename: the file with the ohlcv columns
    :param csv_dict: the dict with the names of the columns
    :return: the dataframe
    """
    filepath = file_exists(filename, dirname(realpath(__file__)))
    df = pd.read_csv(filepath, **kwargs)

    # Reorder and rename
    columns_order = [csv_dict[colname] for colname in csv_dict]
    df = df[columns_order]
    df.columns = csv_dict.keys()

    return df


def read_json(filename):
    """ Reads a JSON file, returns a dict. If file does not exists
    returns None """
    if os.path.exists(filename):
        with open(filename) as json_file:
            data = json.load(json_file)
        return data
    else:
        return None
