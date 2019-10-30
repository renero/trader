import sys
from os.path import splitext, basename

import pandas as pd

from dictionary import Dictionary
from file_io import read_ohlc, save_dataframe
from idx_dictionary import IDXDictionary
from konkorde import Konkorde
from logger import Logger

if __name__ == "__main__":
    params = IDXDictionary(args=sys.argv)
    log: Logger = params.log

    input_data = read_ohlc(params.input_file, params.separator, params.csv_dict)
    log.info('Read file: {}, {} rows, {} cols.'.format(
        params.input_file, input_data.shape[0], input_data.shape[1]))

    # Initialize result, index name, and column names for this indicator
    result = None
    idx_name = params.indicator_name
    ind_columns = params.indicator_dict[idx_name]

    # Check what index to compute
    if params.konkorde:
        log.info('Computing index {}'.format(idx_name))
        konkorde = Konkorde(params)
        result = konkorde.compute(input_data)

    # Decide what to do with the result
    if params.append is True:
        output = save_dataframe(
            '{}_{}'.format(idx_name, splitext(basename(params.input_data))[0]),
            result[['open', 'high', 'low', 'close', 'volume', 'verde', 'azul']],
            params.output_path,
            cols_to_scale=ind_columns)
        params.log.info('Saved index to file {}'.format(output))
    elif params.merge is not None:
        mergeable_data = pd.read_csv(params.merge_file,
                                     delimiter=params.separator)
        indicator_data = pd.DataFrame()
        indicator_data[ind_columns] = result[ind_columns].copy(deep=True)
        indicator_data = indicator_data.reset_index(drop=True)
        df = pd.concat([mergeable_data, indicator_data], axis=1)
        fused = save_dataframe(
            '{}_{}'.format(
                splitext(basename(params.forecast_file))[0], idx_name),
            df,
            params.output_path,
            cols_to_scale=ind_columns)
        params.log.info('Saved forecast and index FUSED: {}'.format(fused))
