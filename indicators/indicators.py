from os.path import splitext, basename

import pandas as pd

from dictionary import Dictionary
from file_io import read_ohlc, save_dataframe
from idx_dictionary import IDXDictionary
from konkorde import Konkorde
from logger import Logger

if __name__ == "__main__":
    params = IDXDictionary(args=sys.argv)

    input_data = read_ohlc(params.input_data, params.separator, params.csv_dict)
    params.log.info('Read file: {}'.format(params.input_data))

    konkorde = Konkorde(params)
    result = konkorde.compute(input_data)

    output = save_dataframe(
        'konkorde_{}'.format(splitext(basename(params.input_data))[0]),
        result[['open', 'high', 'low', 'close', 'volume', 'verde', 'azul']],
        params.output_path,
        cols_to_scale=['verde', 'azul'])
    params.log.info('Saved index to file {}'.format(output))

    forecast = pd.read_csv(params.forecast_file)
    kdata = pd.DataFrame()
    kdata['verde'] = result['verde']
    kdata['azul'] = result['azul']
    kdata = kdata.reset_index(drop=True)
    df = pd.concat([forecast, kdata], axis=1)
    fused = save_dataframe(
        '{}_Konkorde'.format(splitext(basename(params.forecast_file))[0]),
        df,
        params.output_path,
        cols_to_scale=['verde', 'azul'])
    params.log.info('Saved forecast and index FUSED: {}'.format(fused))
