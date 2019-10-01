from os.path import basename, splitext

from file_io import save_dataframe, read_ohlc
from dictionary import Dictionary
from konkorde import Konkorde
from logger import Logger
from plots import plot_konkorde

if __name__ == "__main__":
    conf = Dictionary()
    log = Logger(conf.log_level)
    input_data = read_ohlc(conf.input_data, conf.separator, conf.csv_dict)
    log.info('Read file: {}'.format(conf.input_data))

    konkorde = Konkorde(conf)
    result = konkorde.compute(input_data)

    output = save_dataframe(
        'konkorde_{}'.format(splitext(basename(conf.input_data))[0]),
        result[['open', 'high', 'low', 'close', 'volume', 'verde', 'azul']],
        conf.output_path,
        cols_to_scale=['verde', 'azul'])
    log.info('Saved index to file {}'.format(output))

    # debug trail
    # print(result.iloc[:20].head(20))
    # plot_konkorde(konkorde.cleanup(result))
