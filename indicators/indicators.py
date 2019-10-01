from os.path import basename, splitext

from file_io import save_dataframe, read_ohlc
from dictionary import Dictionary
from konkorde import Konkorde
from preview import plot_result


if __name__ == "__main__":
    conf = Dictionary()
    input_data = read_ohlc(conf.input_data, conf.separator, conf.csv_dict)
    konkorde = Konkorde(conf)

    result = konkorde.compute(input_data)
    save_dataframe(
        'konkorde_{}'.format(splitext(basename(conf.input_data))[0]),
        result[['open', 'high', 'low', 'close', 'volume', 'verde', 'azul']],
        conf.output_path,
        cols_to_scale=['verde', 'azul'])

    # debug trail
    print(result.iloc[:20].head(20))
    plot_result(konkorde.cleanup(result))
