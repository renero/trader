from os.path import basename

from data import read_data, save_indicator, read_ohlc
from dictionary import Dictionary
from konkorde import Konkorde
from preview import plot_result

if __name__ == "__main__":
    conf = Dictionary()
    input_data = read_ohlc(conf)
    konkorde = Konkorde(conf)

    result = konkorde.compute(input_data)
    result = konkorde.cleanup(result)
    save_indicator(
        'konkorde',
        result.loc[:, ['date', 'verde', 'azul']],
        file_prefix=basename(conf.input_data),
        scale=True)

    # debug trail
    print(result.iloc[:, [0, 1, -3, -2, -1]].head(20))
    plot_result(result)
