from data import read_data
from dictionary import Dictionary
from konkorde import Konkorde
from preview import plot_result

if __name__ == "__main__":
    conf = Dictionary()
    input_data = read_data(conf.input_data, conf.separator)
    konkorde = Konkorde(conf)

    result = konkorde.compute(input_data, date=conf.date,
                              close=conf.close, high=conf.high, low=conf.low)
    result = konkorde.cleanup(result, close=conf.close, start_pos=20)

    print(result.iloc[:, [0, 1, -3, -2, -1]].head(20))
    plot_result(result, close=conf.close)
