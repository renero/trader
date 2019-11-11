"""
    retriever.py
    (c) J. Renero

    This module retrieves latest OHLCV info and updates corresponding file.

"""
import sys

from last import last
from closing import closing
from rt_dictionary import RTDictionary


def main(argv):
    """
    Retrieve the latest stock info about the symbol and check if dates match
    """
    params = RTDictionary(args=argv)

    stock_data = closing.alpha_vantage(api_key=params.api_key,
                                       function=params.function_name,
                                       symbol=params.symbol)
    if stock_data['latest trading day'] != last.working_day():
        msg = 'Latest stock DATE does not match last working day\n'
        msg += '  {} != {}'.format(stock_data['latest trading day'],
                                   last.working_day())
        raise ValueError(msg)

    row = closing.csv_row(stock_data, params.json_columns)
    with open(params.file, 'a') as fd:
        fd.write(row)
    fd.close()


if __name__ == "__main__":
    main(sys.argv)
