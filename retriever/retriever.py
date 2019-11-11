"""
    retriever.py
    (c) J. Renero

    This module retrieves latest OHLCV info and updates corresponding file.

"""
from datetime import datetime
import sys

from last import last
from closing import closing
from rt_dictionary import RTDictionary


def main(argv):
    """
    Retrieve the latest stock info about the symbol and check if dates match
    """
    params = RTDictionary(args=argv)
    today = datetime.today().strftime('%Y-%m-%d')
    stock_data = closing.alpha_vantage(api_key=params.api_key,
                                       function=params.function_name,
                                       symbol=params.symbol)

    # If stock data date does not match last working day, we've a problem...
    if stock_data['latest trading day'] != last.working_day() and \
            stock_data['latest trading day'] != today:
        msg = 'Latest stock DATE does not match last working day\n'
        msg += '  {} != {}'.format(stock_data['latest trading day'],
                                   last.working_day())
        raise ValueError(msg)

    # If data coming in is from today, stop.
    elif stock_data['latest trading day'] == today and \
            last.row_date_is(last.working_day(), params.file) is False:
        msg = 'Stock data from API is TODAY\'s data. Stopping.\n'
        msg += 'Latest row in OHLC file is not last working day\'s date.'
        raise ValueError(msg)
    elif stock_data['latest trading day'] == today and \
            last.row_date_is(last.working_day(), params.file):
        params.log.info('Data is already in the file')
        return
    else:
        params.log.info('Retrieved data for symbol {}'.format(params.symbol))

    row = closing.csv_row(stock_data, params.json_columns)
    closing.append_to_file(row, params.file, last.working_day(), params.log)


if __name__ == "__main__":
    main(sys.argv)
