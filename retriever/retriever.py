"""
    retriever.py
    (c) J. Renero

    This module retrieves latest OHLCV info and updates corresponding file.

"""
import sys
from datetime import datetime

from closing import closing
from last import last
from rt_dictionary import RTDictionary


def main(argv):
    """
    Retrieve the latest stock info about the symbol and check if dates match
    """
    params = RTDictionary(args=argv)
    log = params.log

    # Call the proper service to retrieve stock info.
    stock_data, stock_date = getattr(closing, params.service, )(
        api_key=params.api_key,
        log=log,
        function=params.function_name,
        symbol=params.symbol)
    # stock_date = stock_data['latest trading day']
    today = datetime.today().strftime('%Y-%m-%d')
    last_date_in_file = last.row_date(params.file)
    log.info('Retrieved data for {} by <{}>'.format(params.symbol, stock_date))
    log.info('Last date in file <{}>'.format(last_date_in_file))

    # with open('../data/ana.mc.11.11.json') as json_file:
    #     stock_data = json.load(json_file)['Global Quote']

    # If stock data date does not match last working day, we've a problem...
    if stock_date != last.working_day() and stock_date != today:
        msg = 'Latest stock DATE does not match last working day\n'
        msg += '  {} != {}'.format(stock_date, last.working_day())
        raise ValueError(msg)

    # If data coming in is from today, stop.
    elif stock_date == today and last_date_in_file != last.working_day():
        msg = 'Stock data from API is TODAY\'s data. Stopping.\n'
        msg += 'Latest row in OHLC file is not last working day\'s date.'
        raise ValueError(msg)
    elif stock_date == today and last_date_in_file == last.working_day():
        log.info('Data already in file for date <{}>. Doing nothing'.format(
            last_date_in_file))
        return

    row = closing.csv_row(stock_data, params.json_columns,
                          params.ohlc_columns, params.json_file, params.log)
    if params.file is not None:
        closing.append_to_file(row, params.file, last.working_day(), params.log)
    else:
        print(row)


if __name__ == "__main__":
    main(sys.argv)
