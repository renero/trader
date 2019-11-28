from datetime import date, datetime

import holidays
import pandas as pd
from pandas.tseries.offsets import BDay


class last:

    @staticmethod
    def split_date(date_string: str) -> (int, int, int):
        """
        Simply splits the string date passed in a list with YYYY, MM and DD
        """
        year = int(date_string[:4])
        month = int(date_string[5:7])
        day = int(date_string[8:])
        return [year, month, day]

    @staticmethod
    def business_day(datetime_str: str = None, strformat: bool = True):
        """
        Returns the last business day.
        :params datetime_str: A date in format 'YYYY-MM-DD' from which to
                              compute what is the last business day.
        :params strformat:    Whether to convert the returns value to string.
                              Default YES. Otherwise, the value returned is a
                              datetime object.
        """
        if datetime_str is None:
            datetime_obj = pd.datetime.today()
        else:
            datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%d')

        last_business_day = datetime_obj - BDay(1)
        if strformat:
            return last_business_day.strftime('%Y-%m-%d')
        else:
            return last_business_day.to_pydatetime()

    @staticmethod
    def working_day(today: str = None, country: str = 'ES',
                    max_days_back: int = 10):
        """Find the last working day from the reference date passed in `today`.
        Bank holidays are searched in the country specified in the second arg

        :param today:   The date from which to start searching back for a
                        working day.
                        Default will use today's date.
        :param country: The country to use for bank holidays.
        :param max_days_back: Max nr. of days to search back for a working day.

        :return: The first working day, non-bank holiday, back from
                        reference date.
                        If cannot be found within the max nr. of days back,
                        returns 'UNKNOWN'
        """
        if today is None:
            today = datetime.today().strftime('%Y-%m-%d')
        ref_date = today
        last_business_day_found = False
        loop_iterations = 0
        last_day = None
        while not last_business_day_found and loop_iterations < max_days_back:
            last_day = last.business_day(ref_date)
            if date(*last.split_date(last_day)) in getattr(holidays, country)():
                ref_date = last_day
                loop_iterations += 1
            else:
                last_business_day_found = True
        if loop_iterations < max_days_back:
            return last_day
        return 'UNKNOWN'

    @staticmethod
    def row_date(file: str) -> str:
        existing_data = pd.read_csv(file)
        last_date = existing_data.iloc[-1]['Date']
        return last_date

    @staticmethod
    def row_date_is(for_date: str, file: str) -> bool:
        existing_data = pd.read_csv(file)
        last_date = existing_data.iloc[-1]['Date']
        return last_date == for_date
