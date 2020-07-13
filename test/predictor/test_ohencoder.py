import unittest
from unittest import TestCase

import pandas as pd

from my_dict import MyDict


def do_nothing(*args, **kwargs):
    pass


class TestOHEncoder(TestCase):
    params = MyDict()
    params.log = MyDict()
    params.log.debug = do_nothing
    params.log.info = do_nothing
    params.input_file = 'DAX100.csv'
    params.subtypes = ['body', 'move']

    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test
        use setUpClass() and store the result as class variable """
        super(TestCSEncoder, cls).setUpClass()
        # Defining a type A, G, M, Q, W and Z
        data = pd.DataFrame({
            'o': [50., 80., 10., 80., 10., 100],
            'h': [100, 100, 100, 100, 100, 100],
            'l': [00., 00., 00., 00., 00., 00.],
            'c': [51., 70., 30., 40., 70., 00.],
            'v': [100, 830, 230, 660, 500, 120]
        })
        date_column = pd.DataFrame({
            'Date': pd.date_range('2020-06-01', '2020-06-06', freq='D')
        })
        data = pd.concat([date_column, data], axis=1)
        data = data.set_index('Date')
        cls.data = data


if __name__ == '__main__':
    unittest.main()
