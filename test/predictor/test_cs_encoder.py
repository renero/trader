from unittest import TestCase

import pandas as pd

from cs_encoder import CSEncoder
from my_dict import MyDict


def do_nothing(*args, **kwargs):
    pass


class TestCSEncoder(TestCase):

    params = MyDict()
    params.log = MyDict()
    params.log.debug = do_nothing
    params.log.info = do_nothing
    params.input_file = 'DAX100.csv'
    params.subtypes = ['body', 'move']

    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test use setUpClass()
            and store the result as class variable
        """
        super(TestCSEncoder, cls).setUpClass()
        # Defining a type A, G, M, Q, W and Z
        data = pd.DataFrame({
            'o': [50., 70., 30., 45., 7, 100.],
            'h': [90,  91., 92., 93., 94, 100.],
            'l': [1., 2., 3., 4., 1., 2.],
            'c': [51., 75., 15., 85., 62., 0.],
            'v': [1000, 8300, 2300, 6600, 5000, 1241]
        })
        datec = pd.DataFrame({
            'Date': pd.date_range('2020-06-01', '2020-06-06', freq='D')
        })
        data = pd.concat([datec, data], axis=1)
        data = data.set_index('Date')
        cls.data = data

    def test_fit(self):
        encoder = CSEncoder(self.params)
        encoder = encoder.fit(self.data)
        self.assertEqual(encoder.cse_zero_open, 50.)
        self.assertEqual(encoder.cse_zero_high, 90.)
        self.assertEqual(encoder.cse_zero_low, 1.)
        self.assertEqual(encoder.cse_zero_close, 51.)
        self.assertIs(encoder.fitted, True)
        for subtype in self.params.subtypes:
            self.assertIsNotNone(encoder.onehot[subtype])

    def test_encode_tick(self):
        cse = CSEncoder(self.params, self.data.iloc[0])

        self.assertEqual(cse.open, 50., 'Open')
        self.assertEqual(cse.close, 51., 'Close')
        self.assertEqual(cse.high, 90., 'High')
        self.assertEqual(cse.low, 1., 'Low')
        self.assertEqual(cse.min, 50., 'Min between open and close')
        self.assertEqual(cse.max, 51., 'Max between open and close')

        print()
        print(cse.min_percentile)
        print(cse.max_percentile)
        print(cse.mid_body_percentile)
        print(cse.mid_body_point)
        print(cse.positive)
        print(cse.negative)
        print(cse.has_upper_shadow)
        print(cse.has_lower_shadow)
        print(cse.has_both_shadows)
        print(cse.shadows_symmetric)
        print(cse.body_in_upper_half)
        print(cse.body_in_lower_half)
        print(cse.body_in_center)
        print(cse.hl_interval_width)
        print(cse.upper_shadow_len)
        print(cse.lower_shadow_len)
        print(cse.upper_shadow_percentile)
        print(cse.lower_shadow_percentile)
        print(cse.oc_interval_width)
        print(cse.body_relative_size)
        print(cse.shadows_relative_diff)
