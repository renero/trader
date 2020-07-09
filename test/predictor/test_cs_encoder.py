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
        cls.data = pd.DataFrame({
            'o': [9869.12, 9735.65, 9484.25, 9510.33, 9643.76],
            'h': [9879.53, 9790.26, 9624.65, 9592.37, 9855.42],
            'l': [9687.25, 9468.58, 9382.82, 9459.17, 9607.90],
            'c': [9764.73, 9473.16, 9469.66, 9518.17, 9837.61],
            'v': [67673900, 105538300, 96812300, 82466600, 114825000]
        })

    def test_fit(self):
        encoder = CSEncoder(self.params)
        encoder = encoder.fit(self.data)
        self.assertEqual(encoder.cse_zero_open, 9869.12)
        self.assertEqual(encoder.cse_zero_high, 9879.53)
        self.assertEqual(encoder.cse_zero_low, 9687.25)
        self.assertEqual(encoder.cse_zero_close, 9764.73)
        self.assertIs(encoder.fitted, True)
        for subtype in self.params.subtypes:
            self.assertIsNotNone(encoder.onehot[subtype])

    def test_encode_tick(self):
        cse = CSEncoder(self.params, self.data.iloc[0])

        self.assertEqual(cse.open, 9869.12, 'Open')
        self.assertEqual(cse.close, 9764.73, 'Close')
        self.assertEqual(cse.high, 9879.53, 'High')
        self.assertEqual(cse.low, 9687.25, 'Low')
        self.assertEqual(cse.min, 9764.73, 'Min between open and close')
        self.assertEqual(cse.max, 9869.12, 'Max between open and close')

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
