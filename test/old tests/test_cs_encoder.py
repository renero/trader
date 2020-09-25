from unittest import TestCase

from cs_encoder import CSEncoder
from oh_encoder import OHEncoder
from utils.logger import Logger
from utils.my_dict import MyDict
from utils.test_utils import *


def do_nothing(*args, **kwargs):
    pass


class TestCSEncoder(TestCase):
    params = MyDict()
    params.log = Logger(4)
    # params.log = MyDict()
    # params.log.debug = do_nothing
    # params.log.info = do_nothing
    params.input_file = 'DAX100.csv'
    params.subtypes = ['body', 'move']
    params.csv_dict = {'d': 'Date',
                       'o': 'Open',
                       'h': 'High',
                       'l': 'Low',
                       'c': 'Close'}
    params.cse_tags = ['b', 'o', 'h', 'l', 'c']

    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test
        use setUpClass() and store the result as class variable
        """
        super(TestCSEncoder, cls).setUpClass()
        cls.data = sample_ticks()

    def test_CSEncoder(self):
        """
        Test the constructor. Normally called without any tick in the
        arguments, it
        """
        cse = CSEncoder(self.params)
        self.assertEqual(cse.open, 0.)
        self.assertEqual(cse.close, 0.)
        self.assertEqual(cse.high, 0.)
        self.assertEqual(cse.low, 0.)
        self.assertEqual(cse.min, 0.)
        self.assertEqual(cse.max, 0.)

        # Initialization
        self.assertEqual(cse.encoded_delta_close, 'pA')
        self.assertEqual(cse.encoded_delta_high, 'pA')
        self.assertEqual(cse.encoded_delta_low, 'pA')
        self.assertEqual(cse.encoded_delta_max, 'pA')
        self.assertEqual(cse.encoded_delta_min, 'pA')
        self.assertEqual(cse.encoded_delta_open, 'pA')

    def test_correct_encoding(self):
        """
        Test if this method is correctly capturing that column names reflect
        what the encoding is saying.
        """
        self.assertTrue(
            CSEncoder(self.params, encoding='ohlc')._correct_encoding())
        self.assertTrue(
            CSEncoder(self.params, encoding='OHLC')._correct_encoding())
        self.assertFalse(
            CSEncoder(self.params, encoding='0hlc')._correct_encoding())
        self.assertFalse(
            CSEncoder(self.params, encoding='ohl')._correct_encoding())
        self.assertFalse(
            CSEncoder(self.params, encoding='')._correct_encoding())

    def test_fit(self):
        """
        Measure main indicators for encoding of the second tick in the
        test data. I use the second one to be able to compare it against
        the first one.
        """
        cs = CSEncoder(self.params).fit(self.data)
        self.assertEqual(cs.cse_zero_open, 50.)
        self.assertEqual(cs.cse_zero_high, 100.)
        self.assertEqual(cs.cse_zero_low, 0.)
        self.assertEqual(cs.cse_zero_close, 50.5)
        self.assertTrue(cs.fitted)
        # Check that I've two css and they're the correct type
        self.assertEqual(len(cs.onehot), 2)
        for subtype in self.params.subtypes:
            self.assertIsNotNone(cs.onehot[subtype])

    def test_add_ohencoder(self):
        """ Check that a onehot encoder is created for every subtype """
        cs = CSEncoder(self.params).fit(self.data)
        # Check types
        for subtype in self.params.subtypes:
            self.assertIsInstance(cs.onehot[subtype], OHEncoder)

    def test_calc_parameters(self):
        """
        Test if the parameters computed for a sample tick are correct.
        """
        # Check with the first tick. (50, 100, 0, 50.5)
        cse = CSEncoder(self.params, self.data.iloc[0])
        self.assertTrue(cse.positive)
        self.assertFalse(cse.negative)

        # Percentiles, etc...
        self.assertLessEqual(cse.body_relative_size, 0.05)
        self.assertEqual(cse.hl_interval_width, 100.)
        self.assertEqual(cse.oc_interval_width, 0.5)
        self.assertEqual(cse.mid_body_point, 50.25)
        self.assertEqual(cse.mid_body_percentile, 0.5025)
        self.assertEqual(cse.min_percentile, 0.5)
        self.assertEqual(cse.max_percentile, 0.505)
        self.assertEqual(cse.upper_shadow_len, 49.5)
        self.assertEqual(cse.upper_shadow_percentile, 0.495)
        self.assertEqual(cse.lower_shadow_len, 50.)
        self.assertEqual(cse.lower_shadow_percentile, 0.5)
        self.assertAlmostEqual(cse.shadows_relative_diff, 0.005)
        self.assertEqual(cse.body_relative_size, 0.005)
        self.assertAlmostEqual(cse.shadows_relative_diff, 0.005)

        # Body position.
        self.assertTrue(cse.shadows_symmetric)
        self.assertTrue(cse.body_in_center)
        self.assertFalse(cse.body_in_lower_half)
        self.assertFalse(cse.body_in_upper_half)
        self.assertTrue(cse.has_both_shadows)
        self.assertTrue(cse.has_lower_shadow)
        self.assertTrue(cse.has_upper_shadow)

        # Check with the second tick. (80, 100, 0, 70)
        cse = CSEncoder(self.params, self.data.iloc[1])
        self.assertIsNot(cse.positive, cse.negative)
        self.assertFalse(cse.positive)
        self.assertTrue(cse.negative)

        # Percentiles, etc...
        self.assertLessEqual(cse.body_relative_size, 0.1, 'Body relative size')
        self.assertEqual(cse.hl_interval_width, 100.)
        self.assertEqual(cse.oc_interval_width, 10.)
        self.assertEqual(cse.mid_body_point, 75.)
        self.assertEqual(cse.mid_body_percentile, 0.75)
        self.assertEqual(cse.min_percentile, 0.7)
        self.assertEqual(cse.max_percentile, 0.8)
        self.assertEqual(cse.upper_shadow_len, 20.)
        self.assertEqual(cse.upper_shadow_percentile, 0.2)
        self.assertEqual(cse.lower_shadow_len, 70.)
        self.assertEqual(cse.lower_shadow_percentile, 0.7)
        self.assertEqual(cse.body_relative_size, 0.1)
        self.assertAlmostEqual(cse.shadows_relative_diff, 0.5)

        # Body position.
        self.assertFalse(cse.body_in_center)
        self.assertFalse(cse.body_in_lower_half)
        self.assertTrue(cse.body_in_upper_half)
        self.assertTrue(cse.has_both_shadows)
        self.assertTrue(cse.has_lower_shadow)
        self.assertTrue(cse.has_upper_shadow)
        self.assertFalse(cse.shadows_symmetric)

    def test_encode_with(self):
        """
        Ensure correct encoding with sample data in class and robust
        type checking.
        """
        # Start checking that the first one is correctly encoded.
        cs = CSEncoder(self.params, self.data.iloc[0])
        with self.assertRaises(AssertionError):
            cs._encode_with('123')
        self.assertEqual(cs._encode_with('ABCDE'), 'A')
        # Try with the third one
        cs = CSEncoder(self.params, self.data.iloc[2])
        self.assertEqual(cs._encode_with('KLMNO'), 'M')

    def test__encode_body(self):
        """Ensure a proper encoding of the sample ticks in test_utils"""
        tags = encoded_tags()
        for i in range(self.data.shape[0]):
            self.assertEqual(
                CSEncoder(self.params,
                          self.data.iloc[i])._encode_body(),
                tags.iloc[i]['body'][1])

    def test_encode_body(self):
        """
        Ensure a proper encoding of the sample ticks in test_utils
        """
        tags = encoded_tags()
        for i in range(self.data.shape[0]):
            cs = CSEncoder(self.params, self.data.iloc[i])
            self.assertEqual(cs.encode_body(), tags.iloc[i]['body'])

    def test_transform(self):
        """
        Test the method in charge of transforming an entire list of
        ticks into CSE format. It's only goal is to return an array with
        all of them.
        """
        cse = CSEncoder(self.params).fit_transform(self.data)
        self.assertEqual(len(cse), 6)
        for i in range(len(cse)):
            self.assertIsInstance(cse[i], CSEncoder)

    def test_inverse_transform(self):
        """
        Test that we can reverse a transformation to the original values.
        """
        encoder = CSEncoder(self.params)
        cse = encoder.fit_transform(self.data)
        # Inverse transform needs a dataframe as input, and the first CS.
        df = cs_to_df(cse, self.params.cse_tags)
        inv_cse = encoder.inverse_transform(df, cse[0])
        for i in range(inv_cse.shape[0]):
            self.assertEqual(inv_cse.iloc[i]['o'], self.data.iloc[i]['o'])

    def test_encode_tick(self):
        """
        This one checks if the CSEncoder is built and encoded, given
        that a tick is passed together with its previous one. If the previous
        one is None, then movement is not encoded.
        -------------------------------------------
        0   10  20  30  40  50  60  70  80  90  100
        A   B   C   D   E   F   G   H   I   J   K
        -------------------------------------------
        """
        # Start with the first one, which has no previous tick
        encoder = CSEncoder(self.params).fit(self.data)
        tags = encoded_tags()
        deltas = encoded_deltas()

        previous_row = None
        for i, row in self.data.iterrows():
            cs = encoder._encode_tick(row, previous_row)
            self.assertIsInstance(cs, CSEncoder)
            self.assertEqual(cs.encoded_delta_open,
                             tags.at[i, 'delta_open'])
            self.assertEqual(cs.encoded_delta_high,
                             tags.at[i, 'delta_high'])
            self.assertEqual(cs.encoded_delta_low,
                             tags.at[i, 'delta_low'])
            self.assertEqual(cs.encoded_delta_close,
                             tags.at[i, 'delta_close'])
            self.assertAlmostEqual(cs.delta_open, deltas.at[i, 'delta_open'])
            self.assertAlmostEqual(cs.delta_high, deltas.at[i, 'delta_high'])
            self.assertAlmostEqual(cs.delta_low, deltas.at[i, 'delta_low'])
            self.assertAlmostEqual(cs.delta_close, deltas.at[i, 'delta_close'])
            previous_row = cs

    def test_encode_movement(self):
        """
        Given candlestick (CSEncoder) object, compute how is its movement
        with respect to its previous one, which is passes as argument.
        Since first, second and third are tested in test_encode_tick()
        I will only test 4th against 3rd.
        -------------------------------------------
        0   10  20  30  40  50  60  70  80  90  100
        A   B   C   D   E   F   G   H   I   J   K
        -------------------------------------------
        """
        tags = encoded_tags()
        deltas = encoded_deltas()
        prev_cs = CSEncoder(self.params, self.data.iloc[0])
        for i in range(1, self.data.shape[0]):
            cs = CSEncoder(self.params, self.data.iloc[i])
            cs.encode_movement(prev_cs)
            self.assertAlmostEqual(cs.delta_open, deltas.iloc[i].delta_open)
            self.assertAlmostEqual(cs.delta_high,deltas.iloc[i].delta_high)
            self.assertAlmostEqual(cs.delta_low,deltas.iloc[i].delta_low)
            self.assertAlmostEqual(cs.delta_close, deltas.iloc[i].delta_close)
            self.assertEqual(cs.encoded_delta_open,  tags.iloc[i].delta_open)
            self.assertEqual(cs.encoded_delta_high,  tags.iloc[i].delta_high)
            self.assertEqual(cs.encoded_delta_low,   tags.iloc[i].delta_low)
            self.assertEqual(cs.encoded_delta_close, tags.iloc[i].delta_close)
            prev_cs = cs

    def test_recursive_encode_movement(self):
        """
        A value is search within a range of discrete values(buckets).
        Once found, the corresponding substring at the position of the
        bucket is returned.
        """
        encoder = CSEncoder(self.params).fit(self.data)
        # Check calls with default dictionaries
        self.assertEqual(encoder._encode_movement(value=0.0), 'A')
        self.assertEqual(encoder._encode_movement(value=0.1), 'B')
        self.assertEqual(encoder._encode_movement(value=0.2), 'C')
        self.assertEqual(encoder._encode_movement(value=0.3), 'D')
        self.assertEqual(encoder._encode_movement(value=0.4), 'E')
        self.assertEqual(encoder._encode_movement(value=0.5), 'F')
        self.assertEqual(encoder._encode_movement(value=0.6), 'G')
        self.assertEqual(encoder._encode_movement(value=0.7), 'H')
        self.assertEqual(encoder._encode_movement(value=0.8), 'I')
        self.assertEqual(encoder._encode_movement(value=0.9), 'J')
        self.assertEqual(encoder._encode_movement(value=1.1), 'K')

    def test_decode_cse(self):
        """
        Check that decodes correctly a tick, given the previous one.
        """
        col_names = list(self.params.csv_dict.keys())
        if 'd' in col_names:
            col_names.remove('d')
        encoder = CSEncoder(self.params).fit(self.data)
        cs = encoder.transform(self.data)

        # Check every tick
        for i in range(self.data.shape[0]):
            cs_df = encoded_cs_to_df(cs[i], self.params.cse_tags)
            prev_cs = cs[0] if i == 0 else cs[i - 1]
            # Define tolerance as 10% of the min-max range when reconstructing
            tol = prev_cs.hl_interval_width * 0.1
            # Decode the CS, and check
            tick = encoder._decode_cse(cs_df, prev_cs, col_names)
            self.assertLessEqual(abs(tick[0] - self.data.iloc[i]['o']), tol)
            self.assertLessEqual(abs(tick[1] - self.data.iloc[i]['h']), tol)
            self.assertLessEqual(abs(tick[2] - self.data.iloc[i]['l']), tol)
            self.assertLessEqual(abs(tick[3] - self.data.iloc[i]['c']), tol)
