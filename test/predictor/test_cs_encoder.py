from unittest import TestCase

from predictor.cs_encoder import CSEncoder
from predictor.oh_encoder import OHEncoder
from utils.my_dict import MyDict
from utils.test_utils import sample_ticks


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
        self.assertEqual(
            CSEncoder(self.params, self.data.iloc[0])._encode_body(), 'A')
        self.assertEqual(
            CSEncoder(self.params, self.data.iloc[1])._encode_body(), 'G')
        self.assertEqual(
            CSEncoder(self.params, self.data.iloc[2])._encode_body(), 'M')
        self.assertEqual(
            CSEncoder(self.params, self.data.iloc[3])._encode_body(), 'Q')
        self.assertEqual(
            CSEncoder(self.params, self.data.iloc[4])._encode_body(), 'W')
        self.assertEqual(
            CSEncoder(self.params, self.data.iloc[5])._encode_body(), 'Z')

    def test_encode_body(self):
        """
        Ensure a proper encoding of the sample ticks in test_utils
        """
        cs = CSEncoder(self.params, self.data.iloc[0])
        self.assertEqual(cs.encode_body(), 'pA')
        cs = CSEncoder(self.params, self.data.iloc[1])
        self.assertEqual(cs.encode_body(), 'nG')
        cs = CSEncoder(self.params, self.data.iloc[2])
        self.assertEqual(cs.encode_body(), 'pM')
        cs = CSEncoder(self.params, self.data.iloc[3])
        self.assertEqual(cs.encode_body(), 'nQ')
        cs = CSEncoder(self.params, self.data.iloc[4])
        self.assertEqual(cs.encode_body(), 'pW')
        cs = CSEncoder(self.params, self.data.iloc[5])
        self.assertEqual(cs.encode_body(), 'nZ')

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
        cs0 = encoder._encode_tick(self.data.iloc[0], None)
        self.assertIsInstance(cs0, CSEncoder)
        self.assertAlmostEqual(cs0.delta_open, 0.)
        self.assertAlmostEqual(cs0.delta_high, 0.)
        self.assertAlmostEqual(cs0.delta_low, 0.)
        self.assertAlmostEqual(cs0.delta_close, 0.)
        self.assertEqual(cs0.encoded_delta_open, 'pA')
        self.assertEqual(cs0.encoded_delta_high, 'pA')
        self.assertEqual(cs0.encoded_delta_low, 'pA')
        self.assertEqual(cs0.encoded_delta_close, 'pA')

        # Check second one wrt first one.
        cs1 = encoder._encode_tick(self.data.iloc[1], cs0)
        self.assertAlmostEqual(cs1.delta_open, 0.3)
        self.assertAlmostEqual(cs1.delta_high, 0.)
        self.assertAlmostEqual(cs1.delta_low, 0.)
        self.assertAlmostEqual(cs1.delta_close, 0.195)
        self.assertEqual(cs1.encoded_delta_open, 'pD')
        self.assertEqual(cs1.encoded_delta_high, 'pA')
        self.assertEqual(cs1.encoded_delta_low, 'pA')
        self.assertEqual(cs1.encoded_delta_close, 'pC')

        # Check third one wrt second one.
        cs2 = encoder._encode_tick(self.data.iloc[2], cs1)
        self.assertAlmostEqual(cs2.delta_open, -0.7)
        self.assertAlmostEqual(cs2.delta_high, 0.)
        self.assertAlmostEqual(cs2.delta_low, 0.)
        self.assertAlmostEqual(cs2.delta_close, -0.4)
        self.assertEqual(cs2.encoded_delta_open, 'nH')
        self.assertEqual(cs2.encoded_delta_high, 'pA')
        self.assertEqual(cs2.encoded_delta_low, 'pA')
        self.assertEqual(cs2.encoded_delta_close, 'nE')

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
        cs4 = CSEncoder(self.params, self.data.iloc[4])
        cs3 = CSEncoder(self.params, self.data.iloc[3])
        cs2 = CSEncoder(self.params, self.data.iloc[2])
        cs3.encode_movement(cs2)
        self.assertAlmostEqual(cs3.delta_open, 0.7)
        self.assertAlmostEqual(cs3.delta_high, 0.)
        self.assertAlmostEqual(cs3.delta_low, 0.)
        self.assertAlmostEqual(cs3.delta_close, 0.1)
        self.assertEqual(cs3.encoded_delta_open, 'pH')
        self.assertEqual(cs3.encoded_delta_high, 'pA')
        self.assertEqual(cs3.encoded_delta_low, 'pA')
        self.assertEqual(cs3.encoded_delta_close, 'pB')

        cs4.encode_movement(cs3)
        self.assertAlmostEqual(cs4.delta_open, -0.7)
        self.assertAlmostEqual(cs4.delta_high, 0.)
        self.assertAlmostEqual(cs4.delta_low, 0.)
        self.assertAlmostEqual(cs4.delta_close, 0.4)
        self.assertEqual(cs4.encoded_delta_open, 'nH')
        self.assertEqual(cs4.encoded_delta_high, 'pA')
        self.assertEqual(cs4.encoded_delta_low, 'pA')
        self.assertEqual(cs4.encoded_delta_close, 'pE')

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
