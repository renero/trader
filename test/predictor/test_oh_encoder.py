from unittest import TestCase

import numpy as np
import pandas as pd

from my_dict import MyDict
from oh_encoder import OHEncoder
from test_utils import sample_ticks


def do_nothing(*args, **kwargs):
    pass


class TestOHEncoder(TestCase):
    params = MyDict()
    params.log = MyDict()
    params.log.debug = do_nothing
    params.log.info = do_nothing
    params.input_file = 'DAX100.csv'
    params.subtypes = ['body', 'move']
    params.test_size = 0.1
    params.batch_size = 1
    params.window_size = 1
    params.num_predictions = 1

    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test
        use setUpClass() and store the result as class variable """
        super(TestOHEncoder, cls).setUpClass()
        cls.body_encodings = np.array(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        cls.data = sample_ticks()

    def test_reset(self):
        """Check if this function is properly resetting internal dicts"""
        oh = OHEncoder(self.params).fit(self.body_encodings).reset()
        self.assertFalse(oh.states)
        self.assertFalse(oh.dictionary)
        self.assertFalse(oh.inv_dict)

    def test_fit(self):
        """It builds three dictionaries. Check that they're correctly built"""
        oh = OHEncoder(self.params).fit(self.body_encodings)
        self.assertEqual(len(oh.states), len(self.body_encodings))
        self.assertEqual(len(oh.dictionary), len(self.body_encodings))
        self.assertEqual(len(oh.inv_dict), len(self.body_encodings))
        self.assertEqual(len(oh.dictionary.keys()), len(self.body_encodings))
        self.assertEqual(len(oh.inv_dict.keys()), len(self.body_encodings))
        # 2D arrays missing from this test

    def test_encode(self):
        """
        Encodes an array of strings (signed or unsigned). If they are signed
        each string starts with character 'p' (positive) or 'n' (negative).
        One-hot encoder uses the dictionary passed during object creation
        to determine the length of the binary representation.
        """
        # One dimensinoal, Signed case
        oh = OHEncoder(self.params).fit(self.body_encodings)
        bodies = pd.DataFrame({0: ['pB', 'nC']})
        result = oh.encode(bodies)
        self.assertEqual(result.sum(axis=1).iloc[0], +1.)
        self.assertEqual(result.sum(axis=1).iloc[1], -1.)
        self.assertEqual(result.iloc[0, 0], 0.)
        self.assertEqual(result.iloc[0, 1], 1.)
        self.assertEqual(result.iloc[1, 2], -1.)
        self.assertEqual(result.iloc[0, 3], 0.)

        # One-dimensional, Unsigned case
        oh = OHEncoder(self.params, signed=False).fit(self.body_encodings)
        bodies = pd.DataFrame({0: ['B', 'C']})
        result = oh.encode(bodies)
        self.assertEqual(result.sum(axis=1).iloc[0], +1.)
        self.assertEqual(result.sum(axis=1).iloc[1], +1.)
        self.assertEqual(result.iloc[0, 0], 0.)
        self.assertEqual(result.iloc[0, 1], 1.)
        self.assertEqual(result.iloc[1, 2], 1.)
        self.assertEqual(result.iloc[1, 3], 0.)

        # Bi-dimensional, signed case.
        # Encode pA & pB in a row of 26x2 bits
        # Encode pY & nZ in a row of 26x2 bits
        oh = OHEncoder(self.params).fit(self.body_encodings)
        bodies = pd.DataFrame({0: ['pA', 'pY'], 1: ['pB', 'nZ']})
        result = oh.encode(bodies)
        width = self.body_encodings.shape[0]
        self.assertEqual(result.sum(axis=1).iloc[0], +2.)
        self.assertEqual(result.sum(axis=1).iloc[1], 0.)
        self.assertEqual(result.iloc[0, 0], 1.)
        self.assertEqual(result.iloc[0, 1], 0.)
        self.assertEqual(result.iloc[0, width], 0.)
        self.assertEqual(result.iloc[0, width+1], 1.)
        self.assertEqual(result.iloc[1, width-2], 1.)
        self.assertEqual(result.iloc[1, width-1], 0.)
        self.assertEqual(result.iloc[1, width*2-1], -1.)
        self.assertEqual(result.iloc[1, width*2-2], 0.)

        # Bi-dimensional, unsigned case.
        # Encode pA & pB in a row of 26x2 bits
        # Encode pY & nZ in a row of 26x2 bits
        oh = OHEncoder(self.params, signed=False).fit(self.body_encodings)
        bodies = pd.DataFrame({0: ['A', 'Y'], 1: ['B', 'Z']})
        result = oh.encode(bodies)
        width = self.body_encodings.shape[0]
        self.assertEqual(result.sum(axis=1).iloc[0], +2.)
        self.assertEqual(result.sum(axis=1).iloc[1], 2.)
        self.assertEqual(result.iloc[0, 0], 1.)
        self.assertEqual(result.iloc[0, 1], 0.)
        self.assertEqual(result.iloc[0, width], 0.)
        self.assertEqual(result.iloc[0, width+1], 1.)
        self.assertEqual(result.iloc[1, width-2], 1.)
        self.assertEqual(result.iloc[1, width-1], 0.)
        self.assertEqual(result.iloc[1, width*2-1], 1.)
        self.assertEqual(result.iloc[1, width*2-2], 0.)

    def test_decode(self):
        """Check that decoding result in any of the elements of the
        dictionary."""
        data = np.zeros(26)
        data[0] = 1.
        oh = OHEncoder(self.params).fit(self.body_encodings)
        result = oh.decode(data)
        self.assertEqual(result, ['pA'])
