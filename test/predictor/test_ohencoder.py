import unittest
from unittest import TestCase

import numpy as np

from my_dict import MyDict
from oh_encoder import OHEncoder


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
        super(TestOHEncoder, cls).setUpClass()
        cls.body_encodings = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        cls.move_encodings = 'ABCDEFGHIJK'

    def test_fit(self):
        """It builds three dictionaries. Check that they're correctly built"""
        oh = OHEncoder(self.params).fit(np.array(list(self.body_encodings)))
        self.assertEqual(len(oh.states), len(self.body_encodings))
        self.assertEqual(len(oh.dictionary), len(self.body_encodings))
        self.assertEqual(len(oh.inv_dict), len(self.body_encodings))

        self.assertEqual(len(oh.dictionary.keys()), len(self.body_encodings))
        self.assertEqual(len(oh.inv_dict.keys()), len(self.body_encodings))


if __name__ == '__main__':
    unittest.main()
