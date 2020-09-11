import unittest

import numpy as np

from predictor.cs_dictionary import CSDictionary
from predictor.sequences import sequences
from predictor.ticks import Ticks

from typing import Tuple

DataTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

class TestSequences(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test
        use setUpClass() and store the result as class variable
        """
        super(TestSequences, cls).setUpClass()
        argv = [
            "",
            "-c", "test_params.yaml",
            "-f", "test_ticks.csv",
            "--window", "2",
            "--epochs", "1",
            "train",
        ]
        cls.params = CSDictionary(args=argv)
        cls.ticks = Ticks(cls.params, cls.params['input_file'])

    def test_prepare(self):
        X, y, Xt, yt = sequences.prepare(self.ticks.data,
                                         timesteps=self.params.window_size)
        self.assertIsInstance(X, np.ndarray)
        self.check_shape((X, y, Xt, yt))

    def check_shape(self, data_tuple: DataTuple):
        X, y, Xt, yt = data_tuple
        T_samples, t_samples = self.compute_window_values()

        self.check_Xtrain_shape(X, T_samples)
        self.check_ytrain_shape(y, T_samples)
        self.check_Xtest_shape(Xt, t_samples)
        self.check_ytest_shape(yt, t_samples)

    def compute_window_values(self):
        n_samples = self.ticks.data.shape[0]
        test_size = self.params.test_size
        T_samples = int(n_samples * (1 - test_size))  # samples in training
        t_samples = n_samples - T_samples  # samples in test

        return T_samples, t_samples

    def check_ytest_shape(self, yt, t_samples):
        # Check that shape in y (test) is correct
        self.assertEqual(yt.shape[0], t_samples)
        self.assertEqual(yt.shape[1], 1)

    def check_ytrain_shape(self, y, T_samples):
        # Check that shape in y (training) is correct
        self.assertEqual(y.shape[0], T_samples - self.params.window_size)
        self.assertEqual(y.shape[1], 1)

    def check_Xtest_shape(self, Xt, t_samples):
        # Check that shape in X (test) is correct
        self.assertEqual(Xt.shape[0], t_samples)
        self.assertEqual(Xt.shape[1], self.params.window_size)
        self.assertEqual(Xt.shape[2], self.ticks.raw.shape[1])

    def check_Xtrain_shape(self, X, T_samples):
        # Check that shape in X (training) is correct
        self.assertEqual(X.shape[0], T_samples - self.params.window_size)
        self.assertEqual(X.shape[1], self.params.window_size)
        self.assertEqual(X.shape[2], self.ticks.raw.shape[1])


if __name__ == "__main__":
    unittest.main()
