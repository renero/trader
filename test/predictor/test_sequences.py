import unittest
from typing import Tuple

import numpy as np

from predictor.dictionary import Dictionary
from predictor.sequences import sequences
from predictor.ticks import Ticks

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
            "-d", "0",
            "-c", "test_params.yaml",
            "-f", "test_ticks.csv",
            "--window", "2",
            "--epochs", "1",
            "train",
        ]
        cls.params = Dictionary(args=argv)
        cls.ticks = Ticks(cls.params, cls.params['input_file'])

    def test_prepare(self):
        X, y, Xt, yt = sequences.to_time_windows(self.ticks.data,
                                                 timesteps=self.params.window_size,
                                                 train_columns=list(
                                                     self.ticks.data),
                                                 y_column="close",
                                                 test_size=self.params.test_size)
        self.assertIsInstance(X, np.ndarray)
        self.check_shape((X, y, Xt, yt))

    def test_last_in_training(self):
        lit = sequences._last_index_in_training(
            self.ticks.data, self.params.window_size, self.params.test_size)
        n_tr_samples = np.ceil(
            self.ticks.data.shape[0] * (1. - self.params.test_size)
        )
        self.assertEqual(
            lit,
            n_tr_samples - self.params.window_size + 1)

    def test_first_in_test(self):
        fit = sequences._last_index_in_training(
            self.ticks.data,
            self.params.window_size,
            self.params.test_size) + 1
        n_tst_samples = np.ceil(
            self.ticks.data.shape[0] * self.params.test_size
        )
        self.assertEqual(fit, self.ticks.data.shape[0] - n_tst_samples)

    def test_get_indices(self):
        """Test that the returned column indices correspond to the
        column names passed"""
        X_indices, y_index = sequences._get_indices(
            self.ticks.data,
            train_columns=['open', 'high', 'low'],
            y_label='close')
        self.assertListEqual(X_indices, [0, 1, 2])
        self.assertEqual(y_index, 3)

        # The case with no "Y"
        X_indices = sequences._get_indices(
            self.ticks.data,
            train_columns=['open', 'high', 'low'])
        self.assertListEqual(X_indices, [0, 1, 2])

    def test_aggregate_in_timesteps(self):
        n_rows = self.ticks.data.shape[0]
        n_cols = self.ticks.data.shape[1]
        timesteps = self.params.window_size
        df = sequences._aggregate_in_timesteps(
            self.ticks.data.values,
            self.params.window_size)
        self.assertEqual(df.shape[0], n_rows - timesteps)
        self.assertEqual(df.shape[1], n_cols * (timesteps + 1))
        #
        # Check that the first column to the right of the original columns
        # (EXP.) contains the value of the second row of the first column (ACT.)
        #
        #             Col1  Col2  Col1 ...
        # 2020-01-01    10    17   EXP.
        # 2020-01-02   ACT.   27
        #
        self.assertEqual(df.iloc[0, n_cols + 1], df.iloc[1, 0])

        # Now the case with NO prediction
        df = sequences._aggregate_in_timesteps(
            self.ticks.data.values,
            self.params.window_size, no_prediction=True)
        self.assertEqual(df.shape[0], n_rows - timesteps + 1)
        self.assertEqual(df.shape[1], n_cols * timesteps)

    def test_split(self):
        timesteps = self.params.window_size
        n_rows = self.ticks.data.shape[0]
        train_cols = ['open', 'high', 'low']
        X_indices, y_index = sequences._get_indices(
            self.ticks.data,
            train_columns=train_cols,
            y_label='close')
        df = sequences._aggregate_in_timesteps(
            self.ticks.data.values,
            self.params.window_size)
        result = sequences._split(df.values, timesteps, X_indices, y_index)
        self.assertEqual(result[0].shape[0], n_rows - timesteps)
        self.assertEqual(result[0].shape[1], timesteps)
        self.assertEqual(result[0].shape[2], len(train_cols))
        self.assertEqual(result[1].shape[0], n_rows - timesteps)
        self.assertEqual(result[1].shape[1], 1)

        # The case with no "Y"
        X_indices = sequences._get_indices(
            self.ticks.data,
            train_columns=train_cols)
        df = sequences._aggregate_in_timesteps(
            self.ticks.data.values,
            self.params.window_size,
            no_prediction=True
        )
        result = sequences._split(df.values, timesteps, X_indices)
        self.assertEqual(result.shape[0], n_rows - timesteps + 1)
        self.assertEqual(result.shape[1], timesteps)
        self.assertEqual(result.shape[2], len(train_cols))

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
        print(f'\n{n_samples} samples, test size: {test_size}')
        print(f'Training {T_samples}; test {t_samples}')

        return T_samples, t_samples

    def check_Xtrain_shape(self, X, T_samples):
        # Check that shape in X (training) is correct
        self.assertEqual(X.shape[0], T_samples - self.params.window_size)
        self.assertEqual(X.shape[1], self.params.window_size)
        self.assertEqual(X.shape[2], self.ticks.data.shape[1])

    def check_ytrain_shape(self, y, T_samples):
        # Check that shape in y (training) is correct
        self.assertEqual(y.shape[0], T_samples - self.params.window_size)

    def check_ytest_shape(self, yt, t_samples):
        # Check that shape in y (test) is correct
        self.assertEqual(yt.shape[0], t_samples)

    def check_Xtest_shape(self, Xt, t_samples):
        # Check that shape in X (test) is correct
        self.assertEqual(Xt.shape[0], t_samples)
        self.assertEqual(Xt.shape[1], self.params.window_size)
        self.assertEqual(Xt.shape[2], self.ticks.data.shape[1])
