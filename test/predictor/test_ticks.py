import unittest

from dictionaries import CSDictionary
from predictor.ticks import Ticks
from sequences import sequences


class TestTicks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ get_some_resource() is slow, to avoid calling it for each test
        use setUpClass() and store the result as class variable
        """
        super(TestTicks, cls).setUpClass()
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

    def test_Ticks(self):
        """Test the creation of the object"""
        ticks = self.ticks
        self.assertIsInstance(ticks, Ticks)
        # Check column names
        self.assertListEqual(list(ticks.data.columns), self.params.ohlc)
        # Check shape
        self.check_shape(ticks.data)
        self.check_data_is_not_scaled(ticks.data)

    def test_scale(self):
        self.ticks.scale()
        self.check_shape(self.ticks.data)
        self.check_data_is_scaled(self.ticks.data)
        self.check_data_is_not_scaled(self.ticks.raw)

    def check_shape(self, data):
        """Ensure data shape is correct."""
        self.assertEqual(data.shape[0], 10)
        self.assertEqual(data.shape[1], 4)

    def check_data_is_scaled(self, data):
        # Ensure data has been scaled
        self.assertGreater(data.min().min(), -5.0)
        self.assertLess(data.max().max(), 5.0)

    def check_data_is_not_scaled(self, data):
        """Ensure values are not scaled."""
        self.assertGreater(data.min().min(), 9000.0)
        self.assertLess(data.max().max(), 10100.0)

    @classmethod
    def ListAlmostEqual(cls, list1, list2, acc):
        def almost_equal(value_1, value_2, accuracy=5e-1):
            return abs(value_1 - value_2) < accuracy

        return all(almost_equal(*values, acc) for values in zip(list1, list2))

    def test_scale_back(self):
        """Check that inverse transform is almost equal with 0.5 tolerance"""
        self.setUpClass()
        self.ticks.scale()
        inv = self.ticks.scale_back(self.ticks.data)
        self.check_data_is_not_scaled(inv)
        self.assertTrue(self.ListAlmostEqual(
            inv.iloc[0].values,
            self.ticks.raw.iloc[0].values, acc=5e-1))

    def test_append_indicator(self):
        # Check that incorrect indicator name raise exception
        self.assertRaises(ModuleNotFoundError, self.ticks.append_indicator, 'kk')
        # Check that trend is correctly built
        self.ticks.append_indicator('trend')
        self.assertEqual(self.ticks.data.shape[1], 5)

    def test_training_columns(self):
        tc = self.ticks._training_columns(None)
        self.assertListEqual(list(self.ticks.data.columns), tc)

    def test_prepare_for_training(self):
        self.setUpClass()
        self.ticks.append_indicator('trend')
        print(self.ticks.data)
        X, y, Xt, yt = self.ticks.prepare_for_training(predict_column='trend')
        self.assertListEqual(list(X[0][0]), list(self.ticks.data.iloc[0]))
        self.assertEqual(
            y[0],
            self.ticks.data.iloc[self.params.window_size]['trend'])

        first = sequences.first_in_test(
            self.ticks.data,
            self.params.window_size,
            self.params.test_size)
        self.assertListEqual(list(Xt[0][0]), list(first))
        self.assertEqual(
            yt[0],
            self.ticks.data.iloc[-1]['trend'])

        print("\n\n--X[0]--\n", X[0], "\n---------\n\n")
        print("--y[0]--\n", y, "\n--------\n\n")

        print("--Xt[0]--\n", Xt[0], "\n---------\n\n")
        print("--yt[0]--\n", yt, "\n--------\n\n")
