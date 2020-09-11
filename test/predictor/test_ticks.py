import unittest

from predictor.cs_dictionary import CSDictionary
from predictor.ticks import Ticks


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

    def test_transform(self):
        self.ticks.transform()
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

    def test_inverse_transform(self):
        """Check that inverse transform is almost equal with 0.5 tolerance"""
        self.ticks.transform()
        inv = self.ticks.inverse_transform(self.ticks.data)
        self.check_data_is_not_scaled(inv)
        self.assertTrue(self.ListAlmostEqual(
            inv.iloc[0].values,
            self.ticks.raw.iloc[0].values, acc=5e-1))

    def test_get_trend_sign(self):
        """Check that the trend is correctly computed"""
        trend = self.ticks.get_trend_sign()
        self.assertListEqual(list(trend),
                             [1., 0., 1., 1., 1., 0., 1., 1., 0., 1.])


if __name__ == "__main__":
    unittest.main()
