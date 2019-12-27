import unittest
from unittest import TestCase

from positions import Positions


class TestPositions(TestCase):
    params = None

    @staticmethod
    def init_pos(mode='bull'):
        my_positions = Positions(None)
        my_positions.buy_position(0.7, 100, mode=mode)
        my_positions.buy_position(0.1, 90, mode=mode)
        return my_positions

    def test_num_positions(self):
        my_positions = self.init_pos()
        num_positions = my_positions.num_positions()
        self.assertEqual(num_positions, 2)

    def test_num_shares(self):
        my_positions = self.init_pos()
        num_shares = my_positions.num_shares()
        self.assertAlmostEqual(num_shares, 0.8)

    def test_update_value(self):
        my_positions = self.init_pos()
        my_positions.update(110.)
        self.assertEqual(my_positions.book[0].current_price_, 110.)
        self.assertEqual(my_positions.book[1].current_price_, 110.)
        self.assertAlmostEqual(my_positions.book[0].value_, 77., places=2)
        self.assertAlmostEqual(my_positions.book[1].value_, 11., places=2)
        self.assertAlmostEqual(my_positions.book[0].profit_, 7.70, places=2)
        self.assertAlmostEqual(my_positions.book[1].profit_, 2.44, places=2)

    def test_value(self):
        my_positions = self.init_pos()
        p = 80.
        my_positions.update(p)
        total_value = my_positions.value()
        self.assertEqual(total_value, (0.7 * p) + (0.1 * p))

        # Now in bear mode
        my_positions = self.init_pos('bear')
        p = 80.
        my_positions.update(p)
        total_value = my_positions.value()
        self.assertEqual(total_value, (0.7 * p) + (0.1 * p))

    def test_profit(self):
        my_positions = self.init_pos()
        my_positions.update(80.)
        profit = my_positions.profit()
        self.assertAlmostEqual(profit, -12.088, places=2)

        # Bear MODE
        my_positions = self.init_pos('bear')
        my_positions.update(80.)
        profit = my_positions.profit()
        self.assertAlmostEqual(profit, 12.088, places=2)

    def test_sell_all(self):
        my_positions = self.init_pos()
        my_positions.update(80.)
        income, profit = my_positions.sell_all(80.)
        self.assertAlmostEqual(income, 64, places=1)
        self.assertAlmostEqual(profit, -12.088, places=2)

        # Bear MODE
        my_positions = self.init_pos('bear')
        my_positions.update(80.)
        income, profit = my_positions.sell_all(80.)
        self.assertAlmostEqual(income, 64, places=1)
        self.assertAlmostEqual(profit, 12.088, places=2)

    def test_sell_positions_positive_bull(self):
        my_positions = self.init_pos()
        my_positions.update(110.)
        # sell 0.1 from the package bought at 90 (highest performance)
        income, profit = my_positions.sell_positions(0.1, 110)
        self.assertAlmostEqual(income, 11.)
        self.assertAlmostEqual(profit, 2.44, places=2)
        self.assertEqual(my_positions.num_positions(), 1)
        self.assertAlmostEqual(my_positions.num_shares(), 0.7)

        # Sell from the remaining 0.7 shares acquired at 100.
        income, profit = my_positions.sell_positions(0.2, 110)
        self.assertAlmostEqual(income, 22.)
        self.assertAlmostEqual(profit, 2.20, places=2)
        self.assertEqual(my_positions.num_positions(), 1)
        self.assertAlmostEqual(my_positions.num_shares(), 0.5)

    def test_sell_positions_negative_bull(self):
        my_positions = self.init_pos()
        my_positions.update(80.)
        # sell 0.1 from the package bought at 90 (highest performance)
        income, profit = my_positions.sell_positions(0.1, 80)
        self.assertAlmostEqual(income, 8.)
        self.assertAlmostEqual(profit, -0.89, places=2)
        self.assertEqual(my_positions.num_positions(), 1)
        self.assertAlmostEqual(my_positions.num_shares(), 0.7)

        # Sell from the remaining 0.7 shares acquired at 100.
        income, profit = my_positions.sell_positions(0.2, 80)
        self.assertAlmostEqual(income, 16.)
        self.assertAlmostEqual(profit, -3.20, places=2)
        self.assertEqual(my_positions.num_positions(), 1)
        self.assertAlmostEqual(my_positions.num_shares(), 0.5)

    def test_sell_positions_positive_bear(self):
        my_positions = self.init_pos('bear')
        my_positions.update(80.)
        # sell 0.1 from the package bought at 90 (highest performance)
        income, profit = my_positions.sell_positions(0.1, 80.)
        self.assertAlmostEqual(income, 8.)
        self.assertAlmostEqual(profit, 1.60, places=2)
        self.assertEqual(my_positions.num_positions(), 2)
        self.assertAlmostEqual(my_positions.num_shares(), 0.7)

        # Sell 0.2 from the 0.7
        income, profit = my_positions.sell_positions(0.2, 80.)
        self.assertAlmostEqual(income, 16.)
        self.assertAlmostEqual(profit, 3.20, places=2)
        self.assertEqual(my_positions.num_positions(), 2)
        self.assertAlmostEqual(my_positions.num_shares(), 0.5)

    def test_sell_positions_negative_bear(self):
        my_positions = self.init_pos('bear')
        my_positions.update(110.)
        # sell 0.1 from the package bought at 100 (highest performance)
        income, profit = my_positions.sell_positions(0.1, 110.)
        self.assertAlmostEqual(income, 11.)
        self.assertAlmostEqual(profit, -1.10, places=2)
        self.assertEqual(my_positions.num_positions(), 2)
        self.assertAlmostEqual(my_positions.num_shares(), 0.7)

        # Sell 0.2 from the 0.7
        income, profit = my_positions.sell_positions(0.2, 110.)
        self.assertAlmostEqual(income, 22.)
        self.assertAlmostEqual(profit, -2.20, places=2)
        self.assertEqual(my_positions.num_positions(), 2)
        self.assertAlmostEqual(my_positions.num_shares(), 0.5)


if __name__ == '__main__':
    unittest.main()
