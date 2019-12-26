import unittest
from unittest import TestCase

from positions import Positions


class TestPositions(TestCase):
    params = None

    @staticmethod
    def init_pos():
        my_positions = Positions(None)
        my_positions.add_position(0.7, 100)
        my_positions.add_position(0.1, 90)
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
        p = 110.
        my_positions.update(p)
        self.assertEqual(my_positions.book[0].current_price_, p)
        self.assertEqual(my_positions.book[1].current_price_, p)
        self.assertEqual(my_positions.book[0].value_, (0.7 * p))
        self.assertEqual(my_positions.book[1].value_, (0.1 * p))
        self.assertEqual(my_positions.book[0].profit_, (0.7 * p) - (0.7 * 100.))
        self.assertEqual(my_positions.book[1].profit_, (0.1 * p) - (0.1 * 90.))

    def test_value(self):
        my_positions = self.init_pos()
        p = 80.
        my_positions.update(p)
        total_value = my_positions.value()
        self.assertEqual(total_value, (0.7 * p) + (0.1 * p))

        # Now in bear mode
        my_positions = Positions(None)
        my_positions.add_position(0.7, 100, 'bear')
        my_positions.add_position(0.1, 90, 'bear')
        p = 80.
        my_positions.update(p)
        total_value = my_positions.value()
        self.assertEqual(total_value, (0.7 * p) + (0.1 * p))

    def test_profit(self):
        my_positions = self.init_pos()
        p = 80.
        my_positions.update(p)
        profit = my_positions.profit()
        self.assertEqual(
            profit,
            ((0.7 * p) - (0.7 * 100.)) + ((0.1 * p) - (0.1 * 90.))
        )
        # Bear MODE
        my_positions = Positions(None)
        my_positions.add_position(0.7, 100, 'bear')
        my_positions.add_position(0.1, 90, 'bear')
        p = 80.
        my_positions.update(p)
        profit = my_positions.profit()
        self.assertEqual(
            profit,
            ((0.7 * 100.) - (0.7 * p)) + ((0.1 * 90.) - (0.1 * p))
        )

    def test_sell_all(self):
        my_positions = self.init_pos()
        p = 80.
        my_positions.update(p)
        income, profit = my_positions.sell_all(p)
        income_expected = (0.7 * p) + (0.1 * p)
        self.assertEqual(income, income_expected)
        profit_expected = ((0.7 * p) - (0.7 * 100)) + ((0.1 * p) - (0.1 * 90))
        self.assertEqual(profit, profit_expected)

        # Bear mode
        # Bear MODE
        my_positions = Positions(None)
        my_positions.add_position(0.7, 100, 'bear')
        my_positions.add_position(0.1, 90, 'bear')
        p = 80.
        my_positions.update(p)
        income, profit = my_positions.sell_all(p)
        income_expected = (0.7 * p) + (0.1 * p)
        self.assertEqual(income, income_expected)
        profit_expected = ((0.7 * 100) - (0.7 * p)) + ((0.1 * 90) - (0.1 * p))
        self.assertEqual(profit, profit_expected)

    def test_sell_positions(self):
        my_positions = self.init_pos()
        my_positions.update(110.)
        # sell 0.1 from the package bought at 90 (highest performance)
        income, profit = my_positions.sell_positions(0.1, 110)
        self.assertAlmostEqual(income, 11.)
        self.assertAlmostEqual(profit, 2.)
        self.assertEqual(my_positions.num_positions(), 1)
        self.assertAlmostEqual(my_positions.num_shares(), 0.7)

        # Sell from the remaining 0.7 shares acquired at 100.
        income, profit = my_positions.sell_positions(0.1, 110)
        self.assertAlmostEqual(income, 11.)
        self.assertAlmostEqual(profit, 1.)
        self.assertEqual(my_positions.num_positions(), 1)
        self.assertAlmostEqual(my_positions.num_shares(), 0.6)

        my_positions = self.init_pos()
        my_positions.update(80.)
        # Sell 0.1 from the package at 90, and 0.1 from the one at 100
        income, profit = my_positions.sell_positions(0.2, 80.)
        self.assertAlmostEqual(income, 16.)
        self.assertAlmostEqual(profit, -3.)
        self.assertEqual(my_positions.num_positions(), 1)
        self.assertAlmostEqual(my_positions.num_shares(), 0.6)


if __name__ == '__main__':
    unittest.main()
