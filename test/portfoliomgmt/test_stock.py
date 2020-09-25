from unittest import TestCase

from src.portfoliomgmt import Stock
from src.portfoliomgmt import StockPackage
from src.portfoliomgmt.toperation import TOperation
from src.portfoliomgmt.tstrategy import TStrategy


class TestStock(TestCase):

    def test_create_stock(self):
        # Create a empty stock
        stock = Stock("IBEX")
        self.assertIsNot(stock, None)
        print(stock)

    def test_create_add_stock(self):
        # Create a stock  from a stockPackage
        stock_package = StockPackage("2020-07-28", 28.5, 4, TOperation.bear)
        stock = Stock("IBEX", stock_package)
        self.assertIsNotNone(stock)
        self.assertTrue(stock.get_total_packages() == 1)
        self.assertTrue(stock.get_total_shares() == 4)
        print(stock)

        # add another stockPackage over the existing Stock
        stock_package = StockPackage("2020-07-29", 30, 2, TOperation.bear)
        stock.add(stock_package)
        self.assertTrue(stock.get_total_packages() == 2)
        self.assertTrue(stock.get_total_shares() == 6)
        print(stock)

        # add another stockPackage over the existing Stock
        stock_package = StockPackage("2020-07-30", 32, 6, TOperation.bear)
        stock.add(stock_package)
        self.assertTrue(stock.get_total_packages() == 3)
        self.assertTrue(stock.get_total_shares() == 12)
        print(stock)

        # self.assertTrue(stock.sell(1, 23.5, TStrategy.fifo) == 5.)  # Profit after selling 1 share of the first package bought
        self.assertTrue(stock.sell(1, 23.5) == 5.)  # Profit after selling 1 share of the first package bought
        self.assertTrue(stock.get_total_shares() == 11)
        print(stock)

        self.assertTrue(stock.sell(3, 24.5, TStrategy.fifo) == 12.)  # Profit after selling 3 share of the first package bought
        self.assertTrue(stock.get_total_packages() == 3)
        self.assertTrue(stock.get_active_packages() == 2)
        self.assertTrue(stock.get_total_shares() == 8)
        print(stock)

        self.assertTrue(stock.sell(1, 24.5, TStrategy.lifo) == 7.5)  # Profit after selling 1 share of the last package bought
        self.assertTrue(stock.get_total_packages() == 3)
        self.assertTrue(stock.get_active_packages() == 2)
        self.assertTrue(stock.get_total_shares() == 7)
        print(stock)

        self.assertTrue(stock.sell(1, 24.5, TStrategy.lifo, True) == 7.5)  # Simulated operation, don't update any parameter
        self.assertTrue(stock.get_total_packages() == 3)
        self.assertTrue(stock.get_active_packages() == 2)
        self.assertTrue(stock.get_total_shares() == 7)
        print(stock)
        stock.save_to()
