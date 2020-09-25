from unittest import TestCase

from src.portfoliomgmt import Portfolio
from src.portfoliomgmt import StockPackage
from src.portfoliomgmt.toperation import TOperation


class TestPortfolio(TestCase):

    def test_create_portfolio(self):
        # Create a empty portfolio
        portfolio = Portfolio()
        self.assertIsNot(portfolio, None)
        print(portfolio)

    def test_create_add_stock(self):
        portfolio = Portfolio()
        self.assertIsNot(portfolio, None)
        print(portfolio)

        stock_package = StockPackage("2020-07-28", 28.5, 4, TOperation.bear)
        portfolio.add("IBEX", stock_package)
        stock = portfolio.getstock("IBEX")
        print(portfolio)

        self.assertIsNotNone(stock)
        self.assertTrue(stock.get_total_packages() == 1)
        self.assertTrue(stock.get_total_shares() == 4)

        stock_package = StockPackage("2020-07-29", 29, 1, TOperation.bear)
        portfolio.add("IBEX", stock_package)
        print(portfolio)

        self.assertIsNotNone(stock)
        self.assertTrue(stock.get_total_packages() == 2)
        self.assertTrue(stock.get_total_shares() == 5)

        stock_package = StockPackage("2020-07-29", 50.5, 2, TOperation.bear)
        portfolio.add("DAX", stock_package)
        stock = portfolio.getstock("DAX")
        print(portfolio)

        self.assertIsNotNone(stock)
        self.assertTrue(stock.get_total_packages() == 1)
        self.assertTrue(stock.get_total_shares() == 2)

        stock = portfolio.getstock("IBEX")
        stock.sell_all(30)
        print(portfolio)

        self.assertIsNotNone(stock)
        self.assertTrue(stock.get_active_packages() == 0)
        self.assertTrue(stock.get_total_shares() == 0)

        stock = portfolio.getstock("DAX")
        stock.sell_all(51)
        print(portfolio)

        self.assertIsNotNone(stock)
        self.assertTrue(stock.get_active_packages() == 0)
        self.assertTrue(stock.get_total_shares() == 0)

        portfolio.save()
