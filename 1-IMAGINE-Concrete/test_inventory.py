"""Inventory conservation and FIFO checks without running the model pipeline."""
import unittest
import numpy as np

from IMAGINE_Concrete import calculate_inventory, calculate_clinker_inventory


def chronological_inventory(net, lifetime=None):
    """Independent year-by-year accounting oracle, including expired stock."""
    net = np.asarray(net, dtype=np.float64)
    n, years = net.shape
    production = np.zeros_like(net)
    stock = np.zeros_like(net)
    used = np.zeros_like(net)
    expired = np.zeros_like(net)
    balance = np.zeros(n)
    lots = np.zeros((n, lifetime)) if lifetime else None
    for t in range(years):
        demand = np.maximum(net[:, t], 0)
        deposit = np.maximum(-net[:, t], 0)
        if lifetime:
            expired[:, t] = lots[:, 0]
            lots[:, :-1] = lots[:, 1:]
            lots[:, -1] = deposit
            remaining_demand = demand.copy()
            for age in range(lifetime):
                taken = np.minimum(remaining_demand, lots[:, age])
                lots[:, age] -= taken
                remaining_demand -= taken
            production[:, t] = remaining_demand
            used[:, t] = demand - remaining_demand
            stock[:, t] = lots.sum(axis=1)
        else:
            balance += deposit
            used[:, t] = np.minimum(demand, balance)
            balance -= used[:, t]
            production[:, t] = demand - used[:, t]
            stock[:, t] = balance
    return production, stock, used, expired


class InventoryTests(unittest.TestCase):
    def test_deposit_is_consumed_only_once(self):
        production, stock, used = calculate_inventory([[-10, 6, 6, 6]])
        np.testing.assert_array_equal(production, [[0, 0, 2, 6]])
        np.testing.assert_array_equal(stock, [[10, 4, 0, 0]])
        np.testing.assert_array_equal(used, [[0, 6, 4, 0]])

    def test_later_deposits_do_not_undo_earlier_production(self):
        production, stock, used = calculate_inventory([[6, -10, 6, 6]])
        np.testing.assert_array_equal(production, [[6, 0, 0, 2]])
        np.testing.assert_array_equal(stock, [[0, 10, 4, 0]])
        np.testing.assert_array_equal(used, [[0, 0, 6, 4]])

    def test_clinker_expires_before_fourth_year(self):
        production, stock, used = calculate_clinker_inventory([[-10, 6, 0, 6]])
        np.testing.assert_array_equal(production, [[0, 0, 0, 6]])
        np.testing.assert_array_equal(stock, [[10, 4, 4, 0]])
        np.testing.assert_array_equal(used, [[0, 6, 0, 0]])

    def test_fifo_preserves_newer_stock_for_following_year(self):
        production, stock, used = calculate_clinker_inventory([[-10, -10, 10, 10]])
        np.testing.assert_array_equal(production, [[0, 0, 0, 0]])
        np.testing.assert_array_equal(stock, [[10, 20, 10, 0]])
        np.testing.assert_array_equal(used, [[0, 0, 10, 10]])

    def test_clinker_deducts_prior_withdrawals_within_window(self):
        production, stock, used = calculate_clinker_inventory([[-10, 6, 6]])
        np.testing.assert_array_equal(production, [[0, 0, 2]])
        np.testing.assert_array_equal(stock, [[10, 4, 0]])
        np.testing.assert_array_equal(used, [[0, 6, 4]])

    def test_vectorized_matches_chronological_accounting_and_conserves_mass(self):
        rng = np.random.default_rng(42)
        for years in [0, 1, 2, 3, 4, 28, 92]:
            net = rng.normal(0, 100, (256, years))
            for function, lifetime in [(calculate_inventory, None),
                                       (calculate_clinker_inventory, 3)]:
                with self.subTest(years=years, lifetime=lifetime):
                    actual = function(net)
                    expected = chronological_inventory(net, lifetime)
                    for a, b in zip(actual, expected):
                        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-9)
                        self.assertTrue(np.all(a >= 0))
                    production, stock, used = actual
                    np.testing.assert_allclose(production+used, np.maximum(net, 0), atol=1e-9)
                    np.testing.assert_allclose(
                        np.maximum(-net, 0).cumsum(axis=1),
                        stock+used.cumsum(axis=1)+expected[3].cumsum(axis=1), atol=1e-9)

    def test_draws_are_independent_and_input_is_not_modified(self):
        net = np.array([[-10, 6, 6, 6], [1, -2, -3, 7], [8, 0, 0, 2]], dtype=np.float32)
        original = net.copy()
        for function in [calculate_inventory, calculate_clinker_inventory]:
            batch = function(net)
            reversed_batch = function(net[::-1])
            for a, b in zip(batch, reversed_batch):
                np.testing.assert_array_equal(a, b[::-1])
            for i in range(len(net)):
                for a, b in zip(batch, function(net[i:i+1])):
                    np.testing.assert_array_equal(a[i:i+1], b)
            np.testing.assert_array_equal(net, original)

    def test_hong_kong_saved_sand_example(self):
        # Historical draw 1, 2002-2005; kt, copied from the read-only audit.
        net = np.array([[-9835.088515749201, 5822.390794437379,
                          6852.745210289955, 4342.252218723297]])
        production, stock, _ = calculate_inventory(net)
        np.testing.assert_allclose(production, [[0, 0, 2840.0474889781326, 4342.252218723297]])
        np.testing.assert_allclose(stock, [[9835.088515749201, 4012.6977213118225, 0, 0]])


if __name__ == '__main__':
    unittest.main()
