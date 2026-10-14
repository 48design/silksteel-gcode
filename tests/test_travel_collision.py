import unittest

from tools.travel_collision import DepositedPathMap


class DepositedPathMapTests(unittest.TestCase):
    def test_same_layer_crossing_requires_clearance(self):
        roads = DepositedPathMap(0.45)
        roads.add(0, 0, 0.2, 20, 0, 0.2)
        self.assertAlmostEqual(roads.obstacle_top(20, -4, 0, 4, 0.2), 0.2)

    def test_adjacent_and_short_connection_require_no_hop(self):
        roads = DepositedPathMap(0.45)
        roads.add(0, 0, 0.2, 20, 0, 0.2)
        self.assertIsNone(roads.obstacle_top(20, 0.7, 0, 0.7, 0.2))
        self.assertIsNone(roads.obstacle_top(20, 0, 20, 0.6, 0.2))

    def test_older_layer_well_below_nozzle_is_safe(self):
        roads = DepositedPathMap(0.45)
        roads.add(0, 0, 0.2, 20, 0, 0.2)
        self.assertIsNone(roads.obstacle_top(20, -4, 0, 4, 0.4))

    def test_printed_nonplanar_height_is_spatially_local(self):
        roads = DepositedPathMap(0.45)
        roads.add(0, 0, 0.3, 20, 0, 0.6)
        high = roads.obstacle_top(10, -3, 10, 3, 0.55)
        self.assertIsNotNone(high)
        self.assertGreater(high, 0.4)
        self.assertIsNone(roads.obstacle_top(10, 5, 10, 9, 0.55))

    def test_future_paths_cannot_block_travel(self):
        roads = DepositedPathMap(0.45)
        self.assertIsNone(roads.obstacle_top(20, -4, 0, 4, 0.2))
        roads.add(0, 0, 0.2, 20, 0, 0.2)
        self.assertIsNotNone(roads.obstacle_top(20, -4, 0, 4, 0.2))


if __name__ == "__main__":
    unittest.main()
