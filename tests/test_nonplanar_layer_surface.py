"""Regression tests for non-planar infill deposited by preceding layers."""
import unittest

from tools.nonplanar_layer_surface import NonPlanarLayerSurface


class PreviousNonPlanarLayerTests(unittest.TestCase):
    def test_reported_flat_lower_and_upper_paths_overlap_in_xy(self):
        # Exact sampled positions near G-code lines 13178 and 14985.
        # Both have Z1.040. The top path must not be printed at this Z.
        field = NonPlanarLayerSurface(extrusion_width=0.45)
        field.enter_layer(3)
        field.add(148.159, 97.915, 1.04,
                  147.564, 98.075, 1.04)
        field.add(147.564, 98.075, 1.04,
                  146.969, 98.234, 1.04)
        field.enter_layer(4)
        previous_top = field.top_on_path(
            148.023, 98.157, 147.407, 98.322)
        self.assertIsNotNone(previous_top)
        self.assertAlmostEqual(previous_top, 1.04)
        self.assertAlmostEqual(previous_top + 0.28, 1.32)

    def test_same_layer_crossing_does_not_support_itself(self):
        field = NonPlanarLayerSurface(0.45)
        field.enter_layer(3)
        field.add(0, 0, 1.04, 10, 0, 1.04)
        self.assertIsNone(field.top_on_path(5, -1, 5, 1))
        field.enter_layer(4)
        self.assertAlmostEqual(field.top_on_path(5, -1, 5, 1), 1.04)
        field.add(5, -1, 1.32, 5, 1, 1.32)
        self.assertAlmostEqual(field.top_on_path(5, -1, 5, 1), 1.04)
        field.enter_layer(5)
        self.assertAlmostEqual(field.top_on_path(5, -1, 5, 1), 1.32)

    def test_empty_space_and_skipped_layer(self):
        field = NonPlanarLayerSurface(0.45)
        field.enter_layer(3)
        field.add(0, 0, 1.04, 10, 0, 1.04)
        field.enter_layer(4)
        self.assertIsNone(field.top_on_path(5, 2, 5, 4))
        field.enter_layer(6)
        self.assertIsNone(field.top_on_path(5, -1, 5, 1))

    def test_nonplanar_z_slope_is_interpolated_along_strand(self):
        field = NonPlanarLayerSurface(0.45)
        field.enter_layer(3)
        field.add(0, 0, 1.04, 10, 0, 1.60)
        field.enter_layer(4)
        near_peak = field.top_on_path(9, -0.1, 9, 0.1)
        near_start = field.top_on_path(1, -0.1, 1, 0.1)
        self.assertGreater(near_peak, near_start)


if __name__ == "__main__":
    unittest.main()
