"""Reverse/forward non-planar planning must preserve the upper valleys."""
import unittest

from tools.nonplanar_plan import NonPlanarPathPlan


class NonPlanarLookaheadTests(unittest.TestCase):
    def test_upper_valley_is_kept_by_lowering_the_earlier_hill(self):
        p = NonPlanarPathPlan(.45)
        # Actual sparse-infill overlap, with an elevated lower layer and
        # the last layer's valley below the Z1.600 solid roof.
        p.add(3, (100, 0), (0, 0), (1, 0), 1.60, 1.04, 1.60, .28)
        p.add(4, (200, 0), (0, .2), (1, .2), 1.48, 1.04, 1.60, .28)
        z = p.solve()
        self.assertAlmostEqual(z[200, 0], 1.48)
        self.assertLessEqual(z[100, 0], 1.20 + 1e-5)
        self.assertGreater(p.downward, 0)
        self.assertEqual(p.upward, 0)
        self.assertFalse(p.conflicts)

    def test_no_upward_cascade_into_solid_roof(self):
        p = NonPlanarPathPlan(.45)
        for layer, desired in [(3, 1.60), (4, 1.60),
                               (5, 1.60), (6, 1.60)]:
            p.add(layer, (layer, 0), (0, 0), (1, 0),
                  desired, .76, 1.60, .28)
        z = p.solve()
        self.assertAlmostEqual(z[6, 0], 1.60)
        for layer in (3, 4, 5):
            self.assertGreaterEqual(z[layer + 1, 0] -
                                    z[layer, 0], .28 - 1e-5)
        self.assertEqual(p.upward, 0)

    def test_unrelated_xy_strands_retain_full_noise(self):
        p = NonPlanarPathPlan(.45)
        p.add(3, (1, 0), (0, 0), (1, 0), 1.58, 1.04, 1.60, .28)
        p.add(4, (2, 0), (0, 3), (1, 3), 1.10, 1.04, 1.60, .28)
        z = p.solve()
        self.assertAlmostEqual(z[1, 0], 1.58)
        self.assertAlmostEqual(z[2, 0], 1.10)
        self.assertEqual(p.downward, 0)
        self.assertEqual(p.upward, 0)

    def test_impossible_floor_requires_forward_repair(self):
        p = NonPlanarPathPlan(.45)
        p.add(3, (1, 0), (0, 0), (1, 0), 1.04, 1.04, 1.60, .28)
        p.add(4, (2, 0), (0, .2), (1, .2), 1.00, 1.04, 1.60, .28)
        z = p.solve()
        self.assertAlmostEqual(z[1, 0], 1.04)
        self.assertAlmostEqual(z[2, 0], 1.32)
        self.assertGreater(p.upward, 0)

    def test_unresolvable_roof_constraint_is_reported(self):
        p = NonPlanarPathPlan(.45)
        p.add(3, (1, 0), (0, 0), (1, 0), 1.04, 1.04, 1.05, .28)
        p.add(4, (2, 0), (0, .2), (1, .2), 1.05, 1.04, 1.10, .28)
        p.solve()
        self.assertTrue(any(c[2] == "roof_vs_support" for c in p.conflicts))


if __name__ == "__main__":
    unittest.main()
