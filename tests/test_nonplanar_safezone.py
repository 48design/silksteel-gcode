"""Regression tests for continuous safezone-aware Z displacement."""
import math
import unittest

import numpy as np
from tools.nonplanar_safezone import (
    solid_surfaces, bound_offset, constrain_perlin_vertical_gradient
)


class NonPlanarSafezoneTests(unittest.TestCase):
    def test_original_slicer_surfaces_not_deposited_layer_z(self):
        zmap = {2: .76, 3: 1.04, 4: 1.32, 5: 1.60, 6: 1.88}
        regions = {(5, 8): [(0, 2, .2, 1.04), (6, 7, 1.88, 2.44)]}
        for layer in (3, 4, 5):
            self.assertEqual(solid_surfaces(5, 8, layer, regions, zmap, .28),
                             (.76, 1.60))
        self.assertIsNone(solid_surfaces(5, 8, 6, regions, zmap, .28))

    def test_first_hill_and_last_valley_remain(self):
        floor, roof, h = .76, 1.60, .28
        self.assertAlmostEqual(bound_offset(.55, 1.04, h, floor, roof), 1.04)
        self.assertGreater(bound_offset(1.50, 1.04, h, floor, roof), 1.04)
        self.assertAlmostEqual(bound_offset(1.90, 1.60, h, floor, roof), 1.60)
        self.assertLess(bound_offset(1.2, 1.60, h, floor, roof), 1.60)

    def test_middle_keeps_unrestricted_noise(self):
        self.assertAlmostEqual(bound_offset(1.38, 1.32, .28, .76, 1.88), 1.38)
        self.assertAlmostEqual(bound_offset(1.26, 1.32, .28, .76, 1.88), 1.26)

    def test_constant_negative_wave_no_duplicate_z_plateaus(self):
        h, floor, roof = .28, .76, 3.28
        zs = [bound_offset(z - 1.12, z, h, floor, roof)
              for z in (1.04, 1.32, 1.60, 1.88, 2.16, 2.44, 2.72, 3.0, 3.28)]
        self.assertGreaterEqual(
            min(b-a for a,b in zip(zs,zs[1:])), .75*h - 1e-8)
        self.assertAlmostEqual(zs[0], 1.04)
        self.assertLess(zs[-1], 3.28)

    def test_constant_positive_wave_no_duplicate_roof_plateau(self):
        h, floor, roof = .28, .76, 3.28
        zs = [bound_offset(z + 1.12, z, h, floor, roof)
              for z in (1.04, 1.32, 1.60, 1.88, 2.16, 2.44, 2.72, 3.0, 3.28)]
        self.assertGreaterEqual(
            min(b-a for a,b in zip(zs,zs[1:])), .75*h - 1e-8)
        self.assertGreater(zs[0], 1.04)
        self.assertAlmostEqual(zs[-1], 3.28)

    def test_safe_single_layer_gap_becomes_planar(self):
        self.assertAlmostEqual(bound_offset(.2, 1.04, .28, .76, 1.04), 1.04)
        self.assertAlmostEqual(bound_offset(1.8, 1.04, .28, .76, 1.04), 1.04)

    def test_perlin_vertical_projection_bounds_step_without_loss_xy(self):
        rng = np.random.default_rng(2)
        field = rng.uniform(-1, 1, size=(8, 9, 17))
        smooth = constrain_perlin_vertical_gradient(field, .09)
        self.assertEqual(smooth.shape, field.shape)
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(smooth,axis=2)))), .09000000001)
        self.assertLessEqual(float(np.max(smooth)), 1.0)
        self.assertGreaterEqual(float(np.min(smooth)), -1.0)
        self.assertGreater(float(np.std(smooth[:,:,0])), .05)

    def test_bounded_3d_variations_stay_ordered_near_solids(self):
        h, floor, roof = .28, .76, 3.28
        for x,y in ((.1,1.7),(2,3),(3.3,.4),(4.5,5.2)):
            values=[]
            for i in range(9):
                z=1.04+i*h
                n=(math.sin(.8*x)+math.sin(.8*y)+math.sin(.4*z))/3
                values.append(bound_offset(z+1.12*n,z,h,floor,roof))
            self.assertGreater(
                min(b-a for a,b in zip(values,values[1:])),.6*h)

    def test_no_history_or_cascade(self):
        a=bound_offset(1.2,1.04,.28,.76,3.28)
        _=bound_offset(2,1.60,.28,.76,3.28)
        b=bound_offset(1.2,1.04,.28,.76,3.28)
        self.assertEqual(a,b)


if __name__ == "__main__":
    unittest.main()
