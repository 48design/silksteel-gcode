"""Spatial index of *deposited* G-code paths for conservative XY-travel clearance.

Only previously extruded paths are indexed. The map stores the highest deposited
nozzle Z at each sampled XY cell, so memory is bounded by occupied XY area rather
than by layer count. No future geometry, global layer maxima or noise estimates
are treated as physical obstacles.
"""

import math


class DepositedPathMap:
    __slots__ = ("cell_size", "sample_step", "radius_sq", "cells")

    def __init__(self, extrusion_width, cell_size=0.35):
        self.cell_size = cell_size
        self.sample_step = min(0.18, cell_size * 0.5)
        # Do not classify normally adjacent extrusion lines as intersecting.
        # This tests the travel CENTERLINE against deposited bead centers.
        self.radius_sq = (min(0.30, max(0.19, extrusion_width * 0.57))) ** 2
        self.cells = {}

    def add(self, x0, y0, z0, x1, y1, z1):
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 1e-6:
            return
        count = max(1, math.ceil(length / self.sample_step))
        inv_cell = 1.0 / self.cell_size
        cells = self.cells
        for i in range(count + 1):
            t = i / count
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            z = z0 + (z1 - z0) * t
            key = (math.floor(x * inv_cell), math.floor(y * inv_cell))
            previous = cells.get(key)
            if previous is None or z > previous[0]:
                cells[key] = (z, x, y)

    def obstacle_top(self, x0, y0, x1, y1, nozzle_z, height_tolerance=0.08):
        """Highest prior deposit intersected by the *interior* of a travel.

        Small endpoint exclusions prevent every travel from being classified
        as colliding with the strand it just finished printing. A previously
        deposited surface well below the nozzle is not an obstruction.
        """
        distance = math.hypot(x1 - x0, y1 - y0)
        if distance < 0.02 or not self.cells:
            return None
        endpoint = min(0.42, distance * 0.50)
        interior = distance - 2.0 * endpoint
        if interior <= 0:
            return None
        count = max(1, math.ceil(interior / self.sample_step))
        inv_cell = 1.0 / self.cell_size
        radius_sq = self.radius_sq
        threshold = nozzle_z - height_tolerance
        cells = self.cells
        highest = None
        for i in range(count + 1):
            t = (endpoint + interior * i / count) / distance
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            cx, cy = math.floor(x * inv_cell), math.floor(y * inv_cell)
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    deposit = cells.get((cx + ox, cy + oy))
                    if deposit is None or deposit[0] < threshold:
                        continue
                    top, px, py = deposit
                    if (x - px) ** 2 + (y - py) ** 2 > radius_sq:
                        continue
                    if highest is None or top > highest:
                        highest = top
        return highest
