"""Local top surfaces of previously printed non-planar infill.

Only the immediately previous layer is used for inter-layer clearance. The
active layer is indexed separately so a crossing within the same layer does
not become its own artificial substrate. Source solid safezones remain intact.
"""
from __future__ import annotations

import math


class NonPlanarLayerSurface:
    def __init__(self, extrusion_width: float, cell_size: float = 0.35):
        self.cell_size = cell_size
        self.inv_cell = 1.0 / cell_size
        self.step = min(0.17, cell_size * 0.5)
        # A pair of bead footprints intersects when their CENTERLINES are
        # within half-width + half-width. Keep actual sampled XY coordinates
        # instead of treating whole raster cells as occupied.
        radius = max(0.2, extrusion_width)
        self.radius_sq = radius ** 2
        self.neighborhood = range(-math.ceil(radius / cell_size),
                                  math.ceil(radius / cell_size) + 1)
        self.layer: int | None = None
        self.current: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
        self.previous: dict[tuple[int, int], list[tuple[float, float, float]]] = {}

    def enter_layer(self, layer: int) -> None:
        if self.layer == layer:
            return
        if self.layer is not None and layer == self.layer + 1:
            self.previous = self.current
        else:
            self.previous = {}
        self.current = {}
        self.layer = layer

    def add(self, x0: float, y0: float, z0: float,
            x1: float, y1: float, z1: float) -> None:
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 1e-8:
            return
        count = max(1, math.ceil(length / self.step))
        for i in range(count + 1):
            t = i / count
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            z = z0 + (z1 - z0) * t
            key = (math.floor(x * self.inv_cell),
                   math.floor(y * self.inv_cell))
            points = self.current.setdefault(key, [])
            # Typical lines contribute 2-4 probes per cell. Avoid an
            # unbounded list where many crossing paths revisit one cell.
            if len(points) < 12:
                points.append((x, y, z))
            else:
                nearest = min(range(len(points)), key=lambda j:
                              (x - points[j][0])**2 + (y - points[j][1])**2)
                if ((x - points[nearest][0])**2 +
                        (y - points[nearest][1])**2 < 0.035**2):
                    if z > points[nearest][2]:
                        points[nearest] = (x, y, z)
                else:
                    lowest = min(range(len(points)), key=lambda j: points[j][2])
                    if z > points[lowest][2]:
                        points[lowest] = (x, y, z)

    def top_on_path(self, x0: float, y0: float,
                    x1: float, y1: float) -> float | None:
        """Highest previous-layer top overlapping this XY strand footprint."""
        if not self.previous:
            return None
        length = math.hypot(x1 - x0, y1 - y0)
        count = max(1, math.ceil(length / self.step))
        top = None
        for i in range(count + 1):
            t = i / count
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            cx = math.floor(x * self.inv_cell)
            cy = math.floor(y * self.inv_cell)
            for dx in self.neighborhood:
                for dy in self.neighborhood:
                    for px, py, pz in self.previous.get((cx + dx, cy + dy), ()):
                        if (x - px)**2 + (y - py)**2 <= self.radius_sq:
                            if top is None or pz > top:
                                top = pz
        return top
