"""Material-conserving, substrate-aware idealized G-code bead cross sections.

The surface map is only a preview approximation. It stores two elevations for
each occupied XY cell so same-height neighboring paths do not obscure an older
supporting layer. The print bed is treated as the support for the first layer.
"""
from __future__ import annotations

import math


class DepositedSurface:
    """Sparse local XY surface map, updated in deposition order."""

    def __init__(self, resolution: float = 0.40, sample_spacing: float = 0.45):
        self.resolution = resolution
        self.inv = 1.0 / resolution
        self.sample_spacing = sample_spacing
        self._tops: dict[tuple[int, int], tuple[float, float]] = {}

    @staticmethod
    def width_for_area(area: float, height: float) -> float:
        return area / height + (1.0 - math.pi / 4.0) * height

    @staticmethod
    def free_diameter(area: float) -> float:
        return math.sqrt(4.0 * area / math.pi)

    def support_top(self, x: float, y: float, z: float, clearance: float) -> float | None:
        levels = self._tops.get((math.floor(x * self.inv), math.floor(y * self.inv)))
        if levels is None:
            return None
        threshold = z - clearance
        if levels[0] <= threshold:
            return levels[0]
        if math.isfinite(levels[1]) and levels[1] <= threshold:
            return levels[1]
        return None

    def _effective_height(
        self, x: float, y: float, z: float, area: float, nominal: float,
        first_layer: float, layer_z: float | None = None
    ) -> float:
        diameter = self.free_diameter(area)
        clearance = max(0.015, min(0.055, nominal * 0.25))
        support = self.support_top(x, y, z, clearance)
        # The build plate is the known support for the first actual layer.
        if z <= first_layer + 0.025:
            support = max(support, 0.0) if support is not None else 0.0
        # A near-coplanar path is NOT evidence of an additional squeezed
        # layer. Sparse sampling can mistake adjacent tracks for support,
        # producing a 0.02-mm "height" and absurdly wide plastic ribbons.
        # Never squeeze the predicted strand below the programmed pass height.
        minimum_height = min(nominal, diameter)
        if support is None:
            return diameter
        gap = max(0.0, z - support)
        # A raised non-planar path has extra clearance even when coarse XY
        # occupancy accidentally reports a neighboring same-layer strand.
        if layer_z is not None:
            gap = max(gap, nominal + max(0.0, z - layer_z))
        return min(diameter, max(minimum_height, gap))

    def sections(
        self, start: tuple[float, float, float], end: tuple[float, float, float],
        area: float, nominal: float, first_layer: float,
        layer_z: float | None = None
    ) -> list[tuple[float, float, float, float]]:
        """Return t0, t1, width, height. All pieces preserve A=volume/length.

        Probe the whole original motion *before* depositing any new material:
        otherwise a long G1 would incorrectly support itself. Neighboring
        heights are merged into longer sections when they agree closely.
        """
        if area <= 0 or not math.isfinite(area):
            return []
        xy_length = math.hypot(end[0] - start[0], end[1] - start[1])
        if xy_length < 1e-8:
            return []
        samples = max(1, min(8192, math.ceil(xy_length / self.sample_spacing)))
        heights = []
        for i in range(samples):
            t = (i + 0.5) / samples
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            z = start[2] + (end[2] - start[2]) * t
            heights.append(self._effective_height(
                x, y, z, area, nominal, first_layer, layer_z))

        # Group similar cross-sections to keep the GPU instance count small.
        tolerance = max(0.014, min(0.035, nominal * 0.13))
        ranges: list[tuple[float, float, float, float]] = []
        begin, total = 0, heights[0]
        for i in range(1, samples + 1):
            if i < samples and abs(heights[i] - total / (i - begin)) <= tolerance:
                total += heights[i]
                continue
            height = total / (i - begin)
            width = self.width_for_area(area, height)
            ranges.append((begin / samples, i / samples, width, height))
            if i < samples:
                begin, total = i, heights[i]
        return ranges

    def deposit(
        self, start: tuple[float, float, float], end: tuple[float, float, float],
        sections: list[tuple[float, float, float, float]]
    ) -> None:
        """Rasterize deposited top surfaces; use bounded bead footprint width."""
        for t0, t1, width, _height in sections:
            x0 = start[0] + (end[0] - start[0]) * t0
            y0 = start[1] + (end[1] - start[1]) * t0
            x1 = start[0] + (end[0] - start[0]) * t1
            y1 = start[1] + (end[1] - start[1]) * t1
            length = math.hypot(x1 - x0, y1 - y0)
            n = max(1, math.ceil(length / (self.resolution * 0.72)))
            radius = min(2.0, width * 0.5) + self.resolution * 0.55
            grid_radius = math.ceil(radius * self.inv)
            rr = radius * radius
            for step in range(n + 1):
                t = step / n
                x = x0 + (x1 - x0) * t
                y = y0 + (y1 - y0) * t
                z = start[2] + (end[2] - start[2]) * (t0 + (t1 - t0) * t)
                cx = math.floor(x * self.inv)
                cy = math.floor(y * self.inv)
                for dx in range(-grid_radius, grid_radius + 1):
                    gx = cx + dx
                    if ((gx + 0.5) * self.resolution - x) ** 2 > rr:
                        continue
                    for dy in range(-grid_radius, grid_radius + 1):
                        gy = cy + dy
                        if (((gx + 0.5) * self.resolution - x) ** 2 +
                                ((gy + 0.5) * self.resolution - y) ** 2 > rr):
                            continue
                        key = (gx, gy)
                        high, low = self._tops.get(key, (-math.inf, -math.inf))
                        if z > high + 0.025:
                            self._tops[key] = z, high
                        elif z < high - 0.025 and z > low + 0.025:
                            self._tops[key] = high, z
