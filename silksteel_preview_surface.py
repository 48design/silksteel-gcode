"""Material-conserving, substrate-aware idealized G-code bead cross sections.

The surface map is only a preview approximation. It stores two elevations for
each occupied XY cell so same-height neighboring paths do not obscure an older
supporting layer. The print bed is treated as the support for the first layer.
"""
from __future__ import annotations

import math

CORNER_FRACTION = 0.4
CORNER_LOSS = (4.0 - math.pi) * CORNER_FRACTION ** 2


class DepositedSurface:
    """Sparse local XY surface map, updated in deposition order."""

    def __init__(self, resolution: float = 0.40, sample_spacing: float = 0.45):
        self.resolution = resolution
        self.inv = 1.0 / resolution
        self.sample_spacing = sample_spacing
        self._tops: dict[tuple[int, int], tuple[float, float]] = {}
        # The active, connected extrusion is NOT part of its own substrate.
        # Commit only when the strand ends (travel, retraction, feature or
        # layer change). Otherwise ascending non-planar G1 moves see the
        # previous lower G1 as support, while descending moves do not.
        self._pending: dict[tuple[int, int], float] = {}
        self._strand_end: tuple[float, float, float] | None = None
        self._strand_role: int | None = None

    def begin_strand(self, start: tuple[float, float, float], role: int) -> None:
        if (self._strand_end is None or role != self._strand_role or
                math.dist(start, self._strand_end) > 0.012):
            self.finish_strand()
        self._strand_end = start
        self._strand_role = role

    def advance_strand(self, end: tuple[float, float, float]) -> None:
        if self._strand_end is not None:
            self._strand_end = end

    def finish_strand(self) -> None:
        """Expose this strand as support only to later, distinct extrusions."""
        for key, z in self._pending.items():
            high, low = self._tops.get(key, (-math.inf, -math.inf))
            if z > high + 0.025:
                self._tops[key] = (z, high)
            elif z < high - 0.025 and z > low + 0.025:
                self._tops[key] = (high, z)
        self._pending.clear()
        self._strand_end = None
        self._strand_role = None

    @staticmethod
    def area_for_size(width: float, height: float) -> float:
        return width * height - CORNER_LOSS * min(width, height) ** 2

    @staticmethod
    def width_for_area(area: float, height: float) -> float:
        """Exact area for a rounded rectangle: r=0.4*min(w,h)."""
        if area <= 0 or height <= 0:
            return 0.0
        wide = area / height + CORNER_LOSS * height
        if wide >= height:
            return wide
        discriminant = max(0.0, height * height - 4.0 * CORNER_LOSS * area)
        return 2.0 * area / (height + math.sqrt(discriminant))

    @staticmethod
    def free_square_side(area: float) -> float:
        return math.sqrt(area / (1.0 - CORNER_LOSS))

    @staticmethod
    def free_diameter(area: float) -> float:
        """Legacy reference diameter; not the extrusion profile shape."""
        return math.sqrt(4.0 * area / math.pi)

    @classmethod
    def gravity_height(cls, area: float, speed: float) -> float:
        """Conservative heuristic using nozzle speed and volume flow."""
        side = cls.free_square_side(area)
        speed = max(0.2, speed)
        flow = area * speed
        stretch = min(0.9, 0.9 * (18.0 / (9.0 + speed)) *
                      (3.0 / (3.0 + flow)))
        return side * (1.0 + stretch)

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
        first_layer: float, layer_z: float | None = None,
        speed: float = 30.0, desired_height: float | None = None
    ) -> float:
        desired = (self.gravity_height(area, speed) if desired_height is None
                   else desired_height)
        # Physical first-layer bed support; do not invent support at every Z.
        support = self.support_top(
            x, y, z, max(0.015, min(0.04, nominal * 0.15))
        )
        if z <= first_layer + 0.025:
            support = max(support, 0.0) if support is not None else 0.0
        if support is None:
            return desired
        gap = z - support
        # An implausibly tiny gap in this low-resolution XY map is usually a
        # nearby same-layer crossing, not a real supporting surface. Do not
        # create metre-wide extrusion ribbons from this false contact.
        if gap < max(0.075, nominal * 0.45):
            return desired
        # Nozzle top remains at Z. Stretch stops exactly at the support;
        # further compression increases width, not the deposited volume.
        return min(desired, gap)

    def sections(
        self, start: tuple[float, float, float], end: tuple[float, float, float],
        area: float, nominal: float, first_layer: float,
        layer_z: float | None = None, speed: float = 30.0
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
        free_height = self.gravity_height(area, speed)
        for i in range(samples):
            t = (i + 0.5) / samples
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            z = start[2] + (end[2] - start[2]) * t
            heights.append(self._effective_height(
                x, y, z, area, nominal, first_layer, layer_z, speed,
                free_height))

        # The receiver grid is only 0.4 mm wide. At crossing infill paths a
        # center probe may alternate between a supported cell and empty space,
        # producing abrupt bead-width jumps every ~0.45 mm. Reject isolated
        # support spikes first, then transition continuously between sustained
        # supported/free runs. The section area stays strictly A=E/length.
        if len(heights) >= 5 and max(heights) - min(heights) > 0.012:
            heights = self._stabilize_heights(heights)

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

    @staticmethod
    def _stabilize_heights(heights: list[float]) -> list[float]:
        """Denoise sparse support probes without modifying deposited area.

        A five-point median ignores one-cell occupancy spikes. A compact
        normalized triangular filter softens actual support boundaries over
        around two millimetres, avoiding discontinuous stadium radii and
        excessive independent cap meshes. Endpoints are edge-replicated.
        """
        n = len(heights)
        if n < 5:
            return heights[:]
        median = []
        for i in range(n):
            window = sorted(heights[max(0, i - 2):min(n, i + 3)])
            median.append(window[len(window) // 2])
        weights = (1, 2, 3, 2, 1)
        return [
            sum(median[min(n - 1, max(0, i + j - 2))] * weight
                for j, weight in enumerate(weights)) / 9.0
            for i in range(n)
        ]

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
                    x_sq = ((gx + 0.5) * self.resolution - x) ** 2
                    if x_sq > rr:
                        continue
                    for dy in range(-grid_radius, grid_radius + 1):
                        gy = cy + dy
                        if x_sq + ((gy + 0.5) * self.resolution - y) ** 2 > rr:
                            continue
                        key = (gx, gy)
                        if self._strand_end is not None:
                            previous = self._pending.get(key)
                            if previous is None or z > previous:
                                self._pending[key] = z
                        else:
                            high, low = self._tops.get(key, (-math.inf, -math.inf))
                            if z > high + 0.025:
                                self._tops[key] = z, high
                            elif z < high - 0.025 and z > low + 0.025:
                                self._tops[key] = high, z
