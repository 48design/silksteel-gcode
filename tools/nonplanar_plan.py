"""Bidirectional non-planar layer planning, before any material is extruded.

A one-way 'previous top + layer height' clamp accumulates hills and forces upper
layers into the solid roof. Resolve from the future towards the bed instead:
protect desired upper valleys by trimming ONLY lower paths whose actual XY
footprints overlap. A short forward pass addresses infeasible lower-floor
constraints. This is deliberately independent of the material-flow model.
"""
from __future__ import annotations

import math
from collections import defaultdict


class _Footprint:
    def __init__(self, width, cell_size=0.35):
        self.cell_size = cell_size
        self.inv = 1 / cell_size
        self.step = min(0.17, cell_size / 2)
        radius = max(width, 0.2)
        self.radius_sq = radius * radius
        self.neighbors = range(-math.ceil(radius / cell_size),
                               math.ceil(radius / cell_size) + 1)
        self.cells = defaultdict(list)

    def add(self, x0, y0, x1, y1, z):
        length = math.hypot(x1 - x0, y1 - y0)
        count = max(1, math.ceil(length / self.step))
        for n in range(count + 1):
            t = n / count
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            cell = (math.floor(x * self.inv), math.floor(y * self.inv))
            # Keep the full path instead of replacing crossings with a
            # single cell max; multiple nearby tracks may differ in height.
            self.cells[cell].append((x, y, z))

    def nearby(self, x0, y0, x1, y1, minimum):
        """Min future Z, or max earlier Z, along the complete XY footprint."""
        length = math.hypot(x1 - x0, y1 - y0)
        count = max(1, math.ceil(length / self.step))
        result = None
        for n in range(count + 1):
            t = n / count
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            gx = math.floor(x * self.inv)
            gy = math.floor(y * self.inv)
            for dx in self.neighbors:
                for dy in self.neighbors:
                    for px, py, pz in self.cells.get((gx + dx, gy + dy), ()):
                        if (x - px) ** 2 + (y - py) ** 2 <= self.radius_sq:
                            if result is None:
                                result = pz
                            elif minimum and pz < result:
                                result = pz
                            elif not minimum and pz > result:
                                result = pz
        return result


class NonPlanarPathPlan:
    def __init__(self, extrusion_width):
        self.width = extrusion_width
        self.layers = defaultdict(list)
        self.heights = {}
        self.conflicts = []
        self.downward = 0
        self.upward = 0

    def add(self, layer, key, start, end, desired, floor, roof, layer_height):
        self.layers[layer].append((key, start, end, desired, floor, roof))
        self.heights[layer] = layer_height

    def _footprint(self, records, values):
        surface = _Footprint(self.width)
        for key, start, end, _, _, _ in records:
            surface.add(*start, *end, values[key])
        return surface

    def solve(self):
        """Return immutable planned Z values keyed by (source line, subsegment).

        1. Reverse layer traversal preserves the LAST infill layer's negative
           valleys; over-high lower hills move down *before* becoming material.
        2. A forward repair only lifts material where the physical solid floor
           prevented the downward correction. It never touches unconstrained
           segments, so it cannot flatten every upper noise layer.
        """
        planned = {}
        self.conflicts = []
        self.downward = 0
        self.upward = 0
        future = None
        future_layer = None
        for layer in sorted(self.layers, reverse=True):
            rows = self.layers[layer]
            for key, start, end, desired, floor, roof in rows:
                z = max(floor, min(desired, roof))
                if future is not None and future_layer == layer + 1:
                    upper_top = future.nearby(*start, *end, minimum=True)
                    if upper_top is not None:
                        bound = upper_top - self.heights[future_layer]
                        if bound + 1e-5 < floor:
                            self.conflicts.append(
                                (layer, key, 'floor_vs_future', floor, bound))
                        elif bound < z:
                            z = bound
                            self.downward += 1
                planned[key] = z
            future = self._footprint(rows, planned)
            future_layer = layer

        previous = None
        previous_layer = None
        for layer in sorted(self.layers):
            rows = self.layers[layer]
            for key, start, end, _, floor, roof in rows:
                z = planned[key]
                if previous is not None and previous_layer == layer - 1:
                    support = previous.nearby(*start, *end, minimum=False)
                    if support is not None:
                        required = support + self.heights[layer]
                        if required > z + 1e-5:
                            if required > roof + 1e-5:
                                self.conflicts.append(
                                    (layer, key, 'roof_vs_support', required, roof))
                            else:
                                z = required
                                self.upward += 1
                planned[key] = z
            previous = self._footprint(rows, planned)
            previous_layer = layer

        return planned
