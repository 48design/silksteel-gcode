"""Analytical volume-conserving flow profiles for every G-code extrusion.

The nozzle speed and deposited material per millimetre determine the free
strand's idealized height. Available space below the nozzle compresses this
profile, conserving cross-sectional area by expanding its width. The preview
does not simulate prior-bead collisions or maintain a spatial occupancy raster.
"""
from __future__ import annotations

import math

from silksteel_preview_surface import DepositedSurface

CORNER_LOSS = (4.0 - math.pi) * 0.4 ** 2


class IdealizedProfile:
    """Compute one volume-preserving cross section per source path segment."""

    @staticmethod
    def area_for_size(width: float, height: float) -> float:
        return width * height - CORNER_LOSS * min(width, height) ** 2

    @staticmethod
    def width_for_area(area: float, height: float) -> float:
        wide = area / height + CORNER_LOSS * height
        if wide >= height:
            return wide
        return 2.0 * area / (
            height + math.sqrt(max(0.0, height * height - 4.0 * CORNER_LOSS * area))
        )

    @staticmethod
    def free_square_side(area: float) -> float:
        return math.sqrt(area / (1.0 - CORNER_LOSS))

    @staticmethod
    def sections(
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        area: float,
        nominal: float,
        first_layer: float,
        layer_z: float | None = None,
        speed: float = 30.0,
        *,
        role: int = 0,
        brick_pass: int = 0,
        bed_reference: bool = False,
    ) -> list[tuple[float, float, float, float]]:
        """Return (t0, t1, width, height); never subdivide to probe support.

        G-code Z is the top of each rendered bead. The default support plane
        is layer_z - nominal. For explicitly replayed paths, the pass height
        is known and takes precedence over the nominal layer reference.
        """
        if not (0 < area <= 16 and math.isfinite(area) and
                0 < nominal and math.isfinite(nominal)):
            return []
        if math.hypot(end[0] - start[0], end[1] - start[1]) <= 1e-8:
            return []

        nozzle_z = (start[2] + end[2]) * 0.5
        reference_z = nozzle_z if layer_z is None else layer_z

        # One flow model for ALL roles: E / XYZ distance supplies the
        # cross-sectional area, and actual feedrate determines the idealized
        # free-strand shape. Only the available nozzle-to-support gap varies
        # with the feature and its explicitly known pass geometry.
        free_height = DepositedSurface.gravity_height(area, speed)

        if (layer_z is None or bed_reference) and role not in (1, 5, 6) and brick_pass == 0:
            # The only known substrate of an unmarked/first-layer extrusion
            # is the bed at Z=0, not a fictitious nominal-thickness plane.
            available_height = nozzle_z
        elif role == 6:
            # A freely suspended bridge has no supporting substrate.
            available_height = math.inf
        elif role == 1 or brick_pass == 2:
            # Multi-pass paths have an explicit deposited pass thickness.
            available_height = nominal
        elif role == 5:
            # Valley-filling passes use increments <= 0.1 mm.
            available_height = min(nominal, 0.1)
        elif brick_pass == 1:
            available_height = nominal / 0.75
        else:
            # This applies identically to ordinary and non-planar paths.
            # Elevated Z provides more space for extra E to flow DOWN;
            # it doesn't automatically imply an equally tall bead.
            available_height = nominal + (nozzle_z - reference_z)

        height = min(free_height, available_height)
        # Avoid unstable or inverted meshes in exceptionally deep valleys;
        # the same minimum applies to every profile (except thin fill passes).
        height = max(0.05, height)
        if role != 5:
            height = max(nominal * 0.5, height)
        width = IdealizedProfile.width_for_area(area, height)
        if not (math.isfinite(width) and math.isfinite(height) and
                0 < width <= 40 and 0 < height <= 40):
            return []
        return [(0.0, 1.0, width, height)]
