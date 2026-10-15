"""Analytical extrusion profiles for the G-code viewer.

The viewer depicts the commanded material deposit, not a simulated collision
with previously deposited beads. The layer plane is the default substrate;
SilkSteel's explicit multi-pass annotations take precedence. This deliberately
does not maintain or sample a spatial occupancy raster.
"""
from __future__ import annotations

import math

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

        if (layer_z is None or bed_reference) and role not in (1, 5, 6) and brick_pass == 0:
            # Before the first slicer layer *and* in the actual first layer,
            # the only known substrate is the print bed (Z=0). A nominal
            # layer must not be imagined 0.2 mm below an elevated prime line.
            # This is an intentional geometry abstraction: Z gives the
            # nozzle-to-bed gap, regardless of extrusion amount. Cross-section
            # area determines width; do not invent a free-strand height cap.
            height = nozzle_z
        elif role == 1 or brick_pass == 2:
            # Smoothificator / second Bricklayers base pass: already defined
            # by the exact per-pass layer thickness in the processed G-code.
            height = nominal
        elif role == 6:
            # A bridge is assumed to span air, with a rounded free strand.
            height = IdealizedProfile.free_square_side(area)
        elif role == 5:
            # Valley fill is deliberately emitted as <= 0.1 mm Z increments.
            height = min(nominal, 0.1)
        elif brick_pass == 1:
            # First base pass starts at the normal layer height.
            height = nominal / 0.75
        elif role == 4:
            # Non-planar infill follows a 3D toolpath. Its programmed Z is
            # the strand TOP, not an extra layer of solid plastic down to
            # the slicer's planar reference. Keep the nominal cross-section
            # thickness while shifting the entire strand with the nozzle.
            # Source delta E already accounts for XYZ path length.
            height = nominal
        else:
            # Normal path or shifted perimeter:
            # substrate = layer reference minus nominal thickness.
            height = nominal + (nozzle_z - reference_z)

        # A negative deep valley offset would otherwise produce a negative
        # cross-section. Keep its volume represented by a finite thin strand.
        height = max(0.05, nominal * 0.5, height)
        width = IdealizedProfile.width_for_area(area, height)
        if not (math.isfinite(width) and math.isfinite(height) and
                0 < width <= 40 and 0 < height <= 40):
            return []
        return [(0.0, 1.0, width, height)]
