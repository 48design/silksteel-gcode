"""Local non-planar safezone amplitude envelope; no inter-layer state.

All bounds derive from ORIGINAL slicer layer heights and solid occupancy.
Never reuse an adjusted extrusion Z as a floor for the following layer.
"""
from __future__ import annotations

import math

MAX_BOUNDARY_SLOPE = 0.25


def solid_surfaces(gx, gy, layer, solid_regions, z_layer_map, fallback_height):
    """Return (lower solid TOP, upper solid UNDERSIDE) at this infill cell.

    G-code layer Z is the top of its deposited material, not its underside.
    An infill path in a cell actually occupied by a solid is planar.
    Missing lower solids fall back to the bed; missing roofs are unbounded.
    """
    floor = 0.0
    ceiling = None
    for start, end, _, _ in solid_regions.get((gx, gy), ()):
        if start <= layer <= end:
            return None
        if end < layer:
            z = z_layer_map.get(end)
            if z is not None:
                floor = max(floor, z)
        elif start > layer:
            z = z_layer_map.get(start - 1)
            if z is None:
                first = z_layer_map.get(start)
                if first is not None:
                    z = first - fallback_height
            if z is not None:
                ceiling = z if ceiling is None else min(ceiling, z)
    return floor, ceiling


def bound_offset(raw_z, planar_z, layer_height, floor, ceiling,
                 boundary_slope=MAX_BOUNDARY_SLOPE):
    """Gradually reduce only a displacement pointing into a solid boundary.

    A hard absolute Z clamp creates identical-Z plateaus across consecutive
    layers. A limited-slope signed envelope does not: it changes by at most
    boundary_slope * layer_height for a Z-invariant noise value. Noise's own
    Z gradient still needs to be sufficiently gentle (see Perlin LUT limiter).

    No history, recursion, path-wide max, or full-layer preplanning is used.
    The first layer retains positive hills; the last retains negative valleys.
    """
    if not all(math.isfinite(n) for n in
               (raw_z, planar_z, layer_height, floor)):
        return planar_z
    if layer_height <= 0 or not 0 < boundary_slope < 1:
        return planar_z
    if ceiling is not None and not math.isfinite(ceiling):
        return planar_z
    # The commanded Z is the TOP of the new extruded bead.
    lower_limit = floor + layer_height
    if planar_z < lower_limit - 1e-5:
        return planar_z
    if ceiling is not None and planar_z > ceiling + 1e-5:
        return planar_z
    offset = raw_z - planar_z
    if offset < 0:
        # Offset into previous solid: fade the valley toward planar.
        offset = max(offset, -boundary_slope *
                     max(0.0, planar_z - lower_limit))
    elif offset > 0 and ceiling is not None:
        # Offset into future solid: fade hills, not upper-layer valleys.
        offset = min(offset, boundary_slope *
                     max(0.0, ceiling - planar_z))
    return planar_z + offset


def constrain_perlin_vertical_gradient(noise, max_delta):
    """Bound Z differences per LUT slice, preserving detailed XY patterns.

    Forward/backward Lipschitz projections are averaged to avoid biasing
    every Z slice toward the lowest or highest slice. The result stays
    within the source extrema and has |delta Z noise| <= max_delta per cell.
    """
    if max_delta <= 0 or noise.shape[2] < 2:
        return noise
    forward = noise.copy()
    # The source is newly generated for this LUT and may be updated in place.
    # Reuse it for the reverse projection: only one auxiliary 3D array.
    for z in range(1, noise.shape[2]):
        forward[:, :, z].clip(
            forward[:, :, z - 1] - max_delta,
            forward[:, :, z - 1] + max_delta,
            out=forward[:, :, z])
    for z in range(noise.shape[2] - 2, -1, -1):
        noise[:, :, z].clip(
            noise[:, :, z + 1] - max_delta,
            noise[:, :, z + 1] + max_delta,
            out=noise[:, :, z])
    forward += noise
    forward *= 0.5
    return forward
