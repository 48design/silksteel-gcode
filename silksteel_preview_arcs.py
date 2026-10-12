"""Circular G2/G3 interpolation for the SilkSteel geometry preview.

Supports I/J/K centres, signed R arcs, G17/G18/G19 planes, helical
movement and full circles. An invalid arc is reported, never invented.
"""
from __future__ import annotations

import math


_PLANE = {
    "G17": (0, 1, 2, "I", "J"),
    "G18": (2, 0, 1, "K", "I"),
    "G19": (1, 2, 0, "J", "K"),
}


def _sweep(start: float, end: float, clockwise: bool, full: bool = False) -> float:
    if full:
        return -math.tau if clockwise else math.tau
    angle = (end - start) % math.tau
    if clockwise:
        angle = angle - math.tau if angle > 1e-10 else -math.tau
    elif angle < 1e-10:
        angle = math.tau
    return angle


def interpolate_arc(
    start: list[float] | tuple[float, ...],
    end: list[float] | tuple[float, ...],
    params: dict[str, float],
    clockwise: bool,
    plane: str = "G17",
    absolute_center: bool = False,
    resolution: float = 0.7,
    max_segments: int = 8192,
) -> tuple[list[list[float]], float] | None:
    """Return point samples and true spatial path length, or None if malformed."""
    if plane not in _PLANE:
        return None
    u, v, w, cu, cv = _PLANE[plane]
    p0 = (start[u], start[v])
    p1 = (end[u], end[v])
    if not all(math.isfinite(x) for x in (*start, *end, *params.values())):
        return None
    same = math.dist(p0, p1) < 1e-7
    has_center = cu in params or cv in params
    has_radius = "R" in params
    if not has_center and not has_radius:
        return None

    if has_center:
        center = (
            (params.get(cu, p0[0] if absolute_center else 0.0)
             if absolute_center else p0[0] + params.get(cu, 0.0)),
            (params.get(cv, p0[1] if absolute_center else 0.0)
             if absolute_center else p0[1] + params.get(cv, 0.0)),
        )
        radius = math.dist(p0, center)
        if radius <= 1e-7:
            return None
        error = abs(math.dist(p1, center) - radius)
        if not same and error > max(0.06, radius * 0.01):
            return None
    else:
        signed_radius = params["R"]
        radius = abs(signed_radius)
        chord = math.dist(p0, p1)
        if radius <= 1e-7 or chord <= 1e-7 or chord > 2 * radius + 1e-5:
            return None
        middle = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
        height = math.sqrt(max(0.0, radius * radius - chord * chord / 4))
        normal = (-(p1[1] - p0[1]) / chord, (p1[0] - p0[0]) / chord)
        candidates = [
            (middle[0] + sign * normal[0] * height,
             middle[1] + sign * normal[1] * height)
            for sign in (1, -1)
        ]

        def size(candidate: tuple[float, float]) -> float:
            start_angle = math.atan2(p0[1] - candidate[1], p0[0] - candidate[0])
            end_angle = math.atan2(p1[1] - candidate[1], p1[0] - candidate[0])
            return abs(_sweep(start_angle, end_angle, clockwise))

        center = min(candidates, key=size) if signed_radius > 0 else max(candidates, key=size)

    a0 = math.atan2(p0[1] - center[1], p0[0] - center[0])
    a1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
    sweep = _sweep(a0, a1, clockwise, same)
    # P specifies additional full revolutions in common firmware dialects.
    turns = params.get("P", 0.0)
    if not 0 <= turns <= 32 or turns != int(turns):
        return None
    if turns and not same:
        sweep += math.copysign(turns * math.tau, sweep)
    elif same and turns > 1:
        sweep *= int(turns)

    length = math.hypot(radius * abs(sweep), end[w] - start[w])
    if not math.isfinite(length) or length <= 0:
        return None
    samples = max(2, math.ceil(length / resolution),
                  math.ceil(abs(sweep) / (math.pi / 36)))
    # Cap tessellation cost without discarding otherwise valid geometry.
    # The arc remains curved even when the requested resolution cannot fit.
    samples = min(samples, max_segments)
    points = [list(start)]
    for i in range(1, samples):
        t = i / samples
        a = a0 + t * sweep
        point = list(start)
        point[u] = center[0] + radius * math.cos(a)
        point[v] = center[1] + radius * math.sin(a)
        point[w] = start[w] + t * (end[w] - start[w])
        points.append(point)
    points.append(list(end))
    return points, length
