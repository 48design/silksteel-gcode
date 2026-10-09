#!/usr/bin/env python3
"""Audit unchanged wall paths in SilkSteel-processed G-code.

Usage:
    python tools/audit_outer_walls.py result.gcode --layer 25
    python tools/audit_outer_walls.py result.gcode --x 130 --y 90 --radius 15
    python tools/audit_outer_walls.py result.gcode --min-length 5 --top 50

No external dependencies. Read-only: never changes the print file.
"""
import argparse
from collections import Counter
import math
import re


WORD = re.compile(r"([XYZEF])([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))")
LAYER = re.compile(r"^;LAYER:(\d+)")
MARKER = "SMOOTHIFICATOR START"
END = "SMOOTHIFICATOR END"


def _parameters(code):
    return {name: float(value) for name, value in WORD.findall(code)}


def _distance_to_bbox(x, y, run):
    minx, miny, maxx, maxy = run["bbox"]
    return math.hypot(max(minx - x, 0, x - maxx),
                      max(miny - y, 0, y - maxy))


def audit(path):
    """Yield continuous positive-extrusion paths with provenance and source lines."""
    x = y = z = e = 0.0
    relative_e = False
    current_type = "(unmarked)"
    layer = -1
    active_smoothing = False
    run = None

    def finish():
        nonlocal run
        if run is None:
            return None
        result = run
        run = None
        return result

    with open(path, encoding="utf-8", errors="replace") as reader:
        for number, raw in enumerate(reader, 1):
            code = raw.split(";", 1)[0].strip()
            comment = raw.split(";", 1)[1].strip() if ";" in raw else ""
            barrier = False

            if "SMOOTHIFICATOR START:" in comment:
                barrier = True
                active_smoothing = True
            elif END in comment:
                barrier = True
                active_smoothing = False
            elif raw.startswith(";TYPE:"):
                barrier = True
                current_type = raw[6:].strip()
            elif raw.startswith(";LAYER_CHANGE"):
                barrier = True
                layer += 1
            else:
                match = LAYER.match(raw)
                if match:
                    barrier = True
                    layer = int(match.group(1))
                elif code.startswith("M82") and (len(code) == 3 or code[3].isspace()):
                    barrier = True
                    relative_e = False
                elif code.startswith("M83") and (len(code) == 3 or code[3].isspace()):
                    barrier = True
                    relative_e = True
                elif code.startswith("G92") and (len(code) == 3 or code[3].isspace()):
                    barrier = True
                    params = _parameters(code)
                    e = params.get("E", e)

            if barrier:
                ended = finish()
                if ended is not None:
                    yield ended
                continue

            is_motion = re.match(r"^G0?[01](?:\s|$)", code) is not None
            if not is_motion:
                # Ordinary comments and modal controls are allowed within a path.
                continue

            params = _parameters(code)
            nx, ny, nz = params.get("X", x), params.get("Y", y), params.get("Z", z)
            ne = params.get("E")
            amount = (ne if relative_e else ne - e) if ne is not None else 0
            length = math.hypot(nx - x, ny - y)
            is_extrusion = ("E" in params and amount > 1e-8 and length > 1e-6)

            if is_extrusion:
                if (run is None or run["smoothed"] != active_smoothing or
                        run["type"] != current_type):
                    ended = finish()
                    if ended is not None:
                        yield ended
                    run = {
                        "layer": layer, "type": current_type, "smoothed": active_smoothing,
                        "line_start": number, "line_end": number,
                        "z": nz, "start": (x, y), "end": (nx, ny),
                        "length": 0.0, "moves": 0,
                        "bbox": [min(x, nx), min(y, ny), max(x, nx), max(y, ny)],
                    }
                run["line_end"] = number
                run["end"] = (nx, ny)
                run["length"] += length
                run["moves"] += 1
                run["bbox"][0] = min(run["bbox"][0], nx)
                run["bbox"][1] = min(run["bbox"][1], ny)
                run["bbox"][2] = max(run["bbox"][2], nx)
                run["bbox"][3] = max(run["bbox"][3], ny)
            elif (("X" in params or "Y" in params or "Z" in params or "E" in params) and
                  run is not None):
                ended = finish()
                if ended is not None:
                    yield ended

            x, y, z = nx, ny, nz
            if ne is not None:
                e = e + ne if relative_e else ne

    ended = finish()
    if ended is not None:
        yield ended


def summarize(runs, layer=None, x=None, y=None, radius=20, min_length=5, top=50,
              all_types=False):
    filtered = []
    for run in runs:
        if run["smoothed"] or run["length"] < min_length:
            continue
        if layer is not None and run["layer"] != layer:
            continue
        if not all_types and not re.search(r"perimeter|wall|overhang|thin", run["type"], re.I):
            continue
        if x is not None and y is not None and _distance_to_bbox(x, y, run) > radius:
            continue
        filtered.append(run)

    by_type = Counter()
    for run in filtered:
        by_type[run["type"]] += 1

    print("UNMODIFIED POSITIVE-EXTRUSION PATHS")
    print(f"Found {len(filtered)} candidate paths (>= {min_length:g} mm)")
    if by_type:
        print("Types: " + "; ".join(f"{kind}: {count}" for kind, count in by_type.most_common()))
    print("layer   Z(mm)  length   moves    GCODE lines        TYPE / XY bounding box")
    print("-" * 109)
    if x is not None and y is not None:
        filtered.sort(key=lambda item: (_distance_to_bbox(x, y, item), -item["length"]))
    else:
        filtered.sort(key=lambda item: (-item["length"], item["layer"]))
    for run in filtered[:top]:
        x0, y0, x1, y1 = run["bbox"]
        print(f'{run["layer"]:5d}  {run["z"]:6.3f}  {run["length"]:7.1f}  '
              f'{run["moves"]:5d}  {run["line_start"]:8d}-{run["line_end"]:<8d}  '
              f'{run["type"]}  XY[{x0:.2f},{y0:.2f}]-[{x1:.2f},{y1:.2f}]')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gcode", help="G-code generated by SilkSteel")
    parser.add_argument("--layer", type=int, help="Inspect a single slicer layer (zero-based)")
    parser.add_argument("--x", type=float, help="Known X coordinate of the missing stripe")
    parser.add_argument("--y", type=float, help="Known Y coordinate of the missing stripe")
    parser.add_argument("--radius", type=float, default=20, help="XY search radius (mm)")
    parser.add_argument("--min-length", type=float, default=5, help="Minimum candidate path length (mm)")
    parser.add_argument("--top", type=int, default=50, help="Maximum rows")
    parser.add_argument("--all-types", action="store_true", help="Include infill and unlabeled paths")
    args = parser.parse_args(argv)
    if (args.x is None) != (args.y is None):
        parser.error("--x and --y must be provided together")
    summarize(audit(args.gcode), layer=args.layer, x=args.x, y=args.y,
              radius=args.radius, min_length=args.min_length, top=args.top,
              all_types=args.all_types)


if __name__ == "__main__":
    main()
