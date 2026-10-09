#!/usr/bin/env python3
"""Inspect SilkSteel G-code roles and Bricklayers Z moves by source layer.

Usage:
    python tools/audit_feature_layers.py processed.gcode --layer 40
    python tools/audit_feature_layers.py processed.gcode --summary
Read-only, no dependencies.
"""
import argparse
from collections import Counter
import re

LAYER_RE = re.compile(r"^;LAYER:(\d+)")
TYPES = {
    "External perimeter", "Internal perimeter", "Perimeter",
    "Outer wall", "Inner wall", "Overhang perimeter",
    "Internal infill", "Solid infill", "Top solid infill",
    "Gap fill", "Bridge infill", "Internal bridge infill",
    "Custom", "Skirt", "Ironing",
}
COUNTERS = {
    "smooth": "SMOOTHIFICATOR START:",
    "brick_shifted": "Bricklayers shifted block #",
    "brick_base2": "Bricklayers base pass 1/2",
    "brick_base": "Bricklayers base block #",
    "brick_esync": "Bricklayers contour E sync",
    "carried_outer": "SilkSteel: CONTINUED across layer boundary",
}


def audit(path):
    layers = {}
    current = None
    active_type = "(none)"
    with open(path, encoding="utf-8", errors="replace") as stream:
        for no, raw in enumerate(stream, 1):
            line = raw.strip()
            hit = LAYER_RE.match(line)
            if hit:
                number = int(hit.group(1))
                current = layers.setdefault(number, {
                    "line": no, "types": [],
                    "counts": Counter(), "first_xy_e": None,
                    "first_extrusion_type": None,
                    "brick_z": {"base": [], "shifted": []},
                })
                # Slicer viewers may reset the role on a layer boundary.
                active_type = "(none)"
            if current is None:
                if line.startswith(";TYPE:"):
                    active_type = line[len(";TYPE:"):]
                continue
            if line.startswith(";TYPE:"):
                active_type = line[len(";TYPE:"):]
                current["types"].append((no, active_type))
                if active_type not in TYPES:
                    current["counts"]["unrecognized_type_comments"] += 1
            if current["first_xy_e"] is None and re.match(r"^G0?[01](?:\s|$)", line):
                code = line.partition(";")[0]
                if re.search(r"\bE[-+]?\d", code) and re.search(r"\b[XY][-+]?\d", code):
                    current["first_xy_e"] = no
                    current["first_extrusion_type"] = active_type
            for name, needle in COUNTERS.items():
                if needle in line:
                    current["counts"][name] += 1

            # A G-code marker only means the feature ran. The actual
            # interlocking requires different PHYSICAL nozzle Z heights.
            if "Bricklayers shifted block #" in line or "Bricklayers base block #" in line:
                z_match = re.match(r"^G0 Z([-+]?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+))", line)
                if z_match:
                    role = "shifted" if "shifted block #" in line else "base"
                    current["brick_z"][role].append(float(z_match.group(1)))
    return layers


def report(layers, layer=None, summary=False):
    print("SilkSteel feature/Bricklayers audit")
    if layer is not None:
        if layer not in layers:
            print(f"Layer {layer} not found.")
            return
        for n in range(max(0, layer - 1), layer + 2):
            if n not in layers:
                continue
            entry = layers[n]
            print(f"\nLayer {n} (G-code line {entry['line']}):")
            print(f"  First XY/E move: {entry['first_xy_e']}, TYPE: {entry['first_extrusion_type']}")
            print("  Counts: " + ", ".join(f"{key}={value}" for key, value in
                                         entry["counts"].items()))
            z_base = sorted(set(entry["brick_z"]["base"]))
            z_shifted = sorted(set(entry["brick_z"]["shifted"]))
            print(f"  Actual Bricklayers Z: base={z_base}, shifted={z_shifted}")
            if z_base and z_shifted:
                if set(z_base) & set(z_shifted):
                    print("  WARNING: base and shifted blocks share the same Z!")
                else:
                    print("  OK: base and shifted blocks have distinct nozzle heights")
            print("  TYPE comments (first 15):")
            for line, kind in entry["types"][:15]:
                print(f"    {line}: ;TYPE:{kind}")
    if summary or layer is None:
        all_counts = Counter()
        empty = []
        for n, entry in sorted(layers.items()):
            all_counts.update(entry["counts"])
            if not (entry["counts"]["brick_shifted"] or entry["counts"]["brick_base2"]
                    or entry["counts"]["brick_base"]):
                empty.append(n)
        print(f"\nLayers: {len(layers)}")
        print("Total markers: " + ", ".join(
            f"{key}={all_counts[key]}" for key in COUNTERS))
        print(f"Noncanonical TYPE comments: {all_counts['unrecognized_type_comments']}")
        collapsed = [n for n, entry in sorted(layers.items())
                     if (set(entry["brick_z"]["base"]) &
                         set(entry["brick_z"]["shifted"]))]
        print(f"Layers with equal base/shifted Z ({len(collapsed)}): "
              + (", ".join(map(str, collapsed[:100])) or "none")
              + (" ..." if len(collapsed) > 100 else ""))
        print(f"Layers with no Bricklayers markers ({len(empty)}): "
              + ", ".join(map(str, empty[:100]))
              + (" ..." if len(empty) > 100 else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gcode")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--summary", action="store_true")
    a = parser.parse_args()
    report(audit(a.gcode), layer=a.layer, summary=a.summary)


if __name__ == "__main__":
    main()
