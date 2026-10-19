"""Showcase G-code filtering after SilkSteel geometry generation.

The full original geometry is retained for safezone/noise computation; only
deposition from unwanted slicer roles is omitted before travel collision checks.
"""
import re

_ROLE = re.compile(r"^;TYPE:\s*(.+?)\s*$", re.I)
_LAYER = re.compile(r"^;LAYER:(\d+)", re.I)
_BOTTOM = re.compile(r"^;\s*bottom_solid_layers\s*=\s*(\d+)\b", re.I | re.M)
_MOTION = re.compile(r"^G0?([0123])(?:\s|$)", re.I)
_AXIS = re.compile(r"(?<![A-Za-z])([XYZE])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
_G92 = re.compile(r"^G92(?:\s|$)", re.I)
_M82 = re.compile(r"^M82(?:\s|$)", re.I)
_M83 = re.compile(r"^M83(?:\s|$)", re.I)
_G90 = re.compile(r"^G90(?:\s|$)", re.I)
_G91 = re.compile(r"^G91(?:\s|$)", re.I)

_WALLS = {
    "external perimeter", "internal perimeter", "perimeter",
    "overhang perimeter", "outer wall", "inner wall", "thin wall",
    "gap fill", "skirt", "brim",
}
_SOLIDS = {
    "solid infill", "top solid infill", "bottom solid infill",
    "bridge infill", "internal bridge infill",
}


def filter_showcase_gcode(gcode, bottom_layers=None):
    """Return (G-code, statistics) retaining infill and a solid bottom.

    Discarded extrusion is represented by consolidated non-extruding travels,
    preserving modal XYZ and positioning. Absolute/relative extrusion is
    tracked independently; G92 rebases only when the next absolute E is used.
    Retractions and primes remain intact, including those across TYPE blocks.
    """
    if bottom_layers is None:
        match = _BOTTOM.search(gcode)
        bottom_layers = int(match.group(1)) if match else 3
    if bottom_layers < 1:
        raise ValueError("Showcase requires at least one bottom solid layer")

    layer = -1
    role = ""
    relative_e = False
    relative_xyz = False
    source_e = 0.0
    pending_sync = False
    pending_travel = None
    removed = 0
    bottom_kept = 0
    infill_kept = 0
    out = []

    def flush_travel():
        nonlocal pending_travel
        if pending_travel:
            words = " ".join(f"{axis}{value:.5f}" for axis, value in pending_travel.items())
            out.append(f"G0 {words} F8400 ; SilkSteel: showcase travel\n")
            pending_travel = None

    for line in gcode.splitlines(keepends=True):
        stripped = line.strip()
        marker = _LAYER.match(stripped)
        if marker:
            flush_travel()
            layer = int(marker.group(1))
        elif stripped.startswith(";LAYER_CHANGE"):
            flush_travel()
            layer += 1

        typed = _ROLE.match(stripped)
        if typed:
            flush_travel()
            role = typed.group(1).strip().lower()

        code = line.split(";", 1)[0].strip()
        if not code:
            out.append(line)
            continue

        if _G90.match(code):
            flush_travel()
            relative_xyz = False
        elif _G91.match(code):
            flush_travel()
            relative_xyz = True
        if _M82.match(code):
            flush_travel()
            relative_e = False
        elif _M83.match(code):
            flush_travel()
            relative_e = True

        axes = {name.upper(): float(value) for name, value in _AXIS.findall(code)}
        if _G92.match(code):
            flush_travel()
            if "E" in axes:
                source_e = axes["E"]
                pending_sync = False
            out.append(line)
            continue

        move = _MOTION.match(code)
        if not move or "E" not in axes:
            flush_travel()
            out.append(line)
            continue

        before_e = source_e
        source_e = source_e + axes["E"] if relative_e else axes["E"]
        delta_e = source_e - before_e
        is_solid = role in _SOLIDS
        exclude = (role in _WALLS or (is_solid and layer >= bottom_layers))
        skip = exclude and delta_e > 1e-8

        if skip:
            removed += 1
            pending_sync = True
            # Collect consecutive skipped motions into one rapid travel.
            for axis in "XYZ":
                if axis in axes:
                    if pending_travel is None:
                        pending_travel = {}
                    if relative_xyz:
                        pending_travel[axis] = pending_travel.get(axis, 0.0) + axes[axis]
                    else:
                        pending_travel[axis] = axes[axis]
            continue

        flush_travel()
        if pending_sync and not relative_e:
            out.append(f"G92 E{before_e:.5f} ; SilkSteel: showcase E sync\n")
            pending_sync = False

        if delta_e > 1e-8:
            if is_solid and layer < bottom_layers:
                bottom_kept += 1
            elif role == "internal infill":
                infill_kept += 1
        out.append(line)

    flush_travel()
    if bottom_kept == 0:
        raise ValueError(
            "Showcase found no bottom solid extrusion. Slice with bottom solid "
            "layers enabled (e.g. bottom_solid_layers = 3).")
    return "".join(out), {
        "bottom_layers": bottom_layers,
        "removed_extrusions": removed,
        "bottom_extrusions": bottom_kept,
        "infill_extrusions": infill_kept,
    }
