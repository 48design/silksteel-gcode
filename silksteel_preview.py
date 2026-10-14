"""Local, read-only SilkSteel G-code volume preview. Standard library only."""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import shutil
import traceback
import uuid
from urllib.parse import unquote, parse_qs, urlsplit

from silksteel_preview_ideal import IdealizedProfile
from silksteel_preview_arcs import interpolate_arc
from pathlib import Path
import re
import struct
import tempfile
import threading
import time
import webbrowser

TOKEN = re.compile(r"([A-Z])([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)")
FLOAT = re.compile(r"([-+]?\d*\.?\d+)")
PACK = struct.Struct("<8fIIff")  # XYZ endpoints, bead dimensions, role, line, start/end seconds
HOP_PACK = struct.Struct("<6fIIff")  # XYZ endpoints, hop kind, line, start/end seconds
RAW_PACK = struct.Struct("<6fIIff")  # Positive-E moves including skipped viewer geometry
MOVE_PACK = struct.Struct("<ffI")  # Every timed G0/G1/G2/G3/G4: begin/end seconds, source line
SOURCE_INDEX_STEP = 1024

ROLES = ("Standard", "Smoothificator", "Bricklayers base", "Bricklayers shifted",
         "Non-planar infill", "Valley filling", "Bridge")

COLOR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")


def color_settings_path() -> Path:
    base = Path(os.environ["LOCALAPPDATA"]) if os.environ.get("LOCALAPPDATA") else Path.home() / ".config"
    return base / "SilkSteel" / "preview-colors.json"


def read_colors() -> list[str] | None:
    try:
        data = json.loads(color_settings_path().read_text(encoding="utf-8"))
        values = data.get("colors")
        if isinstance(values, list) and len(values) in (len(ROLES),len(ROLES)+1) and all(
            isinstance(v, str) and COLOR_PATTERN.fullmatch(v) for v in values
        ):
            return values if len(values)==len(ROLES)+1 else values+["#ff4655"]
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def write_colors(values: list[str]) -> None:
    if not isinstance(values, list) or len(values) != len(ROLES)+1 or not all(
        isinstance(v, str) and COLOR_PATTERN.fullmatch(v) for v in values
    ):
        raise ValueError("Invalid feature color palette")
    file = color_settings_path()
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(file.name + ".tmp")
    try:
        tmp.write_text(json.dumps({"colors": values}), encoding="utf-8")
        os.replace(tmp, file)
    finally:
        tmp.unlink(missing_ok=True)



SIDEBAR_SECTIONS = ("G-Code laden", "G-Code-Version", "Schichten", "Extrusionstypen",
                    "Schnittansicht", "Diagnose", "Originalvergleich", "Darstellung")


def sidebar_settings_path() -> Path:
    return color_settings_path().with_name("preview-sidebar.json")


def read_sidebar() -> dict[str, bool]:
    try:
        data = json.loads(sidebar_settings_path().read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {key: data[key] for key in SIDEBAR_SECTIONS
                    if key in data and isinstance(data[key], bool)}
    except (OSError, ValueError, TypeError):
        pass
    return {}


def write_sidebar(state: dict[str, bool]) -> None:
    if not isinstance(state, dict) or not all(
        key in SIDEBAR_SECTIONS and isinstance(value, bool)
        for key, value in state.items()
    ):
        raise ValueError("Invalid sidebar settings")
    path = sidebar_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def convert(source: Path, destination: Path, hop_destination: Path | None = None,
            progress=None, raw_destination: Path | None = None,
            source_offsets: list[int] | None = None,
            move_destination: Path | None = None) -> dict:
    """Stream G-code into 48-byte timed segments and separate hop records."""
    hop_destination = hop_destination or destination.with_suffix(".hops.bin")
    raw_destination = raw_destination or destination.with_suffix(".raw.bin")
    move_destination = move_destination or destination.with_suffix(".moves.bin")
    pos = [0.0, 0.0, 0.0]
    e_pos = 0.0
    xyz_absolute, e_absolute = True, True
    arc_center_absolute, arc_plane = False, "G17"
    filament, layer_height, first_height = 1.75, 0.2, None
    layer_z = 0.0
    layer_no = -1
    brick_base_pass = False
    brick_pass_index = 0
    layer = None
    layers = []
    count = ignored_arcs = arc_count = hop_count = raw_count = move_count = omitted_count = partial_count = 0
    feedrate = 1800.0
    elapsed = 0.0
    hop_active = False
    role_type, brick_role, smooth, valley = "", None, False, False
    smooth_height = None
    min_xyz = [math.inf] * 3
    max_xyz = [-math.inf] * 3
    explicit_layers = False
    parsed_chars = 0
    source_size = max(1, source.stat().st_size)

    def begin_layer(number: int) -> None:
        nonlocal layer, layer_no, brick_role, smooth, smooth_height, valley, brick_base_pass, brick_pass_index
        layer_no = number
        layer = {"number": number, "start": count, "count": 0,
                 "rawStart": raw_count, "rawCount": 0,
                 "hopStart": hop_count, "hopCount": 0,
                 "moveStart": move_count, "moveCount": 0,
                 "z": layer_z, "height": layer_height if number else (first_height or layer_height),
                 "timeStart": elapsed, "timeEnd": elapsed}
        layers.append(layer)
        brick_role, smooth, smooth_height, valley, brick_base_pass = None, False, None, False, False
        brick_pass_index = 0

    with (source.open("rb") as src,
          destination.open("wb") as dst, hop_destination.open("wb") as hops,
          raw_destination.open("wb") as raw_tracks,
          move_destination.open("wb") as move_tracks):
        for line_no, raw_bytes in enumerate(src, 1):
            if source_offsets is not None and (line_no - 1) % SOURCE_INDEX_STEP == 0:
                source_offsets.append(parsed_chars)
            parsed_chars += len(raw_bytes)
            raw = raw_bytes.decode("utf-8", errors="replace")
            if progress is not None and line_no % 20000 == 0:
                progress(line_no, min(99, int(parsed_chars * 100 / source_size)))
            code, _, comment = raw.partition(";")
            comment = comment.strip()
            up = comment.upper()
            if comment:
                if up.startswith("FILAMENT_DIAMETER") and "=" in comment:
                    m = FLOAT.search(comment.partition("=")[2])
                    if m and float(m.group(1)) > 0:
                        filament = float(m.group(1))
                elif up.startswith("LAYER_HEIGHT") and "=" in comment:
                    m = FLOAT.search(comment.partition("=")[2])
                    if m and float(m.group(1)) > 0:
                        layer_height = float(m.group(1))
                elif up.startswith("FIRST_LAYER_HEIGHT") and "=" in comment:
                    m = FLOAT.search(comment.partition("=")[2])
                    if m and float(m.group(1)) > 0:
                        first_height = float(m.group(1))
                elif up.startswith("LAYER_CHANGE"):
                    # Purge lines before the first slicer layer belong to
                    # a separate preamble, not to layer zero.
                    if not explicit_layers and layer is not None and len(layers) == 1:
                        layer["number"] = -1
                        layer_no = -1
                    explicit_layers = True
                    begin_layer(layer_no + 1)
                elif re.match(r"LAYER:\s*\d+", comment, re.I):
                    explicit_layers = True
                    number = int(comment.split(":", 1)[1])
                    if layer is None or layer_no != number:
                        begin_layer(number)
                elif up.startswith("HEIGHT:"):
                    try:
                        h = float(comment.partition(":")[2])
                        if 0 < h < 5:
                            layer_height = h
                            if layer:
                                layer["height"] = h
                    except ValueError:
                        pass
                elif up.startswith("Z:") and layer:
                    try:
                        layer_z = float(comment.partition(":")[2])
                        layer["z"] = layer_z
                    except ValueError:
                        pass
                elif up.startswith("TYPE:"):
                    role_type = up[5:].strip()
                    brick_role, smooth, smooth_height, valley, brick_base_pass = None, False, None, False, False
                    brick_pass_index = 0
                if "SMOOTHIFICATOR START:" in up:
                    smooth = True
                    m = re.search(r"AT\s+([\d.]+)MM EACH", up)
                    smooth_height = float(m.group(1)) if m else None
                elif "SMOOTHIFICATOR END" in up:
                    smooth, smooth_height = False, None
                if "BRICKLAYERS SHIFTED BLOCK" in up:
                    brick_role, brick_base_pass, brick_pass_index = 3, False, 0
                elif "BRICKLAYERS BASE PASS" in up or "BRICKLAYERS BASE BLOCK" in up:
                    brick_role = 2
                    brick_base_pass = "BASE PASS" in up
                    pass_match = re.search(r"BASE PASS\s+(\d+)/", up) if brick_base_pass else None
                    brick_pass_index = int(pass_match.group(1)) if pass_match else 0
                elif "RESET Z" in up or "BRICKLAYERS CONTOUR E SYNC" in up:
                    brick_role, brick_base_pass, brick_pass_index = None, False, 0
                if "SILKSTEEL: VALLEY_FILL_START" in up:
                    valley = True
                elif "SILKSTEEL: VALLEY_FILL_END" in up:
                    valley = False

            words = code.strip().upper().split(None, 1)
            if not words:
                continue
            cmd = words[0]
            # Accept G00/G01, and packed "G1X1Y2" syntax.
            if cmd not in ("G0", "G00", "G1", "G01", "G2", "G02", "G3", "G03",
                           "G90", "G91", "G90.1", "G91.1", "G17", "G18", "G19",
                           "M82", "M83", "G92", "G4", "G04"):
                m = re.match(r"^(G0?[01234]|G1[789]|G9[012]|M8[23])(?=[A-Z])", cmd)
                if not m:
                    continue
                cmd = m.group(1)
            cmd = {"G00": "G0", "G01": "G1", "G02": "G2", "G03": "G3", "G04": "G4"}.get(cmd, cmd)
            # Most slicers emit whitespace-separated G0/G1 XYZEF words.
            # Parse those directly; retain the regex parser for packed and
            # nonstandard commands. G-code semantics and output stay identical.
            params = None
            if cmd in ("G0", "G1") and len(words) > 1:
                quick = {}
                for field in words[1].split():
                    key, value = field[:1], field[1:]
                    if (key not in "XYZEF" or not value or
                            value[0] not in "+-.0123456789" or "_" in value):
                        break
                    try:
                        quick[key] = float(value)
                    except ValueError:
                        break
                else:
                    params = quick
            if params is None:
                params = {k: float(v) for k, v in TOKEN.findall(code.upper())
                          if k not in ("G", "M")}
            if cmd == "G4":
                before = elapsed
                elapsed += max(0.0, params.get("P", 0.0)) / 1000.0 + max(0.0, params.get("S", 0.0))
                if elapsed > before:
                    move_tracks.write(MOVE_PACK.pack(before, elapsed, line_no))
                    move_count += 1
            elif cmd == "G90":
                xyz_absolute = True
            elif cmd == "G91":
                xyz_absolute = False
            elif cmd == "G90.1":
                arc_center_absolute = True
            elif cmd == "G91.1":
                arc_center_absolute = False
            elif cmd in ("G17", "G18", "G19"):
                arc_plane = cmd
            elif cmd == "M82":
                e_absolute = True
            elif cmd == "M83":
                e_absolute = False
            elif cmd == "G92":
                for i, ax in enumerate("XYZ"):
                    if ax in params:
                        pos[i] = params[ax]
                if "E" in params:
                    e_pos = params["E"]
            elif cmd in ("G0", "G1", "G2", "G3"):
                new_pos = [pos[i] if ax not in params else
                           (params[ax] if xyz_absolute else pos[i] + params[ax])
                           for i, ax in enumerate("XYZ")]
                delta_e = 0.0
                if "E" in params:
                    delta_e = params["E"] - e_pos if e_absolute else params["E"]
                    e_pos = params["E"] if e_absolute else e_pos + params["E"]
                if "F" in params and math.isfinite(params["F"]) and params["F"] > 0:
                    feedrate = params["F"]
                arc_path = (interpolate_arc(pos, new_pos, params,
                            clockwise=(cmd == "G2"), plane=arc_plane,
                            absolute_center=arc_center_absolute)
                            if cmd in ("G2", "G3") else None)
                if cmd in ("G2", "G3") and delta_e > 0:
                    if arc_path is None:
                        ignored_arcs += 1
                    else:
                        arc_count += 1
                if arc_path is None:
                    path = [pos, new_pos]
                    distance = math.dist(pos, new_pos)
                    path_lengths = [distance]
                else:
                    path = arc_path[0]
                    path_lengths = [math.dist(a, b) for a, b in zip(path, path[1:])]
                    distance = sum(path_lengths)
                motion_seconds = (distance if distance > 1e-7 else abs(delta_e)) * 60.0 / feedrate
                started = elapsed
                elapsed += motion_seconds if math.isfinite(motion_seconds) else 0.0
                if elapsed > started:
                    move_tracks.write(MOVE_PACK.pack(started, elapsed, line_no))
                    move_count += 1

                hop_kind = None
                if "SILKSTEEL: ZHOP_LIFT" in up or "Z-HOP LIFT" in up:
                    hop_kind = 0
                elif "SILKSTEEL: ZHOP_DROP" in up or "Z-HOP DROP" in up:
                    hop_kind = 2
                elif hop_active and distance > 1e-7 and delta_e <= 0:
                    hop_kind = 1
                if hop_kind is not None and distance > 1e-7 and all(map(math.isfinite, pos + new_pos)):
                    if layer is None:
                        begin_layer(0)
                    hops.write(HOP_PACK.pack(*(pos + new_pos), hop_kind, line_no, started, elapsed))
                    layer["hopCount"] += 1
                    hop_count += 1
                if hop_kind == 0:
                    hop_active = True
                elif hop_kind == 2 or delta_e > 0:
                    hop_active = False

                raw_kind = (2 if cmd in ("G2", "G3") and arc_path is None else
                            3 if math.hypot(new_pos[0] - pos[0],
                                            new_pos[1] - pos[1]) <= 1e-7 and
                                 arc_path is None else 1)
                if delta_e > 0 and distance > 1e-7 and all(
                    math.isfinite(value) for value in new_pos + pos
                ):
                    if layer is None:
                        begin_layer(0)
                    if not explicit_layers and new_pos[2] > layer_z + layer_height * 0.5 and count:
                        begin_layer(layer_no + 1)
                    if not explicit_layers:
                        layer_z = max(layer_z, new_pos[2])
                        layer["z"] = layer_z
                    # Keep E / physical path length constant across an arc.
                    # Only XY-traversing subsegments need a bead cross section.
                    extrusion_area = delta_e * math.pi * (filament * 0.5) ** 2 / distance
                    nominal_h = smooth_height if smooth and smooth_height else layer["height"]
                    if brick_base_pass:
                        nominal_h *= 0.75
                    nominal_h = max(0.01, nominal_h)
                    if valley:
                        role = 5
                    elif brick_role is not None:
                        role = brick_role
                    elif smooth:
                        role = 1
                    elif role_type.startswith("BRIDGE") or role_type.startswith("INTERNAL BRIDGE"):
                        role = 6
                    elif "INFILL" in role_type and "Z" in params and abs(new_pos[2] - layer["z"]) > 0.005:
                        role = 4
                    else:
                        role = 0
                    rendered_parts = 0
                    skipped_parts = 0
                    elapsed_distance = 0.0
                    for (part_start, part_end), part_length in zip(
                        zip(path, path[1:]), path_lengths
                    ):
                        segment_begin = started + motion_seconds * elapsed_distance / distance
                        elapsed_distance += part_length
                        segment_end = started + motion_seconds * elapsed_distance / distance
                        if part_length <= 1e-7 or math.hypot(
                            part_end[0] - part_start[0], part_end[1] - part_start[1]
                        ) <= 1e-7:
                            skipped_parts += 1
                            continue
                        # Grossly implausible source extrusion is left in the
                        # diagnostic raw path instead of manufacturing a huge
                        # 3D solid. This is far above ordinary FDM flow.
                        if extrusion_area > 16.0:
                            skipped_parts += 1
                            continue
                        sections = IdealizedProfile.sections(
                            tuple(part_start), tuple(part_end), extrusion_area, nominal_h,
                            first_height or layer_height,
                            layer_z=layer["z"] if explicit_layers or len(layers) > 1 else None,
                            speed=feedrate / 60.0, role=role,
                            brick_pass=brick_pass_index
                        )
                        # Reject only the invalid SUBSECTION, never its valid
                        # neighbors or an entire otherwise valid G-code move.
                        for t0, t1, width, height in sections:
                            if not (math.isfinite(width) and math.isfinite(height) and
                                    0 < height <= 40 and 0 < width <= 40):
                                skipped_parts += 1
                                continue
                            a_pos = [part_start[i] + (part_end[i] - part_start[i]) * t0
                                     for i in range(3)]
                            b_pos = [part_start[i] + (part_end[i] - part_start[i]) * t1
                                     for i in range(3)]
                            t_start = segment_begin + (segment_end - segment_begin) * t0
                            t_end = segment_begin + (segment_end - segment_begin) * t1
                            dst.write(PACK.pack(
                                *(a_pos + b_pos + [width, height]),
                                role, line_no, t_start, t_end
                            ))
                            layer["count"] += 1
                            count += 1
                            rendered_parts += 1
                            for p in (a_pos, b_pos):
                                for i, coordinate in enumerate(p):
                                    min_xyz[i] = min(min_xyz[i], coordinate)
                                    max_xyz[i] = max(max_xyz[i], coordinate)
                    if rendered_parts > 0:
                        raw_kind = (4 if cmd in ("G2", "G3") and arc_path is not None
                                    else 2 if cmd in ("G2", "G3") else 0)
                        if skipped_parts:
                            raw_kind = 1
                            partial_count += 1
                    else:
                        omitted_count += 1
                if delta_e > 0 and distance > 1e-7 and all(
                    math.isfinite(value) for value in new_pos + pos
                ):
                    if layer is None:
                        begin_layer(0)
                    raw_tracks.write(RAW_PACK.pack(
                        *(pos + new_pos), raw_kind, line_no, started, elapsed))
                    layer["rawCount"] += 1
                    raw_count += 1
                pos = new_pos

    if not count:
        raise ValueError(f"No extruding G1 XY/XYZ movements found in {source.name}")
    for index, item in enumerate(layers):
        item["timeEnd"] = (layers[index + 1]["timeStart"]
                           if index + 1 < len(layers) else elapsed)
        item["moveCount"] = ((layers[index + 1]["moveStart"]
                              if index + 1 < len(layers) else move_count)
                             - item["moveStart"])
    if not explicit_layers and layers and layers[0]["number"] == -1:
        layers[0]["number"] = 0
    return {"filename": source.name, "url": "", "count": count, "hopCount": hop_count,
            "arcCount": arc_count,
            "rawCount": raw_count, "moveCount": move_count, "omittedCount": omitted_count,
            "partialCount": partial_count,
            "duration": elapsed, "layers": layers,
            "bounds": [min_xyz, max_xyz], "filament": filament,
            "ignored_arcs": ignored_arcs}


PREVIEW_CACHE_VERSION = 3


def convert_cached(source: Path, destination: Path, progress=None,
                   source_offsets: list[int] | None = None) -> dict:
    """Reuse the exact viewer binary representation for unchanged G-code files."""
    stat = source.stat()
    identity = f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{PREVIEW_CACHE_VERSION}"
    key = hashlib.sha256(os.fsencode(identity)).hexdigest()
    base = (Path(os.environ["LOCALAPPDATA"]) if os.environ.get("LOCALAPPDATA")
            else Path.home() / ".cache") / "SilkSteel" / "preview-cache"
    entry = base / key
    suffixes = (".bin", ".hops.bin", ".raw.bin", ".moves.bin")

    def paths(root: Path) -> list[Path]:
        return [root / ("data" + suffix) for suffix in suffixes]

    started = time.perf_counter()
    try:
        metadata = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
        cached = paths(entry)
        if any(not p.is_file() for p in cached):
            raise OSError("Incomplete viewer cache")
        for src, suffix in zip(cached, suffixes):
            target = destination if suffix == ".bin" else destination.with_suffix(suffix)
            try:
                os.link(src, target)
            except OSError:
                shutil.copyfile(src, target)
        if source_offsets is not None:
            source_offsets.extend(metadata["offsets"])
        print(f"SilkSteel Preview: cache HIT ({time.perf_counter()-started:.2f}s)", flush=True)
        if progress:
            progress(0, 100)
        return metadata["model"]
    except (OSError, ValueError, KeyError, TypeError):
        for suffix in suffixes:
            (destination if suffix == ".bin" else destination.with_suffix(suffix)).unlink(missing_ok=True)

    base.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=key[:12] + "-", dir=base))
    offsets: list[int] = []
    try:
        model = convert(source, paths(temporary)[0],
                        hop_destination=paths(temporary)[1],
                        raw_destination=paths(temporary)[2],
                        move_destination=paths(temporary)[3],
                        progress=progress, source_offsets=offsets)
        (temporary / "meta.json").write_text(
            json.dumps({"model": model, "offsets": offsets}), encoding="utf-8")
        try:
            temporary.rename(entry)
        except FileExistsError:
            pass
        for src, suffix in zip(paths(temporary if temporary.exists() else entry), suffixes):
            target = destination if suffix == ".bin" else destination.with_suffix(suffix)
            try:
                os.link(src, target)
            except OSError:
                shutil.copyfile(src, target)
        if source_offsets is not None:
            source_offsets.extend(offsets)
        print(f"SilkSteel Preview: converted in {time.perf_counter()-started:.2f}s", flush=True)
        return model
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def source_context(source: Path, offsets: list[int], line: int,
                   context: int = 10) -> list[dict]:
    """Read a small numbered source window without loading a giant G-code file."""
    start = max(1, line - context)
    end = line + context
    if not offsets:
        return []
    block = (start - 1) // SOURCE_INDEX_STEP
    block = min(block, len(offsets) - 1)
    current = block * SOURCE_INDEX_STEP + 1
    result = []
    with source.open("rb") as stream:
        stream.seek(offsets[block])
        while current <= end:
            raw = stream.readline()
            if not raw:
                break
            if current >= start:
                text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                result.append({"line": current, "text": text[:2048]})
            current += 1
    return result


def serve(processed: Path | None = None, original: Path | None = None,
          display_name: str | None = None, original_display_name: str | None = None) -> None:
    """Open the browser before parsing; make failures observable to the user."""
    assets = Path(__file__).resolve().parent / "preview"
    with tempfile.TemporaryDirectory(prefix="silksteel-preview-") as tmp:
        root = Path(tmp)
        heartbeat = {"at": time.monotonic()}
        state = {"status": "loading", "message": "G-Code-Analyse wird gestartet…",
                 "progress": 0, "lines": 0, "pendingModels": 0}
        models = []
        sources_by_url: dict[str, tuple[Path, list[int]]] = {}
        ready = threading.Event()
        original_state = {"status": "idle" if original is not None else "absent",
                          "error": ""}
        original_lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/prepare-original":
                    origin = self.headers.get("Origin")
                    if origin and origin != f"http://127.0.0.1:{server.server_port}":
                        self.send_error(403, "Invalid origin")
                        return
                    with original_lock:
                        if original_state["status"] == "idle":
                            original_state["status"] = "parsing"
                            threading.Thread(target=prepare_original, daemon=True).start()
                    payload = json.dumps(original_state).encode("utf-8")
                    self.send_response(202 if original_state["status"] == "parsing" else 200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if self.path == "/settings/sidebar":
                    origin = self.headers.get("Origin")
                    if origin and origin != f"http://127.0.0.1:{server.server_port}":
                        self.send_error(403, "Invalid origin")
                        return
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 0 < length <= 4096:
                            raise ValueError("Invalid sidebar settings size")
                        settings = json.loads(self.rfile.read(length))
                        write_sidebar(settings)
                    except (OSError, ValueError, TypeError) as exc:
                        self.send_error(400, str(exc))
                        return
                    payload = b'{"saved":true}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if self.path == "/settings/colors":
                    origin = self.headers.get("Origin")
                    if origin and origin != f"http://127.0.0.1:{server.server_port}":
                        self.send_error(403, "Invalid origin")
                        return
                    try:
                        size = int(self.headers.get("Content-Length", "0"))
                        if not 0 < size <= 2048:
                            raise ValueError("Invalid settings size")
                        payload = json.loads(self.rfile.read(size))
                        write_colors(payload.get("colors"))
                        response = b'{"saved":true}'
                    except (OSError, ValueError, TypeError, AttributeError) as exc:
                        self.send_error(400, str(exc))
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                if self.path != "/upload":
                    self.send_error(404)
                    return
                origin = self.headers.get("Origin")
                if origin and origin != f"http://127.0.0.1:{server.server_port}":
                    self.send_error(403, "Invalid upload origin")
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.send_error(400, "Invalid file size")
                    return
                if size <= 0 or size > 2 * 1024 ** 3:
                    self.send_error(413, "G-code file must be between 1 byte and 2 GiB")
                    return
                filename = Path(unquote(self.headers.get("X-Gcode-Filename", "uploaded.gcode"))).name
                if not filename.lower().endswith((".gcode", ".gcode.pp", ".gc", ".g")):
                    self.send_error(415, "Only G-code files are supported")
                    return
                upload = root / f"upload-{uuid.uuid4().hex}.gcode"
                try:
                    remaining = size
                    with upload.open("wb") as dst:
                        while remaining:
                            block = self.rfile.read(min(1024 * 1024, remaining))
                            if not block:
                                raise ConnectionError("Incomplete G-code upload")
                            dst.write(block)
                            remaining -= len(block)
                except (OSError, ConnectionError) as exc:
                    upload.unlink(missing_ok=True)
                    self.send_error(400, str(exc))
                    return
                state.update(status="loading", message=f"{filename} wird analysiert…",
                             progress=0, lines=0)
                threading.Thread(target=prepare_upload, args=(upload, filename),
                                 daemon=True).start()
                payload = b'{"accepted":true}'
                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                parsed = urlsplit(self.path)
                path = parsed.path
                if path == "/source":
                    query = parse_qs(parsed.query)
                    source_id = query.get("model", [""])[0]
                    entry = sources_by_url.get(source_id)
                    if not entry:
                        self.send_error(404, "G-code source not found")
                        return
                    try:
                        line = int(query.get("line", ["1"])[0])
                        context = max(0, min(35, int(query.get("context", ["10"])[0])))
                        if line < 1:
                            raise ValueError("Invalid G-code line")
                    except ValueError:
                        self.send_error(400, "Invalid source position")
                        return
                    try:
                        rows = source_context(*entry, line, context)
                        response = json.dumps({"line": line, "rows": rows},
                                              ensure_ascii=False).encode("utf-8")
                    except OSError as exc:
                        self.send_error(404, str(exc))
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(response)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        self.wfile.write(response)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if path == "/ping":
                    heartbeat["at"] = time.monotonic()
                    payload, mime = b"ok", "text/plain"
                elif path == "/meta.json":
                    payload = json.dumps(
                        {"models": models, "roles": ROLES, "colors": read_colors(), "sidebar": read_sidebar(),
                         "originalAvailable": original is not None,
                         "originalState": original_state["status"],
                         "originalError": original_state["error"], **state}
                    ).encode("utf-8")
                    mime = "application/json"
                elif path in ("/", "/viewer.js", "/extrusion_joins.js",
                              "/diagnostics.js", "/comparison.js", "/section.js",
                              "/bead_junctions.js", "/gcode_inspector.js",
                              "/bead_profile.js", "/source_timeline.js", "/loft_geometry.js"):
                    target = assets / ("index.html" if path == "/" else path.lstrip("/"))
                    payload = target.read_bytes()
                    mime = ("text/html; charset=utf-8" if path == "/"
                            else "text/javascript; charset=utf-8")
                elif state["status"] == "ready" and any(
                    path in (m["url"], m["hopUrl"], m["rawUrl"], m["moveUrl"]) for m in models
                ):
                    target = root / path.lstrip("/")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(target.stat().st_size))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        with target.open("rb") as stream:
                            while block := stream.read(1024 * 1024):
                                self.wfile.write(block)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, fmt, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)

        def prepare_upload(upload, display):
            try:
                target = root / f"model-{uuid.uuid4().hex}.bin"
                def update(lines, percent):
                    state["lines"] = lines
                    state["progress"] = percent
                offsets = []
                # A new upload has a unique temporary path: a disk cache only
                # duplicates its data and cannot accelerate the first preview.
                started_parse = time.perf_counter()
                model = convert(upload, target, progress=update, source_offsets=offsets)
                model["parseSeconds"] = round(time.perf_counter() - started_parse, 3)
                print(f"SilkSteel Preview: upload conversion {model['parseSeconds']:.3f}s", flush=True)
                model["name"] = display
                model["filename"] = display
                model["url"] = "/" + target.name
                model["hopUrl"] = "/" + target.with_suffix(".hops.bin").name
                model["rawUrl"] = "/" + target.with_suffix(".raw.bin").name
                model["moveUrl"] = "/" + target.with_suffix(".moves.bin").name
                sources_by_url[model["url"]] = (upload, offsets)
                models.append(model)
                state.update(status="ready", message="Vorschau bereit", progress=100)
                print(f"SilkSteel Preview: loaded {display}", flush=True)
            except Exception as exc:
                traceback.print_exc()
                state.update(status="error", message=f"{type(exc).__name__}: {exc}")
            finally:
                # Keep source available to the G-code inspector until server exit.
                ready.set()

        def prepare_original():
            if original is None:
                return
            try:
                target = root / "original.bin"
                offsets = []
                parser = (convert if
                    (original.parent / ".silksteel-preview-snapshot").exists()
                    else convert_cached)
                started_parse = time.perf_counter()
                model = parser(original, target, source_offsets=offsets)
                model["parseSeconds"] = round(time.perf_counter() - started_parse, 3)
                print(f"SilkSteel Preview: original conversion {model['parseSeconds']:.3f}s", flush=True)
                model["name"], model["url"] = "Original", "/" + target.name
                model["filename"] = original_display_name or original.name
                model["hopUrl"] = "/" + target.with_suffix(".hops.bin").name
                model["rawUrl"] = "/" + target.with_suffix(".raw.bin").name
                model["moveUrl"] = "/" + target.with_suffix(".moves.bin").name
                sources_by_url[model["url"]] = (original, offsets)
                models.append(model)
                original_state["status"] = "ready"
                print("SilkSteel Preview: original ready on demand", flush=True)
            except Exception as exc:
                traceback.print_exc()
                original_state.update(status="error",
                                      error=f"{type(exc).__name__}: {exc}")

        def prepare_files():
            try:
                if processed is not None:
                    state.update(message=f"SilkSteel: {processed.name} wird analysiert…",
                                 progress=0, lines=0)
                    def update(lines, percent):
                        state["lines"] = lines
                        state["progress"] = percent
                    target = root / "silksteel.bin"
                    offsets = []
                    parser = (convert if
                        (processed.parent / ".silksteel-preview-snapshot").exists()
                        else convert_cached)
                    started_parse = time.perf_counter()
                    model = parser(processed, target, progress=update, source_offsets=offsets)
                    model["parseSeconds"] = round(time.perf_counter() - started_parse, 3)
                    print(f"SilkSteel Preview: processed conversion {model['parseSeconds']:.3f}s, {model['count']} strands", flush=True)
                    model["name"], model["url"] = "SilkSteel", "/" + target.name
                    model["filename"] = display_name or processed.name
                    model["hopUrl"] = "/" + target.with_suffix(".hops.bin").name
                    model["rawUrl"] = "/" + target.with_suffix(".raw.bin").name
                    model["moveUrl"] = "/" + target.with_suffix(".moves.bin").name
                    sources_by_url[model["url"]] = (processed, offsets)
                    models.append(model)
                state.update(status="ready", message="Vorschau bereit", progress=100)
                print("SilkSteel Preview: processed model ready", flush=True)
            except Exception as exc:
                traceback.print_exc()
                state.update(status="error", message=f"{type(exc).__name__}: {exc}")
            finally:
                ready.set()

        def idle_watchdog():
            while True:
                time.sleep(20)
                if ready.is_set() and state["status"] != "loading" and time.monotonic() - heartbeat["at"] > 120:
                    server.shutdown()
                    return

        if processed is None:
            state.update(status="ready", message="G-Code per Drag & Drop laden", progress=100)
            ready.set()
        else:
            threading.Thread(target=prepare_files, daemon=True).start()
        threading.Thread(target=idle_watchdog, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"SilkSteel Preview: {url}", flush=True)

        opened = webbrowser.open(url, new=1)
        if not opened and os.name == "nt":
            try:
                os.startfile(url)
            except OSError:
                traceback.print_exc()
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SilkSteel 3D extrusion preview")
    parser.add_argument("processed", nargs="?", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--display-name")
    parser.add_argument("--original-display-name")
    parser.add_argument("--snapshot-dir", type=Path)
    arguments = parser.parse_args()
    try:
        serve(arguments.processed.resolve() if arguments.processed else None,
              arguments.original.resolve() if arguments.original else None,
              arguments.display_name, arguments.original_display_name)
    finally:
        if arguments.snapshot_dir:
            path = arguments.snapshot_dir.resolve()
            temp_root = Path(tempfile.gettempdir()).resolve()
            if (path.parent == temp_root and path.name.startswith("silksteel-preview-input-")
                    and (path / ".silksteel-preview-snapshot").is_file()):
                shutil.rmtree(path, ignore_errors=True)
