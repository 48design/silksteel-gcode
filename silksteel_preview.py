"""Local, read-only SilkSteel G-code volume preview. Standard library only."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import shutil
import traceback
import uuid
from urllib.parse import unquote

from silksteel_preview_surface import DepositedSurface
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



def convert(source: Path, destination: Path, hop_destination: Path | None = None,
            progress=None) -> dict:
    """Stream G-code into 48-byte timed segments and separate hop records."""
    hop_destination = hop_destination or destination.with_suffix(".hops.bin")
    pos = [0.0, 0.0, 0.0]
    e_pos = 0.0
    xyz_absolute, e_absolute = True, True
    filament, layer_height, first_height = 1.75, 0.2, None
    layer_z = 0.0
    layer_no = -1
    brick_base_pass = False
    layer = None
    layers = []
    count = ignored_arcs = hop_count = 0
    feedrate = 1800.0
    elapsed = 0.0
    hop_active = False
    surface = DepositedSurface()
    role_type, brick_role, smooth, valley = "", None, False, False
    smooth_height = None
    min_xyz = [math.inf] * 3
    max_xyz = [-math.inf] * 3
    explicit_layers = False
    parsed_chars = 0
    source_size = max(1, source.stat().st_size)

    def begin_layer(number: int) -> None:
        nonlocal layer, layer_no, brick_role, smooth, smooth_height, valley, brick_base_pass
        layer_no = number
        layer = {"number": number, "start": count, "count": 0,
                 "hopStart": hop_count, "hopCount": 0,
                 "z": layer_z, "height": layer_height if number else (first_height or layer_height),
                 "timeStart": elapsed, "timeEnd": elapsed}
        layers.append(layer)
        brick_role, smooth, smooth_height, valley, brick_base_pass = None, False, None, False, False

    with (source.open("r", encoding="utf-8", errors="replace") as src,
          destination.open("wb") as dst, hop_destination.open("wb") as hops):
        for line_no, raw in enumerate(src, 1):
            parsed_chars += len(raw)
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
                if "SMOOTHIFICATOR START:" in up:
                    smooth = True
                    m = re.search(r"AT\s+([\d.]+)MM EACH", up)
                    smooth_height = float(m.group(1)) if m else None
                elif "SMOOTHIFICATOR END" in up:
                    smooth, smooth_height = False, None
                if "BRICKLAYERS SHIFTED BLOCK" in up:
                    brick_role, brick_base_pass = 3, False
                elif "BRICKLAYERS BASE PASS" in up or "BRICKLAYERS BASE BLOCK" in up:
                    brick_role = 2
                    brick_base_pass = "BASE PASS" in up
                elif "RESET Z" in up or "BRICKLAYERS CONTOUR E SYNC" in up:
                    brick_role, brick_base_pass = None, False
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
                           "G90", "G91", "M82", "M83", "G92", "G4", "G04"):
                m = re.match(r"^(G0?[01234]|G9[012]|M8[23])(?=[A-Z])", cmd)
                if not m:
                    continue
                cmd = m.group(1)
            cmd = {"G00": "G0", "G01": "G1", "G02": "G2", "G03": "G3", "G04": "G4"}.get(cmd, cmd)
            params = {k: float(v) for k, v in TOKEN.findall(code.upper()) if k not in ("G", "M")}
            if cmd == "G4":
                elapsed += max(0.0, params.get("P", 0.0)) / 1000.0 + max(0.0, params.get("S", 0.0))
            elif cmd == "G90":
                xyz_absolute = True
            elif cmd == "G91":
                xyz_absolute = False
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
                distance = math.dist(pos, new_pos)
                motion_seconds = (distance if distance > 1e-7 else abs(delta_e)) * 60.0 / feedrate
                started = elapsed
                elapsed += motion_seconds if math.isfinite(motion_seconds) else 0.0

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

                if cmd in ("G2", "G3"):
                    ignored_arcs += 1
                elif cmd == "G1" and delta_e > 0 and all(map(math.isfinite, new_pos + pos)):
                    length = distance
                    if length > 1e-7 and math.hypot(new_pos[0]-pos[0], new_pos[1]-pos[1]) > 1e-7:
                        if layer is None:
                            begin_layer(0)
                        if not explicit_layers and new_pos[2] > layer_z + layer_height * 0.5 and count:
                            begin_layer(layer_no + 1)
                        if not explicit_layers:
                            layer_z = max(layer_z, new_pos[2])
                            layer["z"] = layer_z
                        nominal_h = smooth_height if smooth and smooth_height else layer["height"]
                        if brick_base_pass:
                            nominal_h *= 0.75
                        nominal_h = max(0.01, nominal_h)
                        area = (delta_e * math.pi * (filament * 0.5) ** 2) / length
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
                        # Determine support along the *entire* raw move before
                        # recording any part of it, avoiding artificial self-support.
                        segments = surface.sections(
                            tuple(pos), tuple(new_pos), area, nominal_h,
                            first_height or layer_height
                        )
                        if segments and all(
                            math.isfinite(w) and math.isfinite(h) and h > 0 and
                            h <= w <= 40 for _, _, w, h in segments
                        ):
                            dx = [new_pos[i] - pos[i] for i in range(3)]
                            for t0, t1, width, height in segments:
                                a_pos = [pos[i] + dx[i] * t0 for i in range(3)]
                                b_pos = [pos[i] + dx[i] * t1 for i in range(3)]
                                start_t = started + (elapsed - started) * t0
                                end_t = started + (elapsed - started) * t1
                                dst.write(PACK.pack(
                                    *(a_pos + b_pos + [width, height]),
                                    role, line_no, start_t, end_t
                                ))
                                layer["count"] += 1
                                count += 1
                            surface.deposit(tuple(pos), tuple(new_pos), segments)
                            for point in (pos, new_pos):
                                for i, coordinate in enumerate(point):
                                    min_xyz[i] = min(min_xyz[i], coordinate)
                                    max_xyz[i] = max(max_xyz[i], coordinate)
                pos = new_pos

    if not count:
        raise ValueError(f"No extruding G1 XY/XYZ movements found in {source.name}")
    for index, item in enumerate(layers):
        item["timeEnd"] = (layers[index + 1]["timeStart"]
                           if index + 1 < len(layers) else elapsed)
    if not explicit_layers and layers and layers[0]["number"] == -1:
        layers[0]["number"] = 0
    return {"filename": source.name, "url": "", "count": count, "hopCount": hop_count,
            "duration": elapsed, "layers": layers,
            "bounds": [min_xyz, max_xyz], "filament": filament,
            "ignored_arcs": ignored_arcs}


def serve(processed: Path | None = None, original: Path | None = None,
          display_name: str | None = None, original_display_name: str | None = None) -> None:
    """Open the browser before parsing; make failures observable to the user."""
    assets = Path(__file__).resolve().parent / "preview"
    with tempfile.TemporaryDirectory(prefix="silksteel-preview-") as tmp:
        root = Path(tmp)
        heartbeat = {"at": time.monotonic()}
        state = {"status": "loading", "message": "G-Code-Analyse wird gestartet…",
                 "progress": 0, "lines": 0}
        models = []
        ready = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
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
                path = self.path.split("?", 1)[0]
                if path == "/ping":
                    heartbeat["at"] = time.monotonic()
                    payload, mime = b"ok", "text/plain"
                elif path == "/meta.json":
                    payload = json.dumps(
                        {"models": models, "roles": ROLES, "colors": read_colors(), **state}
                    ).encode("utf-8")
                    mime = "application/json"
                elif path in ("/", "/viewer.js", "/extrusion_joins.js",
                              "/diagnostics.js", "/comparison.js", "/section.js",
                              "/bead_junctions.js"):
                    target = assets / ("index.html" if path == "/" else path.lstrip("/"))
                    payload = target.read_bytes()
                    mime = ("text/html; charset=utf-8" if path == "/"
                            else "text/javascript; charset=utf-8")
                elif state["status"] == "ready" and any(
                    path in (m["url"], m["hopUrl"]) for m in models
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
                model = convert(upload, target, progress=update)
                model["name"] = display
                model["filename"] = display
                model["url"] = "/" + target.name
                model["hopUrl"] = "/" + target.with_suffix(".hops.bin").name
                models.append(model)
                state.update(status="ready", message="Vorschau bereit", progress=100)
                print(f"SilkSteel Preview: loaded {display}", flush=True)
            except Exception as exc:
                traceback.print_exc()
                state.update(status="error", message=f"{type(exc).__name__}: {exc}")
            finally:
                upload.unlink(missing_ok=True)
                ready.set()

        def prepare_files():
            try:
                sources = [("SilkSteel", processed, display_name or processed.name)] if processed else []
                if original is not None:
                    sources.append(("Original", original, original_display_name or original.name))
                for index, (label, source, display) in enumerate(sources):
                    state["message"] = f"{label}: {source.name} wird analysiert…"
                    state["progress"] = int(index * 100 / len(sources))
                    state["lines"] = 0

                    def update(lines, percent, index=index):
                        state["lines"] = lines
                        state["progress"] = int((index + percent / 100) * 100 / len(sources))

                    target = root / (label.lower() + ".bin")
                    model = convert(source, target, progress=update)
                    model["name"], model["url"] = label, "/" + target.name
                    model["filename"] = display
                    model["hopUrl"] = "/" + target.with_suffix(".hops.bin").name
                    models.append(model)
                state.update(status="ready", message="Vorschau bereit", progress=100)
                print("SilkSteel Preview: finished parsing", flush=True)
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
