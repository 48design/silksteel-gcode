"""Local, read-only SilkSteel G-code volume preview. Standard library only."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import struct
import tempfile
import webbrowser

TOKEN = re.compile(r"([A-Z])([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)")
FLOAT = re.compile(r"([-+]?\d*\.?\d+)")
PACK = struct.Struct("<8fII")

ROLES = ("Standard", "Smoothificator", "Bricklayers base", "Bricklayers shifted",
         "Non-planar infill", "Valley filling", "Bridge")


def convert(source: Path, destination: Path) -> dict:
    """Stream G-code into 40-byte segment records; never buffer the source."""
    pos = [0.0, 0.0, 0.0]
    e_pos = 0.0
    xyz_absolute, e_absolute = True, True
    filament, layer_height, first_height = 1.75, 0.2, 0.2
    layer_z = 0.0
    layer_no = -1
    layer = None
    layers = []
    count = ignored_arcs = 0
    role_type, brick_role, smooth, valley = "", None, False, False
    smooth_height = None
    min_xyz = [math.inf] * 3
    max_xyz = [-math.inf] * 3
    explicit_layers = False

    def begin_layer(number: int) -> None:
        nonlocal layer, layer_no, brick_role, smooth, smooth_height, valley
        layer_no = number
        layer = {"number": number, "start": count, "count": 0,
                 "z": layer_z, "height": layer_height if number else first_height}
        layers.append(layer)
        brick_role, smooth, smooth_height, valley = None, False, None, False

    with source.open("r", encoding="utf-8", errors="replace") as src, destination.open("wb") as dst:
        for line_no, raw in enumerate(src, 1):
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
                    brick_role, smooth, smooth_height, valley = None, False, None, False
                if "SMOOTHIFICATOR START:" in up:
                    smooth = True
                    m = re.search(r"AT\s+([\d.]+)MM EACH", up)
                    smooth_height = float(m.group(1)) if m else None
                elif "SMOOTHIFICATOR END" in up:
                    smooth, smooth_height = False, None
                if "BRICKLAYERS SHIFTED BLOCK" in up:
                    brick_role = 3
                elif "BRICKLAYERS BASE PASS" in up or "BRICKLAYERS BASE BLOCK" in up:
                    brick_role = 2
                elif "RESET Z" in up or "BRICKLAYERS CONTOUR E SYNC" in up:
                    brick_role = None
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
                           "G90", "G91", "M82", "M83", "G92"):
                m = re.match(r"^(G0?[0123]|G9[012]|M8[23])(?=[A-Z])", cmd)
                if not m:
                    continue
                cmd = m.group(1)
            cmd = {"G00": "G0", "G01": "G1", "G02": "G2", "G03": "G3"}.get(cmd, cmd)
            params = {k: float(v) for k, v in TOKEN.findall(code.upper()) if k not in ("G", "M")}
            if cmd == "G90":
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
                new_pos = [params[ax] if xyz_absolute else pos[i] + params[ax]
                           if ax in params else pos[i] for i, ax in enumerate("XYZ")]
                # Avoid binding the conditional to the wrong branch for omitted axes.
                for i, ax in enumerate("XYZ"):
                    if ax not in params:
                        new_pos[i] = pos[i]
                delta_e = 0.0
                if "E" in params:
                    delta_e = params["E"] - e_pos if e_absolute else params["E"]
                    e_pos = params["E"] if e_absolute else e_pos + params["E"]
                if cmd in ("G2", "G3"):
                    ignored_arcs += 1
                elif cmd == "G1" and delta_e > 0 and all(map(math.isfinite, new_pos + pos)):
                    length = math.dist(pos, new_pos)
                    if length > 1e-7:
                        if layer is None:
                            begin_layer(0)
                        if not explicit_layers and new_pos[2] > layer_z + layer_height * 0.5 and count:
                            begin_layer(layer_no + 1)
                        if not explicit_layers:
                            layer_z = max(layer_z, new_pos[2])
                            layer["z"] = layer_z
                        h = smooth_height if smooth and smooth_height else layer["height"]
                        if brick_role == 2 and "BASE PASS" in up:
                            h *= 0.75
                        h = max(0.01, h)
                        area = (delta_e * math.pi * (filament * 0.5) ** 2) / length
                        width = area / h + h * (1 - math.pi / 4)
                        if math.isfinite(width) and width >= h * 0.25 and width <= 40:
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
                            dst.write(PACK.pack(*(pos + new_pos + [width, h]), role, line_no))
                            layer["count"] += 1
                            count += 1
                            for p in (pos, new_pos):
                                for i, c in enumerate(p):
                                    min_xyz[i] = min(min_xyz[i], c)
                                    max_xyz[i] = max(max_xyz[i], c)
                pos = new_pos

    if not count:
        raise ValueError(f"No extruding G1 XY/XYZ movements found in {source.name}")
    return {"filename": source.name, "url": "", "count": count, "layers": layers,
            "bounds": [min_xyz, max_xyz], "filament": filament,
            "ignored_arcs": ignored_arcs}


def serve(processed: Path, original: Path | None = None) -> None:
    assets = Path(__file__).resolve().parent / "preview"
    with tempfile.TemporaryDirectory(prefix="silksteel-preview-") as tmp:
        root = Path(tmp)
        models = []
        for label, source in (("SilkSteel", processed), ("Original", original)):
            if source is None:
                continue
            target = root / (label.lower() + ".bin")
            model = convert(source, target)
            model["name"], model["url"] = label, "/" + target.name
            models.append(model)
        info = json.dumps({"models": models, "roles": ROLES}).encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/meta.json":
                    payload, mime = info, "application/json"
                elif path in ("/", "/viewer.js"):
                    target = assets / ("index.html" if path == "/" else "viewer.js")
                    payload = target.read_bytes()
                    mime = "text/html; charset=utf-8" if path == "/" else "text/javascript; charset=utf-8"
                elif path in ("/silksteel.bin", "/original.bin") and any(m["url"] == path for m in models):
                    target = root / path.lstrip("/")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(target.stat().st_size))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    with target.open("rb") as stream:
                        while block := stream.read(1024 * 1024):
                            self.wfile.write(block)
                    return
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, fmt, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"SilkSteel Preview: {url}", flush=True)
        webbrowser.open(url)
        try:
            server.serve_forever(poll_interval=0.25)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SilkSteel 3D extrusion preview")
    parser.add_argument("processed", type=Path)
    parser.add_argument("--original", type=Path)
    arguments = parser.parse_args()
    serve(arguments.processed.resolve(),
          arguments.original.resolve() if arguments.original else None)
