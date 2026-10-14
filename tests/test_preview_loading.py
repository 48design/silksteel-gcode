"""Regression tests for the fast viewer parser and binary handoff."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from silksteel_preview import convert, convert_cached


class PreviewLoadingTests(unittest.TestCase):
    COMMANDS = (
        "G90",
        "M83",
        ";LAYER_CHANGE",
        ";Z:0.2",
        ";HEIGHT:0.2",
        "G0 X0 Y0 Z0.2 F1200",
        "G1 X20 Y0 E0.3 F1200",
        "G1 X20 Y20 E0.3",
        "G1 X0 Y20 E0.3",
        "G0 X0 Y0 F3000",
    )

    def render(self, root, name, lines):
        source = root / (name + ".gcode")
        target = root / (name + ".bin")
        source.write_text("\n".join(lines) + "\n", encoding="utf-8")
        offsets = []
        meta = convert(source, target, source_offsets=offsets)
        files = tuple(target.read_bytes() if suffix == ".bin"
                      else target.with_suffix(suffix).read_bytes()
                      for suffix in (".bin", ".hops.bin", ".raw.bin", ".moves.bin"))
        return meta, files, offsets

    def test_packed_gcode_matches_whitespace_separated_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            regular = self.render(root, "regular", self.COMMANDS)
            packed = [line.replace(" ", "") if line.startswith(("G0 ", "G1 "))
                      else line for line in self.COMMANDS]
            alternative = self.render(root, "packed", packed)
            self.assertEqual(regular[1:], alternative[1:])
            self.assertEqual(regular[0]["layers"], alternative[0]["layers"])
            self.assertEqual(regular[0]["count"], alternative[0]["count"])

    def test_scientific_notation_matches_decimal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            regular = self.render(root, "decimal", self.COMMANDS)
            scientific = list(self.COMMANDS)
            scientific[6] = "G1 X2e1 Y0 E3e-1 F1.2e3"
            self.assertEqual(regular[1:], self.render(root, "scientific", scientific)[1:])

    def test_cache_preserves_all_binary_tracks_and_source_offsets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.gcode"
            source.write_text("\n".join(self.COMMANDS) + "\n", encoding="utf-8")
            with patch.dict(os.environ, {"LOCALAPPDATA": str(root)}):
                first, second = root / "first.bin", root / "second.bin"
                offsets_first, offsets_second = [], []
                meta_first = convert_cached(source, first, source_offsets=offsets_first)
                meta_second = convert_cached(source, second, source_offsets=offsets_second)
                self.assertEqual(meta_first, meta_second)
                self.assertEqual(offsets_first, offsets_second)
                for suffix in (".bin", ".hops.bin", ".raw.bin", ".moves.bin"):
                    a = first if suffix == ".bin" else first.with_suffix(suffix)
                    b = second if suffix == ".bin" else second.with_suffix(suffix)
                    self.assertEqual(a.read_bytes(), b.read_bytes())


if __name__ == "__main__":
    unittest.main()
