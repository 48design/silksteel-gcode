import math
from pathlib import Path
import struct
import tempfile
import unittest

from silksteel_preview import convert, PACK


class PreviewGeometryTests(unittest.TestCase):
    def parse(self, lines):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "test.gcode"
            target = Path(tmp) / "test.bin"
            source.write_text("\n".join(lines) + "\n", encoding="utf-8")
            metadata = convert(source, target)
            data = target.read_bytes()
            rows = [PACK.unpack_from(data, i) for i in range(0, len(data), PACK.size)]
            return metadata, rows

    def test_more_extrusion_is_physically_wider(self):
        meta, rows = self.parse([
            "; filament_diameter = 1.75",
            "; layer_height = 0.2",
            ";LAYER_CHANGE", ";Z:0.2", ";HEIGHT:0.2", ";LAYER:0",
            "G90", "M82", "G92 E0", "G0 X0 Y0 Z0.2",
            "G1 X10 Y0 E0.4",
            "G1 X20 Y0 E1.2",
        ])
        self.assertEqual(meta["count"], 2)
        self.assertEqual(meta["layers"][0]["count"], 2)
        a, b = rows
        self.assertAlmostEqual(a[7], .2)
        self.assertAlmostEqual(b[7], .2)
        self.assertAlmostEqual(b[6] / a[6], 1.918, delta=.02)

    def test_roles_and_modal_e_modes(self):
        meta, rows = self.parse([
            ";LAYER_CHANGE", ";Z:0.2", ";HEIGHT:0.2", ";LAYER:0",
            "G90", "M83", "G0 X0 Y0 Z0.2",
            "G1 X10 Y0 E0.3",
            "G1 X10 Y10 E-1",
            "G1 E1",
            "G0 Z0.3 ; Bricklayers shifted block #1",
            "G1 X20 Y10 E0.3",
            "G1 Z0.2 ; Reset Z",
            "; SilkSteel: VALLEY_FILL_START",
            "G1 X30 Y10 E0.3",
            "; SilkSteel: VALLEY_FILL_END",
            "; ====== SMOOTHIFICATOR START: 2 passes at 0.1000mm each ======",
            "G0 Z0.1 ; Smoothificator pass 1/2",
            "G1 X40 Y10 E0.15",
            "; ====== SMOOTHIFICATOR END ======",
        ])
        self.assertEqual([row[8] for row in rows], [0, 3, 5, 1])
        self.assertAlmostEqual(rows[3][7], .1)

    def test_absolute_reset_and_relative_xy(self):
        _, rows = self.parse([
            "G91", "M82", "G92 E10", "G1 X10 Y0 Z0.2 E10.5",
            "G92 E0", "G1 X10 E0.5",
        ])
        self.assertEqual(len(rows), 2)
        self.assertEqual([r[0] for r in rows], [0, 10])
        self.assertEqual([r[3] for r in rows], [10, 20])
        self.assertAlmostEqual(rows[0][6], rows[1][6])

    def test_brick_base_height_and_layers(self):
        meta, rows = self.parse([
            ";LAYER_CHANGE", ";Z:0.2", ";HEIGHT:0.2", ";LAYER:0",
            "G0 X0 Y0 Z0.2",
            "G0 Z0.2 ; Bricklayers base pass 1/2",
            "G1 X10 Y0 E0.3",
            ";LAYER_CHANGE", ";Z:0.4", ";HEIGHT:0.2", ";LAYER:1",
            "G0 X0 Y0 Z0.4",
            "G1 X10 Y0 E0.6",
        ])
        self.assertEqual([row[8] for row in rows], [2, 0])
        self.assertAlmostEqual(rows[0][7], .15)
        self.assertEqual([layer["count"] for layer in meta["layers"]], [1, 1])
        self.assertEqual(meta["layers"][1]["start"], 1)


    def test_timed_print_and_hop_overlay_tracks(self):
        from silksteel_preview import HOP_PACK
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "hops.gcode"
            dest = Path(tmp) / "hops.bin"
            source.write_text("\n".join([
                "; layer_height = 0.2",
                ";LAYER_CHANGE", ";Z:0.2", ";HEIGHT:0.2", ";LAYER:0",
                "G90", "M83", "G0 X0 Y0 Z0.2 F1200",
                "G1 X10 Y0 E0.3 F600",
                "G0 Z0.8 F1200 ; SilkSteel: ZHOP_LIFT",
                "G0 X20 Y0 F1200",
                "G0 X20 Y10",
                "G0 Z0.2 F1200 ; SilkSteel: ZHOP_DROP",
                "G1 X30 Y10 E0.3 F600",
            ])+"\n", encoding="utf-8")
            meta = convert(source, dest)
            rows = [PACK.unpack_from(dest.read_bytes(), i) for i in range(0, dest.stat().st_size, PACK.size)]
            hops_file = dest.with_suffix(".hops.bin")
            payload = hops_file.read_bytes()
            hops = [HOP_PACK.unpack_from(payload, i) for i in range(0, len(payload), HOP_PACK.size)]
            self.assertEqual(meta["count"], 2)
            self.assertEqual(meta["hopCount"], 4)
            self.assertEqual(meta["layers"][0]["hopCount"], 4)
            self.assertEqual([hop[6] for hop in hops], [0, 1, 1, 2])
            self.assertAlmostEqual(rows[0][-1] - rows[0][-2], 1.0, places=4)
            self.assertAlmostEqual(rows[1][-1] - rows[1][-2], 1.0, places=4)
            self.assertAlmostEqual(rows[0][-1], hops[0][-2], places=4)
            self.assertAlmostEqual(rows[1][-2], hops[-1][-1], places=4)
            self.assertAlmostEqual(meta["duration"], 3.07, places=2)

    def test_g4_pause_adds_timeline_time_without_extrusion(self):
        meta, rows = self.parse([
            "G90", "M83", "G0 X0 Y0 Z0.2 F600",
            "G1 X10 E0.25", "G4 P2000",
            "G1 X20 E0.25",
        ])
        self.assertEqual(meta["count"], 2)
        self.assertAlmostEqual(rows[1][-2] - rows[0][-1], 2.0, places=3)
        self.assertAlmostEqual(meta["duration"], 4.02, places=2)

    def test_legacy_zhop_comments_are_supported(self):
        from silksteel_preview import HOP_PACK
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "legacy.gcode"
            bin_file = Path(tmp) / "legacy.bin"
            file.write_text("\n".join([
                "G90", "M83", "G0 X0 Y0 Z0.2",
                "G1 X5 Y0 E0.2",
                "G0 Z0.9 ; Z-hop lift",
                "G0 X10 Y0",
                "G0 Z0.2 ; Z-hop drop",
                "G1 X15 Y0 E0.2",
            ])+"\n", encoding="utf-8")
            meta = convert(file, bin_file)
            self.assertEqual(meta["hopCount"], 3)
            raw = bin_file.with_suffix(".hops.bin").read_bytes()
            self.assertEqual([HOP_PACK.unpack_from(raw, i)[6] for i in range(0, len(raw), HOP_PACK.size)], [0, 1, 2])

    def test_snapshot_survives_slicer_temp_file_deletion(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slicer_temp = root / ".13112.gcode.pp"
            slicer_temp.write_text("\n".join([
                "; layer_height = 0.2",
                "G90", "M83", "G0 X0 Y0 Z0.2",
                "G1 X10 Y0 E0.4",
            ]) + "\n", encoding="utf-8")
            snapshot = root / "processed.gcode"
            shutil.copyfile(slicer_temp, snapshot)
            slicer_temp.unlink()
            self.assertFalse(slicer_temp.exists())
            meta = convert(snapshot, root / "preview.bin")
            self.assertEqual(meta["count"], 1)
            self.assertEqual(meta["layers"][0]["count"], 1)



if __name__ == "__main__":
    unittest.main()
