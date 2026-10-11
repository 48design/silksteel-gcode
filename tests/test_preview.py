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
        self.assertAlmostEqual(rows[0][7], .2)  # Plate contact at Z=.2 overrides nominal pass height
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



    def test_viewer_uses_timed_record_stride_for_every_segment_role(self):
        from silksteel_preview import HOP_PACK
        self.assertEqual(PACK.size, 48)
        self.assertEqual(HOP_PACK.size, 40)
        viewer = (Path(__file__).resolve().parents[1] / "preview" / "viewer.js").read_text(encoding="utf-8")
        self.assertIn("SEGMENT_SIZE = 48", viewer)
        self.assertIn("index*SEGMENT_SIZE+32", viewer)
        self.assertIn("const role=readRole(i);", viewer)
        self.assertNotIn("i*40+32", viewer)



    def test_layer_time_windows_include_travel_and_dwell(self):
        meta, rows = self.parse([
            "; layer_height = 0.2",
            ";LAYER_CHANGE", ";Z:0.2", ";HEIGHT:0.2", ";LAYER:0",
            "G90", "M83", "G0 X0 Y0 Z0.2 F600",
            "G1 X10 Y0 E0.3 F600",
            "G4 P500",
            ";LAYER_CHANGE", ";Z:0.4", ";HEIGHT:0.2", ";LAYER:1",
            "G0 Z0.4 F600",
            "G1 X20 Y0 E0.3",
        ])
        first, second = meta["layers"]
        self.assertAlmostEqual(first["timeStart"], 0.0, places=4)
        self.assertAlmostEqual(first["timeEnd"], 1.52, places=3)
        self.assertAlmostEqual(second["timeStart"], first["timeEnd"], places=4)
        self.assertAlmostEqual(second["timeEnd"], meta["duration"], places=4)
        self.assertAlmostEqual(second["timeEnd"] - second["timeStart"], 1.02, places=3)

    def test_colors_are_validated_and_persisted_outside_viewer_port(self):
        from unittest import mock
        import silksteel_preview as preview
        palette = ["#111111", "#AABBCC", "#335577", "#fedcba",
                   "#550011", "#447799", "#123456", "#ff4655"]
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict("os.environ", {"LOCALAPPDATA": tmp}):
                preview.write_colors(palette)
                self.assertEqual(preview.read_colors(), palette)
                file = preview.color_settings_path()
                self.assertTrue(file.is_file())
                with self.assertRaises(ValueError):
                    preview.write_colors(palette[:-1])
                with self.assertRaises(ValueError):
                    preview.write_colors(palette[:-1] + ["not-a-color"])
                self.assertEqual(preview.read_colors(), palette)
                file.write_text(__import__("json").dumps({"colors":palette[:-1]}), encoding="utf-8")
                self.assertEqual(preview.read_colors(), palette)

    def test_full_width_timeline_and_editable_colors_are_wired(self):
        root = Path(__file__).resolve().parents[1]
        viewer = (root / "preview" / "viewer.js").read_text(encoding="utf-8")
        markup = (root / "preview" / "index.html").read_text(encoding="utf-8")
        self.assertIn('TIMELINE_STEPS = 1000000', viewer)
        self.assertIn("function layerRange(", viewer)
        self.assertIn("function activeRange(", viewer)
        self.assertIn('changeTimelineMode("layer")', viewer)
        self.assertIn('fetch("/settings/colors"', viewer)
        self.assertIn('id="playbackDock"', markup)
        self.assertIn('id="layerRail"', markup)
        self.assertIn('id="timelineTicks"', markup)
        self.assertIn('id="resetColors"', markup)
        self.assertIn('id="hopColor"', markup)


    def test_free_vs_supported_cross_sections_conserve_material(self):
        from silksteel_preview_surface import DepositedSurface
        field = DepositedSurface()
        area, nominal = .082, .2
        ground = (0.0, 0.0, .2), ground_end = (10.0, 0.0, .2)
        first = field.sections(ground, ground_end, area, nominal, .2)
        self.assertAlmostEqual(first[0][3], .2, places=4)
        field.deposit(ground, ground_end, first)

        # The same G1 crosses an already-printed stripe and then open air.
        start, end = (0.0, 0.0, .4), (20.0, 0.0, .4)
        pieces = field.sections(start, end, area, nominal, .2)
        self.assertGreaterEqual(len(pieces), 2)
        self.assertAlmostEqual(pieces[0][3], .2, places=3)
        free_d = field.free_diameter(area)
        self.assertAlmostEqual(pieces[-1][3], free_d, places=3)
        self.assertAlmostEqual(pieces[-1][2], free_d, places=3)
        self.assertGreater(pieces[0][2], pieces[-1][2])

        self.assertAlmostEqual(
            sum((t1-t0)*20.0 *
                (width*height-(1-math.pi/4)*height*height)
                for t0,t1,width,height in pieces),
            20.0*area, places=5
        )

    def test_same_height_neighbors_do_not_erase_previous_layer_support(self):
        from silksteel_preview_surface import DepositedSurface
        field = DepositedSurface()
        area = .085
        for z in (.2, .4):
            start, end = (0.0,0.0,z),(8.0,0.0,z)
            sections = field.sections(start,end,area,.2,.2)
            field.deposit(start,end,sections)
        # The z=.4 strand must not be treated as a supporting surface
        # for another strand printed at exactly the same Z.
        same_height = field.sections((0,0,.4),(8,0,.4),area,.2,.2)
        self.assertAlmostEqual(same_height[0][3], .2, places=3)

    def test_substrate_distance_shifts_stadium_toward_round(self):
        from silksteel_preview_surface import DepositedSurface
        field = DepositedSurface()
        area = .09
        old = field.sections((0,0,.2),(12,0,.2),area,.2,.2)
        field.deposit((0,0,.2),(12,0,.2),old)
        lifted = field.sections((0,0,.4),(12,0,.62),area,.2,.2)
        self.assertGreaterEqual(len(lifted), 2)
        self.assertLess(lifted[0][3],lifted[-1][3])
        self.assertGreater(lifted[0][2],lifted[-1][2])
        self.assertLessEqual(max(p[3] for p in lifted),field.free_diameter(area)+1e-6)

    def test_parser_splits_one_g1_only_at_real_substrate_change(self):
        meta, rows = self.parse([
            "; layer_height = 0.2", "; first_layer_height = 0.2",
            ";LAYER_CHANGE", ";Z:0.2", ";HEIGHT:0.2", ";LAYER:0",
            "G90", "M83", "G0 X0 Y0 Z0.2 F1200",
            "G1 X10 Y0 E0.4",
            ";LAYER_CHANGE", ";Z:0.4", ";HEIGHT:0.2", ";LAYER:1",
            "G0 X0 Y0 Z0.4",
            "G1 X20 Y0 E0.8",
        ])
        upper = rows[meta["layers"][1]["start"]:]
        self.assertGreater(len(upper),1)
        self.assertAlmostEqual(upper[0][7], .2, places=3)
        diameter = math.sqrt(4*(.8*math.pi*(1.75/2)**2/20)/math.pi)
        self.assertAlmostEqual(upper[-1][7],diameter,places=3)
        self.assertAlmostEqual(sum(row[-1]-row[-2] for row in upper),1.0,places=3)
        self.assertEqual(set(int(row[8]) for row in upper),{0})
        self.assertEqual(len(set(int(row[9]) for row in upper)),1)

    def test_feature_color_survives_local_cross_section_splitting(self):
        meta, rows = self.parse([
            "; layer_height = 0.2",
            ";LAYER_CHANGE", ";Z:0.2", ";HEIGHT:0.2", ";LAYER:0",
            "G90", "M83", "G0 X0 Y0 Z0.2 F1200",
            "G1 X10 Y0 E0.4",
            ";LAYER_CHANGE", ";Z:0.4", ";HEIGHT:0.2", ";LAYER:1",
            "G0 X0 Y0 Z0.4",
            "; SilkSteel: VALLEY_FILL_START",
            "G1 X20 Y0 E0.8",
            "; SilkSteel: VALLEY_FILL_END",
        ])
        parts = rows[meta["layers"][1]["start"]:]
        self.assertGreater(len(parts),1)
        self.assertEqual({int(row[8]) for row in parts},{5})
        self.assertEqual(len({int(row[9]) for row in parts}),1)
        self.assertTrue(all(row[-1]>row[-2] for row in parts))


    def test_advanced_viewer_modules_and_local_routes_are_connected(self):
        root = Path(__file__).resolve().parents[1]
        viewer = (root / "preview" / "viewer.js").read_text(encoding="utf-8")
        html = (root / "preview" / "index.html").read_text(encoding="utf-8")
        backend = (root / "silksteel_preview.py").read_text(encoding="utf-8")
        for module in ("extrusion_joins", "diagnostics", "comparison", "section"):
            self.assertTrue((root / "preview" / (module + ".js")).is_file())
            self.assertIn('"' + module + '.js"', backend)
            self.assertIn('from "./' + module + '.js"', viewer)
        for control in ("cutEnabled", "cutAxis", "cutPosition", "cutInvert",
                        "diagnosticsToggle", "diagnosticsList", "compareToggle",
                        "comparisonSummary"):
            self.assertIn('id="' + control + '"', html)
            self.assertIn('"'+control+'"', viewer)
        self.assertIn("buildJoinAttributes(items,joinRead,role)", viewer)
        self.assertIn("clipping_planes_pars_fragment", viewer)
        self.assertIn("function refreshComparison()", viewer)
        self.assertIn("function showDiagnostics()", viewer)
        self.assertIn("const SEGMENT_SIZE = 48, HOP_SIZE = 40", viewer)



if __name__ == "__main__":
    unittest.main()
