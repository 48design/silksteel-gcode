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

    def test_g2_g3_slicer_arcs_are_rendered_as_actual_toolpaths(self):
        # Ordinary Arc-Fitting output previously disappeared in its entirety.
        from silksteel_preview import RAW_PACK
        cases = [
            ("G3 X0 Y10 I-10 J0 E0.6 F1200", 10.0, 0.0, 0.0, 10.0),
            ("G3 X0 Y10 R10 E0.6 F1200", 10.0, 0.0, 0.0, 10.0),
        ]
        for command, x0, y0, x1, y1 in cases:
            with self.subTest(command=command):
                with tempfile.TemporaryDirectory() as tmp:
                    source = Path(tmp) / "arc.gcode"
                    target = Path(tmp) / "arc.bin"
                    source.write_text("\n".join([
                        "; layer_height = 0.2",
                        ";LAYER_CHANGE", ";Z:0.2", ";HEIGHT:0.2", ";LAYER:0",
                        "G90", "M83", f"G0 X{x0} Y{y0} Z0.2",
                        command,
                    ]) + "\n", encoding="utf-8")
                    meta=convert(source,target)
                    data=target.read_bytes()
                    rows=[PACK.unpack_from(data,i)
                          for i in range(0,len(data),PACK.size)]
                    self.assertGreater(meta["count"], 10)
                    self.assertEqual(meta["arcCount"], 1)
                    self.assertEqual(meta["ignored_arcs"], 0)
                    self.assertEqual(meta["omittedCount"], 0)
                    self.assertEqual(meta["rawCount"], 1)
                    self.assertEqual(meta["layers"][0]["count"], meta["count"])
                    self.assertAlmostEqual(rows[0][0],x0,places=4)
                    self.assertAlmostEqual(rows[-1][3],x1,places=4)
                    self.assertAlmostEqual(rows[-1][4],y1,places=4)
                    total_length=sum(math.dist(row[0:3],row[3:6]) for row in rows)
                    self.assertAlmostEqual(total_length, math.pi*5,delta=.035)
                    self.assertTrue(any(row[0]!=row[3] and row[1]!=row[4] for row in rows))
                    raw=target.with_suffix(".raw.bin").read_bytes()
                    self.assertEqual(RAW_PACK.unpack_from(raw,0)[6],0)
                    filament_area=math.pi*(meta["filament"]/2)**2
                    estimated_e=sum(
                        (row[6]*row[7]-(1-math.pi/4)*row[7]**2)
                        * math.dist(row[0:3],row[3:6])/filament_area
                        for row in rows)
                    self.assertAlmostEqual(estimated_e,.6,places=3)

    def test_arc_radius_full_circle_and_helical_coordinates(self):
        from silksteel_preview_arcs import interpolate_arc
        base=[10.0,0.0,.2]
        ending=[0.0,10.0,.6]
        ccw=interpolate_arc(base,ending,{"I":-10.0,"J":0.0},False)
        self.assertIsNotNone(ccw)
        points,length=ccw
        self.assertAlmostEqual(length,math.hypot(math.pi*5,.4),places=5)
        self.assertAlmostEqual(points[-1][2],.6)
        self.assertTrue(all(math.isclose(math.hypot(p[0],p[1]),10,abs_tol=1e-4) for p in points))
        reflex=interpolate_arc(base,ending,{"R":-10.0},False)
        self.assertAlmostEqual(reflex[1],15*math.pi,places=4)
        full=interpolate_arc(base,base,{"I":-10.0,"J":0.0},True)
        self.assertAlmostEqual(full[1],20*math.pi,places=4)
        self.assertGreater(len(full[0]),60)
        self.assertIsNone(interpolate_arc(base,ending,{},False))

    def test_arc_center_modes_and_plane_selection(self):
        from silksteel_preview_arcs import interpolate_arc
        start=[10.,0.,.2]
        end=[0.,10.,.2]
        relative=interpolate_arc(start,end,{"I":-10.,"J":0.},False)
        absolute=interpolate_arc(start,end,{"I":0.,"J":0.},False,
                                 absolute_center=True)
        self.assertAlmostEqual(relative[1],absolute[1],places=5)
        g18=interpolate_arc([0.,0.,10.],[10.,0.,0.],
                            {"K":-10.,"I":0.},False,plane="G18")
        self.assertIsNotNone(g18)
        self.assertAlmostEqual(g18[1],math.pi*5,places=5)
        self.assertTrue(all(p[1]==0. for p in g18[0]))

    def test_invalid_subsection_cannot_hide_entire_valid_g1(self):
        from unittest.mock import patch
        from silksteel_preview_surface import DepositedSurface
        actual=DepositedSurface.sections

        def patched(self,start,end,area,nominal,first_layer,layer_z=None):
            result=actual(self,start,end,area,nominal,first_layer,layer_z)
            if len(result)==1:
                t0,t1,w,h=result[0]
                return [(0.,.5,41.,h),(.5,1.,w,h)]
            return result

        with patch.object(DepositedSurface,"sections",patched):
            meta,rows=self.parse(["G90","M83","G0 X0 Y0 Z0.2",
                                   "G1 X10 Y0 E0.25"])
        self.assertEqual(meta["count"],1)
        self.assertEqual(meta["partialCount"],1)
        self.assertEqual(meta["omittedCount"],0)
        self.assertAlmostEqual(rows[0][0],5.0)
        self.assertAlmostEqual(rows[0][3],10.0)

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
        for module in ("extrusion_joins", "bead_junctions", "diagnostics", "comparison", "section"):
            self.assertTrue((root / "preview" / (module + ".js")).is_file())
            self.assertIn('"' + module + '.js"', backend)
            self.assertIn('from "./' + module + '.js"', viewer)
        for control in ("cutEnabled", "cutAxis", "cutPosition", "cutInvert",
                        "diagnosticsToggle", "diagnosticsList", "compareToggle",
                        "comparisonSummary"):
            self.assertIn('id="' + control + '"', html)
            self.assertIn('"'+control+'"', viewer)
        self.assertIn("buildJoinAttributes(items,joinRead,role)", viewer)
        self.assertIn("makeJunctionGeometry(THREE,join.fallbacks)", viewer)
        self.assertIn("vec3 p=basePoint + joinVector*lateral", viewer)
        self.assertNotIn("aCap>0.5 && join.z>0.5", viewer)
        self.assertIn("clipping_planes_pars_fragment", viewer)
        self.assertIn("function refreshComparison()", viewer)
        self.assertIn("function showDiagnostics()", viewer)
        self.assertIn("const SEGMENT_SIZE = 48, HOP_SIZE = 40", viewer)



    def test_preview_preserves_original_before_in_place_processing(self):
        script = (Path(__file__).resolve().parents[1] / "SilkSteel.py").read_text(encoding="utf-8")
        begin = script.index("preview_snapshot_dir = None")
        process = script.index('logging.info("Calling process_gcode()...")')
        preserve = script.index('os.path.join(preview_snapshot_dir, "original.gcode")')
        self.assertLess(begin, preserve)
        self.assertLess(preserve, process)
        self.assertIn('if os.path.isfile(original_snapshot):',script)
        self.assertIn('command += ["--original", original_snapshot,',script)


    def test_tiny_detected_gap_never_creates_flat_extrusion_spikes(self):
        from silksteel_preview_surface import DepositedSurface
        field = DepositedSurface()
        field._tops[(0, 0)] = (0.34, 0.2)
        area, nominal = .12, .2
        h = field._effective_height(.1, .1, .4, area, nominal, .2, .4)
        self.assertAlmostEqual(h, nominal, places=5)
        w = field.width_for_area(area, h)
        self.assertLess(w, 0.7)
        self.assertAlmostEqual(w*h-(1-math.pi/4)*h*h, area)

    def test_raised_nozzle_grows_height_not_width_even_near_adjacent_tracks(self):
        from silksteel_preview_surface import DepositedSurface
        field = DepositedSurface()
        field._tops[(0, 0)] = (.37, .2)
        area, nominal = .12, .2
        at_layer = field._effective_height(.1, .1, .4, area, nominal, .2, .4)
        raised = field._effective_height(.1, .1, .45, area, nominal, .2, .4)
        self.assertAlmostEqual(at_layer, nominal)
        self.assertAlmostEqual(raised, .25)
        self.assertGreater(raised, at_layer)
        self.assertLess(field.width_for_area(area, raised),
                        field.width_for_area(area, at_layer))
        self.assertLessEqual(raised, field.free_diameter(area))

    def test_layer_reference_z_controls_adaptive_width_in_converter(self):
        meta, rows = self.parse([
            "; layer_height = 0.2", "; first_layer_height = 0.2",
            ";LAYER_CHANGE", ";Z:0.2", ";HEIGHT:0.2", ";LAYER:0",
            "G90", "M83", "G0 X0 Y0 Z0.2 F1200",
            "G1 X10 Y0 E0.5",
            ";LAYER_CHANGE", ";Z:0.4", ";HEIGHT:0.2", ";LAYER:1",
            "G0 X0 Y0 Z0.4",
            "G1 X10 Y0 E0.5",
            "G0 Z0.5",
            "G1 X20 Y0 E0.5",
        ])
        upper = rows[meta["layers"][1]["start"]:]
        # Greater free Z must produce at least one taller, narrower bead,
        # not a collapsed .02 mm section with an inflated width.
        self.assertTrue(all(p[7] >= min(.2, p[6])-1e-5 for p in upper))
        base = next(p for p in upper if p[2] <= .4 + .001)
        lifted = next(p for p in upper if p[2] >= .5 - .001)
        self.assertGreaterEqual(lifted[7], base[7]-1e-4)
        self.assertTrue(all(p[7] <= p[6]+1e-5 for p in upper))


    def test_source_windows_use_sparse_offsets_without_loading_full_file(self):
        from silksteel_preview import source_context, SOURCE_INDEX_STEP
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "big.gcode"
            lines = [f"G1 X{i} Y0 E{i/1000:.3f} ; line {i}" for i in range(2500)]
            payload = "\n".join(lines) + "\n"
            file.write_text(payload, encoding="utf-8")
            offsets, size = [], 0
            for index, line in enumerate(lines, 1):
                if (index - 1) % SOURCE_INDEX_STEP == 0:
                    offsets.append(size)
                size += len((line + "\n").encode("utf-8"))
            result = source_context(file, offsets, 2010, 3)
            self.assertEqual([row["line"] for row in result],
                             list(range(2007, 2014)))
            self.assertEqual(result[3]["text"], lines[2009])
            self.assertEqual(source_context(file, offsets, 1, 1)[0]["line"], 1)

    def test_missing_positive_extrusion_is_flagged_in_raw_track(self):
        from silksteel_preview import RAW_PACK
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root/"missing.gcode", root/"missing.bin"
            source.write_text("\n".join([
                ";LAYER_CHANGE", ";Z:0.2", ";HEIGHT:0.2", ";LAYER:0",
                "G90", "M83", "G0 X0 Y0 Z0.2 F1200",
                "G1 X10 Y0 E0.25",
                "G1 X20 Y0 E300",
                "G0 X25 Y0 E0.25",
                "G1 X25 Y0 Z0.5 E0.1",
            ]) + "\n", encoding="utf-8")
            meta = convert(source, target)
            raw_bytes = target.with_suffix(".raw.bin").read_bytes()
            raw = [RAW_PACK.unpack_from(raw_bytes,i)
                   for i in range(0,len(raw_bytes),RAW_PACK.size)]
            self.assertEqual(meta["rawCount"],4)
            self.assertEqual(meta["layers"][0]["rawCount"],4)
            self.assertEqual([row[6] for row in raw],[0,1,0,3])
            self.assertEqual([row[7] for row in raw],[8,9,10,11])
            self.assertEqual(meta["omittedCount"],1)
            self.assertEqual(meta["count"],2)

    def test_sidebar_preferences_survive_local_preview_port_changes(self):
        from unittest import mock
        from silksteel_preview import read_sidebar, write_sidebar, sidebar_settings_path
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict("os.environ", {"LOCALAPPDATA": tmp}):
                values={"Diagnose": True, "Schichten": False}
                write_sidebar(values)
                self.assertEqual(read_sidebar(), values)
                self.assertTrue(sidebar_settings_path().is_file())
                with self.assertRaises(ValueError):
                    write_sidebar({"unknown": True})
                with self.assertRaises(ValueError):
                    write_sidebar({"Diagnose": "yes"})

    def test_sidebar_groups_and_source_inspector_in_viewer(self):
        root=Path(__file__).resolve().parents[1]
        markup=(root/"preview"/"index.html").read_text(encoding="utf-8")
        script=(root/"preview"/"viewer.js").read_text(encoding="utf-8")
        backend=(root/"silksteel_preview.py").read_text(encoding="utf-8")
        self.assertEqual(markup.count('<details class="side-group"'),8)
        for name in ("sourceDrawer","sourceLines","sourceLine","sourceGo",
                     "sourceClose","rawOmitted","rawAll","rawInfo"):
            self.assertIn('id="'+name+'"',markup)
        self.assertIn('openSourceLine(pick.line)',script)
        self.assertIn('function createRawOverlay(',script)
        self.assertIn('silksteel.sidebar.',script)
        self.assertIn('if path == "/source":',backend)
        self.assertIn('source_context(*entry, line, context)',backend)
        self.assertIn('"/gcode_inspector.js"',backend)



if __name__ == "__main__":
    unittest.main()
