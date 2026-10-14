"""Analytical preview geometry: no substrate grid, exactly conserved volume."""
import math
from pathlib import Path
import tempfile
import unittest

from silksteel_preview import PACK, convert
from silksteel_preview_ideal import IdealizedProfile


class AnalyticViewerTests(unittest.TestCase):
    def test_actual_prime_line_uses_bed_gap_not_nominal_layer_thickness(self):
        # Reproduces the G-code preamble: 30 mm filament in a 30 mm line,
        # Z=0.8, before any ;LAYER_CHANGE. It was rendered as a 12 mm-wide
        # sheet because the preview incorrectly assumed a 0.2 mm substrate.
        lines = [
            "G90", "M82", "G92 E0", "G92 X15 Y11",
            "G1 X141.635 Y91.06 Z0.8 F6000",
            "G1 X171.635 Y91.06 E30 F360",
            "G92 E0", "G1 E-0.5 F2100",
            "G1 X181.635 F6000",
            ";LAYER_CHANGE", ";Z:0.2", ";HEIGHT:0.2", ";LAYER:0",
            "G1 Z0.2 F8400", "G92 E0",
            "G1 X144.305 Y103.919 F8400",
            "G1 X165.695 Y103.919 E0.7 F1200",
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "prime.gcode"
            target = root / "prime.bin"
            source.write_text("\n".join(lines) + "\n", encoding="utf-8")
            meta = convert(source, target)
            binary = target.read_bytes()
            rows = [PACK.unpack_from(binary,i)
                    for i in range(0,len(binary),PACK.size)]
        self.assertEqual(meta["count"], 2)
        self.assertEqual([layer["number"] for layer in meta["layers"]],[-1,0])
        priming, outline = rows
        self.assertEqual(int(priming[9]),6)
        self.assertAlmostEqual(priming[2], .8, places=6)
        self.assertAlmostEqual(priming[7], .8, places=6)
        self.assertAlmostEqual(priming[2]-priming[7], 0., places=6)
        self.assertGreater(priming[6], 3.0)
        self.assertLess(priming[6], 3.2)
        expected_area = math.pi * (.875**2)
        self.assertAlmostEqual(IdealizedProfile.area_for_size(priming[6],priming[7]),
                               expected_area,places=5)
        self.assertAlmostEqual(outline[7], .2,places=6)
        self.assertEqual(meta["omittedCount"],0)

    def test_first_explicit_layer_uses_exact_nozzle_to_bed_distance(self):
        # The viewer models the commanded geometry, not physical sagging.
        # Height is exactly the known Z=.8 even for low extrusion flow.
        area=.4*math.pi*(.875**2)/20
        sections=IdealizedProfile.sections(
            (0.,0.,.8),(20.,0.,.8),area,.2,.2,
            layer_z=.8,role=0,bed_reference=True)
        self.assertEqual(len(sections),1)
        _,_,width,height=sections[0]
        self.assertAlmostEqual(height,.8,places=7)
        self.assertLess(width,height)
        self.assertAlmostEqual(IdealizedProfile.area_for_size(width,height),
                               area,places=9)

    def test_unmarked_second_layer_no_longer_uses_printbed_profile(self):
        lines=[
            "G90","M83","G0 Z0.8 X0 Y0",
            "G1 X10 Y0 E4",
            "G0 Z1.0",
            "G1 X20 Y0 E0.4",
        ]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/"unmarked.gcode"
            target=root/"unmarked.bin"
            source.write_text("\n".join(lines)+"\n",encoding="utf-8")
            meta=convert(source,target)
            data=target.read_bytes()
            rows=[PACK.unpack_from(data,i)
                  for i in range(0,len(data),PACK.size)]
        self.assertEqual(len(rows),2)
        self.assertEqual(len(meta["layers"]),2)
        self.assertAlmostEqual(rows[1][7],.2,places=6)

    def test_profiles_preserve_volume_for_every_feature(self):
        area=.11
        cases=[
            (0,0,.2,.4,.4,.2),   # ordinary layer
            (0,0,.2,.6,.4,.4),   # non-planar/shifted nozzle +0.2 mm
            (0,0,.2,.3,.4,.1),   # valley: nozzle below layer reference
            (1,0,.1,.3,.4,.1),   # Smoothificator thickness
            (2,1,.15,.4,.4,.2),  # Bricklayers base 1/2
            (2,2,.15,.55,.4,.15),# Bricklayers base 2/2
            (5,0,.2,.4,.4,.1),   # valley filling stroke
            (6,0,.2,.4,.4,IdealizedProfile.free_square_side(area)),
        ]
        for role,brick_pass,nominal,z,ref,expected in cases:
            with self.subTest(role=role,brick_pass=brick_pass,z=z):
                sections=IdealizedProfile.sections(
                    (0.,0.,z),(10.,0.,z),area,nominal,.2,
                    layer_z=ref,role=role,brick_pass=brick_pass)
                self.assertEqual(len(sections),1)
                _,_,width,height=sections[0]
                self.assertAlmostEqual(height,expected,places=6)
                self.assertAlmostEqual(IdealizedProfile.area_for_size(width,height),
                                       area,places=9)

    def test_analytical_profile_does_not_depend_on_prior_paths(self):
        area=.08
        first=IdealizedProfile.sections(
            (0.,0.,.4),(10.,0.,.4),area,.2,.2,layer_z=.4)
        # The same path following any other extrusion has the same geometry.
        second=IdealizedProfile.sections(
            (0.,0.,.4),(10.,0.,.4),area,.2,.2,layer_z=.4)
        self.assertEqual(first,second)
        self.assertEqual(len(first),1)
        source=(Path(__file__).resolve().parents[1] /
                "silksteel_preview.py").read_text(encoding="utf-8")
        self.assertIn("IdealizedProfile.sections(",source)
        self.assertNotIn("surface.deposit(",source)
        self.assertNotIn("surface.support_top(",source)

    def test_processed_feature_markers_define_pass_heights(self):
        lines=[
            "; layer_height = 0.2", ";LAYER_CHANGE", ";Z:0.4",
            ";HEIGHT:0.2", ";LAYER:1", "G90", "M83",
            "G0 X0 Y0 Z0.4 F1200", "G1 X10 Y0 E0.4",
            "; ====== SMOOTHIFICATOR START: 2 passes at 0.1000mm each ======",
            "G0 Z0.3 ; Smoothificator pass 1/2", "G1 X20 Y0 E0.2",
            "G0 Z0.4 ; Smoothificator pass 2/2", "G1 X30 Y0 E0.2",
            "; ====== SMOOTHIFICATOR END ======",
            "G0 Z0.4 ; Bricklayers base pass 1/2", "G1 X40 Y0 E0.3",
            "G0 Z0.55 ; Bricklayers base pass 2/2", "G1 X50 Y0 E0.3",
            "G1 Z0.4 ; Reset Z",
            ";TYPE:Internal infill", "G0 Z0.8", "G1 X60 Y0 Z0.9 E0.2",
            ";TYPE:Bridge infill", "G0 Z0.4", "G1 X70 Y0 E0.2",
            ";TYPE:Internal infill", "; SilkSteel: VALLEY_FILL_START",
            "G0 Z0.3", "G1 X80 Y0 E0.2",
            "; SilkSteel: VALLEY_FILL_END",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/"processed.gcode"
            target=root/"preview.bin"
            source.write_text("\n".join(lines)+"\n",encoding="utf-8")
            meta=convert(source,target)
            data=target.read_bytes()
            rows=[PACK.unpack_from(data,i) for i in range(0,len(data),PACK.size)]
        self.assertEqual(meta["count"],8)
        self.assertEqual([int(p[8]) for p in rows],[0,1,1,2,2,4,6,5])
        for row,expected in zip(rows[:6],[.2,.1,.1,.2,.15,.65]):
            self.assertAlmostEqual(row[7],expected,places=5)
        self.assertAlmostEqual(rows[7][7],.1,places=5)
        for row in rows:
            self.assertTrue(0<row[6]<=40 and 0<row[7]<=40)
        self.assertEqual(meta["omittedCount"],0)


if __name__=="__main__":
    unittest.main()
