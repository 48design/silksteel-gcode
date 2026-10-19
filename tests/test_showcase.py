import contextlib
import io
import os
import tempfile
import unittest

import SilkSteel as silk
from tools.showcase import filter_showcase_gcode


def fixture(relative=False):
    lines = [
        "; layer_height = 0.2",
        "; first_layer_height = 0.2",
        "; bottom_solid_layers = 2",
        "; extrusion_width = 0.45",
        "G90", "M83" if relative else "M82", "G92 E0",
    ]
    e = 0.0

    def extrude(x, y, layer_z, amount=0.3):
        nonlocal e
        if not relative:
            e += amount
        else:
            e = amount
        return f"G1 X{x} Y{y} Z{layer_z:.2f} E{e:.5f} F1500"

    for layer, z in enumerate((0.2, 0.4, 0.6, 0.8)):
        lines += [
            ";LAYER_CHANGE", f";Z:{z:.2f}", ";HEIGHT:0.2", f";LAYER:{layer}",
            f"G1 Z{z:.2f} F1200",
            ";TYPE:External perimeter",
            f"G0 X0 Y{layer} F8400",
            extrude(12, layer, z),
            extrude(12, layer + 8, z),
            ";TYPE:Internal perimeter",
            extrude(1, layer, z),
            extrude(1, layer + 8, z),
            ";TYPE:Solid infill",
            f"G0 X2 Y{layer + 2} F8400",
            extrude(11, layer + 2, z),
            ";TYPE:Internal infill",
            f"G0 X2 Y{layer + 4} F8400",
            extrude(11, layer + 4, z),
            ";TYPE:Top solid infill",
            extrude(9, layer + 5, z),
        ]
    return "\n".join(lines) + "\n"


class ShowcaseTests(unittest.TestCase):
    def test_keep_bottom_and_infill_and_preserve_positive_e_deltas(self):
        for relative in (False, True):
            with self.subTest(relative=relative):
                source = fixture(relative)
                result, stats = filter_showcase_gcode(source)
                self.assertEqual(stats["bottom_layers"], 2)
                self.assertEqual(stats["bottom_extrusions"], 2)
                self.assertEqual(stats["infill_extrusions"], 4)
                self.assertEqual(stats["removed_extrusions"], 4 * 4 + 2 + 4)
                self.assertGreater(stats["removed_motions"], stats["removed_extrusions"])
                self.assertNotIn("showcase travel", result)

                lines = result.splitlines()
                deltas, _, _ = silk.scan_source_extrusion(lines)
                positive = [
                    delta for line, delta in zip(lines, deltas)
                    if line.startswith(("G0 ", "G1 ")) and delta > 1e-8
                ]
                self.assertEqual(len(positive), 6)
                self.assertTrue(all(abs(value - 0.3) < 1e-5 for value in positive))
                for role in ("External perimeter", "Internal perimeter", "Top solid infill"):
                    self.assertNotIn(";TYPE:" + role, result)
                if relative:
                    self.assertNotIn("showcase E sync", result)
                else:
                    self.assertIn("showcase E sync", result)

    def test_explicit_bottom_count_and_abs_relative_motion(self):
        source = "\n".join([
            "; bottom_solid_layers = 1", "G90", "M82", "G92 E0",
            ";LAYER:0", ";TYPE:Solid infill",
            "G1 X0 Y0 E0.5",
            ";LAYER:1", "G91", ";TYPE:External perimeter",
            "G1 X3 E1.0", "G1 Y4 E1.5",
            "G90", "M83", ";TYPE:Internal infill",
            "G1 X10 Y10 E0.2",
            "M82", "G1 X11 Y10 E1.9",
        ]) + "\n"
        output, stats = filter_showcase_gcode(source)
        self.assertIn("G0 X3.00000 Y4.00000 F8400", output)
        self.assertNotIn("G1 X3 E1.0", output)
        self.assertNotIn("G1 Y4 E1.5", output)
        self.assertIn("G92 E1.70000 ; SilkSteel: showcase E sync", output)
        self.assertEqual(stats["bottom_layers"], 1)
        lines = output.splitlines()
        deltas, _, _ = silk.scan_source_extrusion(lines)
        kept = [
            d for line, d in zip(lines, deltas)
            if line.startswith("G1 ") and ("E0.2" in line or "E1.9" in line)
        ]
        self.assertEqual(len(kept), 2)
        self.assertTrue(all(abs(d - 0.2) < 1e-6 for d in kept))

    def test_prunes_all_removed_wall_and_top_layer_motion(self):
        source = fixture() + "\n".join([
            ";LAYER_CHANGE", ";Z:1.00", ";HEIGHT:0.2", ";LAYER:4",
            ";TYPE:External perimeter",
            "G0 Z5 ; wall-only hop",
            "G0 X100 Y100 ; wall-only travel",
            "G1 X110 Y100 E20",
            ";LAYER_CHANGE", ";Z:1.20", ";LAYER:5",
            "G0 Z6 ; empty top hop",
            "G1 X120 Y120 E21",
            "; end gcode",
            "M104 S0",
            "G0 Z20 ; park",
            "G0 X0 Y200",
            "M84",
        ]) + "\n"
        output, stats = filter_showcase_gcode(source)
        self.assertEqual(stats["empty_layers"], 2)
        self.assertNotIn(";LAYER:4", output)
        self.assertNotIn(";LAYER:5", output)
        self.assertNotIn("wall-only", output)
        self.assertNotIn("empty top hop", output)
        self.assertIn("G0 Z20 ; park", output)
        self.assertIn("G0 X0 Y200", output)
        self.assertIn("M84", output)

    def test_eliminates_wall_hops_and_reconnects_infill(self):
        source = "\n".join([
            "; bottom_solid_layers = 1", "G90", "M82", "G92 E0",
            ";LAYER:0", ";TYPE:Solid infill", "G0 X0 Y0",
            "G1 X10 Y0 E0.4",
            ";TYPE:Internal perimeter",
            "G0 Z1.4 ; deleted wall hop",
            "G0 X100 Y100 ; deleted wall travel",
            "G0 Z0.2", "G1 X110 Y100 E0.8",
            ";TYPE:Internal infill",
            "G0 Z1.4 ; old infill hop",
            "G0 X3 Y3", "G0 Z0.2",
            "G1 X8 Y3 E1.2",
        ]) + "\n"
        output, _ = filter_showcase_gcode(source)
        for unwanted in ("deleted wall hop", "deleted wall travel", "old infill hop", "X100 Y100"):
            self.assertNotIn(unwanted, output)
        self.assertIn("G0 X3.00000 Y3.00000", output)
        self.assertIn("G92 E0.80000", output)
        deltas, _, _ = silk.scan_source_extrusion(output.splitlines())
        self.assertEqual([round(d, 3) for d in deltas if d > 1e-8], [0.4, 0.4])

    def test_deleted_wall_pressure_and_sync_commands_do_not_survive(self):
        source = "\n".join([
            "; bottom_solid_layers = 1", "G90", "M82", "G92 E0",
            ";LAYER:0", ";TYPE:Solid infill",
            "G1 X10 Y0 E0.5",
            "G1 E-0.5 F1800 ; retained retract",
            ";TYPE:External perimeter",
            "G0 X12 Y12",
            "G1 E0.5 F1800 ; wall prime",
            "G1 X20 Y12 E1.0",
            ";TYPE:Internal infill",
            "G0 X0 Y5", "G1 X10 Y5 E1.5",
        ]) + "\n"
        output, _ = filter_showcase_gcode(source)
        self.assertNotIn("G0 X12 Y12", output)
        self.assertNotIn("G1 X20 Y12 E1.0", output)
        self.assertNotIn("retained retract", output)
        self.assertNotIn("wall prime", output)
        self.assertNotIn("showcase pressure", output)
        deltas, _, _ = silk.scan_source_extrusion(output.splitlines())
        self.assertEqual([round(d, 2) for d in deltas if abs(d) > 1e-8],
                         [0.5, 0.5])

    def test_no_smoothificator_bricklayers_pressure_ghosts(self):
        source = "\n".join([
            "; layer_height = 0.28", "; bottom_solid_layers = 1",
            "G90", "M82", "G92 E0",
            ";LAYER_CHANGE", ";Z:0.28", ";HEIGHT:0.28", ";LAYER:0",
            ";TYPE:Solid infill", "G0 X0 Y0", "G1 X10 Y0 E0.3",
            ";LAYER_CHANGE", ";Z:0.56", ";HEIGHT:0.28", ";LAYER:1",
            "G92 E0", "G1 E-10.000000 F3900", "M117 Time Left 0h22m33s",
            "G1 E0.000000 F3900",
            ";TYPE:External perimeter",
            "; ====== SMOOTHIFICATOR START: 3 passes at 0.0933mm each ======",
            "; Smoothificator pass 1/3",
            "G1 X100 Y100 E2.09026",
            "; Smoothificator pass 2/3",
            "G92 E2.09026 ; SilkSteel: showcase E sync",
            "G1 E2.09026 ; SilkSteel: showcase pressure",
            "; Smoothificator pass 3/3",
            "G92 E4.18052 ; SilkSteel: showcase E sync",
            "G1 E4.18052 ; SilkSteel: showcase pressure",
            "G92 E6.27078 ; Smoothificator E sync",
            "; ====== SMOOTHIFICATOR END ======",
            ";WIPE_START", ";WIPE_END",
            ";TYPE:Internal perimeter",
            "G92 E12.40327 ; Bricklayers contour E sync",
            "G1 X120 Y100 E13.4",
            "G92 E18.39416 ; Bricklayers contour E sync",
            "G92 E0", "G1 E-10 F3900", "G1 E0 F3900",
            ";TYPE:Internal infill",
            "G0 X20 Y20", "G1 X30 Y20 Z0.56 E0.35",
        ]) + "\n"
        result, stats = filter_showcase_gcode(source)
        self.assertGreater(stats["removed_extrusions"], 0)
        for garbage in ("SMOOTHIFICATOR", "Smoothificator pass",
                        "Bricklayers", "showcase pressure",
                        "G1 E2.09026", "G1 E4.18052",
                        "G1 E-10", "M117 Time Left"):
            self.assertNotIn(garbage, result)
        self.assertNotIn(";TYPE:External perimeter", result)
        self.assertNotIn(";TYPE:Internal perimeter", result)
        self.assertEqual(result.count("G92 E"), 2)  # startup + one infill rebase
        deltas, _, _ = silk.scan_source_extrusion(result.splitlines())
        self.assertEqual([round(d, 3) for d in deltas if abs(d) > 1e-8], [0.3, 0.35])

    def test_requires_printable_bottom(self):
        with self.assertRaisesRegex(ValueError, "bottom solid"):
            filter_showcase_gcode(
                "; bottom_solid_layers = 2\n;LAYER:0\n"
                ";TYPE:Internal infill\nG1 X10 Y1 E0.2\n")

    def test_complete_processing_retains_nonplanar_and_bottom_only(self):
        with tempfile.TemporaryDirectory() as directory:
            src = os.path.join(directory, "input.gcode")
            dst = os.path.join(directory, "showcase.gcode")
            with open(src, "w", encoding="utf-8") as file:
                file.write(fixture())
            with contextlib.redirect_stdout(io.StringIO()):
                silk.process_gcode(
                    src, dst, enable_nonplanar=True, showcase=True,
                    enable_smoothificator=True, enable_bricklayers=True,
                    enable_safe_z_hop=True, amplitude=0.2,
                    frequency=6, segment_length=1.0)
            with open(dst, encoding="utf-8") as file:
                output = file.read()
            self.assertIn("SilkSteel: showcase reconnect", output)
            self.assertIn(";TYPE:Internal infill", output)
            self.assertIn(";TYPE:Solid infill", output)
            self.assertIn("showcase E sync", output)
            self.assertNotIn("G1 X12 Y0 Z0.20 E0.30000 F1500", output)


if __name__ == "__main__":
    unittest.main()
