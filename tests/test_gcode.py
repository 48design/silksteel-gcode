import contextlib
import io
import os
import tempfile
import unittest

import SilkSteel as silk


def fixture(relative=False):
    lines = [
        "; layer_height = 0.2",
        "; first_layer_height = 0.2",
        "; extrusion_width = 0.45",
        "G90",
        "M83" if relative else "M82",
        "G92 E0" if relative else "G92 E100",
    ]
    e = 0.0 if relative else 100.0

    def extrusion(x, y, amount=0.4, feed=None):
        nonlocal e
        e = amount if relative else e + amount
        return f"G1 X{x} Y{y} E{e:.5f}" + (f" F{feed}" if feed else "")

    for layer, z in enumerate((0.2, 0.4, 0.6)):
        lines += [
            ";LAYER_CHANGE", f";Z:{z:.2f}", ";HEIGHT:0.2", f";LAYER:{layer}",
            f"G1 Z{z:.2f} F1200",
            ";TYPE:External perimeter",
            "G1 X0 Y0 F6000",
            extrusion(10, 0, feed=1200),
            extrusion(10, 10),
            extrusion(0, 10),
            extrusion(0, 0),
            ";TYPE:Internal infill",
            "G1 X0 Y5 F8400",
            extrusion(10, 5, feed=1500),
            "G1 X0 Y6 F8400",
            extrusion(10, 6, feed=1500),
            ";TYPE:Solid infill",
            "G1 X0 Y2 F8400",
            extrusion(10, 2, feed=900),
        ]
    return "\n".join(lines) + "\n"


class GCodeSafetyTests(unittest.TestCase):
    def process(self, gcode, **settings):
        with tempfile.TemporaryDirectory() as root:
            src = os.path.join(root, "input.gcode")
            dst = os.path.join(root, "output.gcode")
            with open(src, "w", encoding="utf-8") as stream:
                stream.write(gcode)
            with contextlib.redirect_stdout(io.StringIO()):
                silk.process_gcode(src, dst, outer_layer_height=0.1,
                                   enable_safe_z_hop=False, **settings)
            with open(dst, encoding="utf-8") as stream:
                return stream.read()

    def test_modal_feedrate_does_not_inherit_fast_travel(self):
        src = "G1 X10 Y0 E1 F1200\nG0 Z0.5 F8400\nG1 X20 Y0 E2\n"
        fixed = silk.restore_extrusion_feedrates(src)
        self.assertIn("G1 X20 Y0 E2 F1200", fixed)
        self.assertEqual(fixed.count("F1200"), 2)

    def test_source_e_deltas_not_absolute_coordinates(self):
        source = ["M82\n", "G92 E100\n", "G1 X10 E100.5\n",
                  "G1 X20 E101\n", "M83\n", "G1 X30 E0.25\n",
                  "G1 E-0.8\n", "M82\n", "G92 E10\n", "G1 X40 E10.4\n"]
        deltas, targets, modes = silk.scan_source_extrusion(source)
        self.assertAlmostEqual(deltas[2], 0.5)
        self.assertAlmostEqual(deltas[3], 0.5)
        self.assertAlmostEqual(deltas[5], 0.25)
        self.assertAlmostEqual(deltas[6], -0.8)
        self.assertAlmostEqual(deltas[9], 0.4)
        self.assertFalse(modes[3])
        self.assertTrue(modes[5])
        self.assertAlmostEqual(targets[9], 10.4)

    def test_smoothificator_absolute_e_preserves_extrusion(self):
        output = self.process(fixture(), enable_smoothificator=True,
                              enable_nonplanar=False)
        self.assertIn("Smoothificator E sync", output)
        segments = output.split("; ====== SMOOTHIFICATOR START:")[1].split("Smoothificator E sync")[0]
        e_values = [silk.extract_e(line) for line in segments.splitlines()
                    if line.startswith("G1 X") and silk.extract_e(line) is not None]
        self.assertTrue(e_values)
        self.assertGreater(min(e_values), 100)
        self.assertTrue(all(b >= a for a, b in zip(e_values, e_values[1:])))

    def test_smoothificator_relative_e_uses_segment_deltas(self):
        output = self.process(fixture(relative=True), enable_smoothificator=True,
                              enable_nonplanar=False)
        self.assertIn("SMOOTHIFICATOR START", output)
        segments = output.split("; ====== SMOOTHIFICATOR START:")[1].split(";TYPE:Internal infill")[0]
        e_values = [silk.extract_e(line) for line in segments.splitlines()
                    if line.startswith("G1 X") and silk.extract_e(line) is not None]
        self.assertGreater(len(e_values), 4)
        self.assertTrue(all(0 < value <= 0.4 for value in e_values))
        self.assertNotIn("Smoothificator E sync", segments)

    def test_nonplanar_absolute_e_resyncs_before_next_type(self):
        output = self.process(fixture(), enable_smoothificator=False,
                              enable_nonplanar=True, segment_length=1.0,
                              amplitude=0.25, frequency=6.0)
        self.assertIn("Non-planar E sync", output)
        lines = output.splitlines()
        idx = next(i for i, line in enumerate(lines) if "Non-planar E sync" in line)
        synced_e = silk.extract_e(lines[idx])
        next_e = next(silk.extract_e(line) for line in lines[idx+1:]
                      if line.startswith("G1 X") and silk.extract_e(line) is not None)
        self.assertAlmostEqual(next_e - synced_e, 0.4, places=3)

    def test_nonplanar_relative_e_keeps_small_segments(self):
        output = self.process(fixture(relative=True), enable_smoothificator=False,
                              enable_nonplanar=True, segment_length=1.0,
                              amplitude=0.25, frequency=6.0)
        segments = [silk.extract_e(line) for line in output.splitlines()
                    if line.startswith("G1 X") and " Z" in line
                    and silk.extract_e(line) is not None]
        self.assertGreater(len(segments), 10)
        self.assertTrue(all(0 <= value < 1 for value in segments))
        self.assertNotIn("Non-planar E sync", output)


    def test_partial_axis_nonplanar_moves_are_processed(self):
        source = fixture().replace("G1 X10 Y5 E", "G1 X10 E").replace(
            "G1 X10 Y6 E", "G1 X10 E")
        output = self.process(source, enable_smoothificator=False,
                              enable_nonplanar=True, segment_length=1,
                              amplitude=0.25, frequency=6)
        self.assertGreater(output.count("G1 X1.000"), 0)
        self.assertIn("Non-planar E sync", output)

    def test_invalid_nonplanar_segment_length_is_rejected(self):
        with self.assertRaises(ValueError):
            self.process(fixture(), enable_smoothificator=False,
                         enable_nonplanar=True, segment_length=0)

    def test_bricklayers_does_not_scale_absolute_e_coordinate(self):
        lines = [
            "; layer_height = 0.2", "; first_layer_height = 0.2",
            "; extrusion_width = 0.45", "G90", "M82", "G92 E100",
        ]
        e = 100.0
        for layer, z in enumerate((0.2, 0.4, 0.6, 0.8)):
            lines += [
                ";LAYER_CHANGE", f";Z:{z:.2f}",
                ";HEIGHT:0.2", f";LAYER:{layer}",
                f"G1 Z{z:.2f} F1200", ";TYPE:Internal perimeter",
                "G0 X0 Y0 F8400",
            ]
            for x, y in ((10, 0), (10, 10), (0, 10), (0, 0)):
                e += 0.4
                lines.append(f"G1 X{x} Y{y} E{e:.5f} F1200")
            lines += [";TYPE:Internal infill", "G0 X3 Y5 F8400"]
            e += 0.4
            lines.append(f"G1 X7 Y5 E{e:.5f} F1500")
        output = self.process("\n".join(lines) + "\n",
                              enable_smoothificator=False,
                              enable_bricklayers=True, enable_nonplanar=True,
                              amplitude=0.1, frequency=6,
                              segment_length=1.0)
        self.assertIn("Bricklayers E sync", output)
        self.assertIn("Bricklayers base pass", output)
        e_deltas, _, modes = silk.scan_source_extrusion(output.splitlines())
        self.assertFalse(any(modes))
        positive = [delta for line, delta in zip(output.splitlines(), e_deltas)
                    if line.startswith("G1 ") and silk.extract_x(line) is not None and delta > 0]
        self.assertTrue(positive)
        self.assertLess(max(positive), 1.0)

    def test_feedrate_after_z_hop_before_extrusion(self):
        source = "G1 X0 Y0 E1 F1350\nG0 Z2 F8400\nG0 X2 Y2 F8400\nG1 X8 Y2 E1.3\n"
        fixed = silk.restore_extrusion_feedrates(source)
        self.assertIn("G1 X8 Y2 E1.3 F1350", fixed)



    def test_segment_line_respects_max_length(self):
        segments = silk.segment_line(0, 0, 2.1, 0, 0.5)
        self.assertGreater(len(segments), 4)
        self.assertNotEqual(segments[0], (0.0, 0.0))
        previous = (0, 0)
        for p in segments:
            self.assertLessEqual(abs(p[0] - previous[0]), 0.500001)
            previous = p
        self.assertEqual(segments[-1], (2.1, 0))
        self.assertEqual(silk.segment_line(2, 2, 2, 2, 0.5), [])

    def test_zero_length_xy_priming_preserves_extrusion(self):
        source = fixture(relative=True).replace(
            "G1 X0 Y5 F8400",
            "G1 X0 Y5 F8400\nG1 X0 Y5 E0.25 F800",
            1
        )
        output = self.process(source, enable_smoothificator=False,
                              enable_nonplanar=True, segment_length=0.5,
                              amplitude=0.2, frequency=6)
        self.assertIn("G1 X0 Y5 E0.25 F800", output)

    def test_relative_bridge_densifier_is_safely_skipped(self):
        source = fixture(relative=True).replace(
            ";TYPE:Solid infill", ";TYPE:Bridge infill")
        output = self.process(source, enable_smoothificator=False,
                              enable_nonplanar=False,
                              enable_bridge_densifier=True)
        self.assertNotIn("[Bridge Densifier]", output)
        self.assertIn(";TYPE:Bridge infill", output)


if __name__ == "__main__":
    unittest.main()
