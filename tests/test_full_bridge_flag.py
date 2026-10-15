import os
import subprocess
import sys
import tempfile
import unittest


class FullModeBridgeTests(unittest.TestCase):
    def test_full_enables_bridge_densifier_and_explicit_disable_wins(self):
        # Real CLI invocation: FULL enables bridge serpentine even if other
        # optional features are disabled. Explicit -disableBridgeDensifier
        # must win, irrespective of argument order.
        source = "\n".join([
            "; layer_height = 0.2", "; first_layer_height = 0.2",
            "; extrusion_width = 0.45", "G90", "M82", "G92 E0",
            ";LAYER_CHANGE", ";Z:0.2", ";HEIGHT:0.2", ";LAYER:0",
            "G1 Z0.2 F8400", ";TYPE:Internal perimeter",
            "G0 X0 Y10 F8400", "G1 X10 Y10 E0.8 F1500",
            ";TYPE:Internal infill",
            ";LAYER_CHANGE", ";Z:0.4", ";HEIGHT:0.2", ";LAYER:1",
            "G1 Z0.4 F8400", "G92 E0",
            ";TYPE:Bridge infill", ";WIDTH:0.4", "M107",
            "G0 X0 Y0 F8400", "G1 F1800",
            "G1 X10 Y0 E1.00000",
            "G1 X10 Y0.4 E1.05000",
            "G1 X0 Y0.4 E2.05000",
            "G1 E1.75000 F3900",
            ";TYPE:Internal infill", "G0 X5 Y5 F8400",
        ]) + "\n"
        script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "SilkSteel.py")
        common = [
            "-disableNonPlanar", "-disableBricklayers",
            "-disableSafeZHop", "-disableSmoothificator",
        ]
        with tempfile.TemporaryDirectory() as folder:
            src = os.path.join(folder, "input.gcode")
            with open(src, "w", encoding="utf-8") as stream:
                stream.write(source)
            cases = [
                ("full", ["-full"], True),
                ("disabled_after", ["-full", "-disableBridgeDensifier"], False),
                ("disabled_before", ["-disableBridgeDensifier", "-full"], False),
            ]
            for name, flags, expected in cases:
                with self.subTest(case=name):
                    dst = os.path.join(folder, name + ".gcode")
                    run = subprocess.run(
                        [sys.executable, script, src, "-o", dst, *common, *flags],
                        capture_output=True, text=True, timeout=40
                    )
                    self.assertEqual(run.returncode, 0, run.stdout + "\n" + run.stderr)
                    with open(dst, encoding="utf-8") as stream:
                        output = stream.read()
                    self.assertEqual("Bridge serpentine intermediate" in output, expected)
                    self.assertEqual(output.count("G1 E1.75000 F3900"), 1)


if __name__ == "__main__":
    unittest.main()
