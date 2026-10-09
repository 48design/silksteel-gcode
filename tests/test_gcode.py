import contextlib
import io
import math
import ast
import os
import tempfile
import unittest

import SilkSteel as silk
from tools import audit_outer_walls as wall_audit


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
            use_zhop = settings.pop("enable_safe_z_hop", False)
            with contextlib.redirect_stdout(io.StringIO()):
                silk.process_gcode(src, dst, outer_layer_height=0.1,
                                   enable_safe_z_hop=use_zhop, **settings)
            with open(dst, encoding="utf-8") as stream:
                return stream.read()

    def test_bricklayers_continue_across_retract_and_g92_in_type_section(self):
        # Original full-block safety guard discarded an entire TYPE section
        # if there was even one E-only retract or a G92 reset. Test an
        # identical pair of stacked internal contours across four layers.
        for relative in (False, True):
            with self.subTest(relative=relative):
                lines = [
                    "; layer_height = 0.2", "; first_layer_height = 0.2",
                    "; extrusion_width = 0.45", "G90",
                    "M83" if relative else "M82",
                ]
                for layer, z in enumerate((0.2, 0.4, 0.6, 0.8)):
                    lines += [
                        ";LAYER_CHANGE", f";Z:{z:.2f}",
                        ";HEIGHT:0.2", f";LAYER:{layer}",
                        f"G1 Z{z:.2f} F1200", ";TYPE:Internal perimeter",
                    ]
                    for contour, x0 in enumerate((0, 20)):
                        lines += [
                            "G92 E0",
                            "G1 E-1.00000 F3900",
                            f"G0 X{x0} Y0 F8400",
                            "G1 E1.00000 F3900" if relative else "G1 E0.00000 F3900",
                        ]
                        current_e = 0
                        for x, y in ((x0+10, 0), (x0+10, 10), (x0, 10), (x0, 0)):
                            current_e += 0.4
                            e = 0.4 if relative else current_e
                            lines.append(f"G1 X{x} Y{y} E{e:.5f} F1200")
                            lines.append("M117 Printing")
                            # Fan commands are interspersed by SuperSlicer
                            # even in one uninterrupted extrusion loop.
                            if x == x0+10 and y == 10:
                                lines.append("M107")
                            if x == x0 and y == 10:
                                lines.append("M106 S63.75")
                        lines.append("G1 E-0.75000 F3900 ; post-contour retract"
                                     if relative else
                                     "G1 E0.85000 F3900 ; post-contour retract")
                        lines.append("G1 E0.75000 F3900 ; post-contour prime"
                                     if relative else
                                     "G1 E1.60000 F3900 ; post-contour prime")
                    lines += [
                        ";TYPE:Internal infill", "G0 X3 Y5 F8400",
                        "G1 X7 Y5 E0.4 F1500",
                    ]
                output = self.process("\n".join(lines) + "\n",
                                      enable_smoothificator=False,
                                      enable_bricklayers=True,
                                      enable_nonplanar=True,
                                      amplitude=0.1, frequency=6,
                                      segment_length=1.0)
                self.assertIn("Bricklayers base pass", output)
                layer_one = output.split(";LAYER:1", 1)[1].split(";LAYER_CHANGE", 1)[0]
                # With two stackable internal contours we must see both
                # parity states on a non-base layer, not always odd.
                self.assertIn("Bricklayers shifted block #1", layer_one)
                self.assertIn("Bricklayers base block #2", layer_one)
                self.assertIn("Bricklayers contour E sync" if not relative
                              else "Bricklayers base pass", output)
                self.assertEqual(output.count("G1 E-1.00000 F3900"), 8)
                self.assertEqual(output.count("M117 Printing"), 4 * 2 * 4)
                self.assertEqual(output.count("M107"), 8)
                self.assertEqual(output.count("M106 S63.75"), 8)
                deltas, _, _ = silk.scan_source_extrusion(output.splitlines())
                retracts = [d for line, d in zip(output.splitlines(), deltas)
                            if "; post-contour retract" in line]
                primes = [d for line, d in zip(output.splitlines(), deltas)
                          if "; post-contour prime" in line]
                self.assertEqual(len(retracts), 8)
                self.assertTrue(all(abs(delta + 0.75) < 1e-5 for delta in retracts))
                self.assertTrue(all(abs(delta - 0.75) < 1e-5 for delta in primes))

    def test_implicit_external_perimeter_across_layer_boundary(self):
        # Reproduces the actual slicer pattern at layer 40: the previous
        # layer ends with an external wall, the next has ;LAYER_CHANGE
        # and priming/travel/M117 but no renewed ;TYPE:External perimeter.
        # Without the injected marker its entire first exterior contour
        # (hundreds of XY extrusion moves) bypasses Smoothificator.
        for relative in (False, True):
            with self.subTest(relative=relative):
                lines = [
                    "; layer_height = 0.28", "; first_layer_height = 0.2",
                    "; extrusion_width = 0.45", "G90",
                    "M83" if relative else "M82",
                ]
                for layer, z in enumerate((0.2, 0.48)):
                    lines.extend([
                        ";LAYER_CHANGE", f";Z:{z:.2f}", ";HEIGHT:0.28",
                        f";LAYER:{layer}", f"G1 Z{z:.2f} F1200", "G92 E0",
                        "G1 E-1 F3900", "M117 Time Left 3h18m",
                        "G1 X20 Y12 F8400", "G1 E1 F3900" if relative else "G1 E0 F3900",
                        ";WIDTH:0.45", "M106 S63.75", "G1 F2700",
                    ])
                    if layer == 0:
                        lines.append(";TYPE:External perimeter")
                    source_e = 0.0
                    for point in range(1, 181):
                        angle = 2 * math.pi * point / 180
                        x = 12 + 8 * math.cos(angle)
                        y = 12 + 8 * math.sin(angle)
                        source_e += 0.04
                        amount = 0.04 if relative else source_e
                        lines.append(f"G1 X{x:.4f} Y{y:.4f} E{amount:.5f}")
                        if point % 11 == 0:
                            lines.append("M117 Time Left 3h18m")
                    lines += [
                        ";WIPE_START", "G1 X19.9 Y12 F8400", ";WIPE_END",
                    ]
                lines.extend([
                    ";TYPE:Internal perimeter", "G1 X19 Y12 F8400",
                    "G1 X18 Y12 E0.06",
                ])
                source = "\n".join(lines) + "\n"
                output = self.process(source, enable_smoothificator=True)
                layer_2 = output.split(";LAYER:1", 1)[1]
                self.assertEqual(layer_2.count("CONTINUED across layer boundary"), 1)
                self.assertIn("; SilkSteel: CONTINUED across layer boundary\n"
                              ";TYPE:External perimeter\n", layer_2)
                self.assertIn(";LAYER:1\n"
                              "; SilkSteel: CONTINUED across layer boundary\n"
                              ";TYPE:External perimeter\n"
                              "G1 Z0.48", output)
                self.assertNotIn(";TYPE:External perimeter ;", layer_2)
                self.assertEqual(layer_2.count("SMOOTHIFICATOR START: 3 passes"), 1)
                first_internal = layer_2.split(";TYPE:Internal perimeter")[0]
                self.assertEqual(len([
                    l for l in first_internal.splitlines()
                    if l.startswith("G1 ") and silk.extract_e(l) is not None
                    and silk.extract_x(l) is not None
                ]), 180 * 3)
                self.assertEqual(first_internal.count("M117 Time Left 3h18m"), 17)
                self.assertEqual(first_internal.count(";WIPE_START"), 1)
                # A synthetic marker does not create or change E moves;
                # across all 3 passes extrusion is still 180 * 0.04 mm.
                prefix = ["M83" if relative else "M82"]
                deltas, _, _ = silk.scan_source_extrusion(prefix + first_internal.splitlines())
                e_total = sum(delta for line, delta in
                              zip(first_internal.splitlines(), deltas[1:])
                              if line.startswith("G1 ") and
                              silk.extract_e(line) is not None and
                              silk.extract_x(line) is not None and delta > 0)
                self.assertAlmostEqual(e_total, 7.2, places=3)

    def test_inherited_outer_type_does_not_override_explicit_inner_type(self):
        lines = [
            ";TYPE:External perimeter", "G1 X1 Y1 E0.1",
            ";LAYER_CHANGE", ";Z:0.48", ";HEIGHT:0.28", ";LAYER:1",
            "G92 E0", "G1 E-1 F3900",
            ";TYPE:Internal perimeter", "G1 X3 Y3 E0.1",
        ]
        updated, added = silk.restore_layer_continued_wall_types(lines)
        self.assertEqual(added, 0)
        self.assertEqual(updated, lines)

    def test_long_relative_e_orphan_loop_is_smoothified(self):
        # Previously only the first 100 segments were inspected, and the
        # heuristic assumed absolute E increases monotonically. Both fail
        # for densely sampled closed contours with relative E (M83).
        source = [
            "; layer_height = 0.28", "; first_layer_height = 0.2",
            "; extrusion_width = 0.45", "G90", "M83",
        ]
        for layer, z in enumerate((0.2, 0.48)):
            source.extend([
                ";LAYER_CHANGE", f";Z:{z}", ";HEIGHT:0.28", f";LAYER:{layer}",
                f"G1 Z{z} F1200", ";TYPE:Custom unlabeled wall",
                "G1 X20 Y12 F8400",
            ])
            for n in range(1, 181):
                angle = n * 2.0 * math.pi / 180
                x = 12 + 8 * math.cos(angle)
                y = 12 + 8 * math.sin(angle)
                amount = 0.04 + (n % 4) * 0.005
                source.append(f"G1 X{x:.4f} Y{y:.4f} E{amount:.5f} F1200")
            source.append(";TYPE:Solid infill")
        output = self.process("\n".join(source) + "\n",
                              enable_smoothificator=True)
        self.assertGreaterEqual(output.count("AUTO-ADDED by Smoothificator"), 2)
        self.assertIn("; SilkSteel: AUTO-ADDED by Smoothificator (heuristic)\n"
                      ";TYPE:External perimeter\n", output)
        self.assertNotIn(";TYPE:External perimeter ;", output)
        layer2 = output.split(";LAYER:1")[-1]
        self.assertIn("SMOOTHIFICATOR START: 3 passes", layer2)

        # Explicitly labelled internal perimeters must NEVER be guessed to
        # be outer just because they happen to trace a closed contour.
        internal = "\n".join(source).replace(
            ";TYPE:Custom unlabeled wall", ";TYPE:Internal perimeter") + "\n"
        internal_output = self.process(internal, enable_smoothificator=True)
        self.assertNotIn("AUTO-ADDED by Smoothificator", internal_output)

    def test_outer_wall_audit_locates_untouched_slicer_feature(self):
        original = fixture(relative=True)
        # Make the second-layer contour a generic perimeter; the
        # Smoothificator cannot safely assume it is an outer wall.
        first = original.find(";TYPE:External perimeter")
        second = original.find(";TYPE:External perimeter", first + 1)
        self.assertGreater(second, first)
        original = (original[:second] + original[second:].replace(
            ";TYPE:External perimeter", ";TYPE:Perimeter", 1))
        output = self.process(original, enable_smoothificator=True)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "analyzed.gcode")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(output)
            runs = list(wall_audit.audit(path))
        skipped = [r for r in runs if r["layer"] == 1 and
                   r["type"] == "Perimeter" and not r["smoothed"]]
        self.assertTrue(skipped, "Audit should identify unsmoothed generic perimeters")
        self.assertGreater(sum(r["length"] for r in skipped), 0)
        smoothed = [r for r in runs if r["layer"] == 2 and
                    "External perimeter" in r["type"] and r["smoothed"]]
        self.assertTrue(smoothed, "Audit must recognize transformed paths")

    def test_cli_has_no_interactive_enter_pause(self):
        # Slicer post-processors must never wait for keyboard interaction.
        with open(silk.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "input"]
        self.assertEqual(calls, [])

    def test_smoothificator_retract_does_not_skip_whole_outer_wall(self):
        # One TYPE section can contain multiple walls with retracts between
        # them. Both walls must receive 3 thinner passes, but pressure
        # changes and the inter-wall travel must occur exactly once.
        for relative in (False, True):
            with self.subTest(relative=relative):
                lines = [
                    "; layer_height = 0.28", "; first_layer_height = 0.2",
                    "; extrusion_width = 0.45", "G90",
                    "M83" if relative else "M82",
                ]
                for layer, z in enumerate((0.2, 0.48)):
                    lines += [
                        ";LAYER_CHANGE", f";Z:{z}", f";HEIGHT:{0.2 if layer == 0 else 0.28}",
                        f";LAYER:{layer}", f"G1 Z{z} F1200",
                        "G92 E0" if relative else "G92 E100",
                        ";TYPE:External perimeter", "G0 X0 Y0 F8400",
                    ]
                    if relative:
                        lines += [
                            "G1 X10 Y0 E0.5 F1200", "G1 X10 Y10 E0.5",
                            "G1 E-0.8 F1800", "G0 X20 Y0 F8400",
                            "G1 E0.8 F1800", "G1 X30 Y0 E0.5 F1200",
                            "G1 X30 Y10 E0.5",
                        ]
                    else:
                        lines += [
                            "G1 X10 Y0 E100.5 F1200", "G1 X10 Y10 E101.0",
                            "G1 E100.2 F1800", "G0 X20 Y0 F8400",
                            "G1 E101.0 F1800", "G1 X30 Y0 E101.5 F1200",
                            "G1 X30 Y10 E102.0",
                        ]
                    lines.append(";TYPE:Solid infill")

                output = self.process("\n".join(lines) + "\n",
                                      enable_smoothificator=True)
                layer2 = output.split(";LAYER:1")[-1]
                self.assertEqual(layer2.count("SMOOTHIFICATOR START: 3 passes"), 2)
                self.assertEqual(layer2.count("G0 X20 Y0 F8400"), 1)
                self.assertEqual(layer2.count("G1 E-0.8 F1800" if relative else "G1 E100.2 F1800"), 1)
                self.assertEqual(layer2.count("G1 E0.8 F1800" if relative else "G1 E101.0 F1800"), 1)
                # Source has 2mm of deposited filament across 4 XY moves.
                # The 3 passes must preserve the same total XY extrusion.
                # Include the file's modal E mode in this layer excerpt.
                deltas, _, _ = silk.scan_source_extrusion(
                    ["M83" if relative else "M82"] + layer2.splitlines())
                wall_e = sum(delta for line, delta in zip(layer2.splitlines(), deltas[1:])
                             if line.startswith("G1 ") and silk.extract_x(line) is not None
                             and silk.extract_e(line) is not None and delta > 0)
                self.assertAlmostEqual(wall_e, 2.0, places=3)

    def test_zhop_drops_before_z_bearing_nonplanar_extrusion(self):
        # Regression: G1 X/Y/Z/E moves all axes simultaneously. A segment's
        # explicit Z must NOT cancel a pending hop without a separate Z drop.
        output = self.process(fixture(), enable_smoothificator=False,
                              enable_nonplanar=True, enable_safe_z_hop=True,
                              segment_length=1.0, amplitude=0.25, frequency=6.0)
        lines = output.splitlines()
        lifted = False
        restored_before_z_extrusion = 0
        for idx, line in enumerate(lines):
            if "; Z-hop lift" in line:
                lifted = True
            elif "; Z-hop drop" in line:
                self.assertTrue(lifted)
                lifted = False
            elif (";LAYER_CHANGE" in line or ";LAYER:" in line or
                  line.startswith(("G0 ", "G1 ")) and
                  silk.extract_z(line) is not None and
                  silk.extract_x(line) is None and silk.extract_y(line) is None and
                  silk.extract_e(line) is None):
                # An independent Z command explicitly repositions the nozzle.
                lifted = False
            if (line.startswith("G1 ") and silk.extract_e(line) is not None and
                silk.extract_z(line) is not None and
                (silk.extract_x(line) is not None or
                 silk.extract_y(line) is not None)):
                self.assertFalse(lifted, f"Extrusion begins above print at line {idx}: {line}")
                if idx > 0 and "; Z-hop drop" in lines[idx - 1]:
                    restored_before_z_extrusion += 1
        self.assertGreater(restored_before_z_extrusion, 0)

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
