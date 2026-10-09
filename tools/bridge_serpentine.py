"""Conservative serpentine reconstruction of connected parallel bridge raster runs.

Only reconstruct consecutive reverse-parallel long bridge strokes connected by
short U turns. Unknown commands pass through unchanged. Added strands stay
inside the trapezoid between two existing source strands.
"""
import math
import re

_MOTION = re.compile(r'^G0?[01](?:\s|$)')
_ALLOWED_CONTROLS = re.compile(r'^M(?:73|106|107|117)(?:\s|$)')


def build_bridge_serpentine(lines, initial_z, initial_e, start_x, start_y,
                            connector_max_length, parser, *, initial_relative=False,
                            min_length=2.0, extrusion_width=0.45,
                            flow_fraction=0.25, slowdown=0.6):
    """Return generated lines, final output E, final XY, added strand count."""
    if any(re.match(r'^G91(?:\s|$)', line.split(';', 1)[0].strip())
           for line in lines):
        return list(lines), initial_e, (start_x, start_y), 0

    x, y, z, e = start_x, start_y, initial_z, initial_e
    relative = initial_relative
    current_f = None
    events = []

    for line in lines:
        code = line.partition(';')[0].strip()
        event = {'line': line, 'code': code, 'f': current_f, 'kind': 'barrier'}
        if re.match(r'^M83(?:\s|$)', code):
            relative = True
        elif re.match(r'^M82(?:\s|$)', code):
            relative = False
        elif re.match(r'^G92(?:\s|$)', code):
            reset = parser(code)['e']
            if reset is not None:
                e = reset
        elif _MOTION.match(code):
            p = parser(code)
            xx = x if p['x'] is None else p['x']
            yy = y if p['y'] is None else p['y']
            zz = z if p['z'] is None else p['z']
            length = math.hypot(xx-x, yy-y)
            delta = (p['e'] if relative else p['e']-e) if p['e'] is not None else 0.0
            if p['f'] is not None:
                current_f = p['f']
            event.update(a=(x, y), b=(xx, yy), length=length, de=delta,
                         mode=relative, before_e=e,
                         after_e=e+delta if relative else (
                             p['e'] if p['e'] is not None else e),
                         f=current_f)
            if (p['x'] is None and p['y'] is None and p['z'] is None and
                p['e'] is None and p['f'] is not None):
                event['kind'] = 'metadata'
            elif (code.startswith('G1') and p['z'] is None and delta > 1e-8 and
                  length >= min_length and
                  (p['x'] is not None or p['y'] is not None)):
                event['kind'] = 'strand'
            elif (p['z'] is None and length > 0 and
                  length <= max(connector_max_length*2.6, 2.0) and
                  delta >= 0 and
                  (p['x'] is not None or p['y'] is not None)):
                event['kind'] = 'connector'
            x, y, z = xx, yy, zz
            if p['e'] is not None:
                e = e+delta if relative else p['e']
        elif (not code and not any(k in line.upper() for k in
                                  (';WIPE', ';LAYER', ';TYPE:'))) or \
                _ALLOWED_CONTROLS.match(code):
            event['kind'] = 'metadata'
        events.append(event)

    result = []
    run = []
    count = 0
    actual_e = initial_e
    mode = initial_relative

    def emit_unmodified(nodes):
        nonlocal actual_e, mode
        for node in nodes:
            result.append(node['line'])
            if re.match(r'^M83(?:\s|$)', node['code']):
                mode = True
            elif re.match(r'^M82(?:\s|$)', node['code']):
                mode = False
            elif re.match(r'^G92(?:\s|$)', node['code']):
                reset = parser(node['code'])['e']
                if reset is not None:
                    actual_e = reset
            elif 'de' in node:
                actual_e = actual_e+node['de'] if mode else node['after_e']

    def flush_run():
        nonlocal run, count, actual_e
        if not run:
            return
        strokes = [node for node in run if node['kind'] == 'strand']
        if len(strokes) < 2:
            emit_unmodified(run)
            run = []
            return

        first, last = run.index(strokes[0]), run.index(strokes[-1])
        prefix, raster, suffix = run[:first], run[first:last+1], run[last+1:]
        emit_unmodified(prefix)
        strokes = [node for node in raster if node['kind'] == 'strand']
        gaps, valid, sign = [], True, 0

        for left, right in zip(strokes, strokes[1:]):
            between = raster[raster.index(left)+1:raster.index(right)]
            if any(node['kind'] not in ('metadata', 'connector')
                   for node in between):
                valid = False
                break

            u = (left['b'][0]-left['a'][0], left['b'][1]-left['a'][1])
            v = (right['b'][0]-right['a'][0], right['b'][1]-right['a'][1])
            cosine = (u[0]*v[0]+u[1]*v[1])/(left['length']*right['length'])
            lateral = (u[0]*(right['b'][1]-left['a'][1]) -
                       u[1]*(right['b'][0]-left['a'][0]))/left['length']

            first_axis = (strokes[0]['b'][0]-strokes[0]['a'][0],
                          strokes[0]['b'][1]-strokes[0]['a'][1])
            first_offset = (
                first_axis[0]*(right['a'][1]-left['a'][1]) -
                first_axis[1]*(right['a'][0]-left['a'][0])
            )/strokes[0]['length']
            local_sign = 1 if first_offset > 0 else -1
            perpendicular = abs(lateral)
            near = math.hypot(left['b'][0]-right['a'][0],
                              left['b'][1]-right['a'][1])
            far = math.hypot(left['a'][0]-right['b'][0],
                             left['a'][1]-right['b'][1])

            if (cosine > -0.98 or perpendicular < max(0.30, extrusion_width*0.70) or
                perpendicular > 2.4 or
                near > max(2.8, 4.5*extrusion_width) or
                far > max(3.5, 0.35*min(left['length'], right['length'])) or
                (sign and sign != local_sign) or
                any(node['kind'] == 'connector' and
                    node['length'] > max(2.8, 4.5*extrusion_width)
                    for node in between)):
                valid = False
                break
            sign = local_sign
            # Pick as few intermediate strands as gap coverage permits.
            # Reconcile endpoint parity over the WHOLE raster below, rather
            # than forcing two new strands into every individual gap.
            intermediate_count = max(
                1, math.ceil(perpendicular/max(0.2, extrusion_width*0.52))-1)
            intermediate_count = min(intermediate_count, 8)
            gaps.append((between, intermediate_count))

        if not valid:
            emit_unmodified(raster + suffix)
            run = []
            return

        # The entire raster must end at the source endpoint, otherwise the
        # following source G-code can start at the wrong XY. Alternating
        # directions means TOTAL added-strand count must be even. Add just
        # ONE strand to the widest gap when parity requires it.
        if sum(extra for _, extra in gaps) % 2:
            widest = max(range(len(gaps)), key=lambda index: gaps[index][1])
            between, extra = gaps[widest]
            if extra >= 8:
                emit_unmodified(raster + suffix)
                run = []
                return
            gaps[widest] = (between, extra + 1)

        # Convert only the validated raster, keeping every original long
        # extrusion delta. For M82 a G92 restores the source E coordinate;
        # M83 correctly accumulates the extra interior material.
        if not mode:
            result.append('M83 ; Bridge serpentine temporary relative E\n')

        first_stroke = strokes[0]
        position = first_stroke['b']
        result.append(
            f'G1 X{position[0]:.3f} Y{position[1]:.3f} '
            f'E{first_stroke["de"]:.5f} F{first_stroke["f"] or 1800:g} '
            '; Bridge serpentine original strand\n')
        if mode:
            actual_e += first_stroke['de']

        for i in range(1, len(strokes)):
            previous, current = strokes[i-1], strokes[i]
            between, k = gaps[i-1]
            for node in between:
                if node['kind'] == 'metadata':
                    result.append(node['line'])
            original_turn_e = sum(
                node['de'] for node in between if node['kind'] == 'connector')

            # Plan a complete alternating raster through both the original
            # long strokes and however many interior strands are required.
            # Some original strokes reverse direction; their E delta is
            # unchanged, unlike the original absolute E coordinate.
            lanes = []
            for j in range(1, k+1):
                t = j/(k+1)
                a = (previous['a'][0]*(1-t)+current['b'][0]*t,
                     previous['a'][1]*(1-t)+current['b'][1]*t)
                b = (previous['b'][0]*(1-t)+current['a'][0]*t,
                     previous['b'][1]*(1-t)+current['a'][1]*t)
                length = math.hypot(b[0]-a[0], b[1]-a[1])
                per_mm = (previous['de']/previous['length'] +
                          current['de']/current['length'])/2
                lanes.append((a, b, round(length*per_mm*flow_fraction, 5),
                              True))
            lanes.append((current['a'], current['b'], current['de'], False))

            planned = []
            for a, b, volume, is_new in lanes:
                if math.hypot(position[0]-a[0], position[1]-a[1]) <= \
                   math.hypot(position[0]-b[0], position[1]-b[1]):
                    entry, exit_point = a, b
                else:
                    entry, exit_point = b, a
                planned.append((position, entry, exit_point, volume, is_new))
                position = exit_point

            turnlengths = [
                math.hypot(entry[0]-start_point[0],
                           entry[1]-start_point[1])
                for start_point, entry, _, _, _ in planned
            ]
            total_turn_length = sum(turnlengths)
            remaining_turn_e = original_turn_e

            for j, (start_point, entry, exit_point, volume, is_new) in \
                    enumerate(planned):
                bridge_f = max(60, int((current['f'] or 1800)*slowdown))
                turnlength = turnlengths[j]
                if turnlength > 0.0005:
                    turn_e = (round(original_turn_e * turnlength /
                                    total_turn_length, 5)
                              if j < len(planned)-1 and total_turn_length
                              else remaining_turn_e)
                    remaining_turn_e -= turn_e
                    if turn_e > 0:
                        result.append(
                            f'G1 X{entry[0]:.3f} Y{entry[1]:.3f} '
                            f'E{turn_e:.5f} F{bridge_f} '
                            '; Bridge serpentine U-turn\n')
                        if mode:
                            actual_e += turn_e
                    else:
                        result.append(
                            f'G0 X{entry[0]:.3f} Y{entry[1]:.3f} F8400 '
                            '; Bridge serpentine short transition\n')

                print_f = bridge_f if is_new else (current['f'] or 1800)
                label = 'intermediate' if is_new else 'original strand'
                result.append(
                    f'G1 X{exit_point[0]:.3f} Y{exit_point[1]:.3f} '
                    f'E{volume:.5f} F{print_f:g} '
                    f'; Bridge serpentine {label}\n')
                if mode:
                    actual_e += volume
                if is_new:
                    count += 1

        # Never allow a planned raster to leave the printer at an
        # unintended source endpoint. Stop instead of generating a long
        # return travel or silently corrupting the next move.
        if math.hypot(position[0]-strokes[-1]['b'][0],
                      position[1]-strokes[-1]['b'][1]) > 0.002:
            raise ValueError('Bridge serpentine endpoint parity mismatch')

        if not mode:
            result.append('M82 ; Bridge serpentine restore M82\n')
            actual_e = strokes[-1]['after_e']
            result.append(
                f'G92 E{actual_e:.5f} ; Bridge serpentine source E sync\n')
        last_f = strokes[-1]['f']
        if last_f is not None:
            result.append(f'G1 F{last_f:g} ; Bridge serpentine restore feedrate\n')
        emit_unmodified(suffix)
        run = []

    for node in events:
        if node['kind'] in ('strand', 'connector', 'metadata'):
            if not run and node['kind'] != 'strand':
                emit_unmodified([node])
                continue
            run.append(node)
        else:
            flush_run()
            emit_unmodified([node])
    flush_run()
    return result, actual_e, (x, y), count
