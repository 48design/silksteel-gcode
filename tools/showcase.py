"""Showcase toolpath pruning after complete SilkSteel feature processing.

Keep non-planar infill with a solid base; discard wall/top deposition *and*
its redundant travel/Z motions, before collision-aware final Z-hop generation.
"""
import re

_ROLE = re.compile(r'^;TYPE:\s*(.+?)\s*$', re.I)
_LAYER = re.compile(r'^;LAYER:(\d+)', re.I)
_BOTTOM = re.compile(r'^;\s*bottom_solid_layers\s*=\s*(\d+)\b', re.I | re.M)
_MOTION = re.compile(r'^G0?([0123])(?:\s|$)', re.I)
_AXIS = re.compile(r'(?<![A-Za-z])([XYZE])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
_G92 = re.compile(r'^G92(?:\s|$)', re.I)
_M82 = re.compile(r'^M82(?:\s|$)', re.I)
_M83 = re.compile(r'^M83(?:\s|$)', re.I)
_G90 = re.compile(r'^G90(?:\s|$)', re.I)
_G91 = re.compile(r'^G91(?:\s|$)', re.I)
_END = re.compile(r'^(?:;\s*(?:filament.*end\s*gcode|end\s*gcode|end_print|print_end)|END_PRINT(?:\s|$)|M(?:104|140)\s+S0\b|M(?:84|18)(?:\s|$))', re.I)
_WALLS = {'external perimeter', 'internal perimeter', 'perimeter',
          'overhang perimeter', 'outer wall', 'inner wall', 'thin wall',
          'gap fill', 'skirt', 'brim'}
_BOTTOM_TYPES = {'solid infill', 'bottom solid infill',
                 'bridge infill', 'internal bridge infill'}


def filter_showcase_gcode(gcode, bottom_layers=None):
    """Prune wall and roof toolpaths, preserving slicer base and all infill.

    Track both original and emitted XYZ/E states, so a retained extrusion
    always begins at its true source starting point even if skipped toolpaths
    once led there. M82 and M83 are both supported. No phantom travel through
    the removed walls and no empty, motion-only top layers are emitted.
    """
    if bottom_layers is None:
        header = _BOTTOM.search(gcode)
        bottom_layers = int(header.group(1)) if header else 3
    if bottom_layers < 1:
        raise ValueError('Showcase requires at least one bottom solid layer')

    def records():
        layer, role, relative_e, relative_xyz, source_e = -1, '', False, False, 0.0
        pos = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        for index, line in enumerate(gcode.splitlines(keepends=True)):
            stripped = line.strip()
            marker = _LAYER.match(stripped)
            if marker:
                layer = int(marker.group(1))
            elif stripped.startswith(';LAYER_CHANGE'):
                layer += 1
            match = _ROLE.match(stripped)
            if match:
                role = match.group(1).strip().lower()
            code = line.split(';', 1)[0].strip()
            if _M82.match(code):
                relative_e = False
            elif _M83.match(code):
                relative_e = True
            if _G90.match(code):
                relative_xyz = False
            elif _G91.match(code):
                relative_xyz = True
            axes = {key.upper(): float(value) for key, value in _AXIS.findall(code)}
            start = pos.copy()
            if _G92.match(code):
                if 'E' in axes:
                    source_e = axes['E']
                for axis in 'XYZ':
                    if axis in axes:
                        pos[axis] = axes[axis]
            motion = bool(_MOTION.match(code))
            delta = 0.0
            if motion:
                if 'E' in axes:
                    delta = axes['E'] if relative_e else axes['E'] - source_e
                    source_e += delta
                for axis in 'XYZ':
                    if axis in axes:
                        pos[axis] = pos[axis] + axes[axis] if relative_xyz else axes[axis]
            depositing = motion and delta > 1e-8 and any(axis in axes for axis in 'XY')
            keep_type = role == 'internal infill' or (layer < bottom_layers and role in _BOTTOM_TYPES)
            yield dict(line=line, stripped=stripped, code=code,
                       index=index, layer=layer, role=role,
                       axes=axes, start=start, end=pos.copy(),
                       motion=motion, e_before=source_e - delta,
                       e_delta=delta, rel_e=relative_e,
                       rel_xyz=relative_xyz, keep_type=keep_type,
                       depositing=depositing)

    print_layers = set()
    first_feature = None
    last_deposition = -1
    bottom_count = infill_count = 0
    for item in records():
        if _ROLE.match(item['stripped']) and first_feature is None:
            first_feature = item['index']
        if item['depositing']:
            last_deposition = item['index']
            if item['keep_type']:
                print_layers.add(item['layer'])
                if item['role'] == 'internal infill':
                    infill_count += 1
                else:
                    bottom_count += 1

    if not bottom_count:
        raise ValueError('Showcase found no bottom solid extrusion. Slice with bottom solid layers enabled.')

    out, emitted_pos = [], {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
    emitted_e, e_mode_rel = 0.0, False
    removed, removed_motion, empty_layers, reconnections = 0, 0, 0, 0
    end_code = False

    def emit_move(target, relative, speed=8400):
        """Reposition with one direct movement instead of replaying wall path."""
        nonlocal reconnections
        dz = target['Z'] - emitted_pos['Z']
        if dz > 0.00001:
            z = dz if relative else target['Z']
            out.append(f'G0 Z{z:.5f} F{speed} ; SilkSteel: showcase raise\n')
            emitted_pos['Z'] = target['Z']
        xy = []
        for axis in 'XY':
            delta = target[axis] - emitted_pos[axis]
            if abs(delta) > 0.00001:
                value = delta if relative else target[axis]
                xy.append(f'{axis}{value:.5f}')
        if xy:
            out.append('G0 ' + ' '.join(xy) + f' F{speed} ; SilkSteel: showcase reconnect\n')
            emitted_pos['X'], emitted_pos['Y'] = target['X'], target['Y']
        dz = target['Z'] - emitted_pos['Z']
        if dz < -0.00001:
            z = dz if relative else target['Z']
            out.append(f'G0 Z{z:.5f} F{speed} ; SilkSteel: showcase lower\n')
            emitted_pos['Z'] = target['Z']
        reconnections += 1

    for item in records():
        line, code, axes = item['line'], item['code'], item['axes']
        index, nlayer = item['index'], item['layer']
        # Retain the actual shutdown and parking script (not the preceding
        # empty slicer top layers). Recognize both markers and heater shutdown.
        if index > last_deposition and _END.match(item['stripped']):
            end_code = True
        in_start = first_feature is None or index < first_feature
        printed_layer = nlayer in print_layers
        retained = item['keep_type'] and printed_layer

        if item['motion']:
            if in_start or end_code:
                # Real startup and shutdown scripts are intentionally untouched.
                if 'E' in axes:
                    if not e_mode_rel and abs(emitted_e - item['e_before']) > 0.00001:
                        out.append(f'G92 E{item["e_before"]:.5f} ; SilkSteel: showcase E sync\n')
                        emitted_e = item['e_before']
                    emitted_e = emitted_e + axes['E'] if e_mode_rel else axes['E']
                out.append(line)
                for axis in 'XYZ':
                    if axis in axes:
                        emitted_pos[axis] = (emitted_pos[axis] + axes[axis]
                                             if item['rel_xyz'] else axes[axis])
                continue
            if not retained:
                # Never forward retracts/primes or extrusion-only commands
                # belonging to a deleted feature. In absolute E mode even
                # G1 E+2 without XY is a potentially enormous filament dump.
                removed_motion += 1
                if item['depositing']:
                    removed += 1
                continue
            if not item['depositing']:
                # Reconstruct only the actual printable segments. Discard
                # slicer retract/prime, wipe and Z-hop sequences (including
                # those that straddle TYPE boundaries), rather than leaving
                # unpaired pressure changes when neighboring walls are gone.
                # G92 is restored just before the next kept extrusion.
                if 'E' not in axes and not any(axis in axes for axis in 'XYZ') and re.search(r'\bF[-+]?\d', code):
                    out.append(line)
                else:
                    removed_motion += 1
                continue
            # The only remaining XY paths are genuine filament deposition.
            # Return directly to the original start, not deleted wall paths.
            if any(abs(item['start'][a] - emitted_pos[a]) > 0.00001 for a in 'XYZ'):
                emit_move(item['start'], item['rel_xyz'])
            if 'E' in axes:
                if not e_mode_rel and abs(emitted_e - item['e_before']) > 0.00001:
                    out.append(f'G92 E{item["e_before"]:.5f} ; SilkSteel: showcase E sync\n')
                    emitted_e = item['e_before']
                emitted_e = emitted_e + axes['E'] if e_mode_rel else axes['E']
            out.append(line)
            emitted_pos.update(item['end'])
            continue

        # Layer change / height / TYPE comments from empty layers must not
        # create phantom preview layers or advance slicer progress.
        if (item['stripped'].startswith((';LAYER_CHANGE', ';LAYER:', ';Z:', ';HEIGHT:'))
                and nlayer >= 0 and not printed_layer and not end_code):
            if item['stripped'].startswith(';LAYER:'):
                empty_layers += 1
            continue
        if _ROLE.match(item['stripped']) and not retained and not in_start and not end_code:
            continue
        if not code:
            # Smoothificator/Bricklayers comments, wipes and toolpath notes
            # have no meaning without the eliminated geometry.
            if re.search(r'smoothificator|bricklayers|\bWIPE_(?:START|END)|\bCONTINUED across layer', item['stripped'], re.I):
                continue
            if in_start or end_code or retained or (
                    printed_layer and not item['stripped'].startswith(';TYPE:')):
                out.append(line)
            continue
        if _G92.match(code):
            # Full-feature processing adds many wall-only G92 E syncs.
            # Ignore them, including Bricklayers/Non-planar syncs. Source E
            # stays tracked by records(); the next retained extrusion gets
            # one rebase if needed, without any physical filament movement.
            xyz = [a for a in 'XYZ' if a in axes]
            if xyz:
                emitted_pos.update({a: axes[a] for a in xyz})
                out.append('G92 ' + ' '.join(f'{a}{axes[a]:g}' for a in xyz) + '\n')
            continue
        if re.match(r'^M117\s+Time Left\b', code, re.I) and not in_start and not end_code:
            continue
        if _M82.match(code):
            e_mode_rel = False
        elif _M83.match(code):
            e_mode_rel = True
        # Preserve fan/heater/acceleration/tool/mode commands, even if they
        # occur between print segments.
        out.append(line)

    return ''.join(out), dict(bottom_layers=bottom_layers,
                              removed_extrusions=removed,
                              bottom_extrusions=bottom_count,
                              infill_extrusions=infill_count,
                              removed_motions=removed_motion,
                              empty_layers=empty_layers,
                              reconnections=reconnections)
