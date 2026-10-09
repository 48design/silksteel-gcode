
# SilkSteel - Advanced G-code Post-Processor
# "Smooth on the outside, strong on the inside"
# 
# Combines multiple advanced features for superior 3D print quality:
# - Smoothificator: Multi-pass external perimeters for silk-smooth surfaces
# - Bricklayers: Z-shifted internal perimeters for steel-strong layer bonding
# - Non-planar Infill: Z-modulated infill for improved interlayer adhesion
# - Safe Z-hop: Intelligent collision avoidance during travel moves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Original concepts inspired by Roman Tenger's work on smoothificator, 
# bricklayers, and non-planar infill techniques.
# Extensively rewritten, optimized, and extended by 48DESIGN GmbH [Fabian Groß]
# Copyright (c) [2025] [48DESIGN GmbH]
#
import re
import sys
import logging
import os
import argparse
import math
import numpy as np  # For 3D noise lookup table
from io import StringIO

# Check PIL/Pillow availability once at module level (for debug visualization)
HAS_PIL = False
try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    pass  # Optional dependency: never install packages when processing print files.

reclassified_bridge_count = 0

# =============================================================================
# PRE-COMPILED REGEX PATTERNS (for performance)
# =============================================================================
# These patterns are compiled ONCE at startup and reused thousands of times.
# Using pre-compiled patterns is ~2-3x faster than re.search() with inline patterns.
# Always use these via the extract_x/y/z/e/f() functions or parse_gcode_line().
REGEX_X = re.compile(r'X([-+]?\d*\.?\d+)')
REGEX_Y = re.compile(r'Y([-+]?\d*\.?\d+)')
REGEX_Z = re.compile(r'Z([-+]?\d*\.?\d+)')
REGEX_E = re.compile(r'E([-+]?\d*\.?\d+)')
REGEX_F = re.compile(r'F([-+]?\d*\.?\d+)')
REGEX_E_SUB = re.compile(r'E[-\d.]+')
REGEX_Z_SUB = re.compile(r'Z[-\d.]+\s*')

# =============================================================================
# GCODE PARSING HELPER FUNCTIONS
# =============================================================================
# Use these functions instead of manual re.search() calls for consistency and performance.

def extract_x(line):
    """Extract X coordinate from G-code line using pre-compiled regex"""
    match = REGEX_X.search(line)
    return float(match.group(1)) if match else None

def extract_y(line):
    """Extract Y coordinate from G-code line"""
    match = REGEX_Y.search(line)
    return float(match.group(1)) if match else None

def extract_z(line):
    """Extract Z coordinate from G-code line"""
    match = REGEX_Z.search(line)
    return float(match.group(1)) if match else None

def extract_e(line):
    """Extract E (extrusion) value from G-code line"""
    match = REGEX_E.search(line)
    return float(match.group(1)) if match else None

def extract_f(line):
    """Extract F (feedrate) value from G-code line"""
    match = REGEX_F.search(line)
    return float(match.group(1)) if match else None

def replace_e(line, new_e):
    """Replace E value in G-code line"""
    return REGEX_E_SUB.sub(f'E{new_e:.5f}', line)

def replace_f(line, new_f):
    """Replace F (feedrate) value in G-code line"""
    return REGEX_F.sub(f'F{new_f}', line, count=1)

def remove_z(line):
    """Remove Z parameter from G-code line"""
    return REGEX_Z_SUB.sub('', line)

def parse_gcode_line(line):
    """
    Parse a G-code line and extract all parameters in one pass.
    Returns a dict with keys: x, y, z, e, f (values are None if not present in line).
    This is more efficient than calling extract_x/y/z/e/f separately.
    
    Example:
        params = parse_gcode_line("G1 X10.5 Y20.3 E0.5 F3600")
        # Returns: {'x': 10.5, 'y': 20.3, 'z': None, 'e': 0.5, 'f': 3600}
        
        # Use to update position only if parameter exists:
        if params['x'] is not None:
            current_x = params['x']
    """
    # Only parse the portion before any comment to avoid capturing things like "Z-hop" or "E-layers".
    code_part = line.split(';', 1)[0]
    result = {'x': None, 'y': None, 'z': None, 'e': None, 'f': None}
    
    x_match = REGEX_X.search(code_part)
    if x_match:
        try:
            result['x'] = float(x_match.group(1))
        except ValueError:
            pass
    
    y_match = REGEX_Y.search(code_part)
    if y_match:
        try:
            result['y'] = float(y_match.group(1))
        except ValueError:
            pass
    
    z_match = REGEX_Z.search(code_part)
    if z_match:
        try:
            result['z'] = float(z_match.group(1))
        except ValueError:
            pass
    
    e_match = REGEX_E.search(code_part)
    if e_match:
        try:
            result['e'] = float(e_match.group(1))
        except ValueError:
            pass
    
    f_match = REGEX_F.search(code_part)
    if f_match:
        result['f'] = float(f_match.group(1))
    
    return result

def scan_source_extrusion(lines):
    """Original E deltas, target coordinates and M82/M83 mode per input line."""
    value = 0.0
    relative = False
    deltas, targets, modes = [], [], []
    for line in lines:
        code = line.split(';', 1)[0].strip()
        if re.match(r'^M83(?:\s|$)', code):
            relative = True
        elif re.match(r'^M82(?:\s|$)', code):
            relative = False
        if re.match(r'^G92(?:\s|$)', code):
            reset = parse_gcode_line(code)['e']
            if reset is not None:
                value = reset
        delta = 0.0
        if re.match(r'^G0?[01](?:\s|$)', code):
            e = parse_gcode_line(code)['e']
            if e is not None:
                delta = e if relative else e - value
                value += delta
        deltas.append(delta)
        targets.append(value)
        modes.append(relative)
    return deltas, targets, modes


def restore_layer_continued_wall_types(lines):
    """Reintroduce omitted TYPE markers at layer starts when the last type
    explicitly set by the slicer was an exterior wall.

    Prusa/Orca may carry ;TYPE:External perimeter across ;LAYER_CHANGE
    without restating it. Smoothificator is TYPE-triggered and would
    otherwise leave the first exterior contour of the next layer untouched.
    Only infer from an *explicit prior exterior TYPE*, never from shape.
    Wait until positive XY extrusion, so G92, prime/retract, travel and
    feedrate lines stay in their original positions before processing.
    """
    original_deltas, _, _ = scan_source_extrusion(lines)
    updated = []
    active_type = None
    pending_outer = None
    inherited = 0
    insertion_index = None
    outer_types = (";TYPE:External perimeter", ";TYPE:Outer wall",
                   ";TYPE:Overhang perimeter")

    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(";TYPE:"):
            active_type = next((kind for kind in outer_types
                                if stripped.startswith(kind)), None)
            pending_outer = None
            insertion_index = None
        elif stripped.startswith(";LAYER_CHANGE") or stripped.startswith(";LAYER:"):
            pending_outer = active_type
            # Place the recovered role right after ;LAYER, BEFORE any
            # G92 / retract / travel commands. Viewer layer attribution
            # can happen before the first XY extrusion is encountered.
            insertion_index = len(updated) + 1
        elif pending_outer and original_deltas[index] > 0:
            code = line.split(";", 1)[0].strip()
            if re.match(r'^G0?[01](?:\s|$)', code):
                p = parse_gcode_line(code)
                if p["x"] is not None or p["y"] is not None:
                    # Keep TYPE comments exact: G-code viewers often
                    # recognize only the canonical slicer feature names.
                    # Provenance belongs on its own comment line.
                    updated[insertion_index:insertion_index] = [
                        "; SilkSteel: CONTINUED across layer boundary\n",
                        pending_outer + "\n",
                    ]
                    inherited += 1
                    pending_outer = None
                    insertion_index = None

        updated.append(line)

    return updated, inherited


def restore_extrusion_feedrates(gcode, safe_fallback=1800):
    """Reassert print speed after a travel command changes modal F."""
    result, modal_f, print_f, changed, restorations = [], None, None, False, 0
    for line in gcode.splitlines(keepends=True):
        code = line.split(';', 1)[0].strip()
        if re.match(r'^G0?[01](?:\s|$)', code):
            params = parse_gcode_line(code)
            xy = params['x'] is not None or params['y'] is not None
            e = params['e'] is not None
            f = params['f']
            if f is not None:
                modal_f = f
                if xy and e:
                    print_f, changed = f, False
                elif not xy and not e and code.startswith('G1'):
                    print_f, changed = f, False
                elif not e:
                    changed = True
            if xy and e and f is None:
                if changed:
                    target = print_f if print_f is not None else safe_fallback
                    if modal_f is None or abs(modal_f - target) > 0.01:
                        head, sep, comment = line.partition(';')
                        line = head.rstrip() + f' F{target:g}' + ((' ;' + comment) if sep else '\n')
                        modal_f = target
                        restorations += 1
                    changed = False
                if print_f is None and modal_f is not None:
                    print_f = modal_f
        result.append(line)
    logging.info("Restored print feedrate after %d travel moves", restorations)
    return ''.join(result)


def write_line(buffer, line):
    """Write a line to output buffer, ensuring it has a newline"""
    if line and not line.endswith('\n'):
        buffer.write(line + '\n')
    else:
        buffer.write(line)

_update_position_for_output = None  # Will be set inside process_gcode once update_position is defined

def write_and_track(buffer, line, recent_buffer, max_size=20):
    """Write line to buffer, update global position if callback set, and add to rolling buffer.
    IMPORTANT: Position tracking is now OUTPUT-DRIVEN. We only update the nozzle position
    based on lines that are ACTUALLY written to the output. This prevents mismatches where
    skipped/removed input lines (e.g., gap fill removal) would previously advance the
    position tracker incorrectly.
    """
    write_line(buffer, line)
    # Update position ONLY from written lines
    if _update_position_for_output:
        _update_position_for_output(line)
    recent_buffer.append(line)
    if len(recent_buffer) > max_size:
        recent_buffer.pop(0)  # Remove oldest line

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Global counters for warnings/errors
_warning_count = 0
_error_count = 0

class CountingHandler(logging.Handler):
    """Custom logging handler that counts warnings and errors"""
    def emit(self, record):
        global _warning_count, _error_count
        if record.levelno >= logging.ERROR:
            _error_count += 1
        elif record.levelno >= logging.WARNING:
            _warning_count += 1

# Configure logging to both console and file
log_file = os.path.join(script_dir, "SilkSteel_log.txt")
counting_handler = CountingHandler()
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),  # UTF-8 for Unicode emojis
        logging.StreamHandler(sys.stdout),          # Also print to console
        counting_handler  # Count warnings/errors
    ]
)

logging.info("=" * 85)
logging.info("SilkSteel started")
logging.info(f"Script directory: {script_dir}")
logging.info(f"Log file: {log_file}")
logging.info(f"Command line args: {sys.argv}")
logging.info("=" * 85)

# Type enumeration for grid cell classification
# Used to track what type of material occupies each grid cell
TYPE_NONE = 0
TYPE_INTERNAL_INFILL = 1
TYPE_SOLID_INFILL = 2
TYPE_TOP_SOLID_INFILL = 3
TYPE_BRIDGE_INFILL = 4
TYPE_INTERNAL_BRIDGE_INFILL = 5
TYPE_INTERNAL_PERIMETER = 6
TYPE_EXTERNAL_PERIMETER = 7
TYPE_OVERHANG_PERIMETER = 8
TYPE_GAP_FILL = 9

# Helper function to get type from TYPE marker string
def get_type_from_marker(type_marker):
    """Convert TYPE marker string to type enum"""
    if 'Internal infill' in type_marker:
        return TYPE_INTERNAL_INFILL
    elif 'Top solid infill' in type_marker:
        return TYPE_TOP_SOLID_INFILL
    elif 'Solid infill' in type_marker:
        return TYPE_SOLID_INFILL
    elif 'Internal bridge infill' in type_marker:
        return TYPE_INTERNAL_BRIDGE_INFILL
    elif 'Bridge infill' in type_marker:
        return TYPE_BRIDGE_INFILL
    elif 'Overhang perimeter' in type_marker:
        return TYPE_OVERHANG_PERIMETER
    elif 'External perimeter' in type_marker or 'Outer wall' in type_marker:
        return TYPE_EXTERNAL_PERIMETER
    elif 'Internal perimeter' in type_marker or 'Inner wall' in type_marker or type_marker == ';TYPE:Perimeter':
        return TYPE_INTERNAL_PERIMETER
    elif 'Gap fill' in type_marker:
        return TYPE_GAP_FILL
    else:
        return TYPE_NONE

# Type colors for visualization (RGB tuples) - matches PrusaSlicer/OrcaSlicer colors
TYPE_COLORS = {
    TYPE_NONE: (0, 0, 0),
    TYPE_INTERNAL_INFILL: (176, 48, 42),
    TYPE_SOLID_INFILL: (214, 50, 214),
    TYPE_TOP_SOLID_INFILL: (254, 26, 26),
    TYPE_BRIDGE_INFILL: (152, 152, 254),
    TYPE_INTERNAL_BRIDGE_INFILL: (169, 169, 220),
    TYPE_INTERNAL_PERIMETER: (254, 254, 102),
    TYPE_EXTERNAL_PERIMETER: (254, 164, 0),
    TYPE_OVERHANG_PERIMETER: (0, 0, 254),
    TYPE_GAP_FILL: (254, 254, 254),
}

# Smoothificator constants
DEFAULT_OUTER_LAYER_HEIGHT = "Auto"  # "Auto" = min(first_layer, base_layer) * 0.5, "Min" = min_layer_height from G-code, or float value (mm)

# Non-planar infill constants
DEFAULT_AMPLITUDE = 4  # Default Z variation in mm [float] or layerheight [int] (reduced for smoother look)
DEFAULT_FREQUENCY = 8  # Default frequency of the sine wave (reduced for longer waves)
DEFAULT_SEGMENT_LENGTH = 0.64  # Split infill lines into segments of this length (mm) - LARGER = fewer segments, smoother motion
DEFAULT_NONPLANAR_FEEDRATE_MULTIPLIER = 1.1  # Boost feedrate by this factor for non-planar 3D moves (2.0 = double speed)
DEFAULT_ENABLE_ADAPTIVE_EXTRUSION = True  # Enable adaptive extrusion multiplier for Z-lift (adds material to droop down and bond)
DEFAULT_ADAPTIVE_EXTRUSION_MULTIPLIER = 1.75  # Base multiplier for adaptive extrusion (e.g., 1.33 = 33% extra material per layer height of lift)

# Grid resolution for solid occupancy detection
# Will be read from G-code if available, otherwise use default
DEFAULT_EXTRUSION_WIDTH = 0.45  # Default extrusion width in mm (typical value)

# Safe Z-hop constants
DEFAULT_ENABLE_SAFE_Z_HOP = True  # Enabled by default
DEFAULT_SAFE_Z_HOP_MARGIN = 0.5  # mm - safety margin above max Z in layer
DEFAULT_Z_HOP_RETRACTION = 1.5  # mm - retraction distance during Z-hop to prevent stringing

# Bridge densifier constants
DEFAULT_ENABLE_BRIDGE_DENSIFIER = False  # Disabled by default (experimental feature)
DEFAULT_BRIDGE_MIN_LENGTH = 2.0  # mm - Only densify lines longer than this (filters out short bridges)
DEFAULT_BRIDGE_MAX_SPACING = 0.6  # mm - Maximum spacing between parallel lines to add intermediate (typical bridge line width is 0.4-0.45mm)
DEFAULT_BRIDGE_EXTRUSION_COMPENSATION = 0.25  # Factor for intermediate extrusion (0.5 = 50%, fills gap not full width - round vs squished)
DEFAULT_BRIDGE_MAX_GAP = 3  # Maximum number of connector lines allowed between parallel long lines (for curves)
DEFAULT_BRIDGE_CONNECTOR_MAX_LENGTH = 0.9  # mm - Fallback value (will be set to 2× actual extrusion width from G-code)

# Gap fill removal constants
DEFAULT_REMOVE_GAP_FILL = False  # Disabled by default - removes all gap fill sections, useful for getting less expansion/contraction of outer walls

def get_layer_height(gcode_lines):
    """Extract layer height from G-code header comments"""
    for line in gcode_lines:
        if "layer_height =" in line.lower():
            match = re.search(r'; layer_height = (\d*\.?\d+)', line, re.IGNORECASE)
            if match:
                return float(match.group(1))
    return None

def get_first_layer_height(gcode_lines):
    """Extract first layer height from G-code header comments"""
    for line in gcode_lines:
        if "first_layer_height =" in line.lower():
            match = re.search(r'; first_layer_height = (\d*\.?\d+)', line, re.IGNORECASE)
            if match:
                return float(match.group(1))
    return None

def get_min_layer_height(gcode_lines):
    """Extract minimum layer height from G-code header comments"""
    for line in gcode_lines:
        if "min_layer_height =" in line.lower():
            match = re.search(r'; min_layer_height = (\d*\.?\d+)', line, re.IGNORECASE)
            if match:
                return float(match.group(1))
    return None

def get_extrusion_width(gcode_lines):
    """Extract extrusion width from G-code header comments"""
    for line in gcode_lines:
        if "extrusion_width =" in line.lower():
            match = re.search(r'; extrusion_width = (\d*\.?\d+)', line, re.IGNORECASE)
            if match:
                return float(match.group(1))
    return None

def parse_outer_layer_height(value):
    """Parse outer layer height argument - can be 'Auto', 'Min', or a float"""
    if isinstance(value, str):
        if value.lower() == 'auto':
            return 'Auto'
        elif value.lower() == 'min':
            return 'Min'
        else:
            try:
                return float(value)
            except ValueError:
                raise argparse.ArgumentTypeError(f"outer-layer-height must be 'Auto', 'Min', or a number, got: {value}")
    return value  # Already parsed as default

def segment_line(x1, y1, x2, y2, segment_length):
    """Divide a line into smaller segments for non-planar infill."""
    segments = []
    total_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if total_length <= 1e-9:
        return []
    num_segments = max(1, math.ceil(total_length / segment_length))

    for i in range(1, num_segments + 1):
        t = i / num_segments
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        segments.append((x, y))
    
    return segments

def generate_perlin_noise_3d(shape, res, seed=None):
    """
    Generate 3D Perlin noise using numpy - improved smooth version.
    
    Args:
        shape: Tuple of (width, height, depth) for output array
        res: Tuple of (res_x, res_y, res_z) - resolution of grid
        seed: Random seed for reproducibility
    
    Returns:
        3D numpy array with Perlin noise values in range [-1, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    def fade(t):
        """Improved fade function (smoothstep)"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def lerp(a, b, t):
        """Linear interpolation"""
        return a + t * (b - a)
    
    # Generate random gradients at grid points
    gradients = np.random.randn(res[0] + 1, res[1] + 1, res[2] + 1, 3)
    # Normalize gradients
    gradients = gradients / (np.linalg.norm(gradients, axis=3, keepdims=True) + 1e-10)
    
    # Create output array
    noise = np.zeros(shape)
    
    # For each point in the output shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                # Map output coordinates to gradient grid coordinates
                x = i * res[0] / shape[0]
                y = j * res[1] / shape[1]
                z = k * res[2] / shape[2]
                
                # Get integer parts (grid cell)
                xi = int(np.floor(x))
                yi = int(np.floor(y))
                zi = int(np.floor(z))
                
                # Get fractional parts (position within cell)
                xf = x - xi
                yf = y - yi
                zf = z - zi
                
                # Clamp to valid gradient indices
                xi = min(xi, res[0] - 1)
                yi = min(yi, res[1] - 1)
                zi = min(zi, res[2] - 1)
                
                # Get the 8 corner gradients
                g000 = gradients[xi,   yi,   zi]
                g100 = gradients[xi+1, yi,   zi]
                g010 = gradients[xi,   yi+1, zi]
                g110 = gradients[xi+1, yi+1, zi]
                g001 = gradients[xi,   yi,   zi+1]
                g101 = gradients[xi+1, yi,   zi+1]
                g011 = gradients[xi,   yi+1, zi+1]
                g111 = gradients[xi+1, yi+1, zi+1]
                
                # Calculate dot products with distance vectors
                n000 = np.dot(g000, [xf,   yf,   zf])
                n100 = np.dot(g100, [xf-1, yf,   zf])
                n010 = np.dot(g010, [xf,   yf-1, zf])
                n110 = np.dot(g110, [xf-1, yf-1, zf])
                n001 = np.dot(g001, [xf,   yf,   zf-1])
                n101 = np.dot(g101, [xf-1, yf,   zf-1])
                n011 = np.dot(g011, [xf,   yf-1, zf-1])
                n111 = np.dot(g111, [xf-1, yf-1, zf-1])
                
                # Apply fade curves
                u = fade(xf)
                v = fade(yf)
                w = fade(zf)
                
                # Trilinear interpolation
                x00 = lerp(n000, n100, u)
                x10 = lerp(n010, n110, u)
                x01 = lerp(n001, n101, u)
                x11 = lerp(n011, n111, u)
                
                y0 = lerp(x00, x10, v)
                y1 = lerp(x01, x11, v)
                
                noise[i, j, k] = lerp(y0, y1, w)
    
    # Normalize to [-1, 1] range to match sine wave behavior
    # Perlin noise typically has range around [-0.7, 0.7], so we normalize it
    noise_min = np.min(noise)
    noise_max = np.max(noise)
    if noise_max > noise_min:
        # Scale to [-1, 1]
        noise = 2 * (noise - noise_min) / (noise_max - noise_min) - 1
    
    return noise

def voxel_traversal(x0, y0, x1, y1, grid_resolution):
    """
    Fast voxel traversal algorithm to find all grid cells crossed by a line segment.
    Uses a DDA-like approach to traverse the grid efficiently.
    
    Args:
        x0, y0: Start point coordinates
        x1, y1: End point coordinates
        grid_resolution: Size of each grid cell
    
    Returns:
        List of (gx, gy) tuples representing grid cells crossed by the line
    """
    # Convert to grid coordinates
    gx0 = int(x0 / grid_resolution)
    gy0 = int(y0 / grid_resolution)
    gx1 = int(x1 / grid_resolution)
    gy1 = int(y1 / grid_resolution)
    
    dx = abs(gx1 - gx0)
    dy = abs(gy1 - gy0)
    
    x = gx0
    y = gy0
    
    n = 1 + dx + dy
    x_inc = 1 if gx1 > gx0 else -1
    y_inc = 1 if gy1 > gy0 else -1
    error = dx - dy
    dx *= 2
    dy *= 2
    
    cells = []
    for _ in range(n):
        cells.append((x, y))
        
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    
    return cells


def detect_bridge_over_air(lines, start_idx, current_layer_num, solid_at_grid, grid_resolution, parse_gcode_line, voxel_traversal, max_lookahead=60, min_points=2, first_n_segments=10):
    """
    Heuristic to decide whether a forthcoming ';TYPE:Bridge infill' block
    is actually spanning air (a real bridge) by sampling the FIRST few
    extrusion segments and checking the layer below at the CENTER cell of
    each segment. Many slicers (e.g., PrusaSlicer) draw the initial bridge
    extrusions anchored over nearby solid walls — checking the first
    segments detects this cheaply.

    Returns True if the bridge appears to be over air (no solid directly below
    in the sampled first segments), False if any sampled center cell has
    supporting solid beneath (treat as internal bridge).

    Notes:
    - We only check the center CELL of each segment (cheap) instead of all
      traversed cells along the segment.
    - If we cannot collect at least `min_points` extrusion coordinates,
      return False (conservative: treat as internal).
    """
    prev_layer = current_layer_num - 1
    if prev_layer < 0:
        # No layer below -> treat as a real bridge (over air)
        return True

    # If the occupancy grid is empty at this point, be conservative and treat
    # as internal (do not densify). This avoids accidentally triggering the
    # densifier when grid-building hasn't populated previous layers yet.
    if not solid_at_grid:
        if globals().get('debug', 0) >= 2:
            logging.info(f"[BRIDGE-DETECT] solid_at_grid empty at layer {current_layer_num}, treating as internal")
        return False

    pts = []
    end_idx = min(len(lines), start_idx + max_lookahead)
    for j in range(start_idx, end_idx):
        l = lines[j]
        # stop at next TYPE/LAYER marker
        if ';TYPE:' in l or ';LAYER:' in l or ';LAYER_CHANGE' in l:
            break
        if l.strip().startswith('G1') and 'X' in l and 'Y' in l and 'E' in l:
            params = parse_gcode_line(l)
            if params['x'] is not None and params['y'] is not None and params['e'] is not None:
                # Only consider positive extrusion (skip retractions)
                if params['e'] >= 0:
                    pts.append((params['x'], params['y']))

    # Not enough sample points -> conservative: treat as internal (do not densify)
    if len(pts) < min_points:
        if globals().get('debug', 0) >= 2:
            logging.info(f"[BRIDGE-DETECT] insufficient extrusion points ({len(pts)}) for bridge detection at layer {current_layer_num}")
        return False

    # We'll inspect up to `first_n_segments` initial segment-center cells,
    # but only counting segments whose euclidean length >= min_segment_length.
    # This ignores tiny bridging steps that are just curve-connectors and
    # focuses on meaningful extrusion segments.
    # Use configured bridge min length to ignore tiny connector segments
    min_segment_length = globals().get('DEFAULT_BRIDGE_MIN_LENGTH', DEFAULT_BRIDGE_MIN_LENGTH)
    if globals().get('debug', 0) >= 2:
        logging.info(f"[BRIDGE-DETECT] evaluating up to {first_n_segments} segments (min_segment_length={min_segment_length}mm) for bridge at layer {current_layer_num}")

    valid_seen = 0
    scanned = 0
    # iterate through consecutive segments until we have evaluated enough valid ones
    for k in range(len(pts) - 1):
        x0, y0 = pts[k]
        x1, y1 = pts[k + 1]
        # distance of this segment
        seg_len = math.hypot(x1 - x0, y1 - y0)
        if seg_len < min_segment_length:
            if globals().get('debug', 0) >= 3:
                logging.info(f"[BRIDGE-DETECT] skipping tiny segment {k} length={seg_len:.3f}mm")
            continue

        # This is a valid segment to consider
        valid_seen += 1
        scanned += 1
        # For longer segments, sample multiple points along the segment (25%,50%,75%)
        # to avoid missing support that only touches near the ends or center.
        sample_points = []
        multi_sample_threshold = 3.0 * min_segment_length
        if seg_len >= multi_sample_threshold:
            # sample at 25%, 50%, 75%
            sample_points = [0.25, 0.5, 0.75]
        else:
            # cheap center-only check
            sample_points = [0.5]

        for frac in sample_points:
            sx = x0 + frac * (x1 - x0)
            sy = y0 + frac * (y1 - y0)
            gx = int(sx / grid_resolution)
            gy = int(sy / grid_resolution)
            key = (gx, gy, prev_layer)

            present = key in solid_at_grid
            cell = solid_at_grid.get(key, {})
            infill_crossings = cell.get('infill_crossings', 0)
            cell_type = cell.get('type', TYPE_NONE)

            if globals().get('debug', 0) >= 2:
                logging.info(f"[BRIDGE-DETECT] valid seg {k} (len={seg_len:.3f}mm) sample {int(frac*100)}% -> center ({sx:.3f},{sy:.3f}) -> cell {key}: present={present}, type={cell_type}, infill_crossings={infill_crossings}")

            # If internal infill exists under any sampled point -> internal bridge
            if infill_crossings > 0 or cell_type == TYPE_INTERNAL_INFILL:
                if globals().get('debug', 0) >= 2:
                    logging.info(f"[BRIDGE-DETECT] detected internal infill under segment {k} at {int(frac*100)}%, classifying as internal bridge")
                return False

            # If sampled cell is absent -> air below; since we did not see internal
            # infill in the first valid segments, this indicates an external bridge
            if not present:
                if globals().get('debug', 0) >= 2:
                    logging.info(f"[BRIDGE-DETECT] detected air under segment {k} at {int(frac*100)}%, classifying as external bridge")
                return True

        # Otherwise cell present but not internal infill -> continue scanning
        if valid_seen >= first_n_segments:
            break

    # If we didn't find any supporting internal infill or air in the first
    # evaluated segments, conservatively treat as internal
    if globals().get('debug', 0) >= 2:
        logging.info(f"[BRIDGE-DETECT] evaluated {valid_seen} valid segments (scanned {scanned}), no internal infill or air found; classifying as internal bridge")
    return False

def is_in_safezone(gx, gy, layer, grid_cell_solid_regions):
    """
    Check if a grid cell is in a safezone (gap between solid regions) at a given layer.
    
    Args:
        gx, gy: Grid cell coordinates
        layer: Layer number to check
        grid_cell_solid_regions: Dictionary mapping (gx,gy) to list of solid regions
    
    Returns:
        True if the cell is in a safezone at this layer, False otherwise
    """
    if (gx, gy) not in grid_cell_solid_regions:
        return False
    
    regions = grid_cell_solid_regions[(gx, gy)]
    if len(regions) < 2:
        return False
    
    # Check if layer is between any two solid regions
    for i in range(len(regions) - 1):
        region_end_below = regions[i][1]
        region_start_above = regions[i + 1][0]
        if region_end_below < layer < region_start_above:
            return True
    return False

def has_solid_above_blurry(solid_at_grid, gx, gy, layer, radius=1):
    """
    Blurry lookup: check a (2r+1)x(2r+1) neighborhood around (gx, gy) for any
    solid type above the given layer that should trigger un-shift.

    TRIGGERING types: Solid infill, top solid, external perimeters, bridges, etc.
    NON-triggering types: TYPE_INTERNAL_PERIMETER (will also be bricklayered), TYPE_NONE (empty space)

    Args:
        solid_at_grid: dict with keys (gx,gy,layer)
        gx, gy: grid coordinates of the center cell
        layer: current layer number (we look at layer+1)
        radius: how many cells to expand in each direction (1 = 8 neighbors)

    Returns:
        True if any neighbor cell at layer+1 contains actual solid material that blocks full shift.
    """
    next_layer = layer + 1
    for nx in range(gx - radius, gx + radius + 1):
        for ny in range(gy - radius, gy + radius + 1):
            key = (nx, ny, next_layer)
            # Only trigger if cell EXISTS and has actual solid material
            if key in solid_at_grid:
                ntype = solid_at_grid[key].get('type', TYPE_NONE)
                # Trigger on solid types (NOT internal perimeter, NOT empty)
                if ntype not in [TYPE_NONE, TYPE_INTERNAL_PERIMETER]:
                    return True
    return False

def add_inline_comment(gcode_line, comment):
    """
    Helper function to add an inline comment to a G-code line.
    
    Args:
        gcode_line: The G-code line (should end with \n)
        comment: The comment text to append
    
    Returns:
        G-code line with inline comment appended
    """
    # Remove trailing newline, add comment, add newline back
    line = gcode_line.rstrip('\n')
    return f"{line} ; {comment}\n"

def is_first_of_safezone(gx, gy, layer, infill_at_grid):
    """
    Check if this infill cell is the first layer of a safezone.
    Adaptive extrusion boost is helpful here!
    
    Args:
        gx, gy: Grid cell coordinates
        layer: Layer number to check
        infill_at_grid: The infill grid dictionary with metadata
    
    Returns:
        True if this is the first infill layer of a safezone, False otherwise
    """
    key = (gx, gy, layer)
    if key not in infill_at_grid:
        return False
    
    cell_data = infill_at_grid[key]
    if isinstance(cell_data, dict):
        return cell_data.get('is_first_of_safezone', False)
    return False

def is_last_of_safezone(gx, gy, layer, infill_at_grid):
    """
    Check if this infill cell is the last layer of a safezone.
    Valley filling is needed here!
    
    Args:
        gx, gy: Grid cell coordinates
        layer: Layer number to check
        infill_at_grid: The infill grid dictionary with metadata
    
    Returns:
        True if this is the last infill layer of a safezone, False otherwise
    """
    key = (gx, gy, layer)
    if key not in infill_at_grid:
        return False
    
    cell_data = infill_at_grid[key]
    if isinstance(cell_data, dict):
        return cell_data.get('is_last_of_safezone', False)
    return False

def calculate_grid_bounds(solid_at_grid):
    """
    Calculate grid bounds from solid_at_grid dictionary.
    
    Args:
        solid_at_grid: Dictionary with (gx, gy, layer) keys
    
    Returns:
        Tuple of (x_min, x_max, y_min, y_max, width, height) or None if empty
    """
    if not solid_at_grid:
        return None
    
    all_gx = [gx for gx, gy, lay in solid_at_grid.keys()]
    all_gy = [gy for gx, gy, lay in solid_at_grid.keys()]
    
    grid_x_min, grid_x_max = min(all_gx), max(all_gx)
    grid_y_min, grid_y_max = min(all_gy), max(all_gy)
    
    grid_width = grid_x_max - grid_x_min + 1
    grid_height = grid_y_max - grid_y_min + 1
    
    return (grid_x_min, grid_x_max, grid_y_min, grid_y_max, grid_width, grid_height)

def get_safezone_bounds(gx, gy, current_layer, grid_cell_solid_regions, base_layer_height):
    """
    Determine which safezone (gap between solid regions) the current layer is in,
    and return the floor and ceiling Z values for that safezone.
    
    Args:
        gx, gy: Grid coordinates
        current_layer: Current layer number
        grid_cell_solid_regions: Dictionary mapping (gx, gy) to list of solid regions
        base_layer_height: Layer height in mm
    
    Returns:
        Tuple of (z_min, z_max, layers_until_ceiling, height_until_ceiling):
        - z_min: Floor Z (top of solid region below, or -999 if none)
        - z_max: Ceiling Z (bottom of solid region above minus one layer, or 999 if none)
        - layers_until_ceiling: Number of layers until solid starts above (0 if none)
        - height_until_ceiling: Remaining height in mm from current layer to ceiling (layers_until_ceiling × layer_height)
    """
    local_z_min = -999  # Floor (top of solid below)
    local_z_max = 999   # Ceiling (bottom of solid above)
    layers_until_ceiling = 0
    
    if (gx, gy) in grid_cell_solid_regions:
        for region_start, region_end, z_bottom, z_top in grid_cell_solid_regions[(gx, gy)]:
            # Check if this solid region is BELOW our current layer
            if region_end < current_layer:
                # This solid is below - use its top as our floor
                local_z_min = max(local_z_min, z_top)
            
            # Check if this solid region is ON or ABOVE our current layer
            elif region_start >= current_layer:
                # This solid is on same layer or above - use its bottom minus layer height as ceiling
                # (infill must stay in the layer BELOW the solid)
                candidate_ceiling = z_bottom - base_layer_height
                if local_z_max == 999:  # Take the LOWEST ceiling
                    local_z_max = candidate_ceiling
                    layers_until_ceiling = region_start - current_layer
                elif candidate_ceiling < local_z_max:
                    local_z_max = candidate_ceiling
                    layers_until_ceiling = region_start - current_layer
    
    # Calculate remaining height until ceiling (how much safezone is left above current layer)
    height_until_ceiling = layers_until_ceiling * base_layer_height
    
    return (local_z_min, local_z_max, layers_until_ceiling, height_until_ceiling)

def generate_fractal_noise_3d(shape, res, octaves=1, persistence=0.5, seed=None):
    """
    Generate 3D fractal Perlin noise with multiple octaves.
    
    Args:
        shape: Tuple of (width, height, depth) for output array
        res: Tuple of (res_x, res_y, res_z) - base resolution
        octaves: Number of octaves (layers of detail)
        persistence: How much each octave contributes (0-1)
        seed: Random seed for reproducibility
    
    Returns:
        3D numpy array with fractal noise values
    """
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    max_amplitude = 0
    
    for octave in range(octaves):
        # Generate Perlin noise at this octave's frequency
        octave_noise = generate_perlin_noise_3d(
            shape, 
            (frequency*res[0], frequency*res[1], frequency*res[2]),
            seed=(seed + octave) if seed is not None else None
        )
        
        # Ensure the octave noise matches our target shape (trim if needed)
        if octave_noise.shape != shape:
            # Trim to match target shape
            octave_noise = octave_noise[:shape[0], :shape[1], :shape[2]]
        
        noise += amplitude * octave_noise
        max_amplitude += amplitude
        frequency *= 2
        amplitude *= persistence
    
    # Normalize to [-1, 1]
    return noise / max_amplitude

def generate_3d_noise_lut(x_min, x_max, y_min, y_max, z_min, z_max, 
                          resolution=1.0, frequency_x=0.5, frequency_y=0.5, frequency_z=0.5,
                          octaves=3, persistence=0.5, seed=42):
    """
    Generate a 3D lookup table for noise/modulation values using Perlin noise.
    
    Args:
        x_min, x_max, y_min, y_max, z_min, z_max: Bounds of the print volume
        resolution: Grid spacing in mm (smaller = more detailed but more memory)
        frequency_x, frequency_y, frequency_z: Base frequencies for each axis
        octaves: Number of noise octaves for fractal detail
        persistence: How much each octave contributes (0-1)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with grid parameters and the 3D array of values
    """
    # Create grid
    x_steps = int((x_max - x_min) / resolution) + 1
    y_steps = int((y_max - y_min) / resolution) + 1
    z_steps = int((z_max - z_min) / resolution) + 1
    
    # Shape for the noise array
    shape = (x_steps, y_steps, z_steps)
    
    # Convert frequency to Perlin resolution
    # Higher frequency = more waves = higher resolution grid
    # Frequency of 1.0 should give roughly 10 grid cells (one full wave per 10mm at resolution=1.0)
    res_x = max(2, int((x_max - x_min) / 10.0 * frequency_x))
    res_y = max(2, int((y_max - y_min) / 10.0 * frequency_y))
    res_z = max(2, int((z_max - z_min) / 10.0 * frequency_z))
    
    logging.info(f"  Perlin resolution: {res_x} x {res_y} x {res_z} grid cells")
    
    # Generate fractal Perlin noise
    noise = generate_fractal_noise_3d(shape, (res_x, res_y, res_z), octaves, persistence, seed)
    
    lut = {
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'z_min': z_min, 'z_max': z_max,
        'resolution': resolution,
        'x_steps': x_steps,
        'y_steps': y_steps,
        'z_steps': z_steps,
        'data': noise
    }
    
    logging.info(f"Generated 3D Perlin noise LUT: {x_steps}x{y_steps}x{z_steps} grid, resolution={resolution}mm, octaves={octaves}")
    return lut

def generate_3d_sine_lut(x_min, x_max, y_min, y_max, z_min, z_max, 
                         resolution=1.0, frequency_x=0.5, frequency_y=0.5, frequency_z=0.5):
    """
    Generate a 3D lookup table for smooth sine wave patterns.
    Creates clean, predictable wave patterns for non-planar infill.
    
    Args:
        x_min, x_max, y_min, y_max, z_min, z_max: Bounds of the print volume
        resolution: Grid spacing in mm (smaller = more detailed but more memory)
        frequency_x, frequency_y, frequency_z: Frequencies for each axis
    
    Returns:
        Dictionary with grid parameters and the 3D array of values
    """
    # Create grid
    x_steps = int((x_max - x_min) / resolution) + 1
    y_steps = int((y_max - y_min) / resolution) + 1
    z_steps = int((z_max - z_min) / resolution) + 1
    
    # Generate coordinates
    x_coords = np.linspace(x_min, x_max, x_steps)
    y_coords = np.linspace(y_min, y_max, y_steps)
    z_coords = np.linspace(z_min, z_max, z_steps)
    
    # Create 3D meshgrid
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    # Generate pure 3D sine wave pattern
    # Simple combination for smooth, regular waves
    sine_pattern = (np.sin(frequency_x * X) + 
                   np.sin(frequency_y * Y) + 
                   np.sin(frequency_z * Z)) / 3.0
    
    lut = {
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'z_min': z_min, 'z_max': z_max,
        'resolution': resolution,
        'x_steps': x_steps,
        'y_steps': y_steps,
        'z_steps': z_steps,
        'data': sine_pattern
    }
    
    logging.info(f"Generated 3D sine LUT: {x_steps}x{y_steps}x{z_steps} grid, resolution={resolution}mm")
    return lut

def sample_3d_noise_lut(lut, x, y, z):
    """
    Sample the 3D noise lookup table at given coordinates with trilinear interpolation.
    
    Args:
        lut: The lookup table dictionary from generate_3d_noise_lut
        x, y, z: Coordinates to sample
    
    Returns:
        Interpolated noise value (normalized -1 to 1)
    """
    # Clamp coordinates to bounds
    x = max(lut['x_min'], min(lut['x_max'], x))
    y = max(lut['y_min'], min(lut['y_max'], y))
    z = max(lut['z_min'], min(lut['z_max'], z))
    
    # Convert to grid indices (floating point)
    x_idx = (x - lut['x_min']) / lut['resolution']
    y_idx = (y - lut['y_min']) / lut['resolution']
    z_idx = (z - lut['z_min']) / lut['resolution']
    
    # Get integer indices and fractional parts
    x0 = int(x_idx)
    y0 = int(y_idx)
    z0 = int(z_idx)
    
    x1 = min(x0 + 1, lut['x_steps'] - 1)
    y1 = min(y0 + 1, lut['y_steps'] - 1)
    z1 = min(z0 + 1, lut['z_steps'] - 1)
    
    xf = x_idx - x0
    yf = y_idx - y0
    zf = z_idx - z0
    
    # Trilinear interpolation
    data = lut['data']
    
    c000 = data[x0, y0, z0]
    c001 = data[x0, y0, z1]
    c010 = data[x0, y1, z0]
    c011 = data[x0, y1, z1]
    c100 = data[x1, y0, z0]
    c101 = data[x1, y0, z1]
    c110 = data[x1, y1, z0]
    c111 = data[x1, y1, z1]
    
    c00 = c000 * (1 - xf) + c100 * xf
    c01 = c001 * (1 - xf) + c101 * xf
    c10 = c010 * (1 - xf) + c110 * xf
    c11 = c011 * (1 - xf) + c111 * xf
    
    c0 = c00 * (1 - yf) + c10 * yf
    c1 = c01 * (1 - yf) + c11 * yf
    
    result = c0 * (1 - zf) + c1 * zf
    
    return result

def calculate_nonplanar_z(noise_lut, x, y, layer_base_z, amplitude, taper_factor=1.0):
    """
    Calculate the actual Z height for non-planar infill at given XY coordinates.
    
    Args:
        noise_lut: The 3D noise lookup table
        x, y: Coordinates to sample
        layer_base_z: Base Z height of the current layer
        amplitude: Noise amplitude (in mm)
        taper_factor: Tapering factor for wall proximity (0.0 to 1.0, default 1.0 = full modulation)
    
    Returns:
        Actual Z height after applying non-planar modulation
    """
    # Sample 3D noise at this point
    noise_value = sample_3d_noise_lut(noise_lut, x, y, layer_base_z)
    
    # Apply amplitude with optional tapering
    z_offset = amplitude * taper_factor * noise_value
    z_actual = layer_base_z + z_offset
    
    return z_actual

def generate_lut_visualization(layer_num, layer_z, noise_lut, amplitude, grid_resolution, 
                                solid_at_grid, grid_cell_solid_regions, script_dir, logging):
    """
    Generate a PNG visualization of the noise LUT for a specific layer.
    Shows noise values across all safezones (gaps between solid regions).
    
    Args:
        layer_num: Layer number for filename
        layer_z: Z height of the layer
        noise_lut: The 3D noise lookup table
        amplitude: Noise amplitude for valley detection
        grid_resolution: Size of each grid cell
        solid_at_grid: Dictionary tracking solid regions
        grid_cell_solid_regions: Dictionary mapping (gx,gy) to list of solid regions
        script_dir: Directory to save image
        logging: Logger instance
    
    Returns:
        True if successful, False otherwise
    """
    # Check if PIL is available
    if not HAS_PIL:
        return False
    
    try:
        # Get FULL grid bounds (cached helper function)
        bounds = calculate_grid_bounds(solid_at_grid)
        if not bounds:
            return False
        
        grid_x_min, grid_x_max, grid_y_min, grid_y_max, grid_width, grid_height = bounds
        
        # Scale up for visibility (each grid cell = 4 pixels)
        scale = 4
        img_width = grid_width * scale
        img_height = grid_height * scale
        
        # Create image (black background)
        img = Image.new('RGB', (img_width, img_height), color='black')
        draw = ImageDraw.Draw(img)
        
        # Calculate noise for ALL cells in safezones (using extracted helper)
        safezone_noise_map = {}
        for gx in range(grid_x_min, grid_x_max + 1):
            for gy in range(grid_y_min, grid_y_max + 1):
                if is_in_safezone(gx, gy, layer_num, grid_cell_solid_regions):
                    # Sample noise at cell center
                    seg_x = (gx + 0.5) * grid_resolution
                    seg_y = (gy + 0.5) * grid_resolution
                    noise_val = sample_3d_noise_lut(noise_lut, seg_x, seg_y, layer_z)
                    safezone_noise_map[(gx, gy)] = noise_val
        
        if not safezone_noise_map:
            return False
        
        # Find max absolute noise value for normalization
        noise_values = list(safezone_noise_map.values())
        noise_range = max(abs(min(noise_values)), abs(max(noise_values)))
        if noise_range == 0:
            noise_range = 1.0
        
        # Draw noise for all safezone cells
        for (gx, gy), noise_val in safezone_noise_map.items():
            # Convert to image coordinates (flip Y axis)
            img_x = (gx - grid_x_min) * scale
            img_y = (grid_y_max - gy) * scale  # Flip Y
            
            # Normalize noise to 0-255
            normalized = abs(noise_val) / noise_range
            intensity = int(normalized * 255)
            
            # Base grayscale from noise
            r = g = b = intensity
            
            # Calculate actual Z for this noise value
            z_offset = amplitude * noise_val
            z_mod = layer_z + z_offset
            
            # Check if this cell has infill extrusions
            cell_key = (gx, gy, layer_num)
            infill_crossings = 0
            if cell_key in solid_at_grid:
                infill_crossings = solid_at_grid[cell_key].get('infill_crossings', 0)
            
            # Overlay scheme on top of grayscale base:
            # - GREEN channel boost: valley (z < layer_z) AND single infill crossing
            # - RED channel boost: valley (z < layer_z) AND multiple infill crossings
            is_valley = z_mod < layer_z
            if is_valley and infill_crossings > 1:
                r = 255  # crossings/intersections
            elif is_valley and infill_crossings == 1:
                g = 255  # single extrusion line
            
            color = (r, g, b)
            
            # Draw filled rectangle for this grid cell
            draw.rectangle(
                [img_x, img_y, img_x + scale - 1, img_y + scale - 1],
                fill=color
            )
        
        # Save image
        img_filename = os.path.join(script_dir, f"lut_layer_{layer_num:03d}_z{layer_z:.2f}.png")
        img.save(img_filename)
        if layer_num % 10 == 0:
            logging.info(f"  Saved LUT visualization: {os.path.basename(img_filename)}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error generating LUT visualization for layer {layer_num}: {e}")
        return False

def process_bridge_section(buffered_lines, current_z, current_e, start_x, start_y,
                           connector_max_length, logging, debug=False,
                           bridge_feedrate_slowdown=0.6, initial_relative=False):
    """Conservatively reinforce gaps BETWEEN consecutive parallel bridge lines.

    The original slicer commands are emitted byte-for-byte and in their
    original order. Only a short, internal intermediate strand is added
    between verified adjacent bridge lines. This avoids the old serpentine
    reconstruction, which dropped commands, duplicated retracts, and
    assumed M82 even under M83.

    Every extra extrusion runs under M83. For M82 sources, restore M82 and
    G92 to the *source* E coordinate before continuing. In M83 sources,
    keep the original relative mode and E-only pressure moves untouched.
    Never extrapolate outside the original bridge outline.
    """
    output = []
    px, py, pz, pe = start_x, start_y, current_z, current_e
    relative_e = initial_relative
    modal_f = None
    previous_long = None
    inserted = 0

    # A relative-XY bridge cannot be safely interpolated using absolute
    # endpoints. Leave it entirely untouched until a modal XY parser is
    # implemented.
    if any(re.match(r'^G91(?:\\s|$)', line.split(';', 1)[0].strip())
           for line in buffered_lines):
        return list(buffered_lines), current_e, (start_x, start_y)

    for line in buffered_lines:
        code = line.split(';', 1)[0].strip()

        if re.match(r'^M83(?:\\s|$)', code):
            relative_e = True
            previous_long = None
        elif re.match(r'^M82(?:\\s|$)', code):
            relative_e = False
            previous_long = None
        elif re.match(r'^G92(?:\\s|$)', code):
            reset = parse_gcode_line(code)['e']
            if reset is not None:
                pe = reset
            previous_long = None

        is_motion = re.match(r'^G0?[01](?:\\s|$)', code) is not None
        if not is_motion:
            output.append(line)
            continue

        p = parse_gcode_line(code)
        nx = p['x'] if p['x'] is not None else px
        ny = p['y'] if p['y'] is not None else py
        nz = p['z'] if p['z'] is not None else pz
        distance = math.hypot(nx - px, ny - py)
        delta = (p['e'] if relative_e else p['e'] - pe) if p['e'] is not None else 0.0
        if p['f'] is not None:
            modal_f = p['f']

        long_extrusion = (
            code.startswith("G1") and
            (p['x'] is not None or p['y'] is not None) and
            p['z'] is None and delta > 0 and
            distance >= DEFAULT_BRIDGE_MIN_LENGTH
        )
        current_long = None
        if long_extrusion:
            current_long = {
                'a': (px, py), 'b': (nx, ny),
                'length': distance, 'volume': delta,
            }

        output.append(line)  # Preserve every original slicer command.

        if current_long and previous_long:
            ax = previous_long['b'][0] - previous_long['a'][0]
            ay = previous_long['b'][1] - previous_long['a'][1]
            bx = nx - px
            by = ny - py
            dot = (ax * bx + ay * by) / (distance * previous_long['length'])
            perp_distance = abs(
                (px - previous_long['a'][0]) * ay -
                (py - previous_long['a'][1]) * ax
            ) / previous_long['length']

            # Measure overlap along the first line, not merely parallelism:
            # unrelated nearby segments must not be connected with plastic.
            ux, uy = ax / previous_long['length'], ay / previous_long['length']
            p0 = (px - previous_long['a'][0]) * ux + (py - previous_long['a'][1]) * uy
            p1 = (nx - previous_long['a'][0]) * ux + (ny - previous_long['a'][1]) * uy
            lower = max(0.0, min(p0, p1))
            upper = min(previous_long['length'], max(p0, p1))
            overlap = max(0.0, upper - lower)

            if (abs(dot) >= 0.98 and
                0.12 <= perp_distance < DEFAULT_BRIDGE_MAX_SPACING and
                overlap >= 0.8 * min(previous_long['length'], distance)):
                # Reconstruct an interior, aligned midpoint strand.
                first0 = previous_long['a']
                first1 = previous_long['b']
                second0 = (px, py) if dot >= 0 else (nx, ny)
                second1 = (nx, ny) if dot >= 0 else (px, py)
                mid0 = ((first0[0] + second0[0]) / 2,
                        (first0[1] + second0[1]) / 2)
                mid1 = ((first1[0] + second1[0]) / 2,
                        (first1[1] + second1[1]) / 2)
                mid_length = math.hypot(mid1[0] - mid0[0], mid1[1] - mid0[1])

                if mid_length > 1.0:
                    e_per_mm = (
                        previous_long['volume'] / previous_long['length'] +
                        delta / distance
                    ) / 2
                    extra_e = round(
                        mid_length * e_per_mm * DEFAULT_BRIDGE_EXTRUSION_COMPENSATION, 5)
                    if extra_e > 0:
                        # The original move ended at (nx, ny). Start the
                        # added path at the closer midpoint endpoint.
                        if math.hypot(nx - mid0[0], ny - mid0[1]) <= math.hypot(
                                nx - mid1[0], ny - mid1[1]):
                            entry, exit_point = mid0, mid1
                        else:
                            entry, exit_point = mid1, mid0

                        output.append(
                            "; SilkSteel: Bridge Densifier intermediate between parallel strands\n")
                        if not relative_e:
                            output.append("M83 ; Bridge Densifier temporary relative E\n")
                        output.append(
                            f"G0 X{entry[0]:.3f} Y{entry[1]:.3f} F8400 ; Bridge intermediate entry\n")
                        bridge_f = max(60, int(
                            (modal_f if modal_f is not None else 1800)
                            * bridge_feedrate_slowdown))
                        output.append(
                            f"G1 X{exit_point[0]:.3f} Y{exit_point[1]:.3f} "
                            f"E{extra_e:.5f} F{bridge_f} ; Bridge intermediate extrusion\n")
                        output.append(
                            f"G0 X{nx:.3f} Y{ny:.3f} F8400 ; Bridge return to source endpoint\n")
                        if not relative_e:
                            output.append("M82 ; Bridge Densifier restore absolute E\n")
                            # Added relative E must NOT shift slicer's M82
                            # coordinate system or next retract/prime.
                            source_target = p['e'] if p['e'] is not None else pe
                            output.append(
                                f"G92 E{source_target:.5f} ; Bridge Densifier E sync\n")
                        if modal_f is not None:
                            output.append(
                                f"G1 F{modal_f:g} ; Bridge Densifier restore feedrate\n")
                        inserted += 1

        # A short connector inside a bridge weave is fine, but travels,
        # Z changes, pressure moves and long/unknown paths break adjacency.
        if current_long:
            previous_long = current_long
        elif (p['z'] is not None or p['e'] is None or delta <= 0 or
              distance > connector_max_length):
            previous_long = None

        px, py, pz = nx, ny, nz
        if p['e'] is not None:
            pe = pe + delta if relative_e else p['e']

    if inserted and debug:
        logging.info("[BRIDGE] Inserted %d safe interior bridge strands", inserted)
    return output, pe, (px, py)


def process_gcode(input_file, output_file=None, outer_layer_height=None,
                 enable_smoothificator=True, smoothificator_skip_first_layer=True,
                 enable_bricklayers=False, bricklayers_extrusion_multiplier=1.0,
                 enable_nonplanar=False, deform_type='sine',
                 segment_length=DEFAULT_SEGMENT_LENGTH, amplitude=DEFAULT_AMPLITUDE, frequency=DEFAULT_FREQUENCY,
                 nonplanar_feedrate_multiplier=DEFAULT_NONPLANAR_FEEDRATE_MULTIPLIER,
                 enable_adaptive_extrusion=DEFAULT_ENABLE_ADAPTIVE_EXTRUSION,
                 adaptive_extrusion_multiplier=DEFAULT_ADAPTIVE_EXTRUSION_MULTIPLIER,
                 enable_safe_z_hop=DEFAULT_ENABLE_SAFE_Z_HOP, safe_z_hop_margin=DEFAULT_SAFE_Z_HOP_MARGIN,
                 z_hop_retraction=DEFAULT_Z_HOP_RETRACTION,
                 enable_bridge_densifier=DEFAULT_ENABLE_BRIDGE_DENSIFIER,
                 remove_gap_fill=DEFAULT_REMOVE_GAP_FILL,
                 debug=False):
    
    # Determine output filename
    # If no output specified, modify in-place (for slicer compatibility)
    # If output specified, write to that file (for manual testing)
    if output_file is None:
        output_file = input_file  # Modify in-place for slicer
        in_place_mode = True
    else:
        in_place_mode = False
    
    logging.info("=" * 85)
    logging.info("SMOOTHIFICATOR ADVANCED - Starting G-code processing")
    logging.info("=" * 85)
    logging.info(f"Input file: {input_file}")
    if in_place_mode:
        logging.info(f"Output mode: IN-PLACE (for slicer compatibility)")
    else:
        logging.info(f"Output file: {output_file}")
    logging.info(f"Features enabled:")
    logging.info(f"  - Smoothificator (External perimeters): {enable_smoothificator}")
    if enable_smoothificator:
        logging.info(f"    - Skip first layer: {smoothificator_skip_first_layer}")
    logging.info(f"  - Bricklayers (Internal perimeters): {enable_bricklayers}")
    logging.info(f"  - Non-planar Infill: {enable_nonplanar}")
    logging.info(f"  - Safe Z-hop: {enable_safe_z_hop}")
    logging.info(f"  - Bridge Densifier: {enable_bridge_densifier}")
    logging.info(f"  - Remove Gap Fill: {remove_gap_fill}")
    
    # Print to console for user visibility
    print("\n" + "=" * 85)
    print("  SILKSTEEL - Advanced G-code Post-Processor")
    print("  \"Smooth on the outside, strong on the inside\"")
    print("=" * 85)
    print(f"  Input:  {os.path.basename(input_file)}")
    if in_place_mode:
        print(f"  Output: [IN-PLACE] {os.path.basename(output_file)}")
    else:
        print(f"  Output: {os.path.basename(output_file)}")
    
    # Show enabled features (settings will be shown after we read the G-code)
    print(f"  Features: ", end="")
    features = []
    if enable_smoothificator:
        features.append("Smoothificator")
    if enable_bricklayers:
        features.append("Bricklayers")
    if enable_nonplanar:
        features.append("Non-planar Infill")
    if enable_safe_z_hop:
        features.append("Safe Z-hop")
    if enable_bridge_densifier:
        features.append("Bridge Densifier")
    print(", ".join(features) if features else "(None)")
    print("=" * 85)
    
    # Read the input G-code
    print("Reading G-code file...")
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    print(f"Loaded {len(lines)} lines")

    if enable_bridge_densifier and any(re.match(r'^\s*M83(?:\s|$)', l) for l in lines):
        logging.warning("Bridge Densifier disabled: relative-E (M83) source is unsupported")
        enable_bridge_densifier = False
    elif enable_bridge_densifier:
        logging.warning("Experimental Bridge Densifier enabled: E-mode/flow requires printer validation")

    # Get layer heights from G-code
    base_layer_height = get_layer_height(lines)
    if base_layer_height is None:
        base_layer_height = 0.2
        logging.warning(f"Could not detect layer height, using default: {base_layer_height}mm")
    else:
        logging.info(f"Detected base layer height: {base_layer_height}mm")
    
    # Determine outer layer height based on mode
    if isinstance(outer_layer_height, str):
        if outer_layer_height == 'Auto':
            # Auto mode: min(first_layer_height, base_layer_height) * 0.5
            first_layer_height = get_first_layer_height(lines)
            if first_layer_height is None:
                first_layer_height = base_layer_height  # Fallback
                logging.warning(f"Could not find first_layer_height, using base_layer_height")
            
            min_height = min(first_layer_height, base_layer_height)
            outer_layer_height = min_height * 0.5
            logging.info(f"Auto mode: first_layer={first_layer_height}mm, base={base_layer_height}mm")
            logging.info(f"          → outer_layer_height = min({first_layer_height}, {base_layer_height}) * 0.5 = {outer_layer_height}mm")
        
        elif outer_layer_height == 'Min':
            # Min mode: use min_layer_height from G-code
            outer_layer_height = get_min_layer_height(lines)
            if outer_layer_height is None:
                outer_layer_height = base_layer_height / 2
                logging.warning(f"Could not find min_layer_height, using half of base: {outer_layer_height}mm")
            else:
                logging.info(f"Min mode: using min_layer_height = {outer_layer_height}mm")
    else:
        # Numeric value provided directly
        logging.info(f"Using specified outer_layer_height = {outer_layer_height}mm")
    
    logging.info(f"Target outer wall height: {outer_layer_height}mm")
    
    # Convert amplitude from layers to mm if it's an integer
    # Integer = number of layers, Float = mm
    if isinstance(amplitude, int) or (isinstance(amplitude, float) and amplitude.is_integer()):
        amplitude_layers = int(amplitude)
        amplitude = amplitude_layers * base_layer_height
        logging.info(f"Amplitude: {amplitude_layers} layers = {amplitude:.2f}mm")
    else:
        logging.info(f"Amplitude: {amplitude:.2f}mm")
    
    if enable_bricklayers:
        logging.info(f"Bricklayers extrusion multiplier: {bricklayers_extrusion_multiplier}")
    
    if enable_nonplanar:
        logging.info(f"Non-planar infill - Deform type: {deform_type}, Amplitude: {amplitude}mm, Frequency: {frequency}, Segment length: {segment_length}mm, Feedrate multiplier: {nonplanar_feedrate_multiplier}x, Adaptive extrusion: {enable_adaptive_extrusion}, Adaptive multiplier: {adaptive_extrusion_multiplier}x")
    
    # Print feature settings summary to console
    print("\nFeature Settings:")
    if enable_smoothificator:
        skip_status = "Yes (preserves first layer tuning)" if smoothificator_skip_first_layer else "No"
        print(f"  • Smoothificator: target layer height = {outer_layer_height:.3f}mm, skip first layer = {skip_status}")
    if enable_bricklayers:
        print(f"  • Bricklayers: extrusion multiplier = {bricklayers_extrusion_multiplier:.2f}x")
    if enable_nonplanar:
        print(f"  • Non-planar Infill: amplitude = {amplitude:.2f}mm, frequency = {frequency:.2f}, type = {deform_type}")
        print(f"                       segment length = {segment_length:.2f}mm, feedrate boost = {nonplanar_feedrate_multiplier:.1f}x")
        print(f"                       adaptive extrusion = {'ON' if enable_adaptive_extrusion else 'OFF'}, multiplier = {adaptive_extrusion_multiplier:.2f}x")
    if enable_safe_z_hop:
        print(f"  • Safe Z-hop: margin = {safe_z_hop_margin:.2f}mm, retraction = {z_hop_retraction:.2f}mm")
    if enable_bridge_densifier:
        print(f"  • Bridge Densifier: min_length = {DEFAULT_BRIDGE_MIN_LENGTH}mm, max_spacing = {DEFAULT_BRIDGE_MAX_SPACING}mm")
    print()
    print("⏳ Processing G-code... This might take a while. Time for coffee? ☕")
    print("   (Or tea, if that's your thing. We don't judge.)")
    print()
    
    # Validate user-supplied parameters before generating machine motion.
    if not math.isfinite(float(outer_layer_height)) or outer_layer_height <= 0:
        logging.error(f"Outer layer height ({outer_layer_height}mm) must be greater than 0")
        sys.exit(1)
    
    if not math.isfinite(float(segment_length)) or segment_length <= 0:
        raise ValueError("segment_length must be a finite positive number")
    if not math.isfinite(float(nonplanar_feedrate_multiplier)) or nonplanar_feedrate_multiplier <= 0:
        raise ValueError("nonplanar_feedrate_multiplier must be finite and positive")
    if not math.isfinite(float(bricklayers_extrusion_multiplier)) or bricklayers_extrusion_multiplier <= 0:
        raise ValueError("bricklayers_extrusion_multiplier must be finite and positive")
    if not math.isfinite(float(amplitude)) or amplitude < 0:
        raise ValueError("amplitude must be finite and nonnegative")
    if not math.isfinite(float(frequency)) or frequency <= 0:
        raise ValueError("frequency must be finite and positive")

    # State variables
    current_layer = 0
    current_z = 0.0
    current_layer_height = 0.0
    actual_output_z = 0.0  # Track what Z was actually written to output
    old_z = 0.0
    
    # Safe Z-hop tracking
    layer_max_z = {}  # layer_num -> maximum Z value seen in that layer (pre-calculated from original)
    actual_layer_max_z = {}  # layer_num -> actual maximum Z written to output (updated during processing)
    current_travel_z = 0.0  # Track current Z during travel moves
    working_z = 0.0  # Track the Z where extrusion should happen (before any hop)
    is_hopped = False  # Track if we're currently hopped up above working Z
    seen_first_layer = False  # Don't apply Z-hop until we've started printing
    has_extruded_on_layer = False  # Track if we've done any extrusions on current layer (Z-hop only after first extrusion)
    current_e = 0.0  # Track current E position for retraction/unretraction
    use_relative_e = False  # Track if using relative E mode (G91)
    
    # Bricklayers variables
    perimeter_block_count = 0
    bricklayers_preserved_count = 0  # Legacy field; no longer skips whole TYPE blocks
    bricklayers_unstackable_count = 0
    is_shifted = False
    
    # Non-planar infill variables
    solid_infill_heights = []
    in_infill = False
    in_bridge_infill = False  # Track when we're in bridge infill (skip Z-hops)
    processed_infill_indices = set()
    
    # Bridge densifier variables
    bridge_buffer = []  # Buffer to collect bridge section lines
    bridge_start_e = 0.0  # E value at start of bridge section
    bridge_start_x = 0.0  # X position at start of bridge section
    bridge_start_y = 0.0  # Y position at start of bridge section
    in_bridge_section = False  # Track when we're buffering a bridge section
    
    # First pass: Build 3D grid showing which Z layers have solid at each XY position
    # Then for each layer, determine the safe Z range (space between solid layers)
    
    # Get extrusion width from G-code or use default
    extrusion_width = get_extrusion_width(lines)
    if extrusion_width is None:
        extrusion_width = DEFAULT_EXTRUSION_WIDTH
        logging.info(f"No extrusion_width found in G-code, using default: {extrusion_width}mm")
    else:
        logging.info(f"Detected extrusion_width from G-code: {extrusion_width}mm")
    
    # Update bridge connector max length based on actual extrusion width (2× extrusion width)
    bridge_connector_max_length = extrusion_width * 2.0
    if enable_bridge_densifier:
        logging.info(f"Bridge densifier connector max length: {bridge_connector_max_length:.3f}mm (2× extrusion width)")
    
    # Set grid resolution to a coarser value for a slightly "blurry" grid
    # Using 1.4444× the extrusion width maintains good diagonal coverage while reducing precision
    grid_resolution = extrusion_width * 1.4444
    
    solid_at_grid = {}  # (grid_x, grid_y, layer_num) -> True if solid exists
    z_layer_map = {}    # layer_num -> Z height
    layer_z_map = {}    # Z height -> layer_num
    
    if enable_nonplanar:
        logging.info("\n" + "="*70)
        logging.info("PASS 1: Building 3D solid occupancy grid")
        logging.info("="*70)
        logging.info(f"Grid resolution: {grid_resolution:.3f}mm (~1.44× extrusion width for coarse grid)")
        logging.info("="*70)
        
        # First, scan to find all Z layers and print bounds
        logging.info("Scanning for layers and print bounds...")
        temp_z = 0.0
        current_layer_num = -1  # Will be set from ;LAYER: marker
        x_coords, y_coords = [], []
        
        for line in lines:
            if ';LAYER:' in line:
                layer_match = re.search(r';LAYER:(\d+)', line)
                if layer_match:
                    current_layer_num = int(layer_match.group(1))
            elif ';LAYER_CHANGE' in line:
                current_layer_num += 1  # Fallback for non-standard markers
            if line.startswith('G1') and 'Z' in line:
                temp_z_val = extract_z(line)
                if temp_z_val is not None:
                    temp_z = temp_z_val
                    if current_layer_num not in z_layer_map:
                        z_layer_map[current_layer_num] = temp_z
                        layer_z_map[temp_z] = current_layer_num
            if line.startswith('G1'):
                params = parse_gcode_line(line)
                if params['x'] is not None:
                    x_coords.append(params['x'])
                if params['y'] is not None:
                    y_coords.append(params['y'])
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        total_layers = max(z_layer_map.keys()) if z_layer_map else 0
        logging.info(f"  Print: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}], {total_layers} layers")
        logging.info(f"  Found {len(z_layer_map)} unique Z heights")
        if debug >= 3:
            print(f"[DEBUG] Found {total_layers} layers")
        
        # Second pass: Mark which grid cells have solid infill at each layer
        # UNIFIED GRID STRUCTURE:
        # solid_at_grid[(gx, gy, layer)] = {
        #   'solid': bool,              # True if solid material exists (perimeters, solid infill, etc.)
        #   'infill_crossings': int     # Number of times internal infill crosses this cell (0 = none, 1 = once, 2+ = multiple crossings)
        # }
        # This replaces the old separate solid_at_grid (bool) and infill_traversal_at_grid (int) dictionaries.
        logging.info("\nScanning solid infill AND internal infill to build occupancy grid...")
        logging.info(f"  Processing {len(lines)} lines...")
        solid_at_grid = {}  # (gx, gy, layer) -> {'solid': bool, 'infill_crossings': int}
        temp_z = 0.0
        prev_layer_z = 0.0
        current_layer_height = base_layer_height  # Default
        current_layer_num = -1  # Will be set from ;LAYER: marker
        in_solid_infill = False
        in_internal_infill = False
        current_type = TYPE_NONE  # Track current TYPE for grid metadata
        last_solid_pos = None  # Track last position to mark all cells along line
        last_solid_coords = None  # Track actual X,Y coordinates for DDA
        last_infill_pos = None  # Track last position for infill
        last_infill_coords = None  # Track actual X,Y coordinates for infill
        grid_build_pos = {'x': 0.0, 'y': 0.0}  # Track global position during grid building
        debug_line_count = 0
        debug_cells_marked = 0
        type_markers_seen = set()  # Track all TYPE markers we encounter
        prev_line = ""  # Track previous line for debugging
        line_number = 0  # Track line number for debugging
        
        for line in lines:
            line_number += 1
            
            # Update global position tracker for grid building
            if line.startswith('G1') or line.startswith('G0'):
                x_match = REGEX_X.search(line)
                y_match = REGEX_Y.search(line)
                if x_match:
                    grid_build_pos['x'] = float(x_match.group(1))
                if y_match:
                    grid_build_pos['y'] = float(y_match.group(1))
            
            if ';LAYER:' in line:
                layer_match = re.search(r';LAYER:(\d+)', line)
                if layer_match:
                    current_layer_num = int(layer_match.group(1))
                    last_solid_pos = None  # Reset on layer change
                    last_solid_coords = None
                    last_infill_pos = None
                    last_infill_coords = None
                    # Calculate layer height for this layer
                    if current_layer_num in z_layer_map:
                        current_z = z_layer_map[current_layer_num]
                        if current_layer_num > 0 and (current_layer_num - 1) in z_layer_map:
                            prev_layer_z = z_layer_map[current_layer_num - 1]
                            current_layer_height = current_z - prev_layer_z
                        else:
                            current_layer_height = base_layer_height  # First layer or fallback
            elif ';LAYER_CHANGE' in line:
                current_layer_num += 1  # Fallback for non-standard markers
                last_solid_pos = None  # Reset on layer change
                last_solid_coords = None
                last_infill_pos = None
                last_infill_coords = None
                # Calculate layer height for this layer
                if current_layer_num in z_layer_map:
                    current_z = z_layer_map[current_layer_num]
                    if current_layer_num > 0 and (current_layer_num - 1) in z_layer_map:
                        prev_layer_z = z_layer_map[current_layer_num - 1]
                        current_layer_height = current_z - prev_layer_z
                    else:
                        current_layer_height = base_layer_height  # First layer or fallback
            if line.startswith('G1') and 'Z' in line:
                temp_z_val = extract_z(line)
                if temp_z_val is not None:
                    temp_z = temp_z_val
            
            # Detect solid infill AND perimeters (both block infill from below)
            if ';TYPE:Solid infill' in line or ';TYPE:Top solid infill' in line or ';TYPE:Bridge infill' in line or \
               ';TYPE:Internal bridge infill' in line or ';TYPE:Overhang perimeter' in line or \
               ';TYPE:External perimeter' in line or ';TYPE:Internal perimeter' in line or ';TYPE:Perimeter' in line or \
               ';TYPE:Outer wall' in line or ';TYPE:Inner wall' in line:

                in_solid_infill = True
                in_internal_infill = False
                current_type = get_type_from_marker(line)
                # If this is a Bridge infill marker from slicer, some slicers (Prusa)
                # don't distinguish internal bridge infill. Detect whether the
                # upcoming bridge run actually spans air by sampling the middle
                # of the extrusion path and checking the layer below. If it's
                # NOT over air, treat it as internal bridge infill to prevent
                # running the densifier.
                try:
                    if current_type == TYPE_BRIDGE_INFILL:
                        # use line_number (1-based) -> next line index is line_number
                        start_idx_for_scan = line_number
                        over_air = detect_bridge_over_air(lines, start_idx_for_scan, current_layer_num, solid_at_grid, grid_resolution, parse_gcode_line, voxel_traversal)
                        if not over_air:
                            current_type = TYPE_INTERNAL_BRIDGE_INFILL
                            if debug >= 2:
                                logging.info(f"[BRIDGE-DETECT] Treated Bridge infill as Internal at layer {current_layer_num} (line {line_number})")
                            # Also rewrite the literal TYPE comment in the source lines so
                            # downstream passes that inspect the raw G-code see the
                            # corrected type (easier to trace in outputs / debug).
                            try:
                                idx = line_number - 1
                                if 0 <= idx < len(lines) and ';TYPE:Bridge infill' in lines[idx]:
                                    lines[idx] = lines[idx].replace(';TYPE:Bridge infill', ';TYPE:Internal bridge infill')
                                    # increment module-level counter (use globals to avoid needing a local global decl)
                                    globals()['reclassified_bridge_count'] = globals().get('reclassified_bridge_count', 0) + 1
                                    if debug >= 2:
                                        logging.info(f"[BRIDGE-DETECT] Rewrote TYPE comment to Internal bridge infill at line {line_number}")
                            except Exception as e:
                                logging.warning(f"[BRIDGE-DETECT] failed to rewrite TYPE comment at line {line_number}: {e}")
                except Exception as e:
                    # Never fail the processing on detection errors — log and continue
                    logging.warning(f"[BRIDGE-DETECT] detection error at line {line_number}: {e}")
                # Initialize solid tracking from current global position
                last_solid_coords = (grid_build_pos['x'], grid_build_pos['y'])
                last_solid_pos = (int(grid_build_pos['x'] / grid_resolution), int(grid_build_pos['y'] / grid_resolution))
                last_infill_pos = None
                last_infill_coords = None
                if temp_z not in solid_infill_heights:
                    solid_infill_heights.append(temp_z)
            elif ';TYPE:Internal infill' in line:
                in_internal_infill = True
                in_solid_infill = False
                current_type = TYPE_INTERNAL_INFILL
                last_solid_pos = None
                last_solid_coords = None
                # Initialize infill tracking from current global position
                last_infill_coords = (grid_build_pos['x'], grid_build_pos['y'])
                last_infill_pos = (int(grid_build_pos['x'] / grid_resolution), int(grid_build_pos['y'] / grid_resolution))
            elif ';TYPE:' in line:
                # Track all TYPE markers for debugging
                type_match = re.search(r';TYPE:([^\n]+)', line)
                if type_match:
                    type_markers_seen.add(type_match.group(1))
                in_solid_infill = False
                in_internal_infill = False
                current_type = TYPE_NONE
                last_solid_pos = None  # Reset when exiting solid infill or perimeter
                last_solid_coords = None
                last_infill_pos = None
                last_infill_coords = None
            
            # Track position during solid infill (both G0 and G1 moves)
            if in_solid_infill:
                # Reset tracking on ANY G0 travel move (breaks continuity)
                if line.startswith('G0'):
                    last_solid_pos = None
                    last_solid_coords = None
                
                # Reset tracking on retraction (NEGATIVE E value only)
                # This prevents connecting across travel moves
                if line.startswith('G1') and 'E' in line:
                    e_check = REGEX_E.search(line)
                    if e_check:
                        e_val = float(e_check.group(1))
                        # Only reset on actual retraction (negative E)
                        if e_val < 0:
                            last_solid_pos = None
                            last_solid_coords = None
                
                # Now process ONLY G1 moves with XY coordinates (G0 is travel, skip it)
                if line.startswith('G1') and ('X' in line or 'Y' in line):
                    # Extract X and Y coordinates (handle X-only, Y-only, or both)
                    x = extract_x(line)
                    y = extract_y(line)
                    
                    # If only one coordinate is present, use last known value for the other
                    if x is None and last_solid_coords is not None:
                        x = last_solid_coords[0]
                    if y is None and last_solid_coords is not None:
                        y = last_solid_coords[1]
                    
                    # Only process if we have both coordinates
                    if x is not None and y is not None:
                        # Use floor division for consistent grid cell assignment
                        gx = int(x / grid_resolution)
                        gy = int(y / grid_resolution)
                        
                        # STRICT extrusion detection: Only mark if E parameter present AND positive
                        # This prevents marking travel moves
                        e_value = extract_e(line)
                        has_extrusion = e_value is not None and e_value >= 0
                        
                        if has_extrusion:
                            # Mark all grid cells from last position to current position
                            if last_solid_pos is not None:
                                last_gx, last_gy = last_solid_pos
                                last_x, last_y = last_solid_coords
                                
                                # PROPER GRID TRAVERSAL: Visit every cell the line crosses
                                # Based on "A Fast Voxel Traversal Algorithm for Ray Tracing"
                                # This guarantees we hit EVERY grid cell the line passes through
                                
                                dx = x - last_x
                                dy = y - last_y
                                
                                # Determine step direction for each axis
                                step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
                                step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
                                
                                # Calculate how far we must move (in units of t) to cross one grid cell
                                t_delta_x = abs(grid_resolution / dx) if dx != 0 else float('inf')
                                t_delta_y = abs(grid_resolution / dy) if dy != 0 else float('inf')
                                
                                # Calculate initial t values to reach the next grid line
                                current_cell_x = int(last_x / grid_resolution)
                                current_cell_y = int(last_y / grid_resolution)
                                
                                # Calculate t for next X and Y grid crossings
                                if dx > 0:
                                    t_max_x = ((current_cell_x + 1) * grid_resolution - last_x) / dx
                                elif dx < 0:
                                    t_max_x = (current_cell_x * grid_resolution - last_x) / dx
                                else:
                                    t_max_x = float('inf')
                                
                                if dy > 0:
                                    t_max_y = ((current_cell_y + 1) * grid_resolution - last_y) / dy
                                elif dy < 0:
                                    t_max_y = (current_cell_y * grid_resolution - last_y) / dy
                                else:
                                    t_max_y = float('inf')
                                
                                # Target cell
                                target_cell_x = int(x / grid_resolution)
                                target_cell_y = int(y / grid_resolution)
                                
                                # Mark cells along the ray
                                cells_marked = 0
                                max_iterations = abs(target_cell_x - current_cell_x) + abs(target_cell_y - current_cell_y) + 1
                                
                                # Grid resolution is sized to match extrusion width, so just mark center cells
                                for _ in range(max_iterations + 10):  # Safety margin
                                    # Mark current cell as solid with type information
                                    cell_key = (current_cell_x, current_cell_y, current_layer_num)
                                    if cell_key not in solid_at_grid:
                                        solid_at_grid[cell_key] = {'solid': True, 'infill_crossings': 0, 'type': current_type, 'bricklayer_type': None}
                                    else:
                                        solid_at_grid[cell_key]['solid'] = True
                                        # PRIORITY: Internal perimeters always overwrite other types
                                        # This ensures bricklayers focuses on perimeter-to-perimeter contact
                                        if current_type == TYPE_INTERNAL_PERIMETER:
                                            solid_at_grid[cell_key]['type'] = current_type
                                        elif 'type' not in solid_at_grid[cell_key]:
                                            solid_at_grid[cell_key]['type'] = current_type
                                    cells_marked += 1
                                    
                                    # Check if we've reached the target
                                    if current_cell_x == target_cell_x and current_cell_y == target_cell_y:
                                        break
                                    
                                    # Step to next cell
                                    if t_max_x < t_max_y:
                                        current_cell_x += step_x
                                        t_max_x += t_delta_x
                                    else:
                                        current_cell_y += step_y
                                        t_max_y += t_delta_y
                                
                                debug_line_count += 1
                                debug_cells_marked += cells_marked
                                
                                # Debug output for first few lines
                                if debug >= 3 and debug_line_count <= 10:
                                    print(f"[DEBUG] Line {debug_line_count}: ({last_gx},{last_gy}) -> ({gx},{gy}) marked {cells_marked} cells (voxel traversal)")
                            else:
                                # First point - just mark the cell
                                cell_key = (gx, gy, current_layer_num)
                                if cell_key not in solid_at_grid:
                                    solid_at_grid[cell_key] = {'solid': True, 'infill_crossings': 0}
                                else:
                                    solid_at_grid[cell_key]['solid'] = True
                            
                            # Save actual coordinates for next iteration
                            last_solid_coords = (x, y)
                            
                            # Update last position for next line (for continuity)
                            last_solid_pos = (gx, gy)
            
            # Track position during INTERNAL infill (to count valley crossings)
            if in_internal_infill:
                # Reset tracking on ANY G0 travel move (breaks continuity)
                if line.startswith('G0'):
                    last_infill_pos = None
                    last_infill_coords = None
                    
                    # BUT track position from G0 travel for next extrusion start
                    if 'X' in line or 'Y' in line:
                        x = extract_x(line)
                        y = extract_y(line)
                        if x is not None and y is not None:
                            last_infill_coords = (x, y)
                            last_infill_pos = (int(x / grid_resolution), int(y / grid_resolution))
                
                # Reset tracking on retraction (NEGATIVE E value only)
                if line.startswith('G1') and 'E' in line:
                    e_check = REGEX_E.search(line)
                    if e_check:
                        e_val = float(e_check.group(1))
                        if e_val < 0:
                            last_infill_pos = None
                            last_infill_coords = None
                
                # Process G1 moves with XY coordinates
                if line.startswith('G1') and ('X' in line or 'Y' in line):
                    x = extract_x(line)
                    y = extract_y(line)
                    
                    if x is None and last_infill_coords is not None:
                        x = last_infill_coords[0]
                    if y is None and last_infill_coords is not None:
                        y = last_infill_coords[1]
                    
                    if x is not None and y is not None:
                        gx = int(x / grid_resolution)
                        gy = int(y / grid_resolution)
                        
                        e_value = extract_e(line)
                        has_extrusion = e_value is not None and e_value >= 0
                        
                        if has_extrusion:
                            # Mark all grid cells from last position to current position
                            if last_infill_pos is not None:
                                last_gx, last_gy = last_infill_pos
                                last_x, last_y = last_infill_coords
                                
                                # Use same voxel traversal as solid
                                dx = x - last_x
                                dy = y - last_y
                                
                                step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
                                step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
                                
                                t_delta_x = abs(grid_resolution / dx) if dx != 0 else float('inf')
                                t_delta_y = abs(grid_resolution / dy) if dy != 0 else float('inf')
                                
                                current_cell_x = int(last_x / grid_resolution)
                                current_cell_y = int(last_y / grid_resolution)
                                
                                if dx > 0:
                                    t_max_x = ((current_cell_x + 1) * grid_resolution - last_x) / dx
                                elif dx < 0:
                                    t_max_x = (current_cell_x * grid_resolution - last_x) / dx
                                else:
                                    t_max_x = float('inf')
                                
                                if dy > 0:
                                    t_max_y = ((current_cell_y + 1) * grid_resolution - last_y) / dy
                                elif dy < 0:
                                    t_max_y = (current_cell_y * grid_resolution - last_y) / dy
                                else:
                                    t_max_y = float('inf')
                                
                                target_cell_x = int(x / grid_resolution)
                                target_cell_y = int(y / grid_resolution)
                                
                                max_iterations = abs(target_cell_x - current_cell_x) + abs(target_cell_y - current_cell_y) + 1
                                
                                for _ in range(max_iterations + 10):
                                    cell_key = (current_cell_x, current_cell_y, current_layer_num)
                                    # Increment crossing count for this cell and store type
                                    if cell_key not in solid_at_grid:
                                        solid_at_grid[cell_key] = {'solid': False, 'infill_crossings': 1, 'type': TYPE_INTERNAL_INFILL}
                                    else:
                                        solid_at_grid[cell_key]['infill_crossings'] += 1
                                        if 'type' not in solid_at_grid[cell_key]:
                                            solid_at_grid[cell_key]['type'] = TYPE_INTERNAL_INFILL
                                    
                                    if current_cell_x == target_cell_x and current_cell_y == target_cell_y:
                                        break
                                    
                                    if t_max_x < t_max_y:
                                        current_cell_x += step_x
                                        t_max_x += t_delta_x
                                    else:
                                        current_cell_y += step_y
                                        t_max_y += t_delta_y
                            else:
                                # First point - just increment count
                                cell_key = (gx, gy, current_layer_num)
                                if cell_key not in solid_at_grid:
                                    solid_at_grid[cell_key] = {'solid': False, 'infill_crossings': 1, 'type': TYPE_INTERNAL_INFILL}
                                else:
                                    solid_at_grid[cell_key]['infill_crossings'] += 1
                                    if 'type' not in solid_at_grid[cell_key]:
                                        solid_at_grid[cell_key]['type'] = TYPE_INTERNAL_INFILL
                        
                        # ALWAYS save coordinates for next iteration (even for travel moves)
                        # This ensures next extrusion knows where it's starting from
                        last_infill_coords = (x, y)
                        last_infill_pos = (gx, gy)
            
            # Track previous line for debugging
            prev_line = line
        
        logging.info(f"Total solid infill layers: {len(solid_infill_heights)}")
        logging.info(f"Grid cells marked: {len(solid_at_grid)}")
        
        # Count solid vs infill cells
        solid_count = sum(1 for cell in solid_at_grid.values() if cell['solid'])
        infill_count = sum(1 for cell in solid_at_grid.values() if cell['infill_crossings'] > 0)
        max_crossings = max((cell['infill_crossings'] for cell in solid_at_grid.values()), default=0)
        
        logging.info(f"  Cells with solid material: {solid_count}")
        logging.info(f"  Cells with infill crossings: {infill_count}")
        logging.info(f"  Maximum crossings at any cell: {max_crossings}")
        
        # Build infill_at_grid: Mark INFILL layers (safezones) with first/last metadata
        logging.info("\nBuilding infill grid with first/last safezone markers...")
        
        infill_at_grid = {}  # (gx, gy, layer) -> metadata for INFILL layers
        
        # Build inverted index for faster lookup (will be reused for safe_z calculation)
        grid_to_layers = {}
        for cell_key, cell_data in solid_at_grid.items():
            gx, gy, layer = cell_key
            if cell_data.get('solid', False):  # Only index solid cells
                if (gx, gy) not in grid_to_layers:
                    grid_to_layers[(gx, gy)] = []
                grid_to_layers[(gx, gy)].append(layer)
        
        # Sort layers for each position
        for key in grid_to_layers:
            grid_to_layers[key].sort()
        
        # Mark infill layers in gaps between solid regions
        first_of_safezone_count = 0
        last_of_safezone_count = 0
        
        for (gx, gy), solid_layers in grid_to_layers.items():
            if len(solid_layers) < 2:
                continue  # Need at least 2 solid regions to have gaps
            
            # Find gaps (safezones) between consecutive solid regions
            for i in range(len(solid_layers) - 1):
                solid_end = solid_layers[i]      # Last solid before gap
                solid_start = solid_layers[i+1]  # First solid after gap
                
                # Check if there's a gap (non-consecutive layers)
                if solid_start > solid_end + 1:
                    # Gap exists! Infill layers are from (solid_end + 1) to (solid_start - 1)
                    infill_start = solid_end + 1
                    infill_end = solid_start - 1
                    
                    # Mark first infill layer of safezone (needs adaptive extrusion)
                    key_first = (gx, gy, infill_start)
                    infill_at_grid[key_first] = {
                        'is_first_of_safezone': True,
                        'prev_solid_layer': solid_end,
                        'next_solid_layer': solid_start
                    }
                    first_of_safezone_count += 1
                    
                    # Mark last infill layer of safezone (needs valley filling)
                    key_last = (gx, gy, infill_end)
                    if key_last in infill_at_grid:
                        # Single-layer safezone - mark as both first AND last
                        infill_at_grid[key_last]['is_last_of_safezone'] = True
                    else:
                        infill_at_grid[key_last] = {
                            'is_last_of_safezone': True,
                            'prev_solid_layer': solid_end,
                            'next_solid_layer': solid_start
                        }
                        last_of_safezone_count += 1
        
        logging.info(f"  Built infill grid with {len(infill_at_grid)} infill cells")
        logging.info(f"  Marked {first_of_safezone_count} 'first of safezone' cells (adaptive extrusion)")
        logging.info(f"  Marked {last_of_safezone_count} 'last of safezone' cells (valley filling)")
        
        if debug >= 3:
            print(f"[DEBUG] Marked {len(solid_at_grid)} grid cells with solid")
            print(f"[DEBUG] Processed {debug_line_count} line segments, avg {debug_cells_marked/max(1,debug_line_count):.1f} cells per line")
            print(f"[DEBUG] Built infill grid with {len(infill_at_grid)} cells ({first_of_safezone_count} first + {last_of_safezone_count} last)")
            print(f"\n[DEBUG] All TYPE markers seen in G-code: {sorted(type_markers_seen)}")
            
            # Show which layers have solid infill (only in high debug mode)
            if debug >= 3:
                layers_with_solid = sorted(set(layer for (gx, gy, layer), cell_data in solid_at_grid.items() if cell_data.get('solid', False)))
                print(f"[DEBUG] Layers with solid infill: {layers_with_solid[:30]}..." if len(layers_with_solid) > 30 else f"[DEBUG] Layers with solid infill: {layers_with_solid}")
                
                # Show coverage per layer for first few layers
                print(f"\n[DEBUG] Grid cell coverage for first 10 layers:")
                for layer in layers_with_solid[:10]:
                    cells_at_layer = [(gx, gy) for (gx, gy, lay), cell_data in solid_at_grid.items() if lay == layer and cell_data.get('solid', False)]
                    layer_z = z_layer_map.get(layer, "unknown")
                    print(f"  Layer {layer} (Z={layer_z}): {len(cells_at_layer)} cells marked")
                    if len(cells_at_layer) < 20:  # If few cells, show them
                        print(f"    Cells: {sorted(cells_at_layer)}")
        
        # Third pass: Calculate safe Z range PER GRID CELL
        # For each grid cell (gx, gy), track the safe Z range between solid layers
        # z_min = Z of last solid layer seen at this cell (bottom of safe range)
        # z_max = Z of next solid layer seen at this cell (top of safe range)
        logging.info("\nCalculating safe Z ranges per grid cell...")
        
        # OPTIMIZATION: Reuse grid_to_layers index from infill_at_grid building
        # (already built and sorted above - no need to rebuild!)
        all_grid_positions = sorted(grid_to_layers.keys())
        logging.info(f"  Processing {len(all_grid_positions)} unique grid positions...")
        
        grid_cell_safe_z = {}  # (gx, gy) -> list of (layer_num, z_min, z_max) tuples
        grid_cell_solid_regions = {}  # (gx, gy) -> list of (layer_start, layer_end, z_bottom, z_top) tuples
        
        # NEW: Enhanced grid metadata stored directly in solid_at_grid
        # Instead of separate dictionaries, we'll mark cells with metadata
        # Format: solid_at_grid[(gx, gy, layer)] = {
        #   'is_solid': True,
        #   'is_first_after_safezone': bool,  # First infill layer after a gap (rising Z)
        #   'is_last_before_safezone': bool,  # Last infill layer before a gap (needs valley fill)
        #   'safezone_above': bool,      # Has safezone above this layer
        #   'safezone_below': bool       # Has safezone below this layer
        # }
        
        # For backward compatibility, keep the old dictionaries for now
        # But prepare to migrate to grid-based metadata
        
        # For each grid cell, scan through layers and find solid regions and safe ranges
        for gx, gy in all_grid_positions:
            # Get all layers where this grid cell has solid infill (already sorted from index)
            solid_layers_at_cell = grid_to_layers[(gx, gy)]
            
            if not solid_layers_at_cell:
                continue
            
            # STEP 1: Identify continuous solid regions
            # A solid region is a group of consecutive layers with solid infill
            solid_regions = []
            region_start = solid_layers_at_cell[0]
            
            for i in range(1, len(solid_layers_at_cell)):
                # Check if there's a gap between this layer and the previous
                if solid_layers_at_cell[i] > solid_layers_at_cell[i-1] + 1:
                    # Gap found - end current region
                    region_end = solid_layers_at_cell[i-1]
                    z_bottom = z_layer_map[region_start]
                    z_top = z_layer_map[region_end] + base_layer_height
                    solid_regions.append((region_start, region_end, z_bottom, z_top))
                    
                    # Start new region
                    region_start = solid_layers_at_cell[i]
            
            # Don't forget the last region
            region_end = solid_layers_at_cell[-1]
            z_bottom = z_layer_map[region_start]
            z_top = z_layer_map[region_end] + base_layer_height
            solid_regions.append((region_start, region_end, z_bottom, z_top))
            
            # Store solid regions for this cell
            grid_cell_solid_regions[(gx, gy)] = solid_regions
            
            # STEP 2: Build safe ranges between solid regions
            safe_ranges = []
            
            # For each pair of consecutive solid regions, the space between is safe
            for i in range(len(solid_regions) - 1):
                _, region1_end, _, z_top_region1 = solid_regions[i]
                region2_start, _, z_bottom_region2, _ = solid_regions[i + 1]
                
                # The safe range is from top of first region to bottom of second region
                z_min_safe = z_top_region1  # Top of lower solid region
                z_max_safe = z_bottom_region2  # Bottom of upper solid region
                
                # For all layers between these two solid regions, the safe range is [z_min_safe, z_max_safe]
                for layer_num in range(region1_end + 1, region2_start):
                    safe_ranges.append((layer_num, z_min_safe, z_max_safe))
            
            # Store all safe ranges for this grid cell
            if safe_ranges:
                grid_cell_safe_z[(gx, gy)] = safe_ranges
        
        logging.info(f"  Calculated safe Z ranges for {len(all_grid_positions)} unique grid cells")
        logging.info(f"  Identified {sum(len(regions) for regions in grid_cell_solid_regions.values())} solid regions")
        total_safe_ranges = sum(len(ranges) for ranges in grid_cell_safe_z.values())
        logging.info(f"  Total safe range entries: {total_safe_ranges}")
        
        # Extract all layer numbers for visualization and debugging
        all_layer_nums = sorted(set(layer for _, _, layer in solid_at_grid.keys()))
        
        if debug >= 3:
            print(f"[DEBUG] Calculated safe Z ranges for {len(all_grid_positions)} unique grid cells")
            print(f"[DEBUG] Identified {sum(len(regions) for regions in grid_cell_solid_regions.values())} solid regions")
            total_safe_ranges = sum(len(ranges) for ranges in grid_cell_safe_z.values())
            print(f"[DEBUG] Total safe range entries: {total_safe_ranges}")
            
            # Show example solid regions and safe ranges for a few grid cells
            print(f"\n[DEBUG VISUALIZATION] Example solid regions for first 5 grid cells:")
            for idx, ((gx, gy), regions) in enumerate(list(grid_cell_solid_regions.items())[:5]):
                print(f"  Grid cell ({gx}, {gy}) at X={gx*grid_resolution:.1f}, Y={gy*grid_resolution:.1f}:")
                for region_start, region_end, z_bottom, z_top in regions:
                    print(f"    Solid region: layers {region_start}-{region_end}, Z={z_bottom:.2f} to {z_top:.2f}")
            
            # Check if we're missing bottom layers
            print(f"\n[DEBUG] Solid layers detected: {all_layer_nums[:20]}..." if len(all_layer_nums) > 20 else f"\n[DEBUG] Solid layers detected: {all_layer_nums}")
            print(f"[DEBUG] First solid layer: {min(all_layer_nums)}, Last: {max(all_layer_nums)}")
            
            print(f"\n[DEBUG VISUALIZATION] Example safe Z ranges for first 5 grid cells:")
            for idx, ((gx, gy), ranges) in enumerate(list(grid_cell_safe_z.items())[:5]):
                print(f"  Grid cell ({gx}, {gy}) at X={gx*grid_resolution:.1f}, Y={gy*grid_resolution:.1f}:")
                for layer_num, z_min, z_max in ranges[:3]:  # Show first 3 ranges
                    layer_z = z_layer_map.get(layer_num, 0)
                    print(f"    Layer {layer_num} (Z={layer_z:.2f}): safe range [{z_min:.2f}, {z_max:.2f}]")
                if len(ranges) > 3:
                    print(f"    ... and {len(ranges) - 3} more ranges")
        
        if debug >= 2:
            # Generate debug PNG images for all layers
            print(f"\n[DEBUG] Generating layer visualization PNGs...")
            if HAS_PIL:
                # Use cached grid bounds helper
                bounds = calculate_grid_bounds(solid_at_grid)
                if bounds:
                    grid_x_min, grid_x_max, grid_y_min, grid_y_max, grid_width, grid_height = bounds
                    
                    # Scale up for visibility (each grid cell = 4 pixels)
                    scale = 4
                    img_width = grid_width * scale
                    img_height = grid_height * scale
                    
                    layers_to_visualize = sorted(all_layer_nums)  # ALL layers
                    print(f"[DEBUG] Generating PNG images for {len(layers_to_visualize)} layers...")
                    
                    for layer in layers_to_visualize:
                        layer_z = z_layer_map.get(layer, 0)
                        
                        # IMAGE 1: Solid only (white/black) - simple solid detection
                        img_solid = Image.new('RGB', (img_width, img_height), color='black')
                        draw_solid = ImageDraw.Draw(img_solid)
                        
                        # IMAGE 2: Type-based colors - shows what type of material
                        img_type = Image.new('RGB', (img_width, img_height), color='black')
                        draw_type = ImageDraw.Draw(img_type)
                        
                        # IMAGE 3: Infill crossings overlay (for debugging)
                        img_infill = Image.new('RGB', (img_width, img_height), color='black')
                        draw_infill = ImageDraw.Draw(img_infill)
                        
                        # Draw all cells for this layer
                        for cell_key, cell_data in solid_at_grid.items():
                            gx, gy, lay = cell_key
                            if lay == layer:
                                # Convert to image coordinates (flip Y axis)
                                img_x = (gx - grid_x_min) * scale
                                img_y = (grid_y_max - gy) * scale  # Flip Y
                                
                                # Solid image: white = solid, black = air
                                if cell_data.get('solid', False):
                                    draw_solid.rectangle(
                                        [img_x, img_y, img_x + scale - 1, img_y + scale - 1],
                                        fill='white'
                                    )
                                
                                # Type image: color-coded by material type
                                cell_type = cell_data.get('type', TYPE_NONE)
                                if cell_type != TYPE_NONE:
                                    type_color = TYPE_COLORS.get(cell_type, (128, 128, 128))
                                    draw_type.rectangle(
                                        [img_x, img_y, img_x + scale - 1, img_y + scale - 1],
                                        fill=type_color
                                    )
                                
                                # Infill crossings image: show infill density
                                infill_crossings = cell_data.get('infill_crossings', 0)
                                if infill_crossings > 0:
                                    # Solid first (white)
                                    if cell_data.get('solid', False):
                                        draw_infill.rectangle(
                                            [img_x, img_y, img_x + scale - 1, img_y + scale - 1],
                                            fill='white'
                                        )
                                    # Infill overlay (dark grey only if not solid)
                                    else:
                                        draw_infill.rectangle(
                                            [img_x, img_y, img_x + scale - 1, img_y + scale - 1],
                                            fill=(30, 30, 30)
                                        )
                        
                        # Save all three images
                        img_solid.save(os.path.join(script_dir, f"layer_solid_{layer:03d}_z{layer_z:.2f}.png"))
                        img_type.save(os.path.join(script_dir, f"layer_type_{layer:03d}_z{layer_z:.2f}.png"))
                        # LUT visualization: shows solid (white) + infill crossings (dark grey)
                        img_infill.save(os.path.join(script_dir, f"layer_lut_{layer:03d}_z{layer_z:.2f}.png"))
                        print(f"  Saved: layer_[solid/type/lut]_{layer:03d}_z{layer_z:.2f}.png")
                    
                    print(f"[DEBUG] Generated {len(layers_to_visualize) * 3} layer visualization PNGs (solid, type, lut)")
            else:
                print(f"[DEBUG] PIL/Pillow not available - skipping PNG generation")
                print(f"[DEBUG] Install with: pip install Pillow")
        
        # Get all layers that have solid or infill material anywhere (for visualization)
        all_solid_layers = sorted(set(layer for (gx, gy, layer) in solid_at_grid.keys()))
        if debug >= 3:
            print(f"[DEBUG] Found solid infill on {len(all_solid_layers)} layers: {all_solid_layers[:10]}..." if len(all_solid_layers) > 10 else f"[DEBUG] Found solid infill on {len(all_solid_layers)} layers: {all_solid_layers}")
        
        # Cache grid bounds (optimization: calculate once, reuse everywhere)
        grid_bounds_cached = calculate_grid_bounds(solid_at_grid)
        
        # Prepare grid visualization G-code to insert at layer 0 (only if full debug enabled)
        grid_visualization_gcode = []
        if debug >= 2 and grid_bounds_cached:
            grid_x_min, grid_x_max, grid_y_min, grid_y_max, grid_width, grid_height = grid_bounds_cached
            
            grid_visualization_gcode.append("; ========================================\n")
            grid_visualization_gcode.append("; GRID VISUALIZATION - Solid Infill Detection & Safe Z Ranges PER CELL\n")
            grid_visualization_gcode.append("; Grid: horizontal and vertical lines showing grid structure\n")
            grid_visualization_gcode.append("; + markers: safe Z range boundaries for each grid cell\n")
            grid_visualization_gcode.append("; ========================================\n")
            grid_visualization_gcode.append("G90 ; Absolute positioning\n")
            grid_visualization_gcode.append("M82 ; Absolute extrusion mode\n")
            
            # Use a small E value for visualization
            grid_e = 0.0
            
            # Draw grid lattice ONCE at first solid layer (horizontal and vertical lines)
            first_layer = sorted(all_solid_layers)[0]
            first_z = z_layer_map[first_layer]
            grid_visualization_gcode.append(f"\n; === GRID LATTICE at Z={first_z:.2f} ===\n")
            grid_visualization_gcode.append(f"; Grid bounds: X[{grid_x_min},{grid_x_max}] Y[{grid_y_min},{grid_y_max}]\n")
            grid_visualization_gcode.append(f"G0 Z{first_z:.2f} F3000\n")
            
            # Offset by 0.5 so grid lines pass through cell centers (markers are at integer grid positions)
            offset = grid_resolution * 0.5
            
            # Draw horizontal lines (lines in X direction, constant Y)
            for gy in range(grid_y_min, grid_y_max + 1):
                y = gy * grid_resolution + offset
                x_start = grid_x_min * grid_resolution + offset
                x_end = grid_x_max * grid_resolution + offset
                
                # Thicker/slower for boundary lines
                if gy == grid_y_min or gy == grid_y_max:
                    grid_visualization_gcode.append(f"G0 X{x_start:.2f} Y{y:.2f} F6000\n")
                    grid_visualization_gcode.append(f"G1 X{x_end:.2f} Y{y:.2f} E{grid_e:.5f} F400\n")  # Slower = thicker
                    grid_e += 0.003
                else:
                    grid_visualization_gcode.append(f"G0 X{x_start:.2f} Y{y:.2f} F6000\n")
                    grid_visualization_gcode.append(f"G1 X{x_end:.2f} Y{y:.2f} E{grid_e:.5f} F1200\n")
                    grid_e += 0.001
            
            # Draw vertical lines (lines in Y direction, constant X)
            for gx in range(grid_x_min, grid_x_max + 1):
                x = gx * grid_resolution + offset
                y_start = grid_y_min * grid_resolution + offset
                y_end = grid_y_max * grid_resolution + offset
                
                # Thicker/slower for boundary lines
                if gx == grid_x_min or gx == grid_x_max:
                    grid_visualization_gcode.append(f"G0 X{x:.2f} Y{y_start:.2f} F6000\n")
                    grid_visualization_gcode.append(f"G1 X{x:.2f} Y{y_end:.2f} E{grid_e:.5f} F400\n")  # Slower = thicker
                    grid_e += 0.003
                else:
                    grid_visualization_gcode.append(f"G0 X{x:.2f} Y{y_start:.2f} F6000\n")
                    grid_visualization_gcode.append(f"G1 X{x:.2f} Y{y_end:.2f} E{grid_e:.5f} F1200\n")
                    grid_e += 0.001
            
            # Now draw safe range boundaries for each grid cell
            # Just draw simple + markers at each grid cell position (at first layer Z)
            grid_visualization_gcode.append(f"\n; === GRID CELL MARKERS (at each cell position) ===\n")
            
            for (gx, gy) in all_grid_positions:
                x = gx * grid_resolution
                y = gy * grid_resolution
                
                # Draw + marker at this grid cell position
                grid_visualization_gcode.append(f"G0 X{x-0.3:.2f} Y{y:.2f} Z{first_z:.2f} F6000\n")
                grid_visualization_gcode.append(f"G1 X{x+0.3:.2f} Y{y:.2f} Z{first_z:.2f} E{grid_e:.5f} F1200\n")
                grid_e += 0.001
                grid_visualization_gcode.append(f"G0 X{x:.2f} Y{y-0.3:.2f} Z{first_z:.2f} F6000\n")
                grid_visualization_gcode.append(f"G1 X{x:.2f} Y{y+0.3:.2f} Z{first_z:.2f} E{grid_e:.5f} F1200\n")
                grid_e += 0.001
            
            # Add cross-section views beside the grid (projected onto Z plane)
            grid_visualization_gcode.append(f"\n; === CROSS-SECTION VIEWS (side views projected to build plate) ===\n")
            
            # Calculate center of grid for cross-sections
            center_gx = (grid_x_min + grid_x_max) // 2
            center_gy = (grid_y_min + grid_y_max) // 2
            
            if debug >= 3:
                msg = f"Cross-section center: gx={center_gx} (X={center_gx * grid_resolution:.1f}mm), gy={center_gy} (Y={center_gy * grid_resolution:.1f}mm)"
                print(f"[DEBUG] {msg}")
                logging.info(f"[DEBUG] {msg}")
                msg = f"Grid bounds: X=[{grid_x_min}, {grid_x_max}], Y=[{grid_y_min}, {grid_y_max}]"
                print(f"[DEBUG] {msg}")
                logging.info(f"[DEBUG] {msg}")
            
            # Cross-section 1: YZ plane (view from +X direction) - shows Y vs Z
            # Draw individual dots for each layer that has solid material
            x_section_x_base = (grid_x_max + 3) * grid_resolution  # Base X position to the right of grid
            z_scale = 1.0  # Use 1:1 scale for Z
            grid_visualization_gcode.append(f"\n; Cross-section YZ plane (looking from +X, through X={center_gx * grid_resolution:.1f}mm)\n")
            grid_visualization_gcode.append(f"; Each dot = solid material at that Y,Z position\n")
            
            # Draw solid cells as individual markers
            yz_cells_count = 0
            for cell_key, cell_data in solid_at_grid.items():
                gx, gy, layer = cell_key
                if gx == center_gx and cell_data.get('solid', False):  # Only solid cells on the center slice
                    yz_cells_count += 1
                    if layer in z_layer_map:
                        layer_z = z_layer_map[layer]
                        y_draw = gy * grid_resolution
                        x_draw = x_section_x_base + (layer_z * z_scale)
                        
                        # Draw a small marker (short line)
                        grid_visualization_gcode.append(f"G0 X{x_draw:.2f} Y{y_draw:.2f} Z{first_z:.2f} F6000\n")
                        grid_visualization_gcode.append(f"G1 X{x_draw + 0.2:.2f} Y{y_draw:.2f} Z{first_z:.2f} E{grid_e:.5f} F300\n")
                        grid_e += 0.001
            
            # Cross-section 2: XZ plane (view from +Y direction) - shows X vs Z
            y_section_y_base = (grid_y_max + 3) * grid_resolution  # Base Y position below grid
            grid_visualization_gcode.append(f"\n; Cross-section XZ plane (looking from +Y, through Y={center_gy * grid_resolution:.1f}mm)\n")
            grid_visualization_gcode.append(f"; Each dot = solid material at that X,Z position\n")
            
            # Draw solid cells as individual markers
            xz_cells_count = 0
            for cell_key, cell_data in solid_at_grid.items():
                gx, gy, layer = cell_key
                if gy == center_gy and cell_data.get('solid', False):  # Only solid cells on the center slice
                    xz_cells_count += 1
                    if layer in z_layer_map:
                        layer_z = z_layer_map[layer]
                        x_draw = gx * grid_resolution
                        y_draw = y_section_y_base + (layer_z * z_scale)
                        
                        # Draw a small marker (short line)
                        grid_visualization_gcode.append(f"G0 X{x_draw:.2f} Y{y_draw:.2f} Z{first_z:.2f} F6000\n")
                        grid_visualization_gcode.append(f"G1 X{x_draw:.2f} Y{y_draw + 0.2:.2f} Z{first_z:.2f} E{grid_e:.5f} F300\n")
                        grid_e += 0.001
            
            grid_visualization_gcode.append(f"G92 E0 ; Reset extruder after grid visualization\n")
            grid_visualization_gcode.append("; === End of grid visualization ===\n\n")
            print(f"[DEBUG] Grid visualization prepared: {len(grid_visualization_gcode)} lines to insert at layer 0")
        
        # Generate deformation lookup table for non-planar infill
        if deform_type == 'sine':
            logging.info("\nGenerating 3D sine wave lookup table...")
        else:
            logging.info("\nGenerating 3D noise lookup table...")
        
        x_coords, y_coords, z_coords = [], [], []
        for line in lines:
            if line.startswith('G1'):
                x_match = re.search(r'X([-+]?\d*\.?\d+)', line)
                y_match = re.search(r'Y([-+]?\d*\.?\d+)', line)
                z_match = re.search(r'Z([-+]?\d*\.?\d+)', line)
                if x_match:
                    x_coords.append(float(x_match.group(1)))
                if y_match:
                    y_coords.append(float(y_match.group(1)))
                if z_match:
                    z_coords.append(float(z_match.group(1)))
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        z_min, z_max = min(z_coords), max(z_coords)
        
        logging.info(f"  Print volume: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}], Z[{z_min:.1f}, {z_max:.1f}]")
        
        # Generate LUT based on deform type
        if deform_type == 'sine':
            # Generate 3D sine wave pattern
            noise_lut = generate_3d_sine_lut(
                x_min, x_max, y_min, y_max, z_min, z_max,
                resolution=1.0,  # 1mm grid spacing
                frequency_x=frequency * 0.1,  # Scale frequency to reasonable range
                frequency_y=frequency * 0.1,
                frequency_z=frequency * 0.05  # Less variation in Z direction
            )
        else:
            # Generate Perlin noise pattern
            noise_lut = generate_3d_noise_lut(
                x_min, x_max, y_min, y_max, z_min, z_max,
                resolution=1.0,  # 1mm grid spacing
                frequency_x=frequency * 0.1,  # Scale frequency to reasonable range
                frequency_y=frequency * 0.1,
                frequency_z=frequency * 0.05  # Less variation in Z direction
            )
    else:
        noise_lut = None
    
    # Main processing pass
    logging.info("\n" + "="*70)
    logging.info("PASS 1: Detect and annotate orphan external perimeters")
    logging.info("="*70)
    
    # Preserve the slicer's feature-type continuity across a layer boundary.
    # This specifically fixes exterior contours that begin immediately after
    # ;LAYER_CHANGE with no repeated ;TYPE:External perimeter marker.
    if enable_smoothificator:
        lines, inherited_outer_types = restore_layer_continued_wall_types(lines)
        logging.info("Restored %d implicit outer-wall TYPE markers at layer boundaries",
                     inherited_outer_types)

    # Pass 1: Heuristic-based orphan detection
    # External perimeters are typically continuous extrusion paths that form loops
    annotated_lines = []
    current_type = None
    orphans_found = 0
    # The input may use M83 (relative E); a rising absolute E coordinate
    # is not evidence of extrusion. Detect actual positive source deltas.
    orphan_e_deltas, _, _ = scan_source_extrusion(lines)
    orphan_scanned_through = -1
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Reset TYPE tracking at layer change
        if ";LAYER_CHANGE" in line:
            current_type = None
        
        # Track TYPE markers
        if ";TYPE:" in line:
            current_type = line.strip()
        
        # Check for potential orphan: extrusion NOT in an external perimeter block
        # ALSO exclude solid infill (can have similar characteristics but shouldn't be smoothified)
        # Never guess that a slicer-labelled internal wall, infill, bridge,
        # gap fill or support is an outside contour. Only consider an
        # unlabeled/unknown feature whose geometry looks like a perimeter.
        recognized_nonouter = (current_type is not None and any(
            name in current_type for name in (
                ";TYPE:Internal perimeter", ";TYPE:Inner wall",
                ";TYPE:Perimeter", ";TYPE:Solid infill",
                ";TYPE:Top solid infill", ";TYPE:Internal infill",
                ";TYPE:Bridge infill", ";TYPE:Internal bridge infill",
                ";TYPE:Gap fill", ";TYPE:Support", ";TYPE:Skirt", ";TYPE:Brim",
                ";TYPE:Ironing", ";TYPE:External perimeter", ";TYPE:Outer wall",
                ";TYPE:Overhang perimeter",
            )))
        if (not recognized_nonouter and i > orphan_scanned_through
            and re.match(r'^G1(?:\s|$)', line)
            and extract_x(line) is not None and extract_y(line) is not None
            and orphan_e_deltas[i] > 0):
            
            # Look back to see if there was a recent travel move (high F value, no E)
            # AND check what TYPE was active BEFORE the travel (to avoid catching infill continuations)
            recent_travel = False
            type_before_travel = current_type  # Default to current type
            for lookback_idx in range(max(0, i-20), i):  # Increased from 10 to 20 lines
                check_line = lines[lookback_idx]
                # Track TYPE markers as we look back
                if ";TYPE:" in check_line and not recent_travel:
                    type_before_travel = check_line.strip()
                if "G1" in check_line and "F" in check_line and "E" not in check_line:
                    f_val = extract_f(check_line)
                    if f_val and f_val >= 7200:  # High speed travel
                        recent_travel = True
                        # Don't break - keep looking back for TYPE before travel
            
            # If the TYPE before travel was any kind of infill OR internal perimeter, don't treat as orphan
            # (it's likely a continuation of that feature after a retract/travel)
            if type_before_travel and ("infill" in type_before_travel.lower() or 
                                       "Internal perimeter" in type_before_travel or
                                       "Inner wall" in type_before_travel):
                recent_travel = False  # Suppress orphan detection
            
            # Collect the extrusion path to analyze
            if recent_travel:
                candidate_path = []
                j = i
                path_x = None
                path_y = None
                # Large round objects can have far more than 100 G1
                # segments before closing. A 100-segment prefix looks open
                # even when the complete perimeter is a closed loop.
                while j < len(lines) and len(candidate_path) < 20000:
                    check_line = lines[j]
                    if ";TYPE:" in check_line or ";LAYER_CHANGE" in check_line or check_line.startswith(";LAYER:"):
                        break
                    code = check_line.split(';', 1)[0].strip()
                    if not code:
                        j += 1
                        continue
                    if not re.match(r'^G1(?:\s|$)', code):
                        break
                    params = parse_gcode_line(code)
                    if (params['e'] is None or orphan_e_deltas[j] <= 0 or
                            (params['x'] is None and params['y'] is None)):
                        break
                    if params['x'] is not None:
                        path_x = params['x']
                    if params['y'] is not None:
                        path_y = params['y']
                    if path_x is None or path_y is None:
                        break
                    candidate_path.append((path_x, path_y))
                    j += 1
                orphan_scanned_through = max(orphan_scanned_through, j - 1)

                # Analyze if this looks like an external perimeter:
                # 1. Has at least 10 points (substantial path)
                # 2. E values continuously increase (no retractions)
                # 3. Forms a closed or nearly-closed loop (first/last distance < 10mm)
                # 4. Has sufficient direction changes (not a straight line)
                #    Note: Some perimeters are open paths, so we use a generous threshold
                is_likely_perimeter = False
                if len(candidate_path) >= 10:
                    # Each line has a positive E delta, including in M83.
                    e_increasing = True
                    
                    # Check if closed or nearly-closed loop
                    first_xy = candidate_path[0]
                    last_xy = candidate_path[-1]
                    distance = ((first_xy[0] - last_xy[0])**2 + (first_xy[1] - last_xy[1])**2)**0.5
                    is_closed = distance < 10.0  # Within 10mm (generous for open perimeters)
                    
                    # Check for direction changes to filter out straight infill lines
                    # Count significant angle changes (> 10 degrees)
                    direction_changes = 0
                    total_turn_degrees = 0.0
                    if len(candidate_path) >= 3:
                        for k in range(1, len(candidate_path) - 1):
                            p1 = candidate_path[k-1]
                            p2 = candidate_path[k]
                            p3 = candidate_path[k+1]
                            
                            # Vectors
                            v1x, v1y = p2[0] - p1[0], p2[1] - p1[1]
                            v2x, v2y = p3[0] - p2[0], p3[1] - p2[1]
                            
                            # Angle between vectors
                            len1 = math.sqrt(v1x**2 + v1y**2)
                            len2 = math.sqrt(v2x**2 + v2y**2)
                            
                            if len1 > 0.001 and len2 > 0.001:
                                dot = (v1x * v2x + v1y * v2y) / (len1 * len2)
                                dot = max(-1.0, min(1.0, dot))  # Clamp to avoid math errors
                                angle_deg = math.degrees(math.acos(dot))
                                total_turn_degrees += angle_deg

                                if angle_deg > 10:  # Significant direction change
                                    direction_changes += 1
                    
                    # Perimeters should have at least 3 direction changes
                    # Straight infill lines will have 0-2
                    has_curvature = (direction_changes >= 3 or total_turn_degrees >= 120.0)
                    
                    if e_increasing and is_closed and has_curvature:
                        is_likely_perimeter = True
                
                if is_likely_perimeter:
                    # This is an orphan external perimeter!
                    annotated_lines.append("; SilkSteel: AUTO-ADDED by Smoothificator (heuristic)\n")
                    annotated_lines.append(";TYPE:External perimeter\n")
                    current_type = ";TYPE:External perimeter"
                    orphans_found += 1
                    if debug >= 1:
                        logging.info("Orphan outer candidate: input line %d, %d G1 segments, TYPE=%s, XY start=(%.3f, %.3f)",
                                     i + 1, len(candidate_path), current_type, first_xy[0], first_xy[1])
                    print(f"  [ORPHAN] Found at line {i}: {len(candidate_path)} points, closed loop")
        
        annotated_lines.append(line)
        i += 1
    
    print(f"Pass 1 complete: Found and marked {orphans_found} orphan external perimeter segments")
    logging.info(f"Pass 1 complete: Found and marked {orphans_found} orphan external perimeter segments")
    
    # Use annotated lines for Pass 2
    lines = annotated_lines
    source_e_deltas, source_e_targets, source_relative_modes = scan_source_extrusion(lines)
    
    # Main processing pass
    logging.info("\n" + "="*70)
    logging.info("PASS 2: Processing all features")
    logging.info("="*70)
    
    # Pre-calculate max Z for each layer if safe Z-hop is enabled
    if enable_safe_z_hop:
        logging.info("\nPre-calculating max Z for each layer (for safe Z-hop)...")
        temp_layer = 0
        for line in lines:
            if ";LAYER_CHANGE" in line or ";LAYER:" in line:
                if ";LAYER:" in line:
                    layer_match = re.search(r';LAYER:(\d+)', line)
                    if layer_match:
                        temp_layer = int(layer_match.group(1))
                else:
                    temp_layer += 1
            
            # Track any Z value in G1 commands
            if line.startswith("G1") and "Z" in line:
                z_value = extract_z(line)
                if z_value is not None:
                    if temp_layer not in layer_max_z or z_value > layer_max_z[temp_layer]:
                        layer_max_z[temp_layer] = z_value
        
        logging.info(f"  Found max Z for {len(layer_max_z)} layers")
        if layer_max_z:
            logging.info(f"  Example: Layer 0 max Z = {layer_max_z.get(0, 0):.3f}mm")
    
    print("Processing layers...")
    logging.info(f"\nStarting main processing loop with {len(lines)} lines...")
    logging.info(f"  Max layer detected: {max(z_layer_map.keys()) if z_layer_map else 0}")
    
    # Calculate max layer number for bricklayers
    max_layer = max(z_layer_map.keys()) if z_layer_map else 0
    
    # Use StringIO for faster output building (avoids 100k+ list.append() calls)
    output_buffer = StringIO()
    i = 0
    current_type = None  # Track current TYPE for Z-hop exclusion logic
    line_count_processed = 0
    total_lines = len(lines)  # Cache length to avoid repeated calls
    force_write_next_z = False  # Flag to force writing next standalone Z move after LAYER_CHANGE
    
    # Rolling buffer for lookback operations on OUTPUT (keep last 50 lines for Smoothificator)
    recent_output_lines = []
    max_recent_output = 50
    
    # Global position tracker - updated with EVERY line we read from input
    # This tracks where the nozzle IS (the starting position for the NEXT move)
    # Uses a dict so it can be modified inside the helper function
    # CRITICAL: All features MUST use this tracker as their entry position when starting a new TYPE section
    # DO NOT do backwards lookups through lines - always trust the global position tracker!
    position = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'e': 0.0}
    output_relative_e = False
    
    def update_position(line_str):
        """Update global position tracker from a G-code line.
        Call this EVERY time you read a line from the input, regardless of processing mode."""
        nonlocal output_relative_e
        if re.match(r'^M83(?:\s|$)', line_str):
            output_relative_e = True
        elif re.match(r'^M82(?:\s|$)', line_str):
            output_relative_e = False
        if line_str.startswith("G1") or line_str.startswith("G0"):
            # Use parse_gcode_line for efficient parameter extraction
            params = parse_gcode_line(line_str)
            
            # Update position only for parameters that are present in the line
            if params['x'] is not None:
                position['x'] = params['x']
            if params['y'] is not None:
                position['y'] = params['y']
            if params['z'] is not None:
                position['z'] = params['z']
            if params['e'] is not None:
                if output_relative_e:
                    position['e'] += params['e']
                else:
                    position['e'] = params['e']
        elif line_str.startswith("G92"):
            # G92 resets positions (usually E0)
            e_val = extract_e(line_str)
            if e_val is not None:
                position['e'] = e_val

    # Register output-driven position update callback
    global _update_position_for_output
    _update_position_for_output = update_position
    # From now on, ANY line written via write_and_track will update the position tracker.
    # Collection phases MUST NOT call update_position directly (input lines can be skipped/modified).
    
    while i < total_lines:
        line = lines[i]
        line_count_processed += 1
        
        # NOTE: We NO LONGER update position tracker from INPUT here.
        # Position is updated ONLY when lines are written (see write_and_track).
        # This ensures position always reflects ACTUAL nozzle state in the modified G-code.
        
        # Track TYPE markers for Z-hop exclusion logic and bridge densifier
        if ";TYPE:" in line:
            current_type = line.strip()
            
            # BRIDGE DENSIFIER: Process buffered bridge section when exiting bridge
            if enable_bridge_densifier and in_bridge_section:
                # Check if we're leaving bridge infill (only "Bridge infill", NOT "Internal bridge infill")
                if "Bridge infill" not in current_type or "Internal bridge infill" in current_type:
                    # Process the buffered bridge section
                    logging.info(f"[BRIDGE] Exiting bridge section, processing {len(bridge_buffer)} buffered lines")
                    densified_lines, final_e, final_pos = process_bridge_section(
                        bridge_buffer, current_z, bridge_start_e, bridge_start_x, bridge_start_y, bridge_connector_max_length, logging, debug, bridge_feedrate_slowdown=0.6
                    )
                    
                    # Output densified bridge lines
                    for densified_line in densified_lines:
                        write_and_track(output_buffer, densified_line, recent_output_lines)
                    
                    # Find where the original G-code expects to continue from (last XY move in bridge buffer)
                    # Also look for un-retract command that needs to be preserved
                    last_x, last_y = None, None
                    unretract_line = None
                    last_move_idx = -1
                    
                    for idx in range(len(bridge_buffer) - 1, -1, -1):
                        buf_line = bridge_buffer[idx]
                        if buf_line.startswith("G1") and "X" in buf_line and "Y" in buf_line:
                            params = parse_gcode_line(buf_line)
                            if params['x'] is not None:
                                last_x = params['x']
                            if params['y'] is not None:
                                last_y = params['y']
                            if last_x is not None and last_y is not None:
                                last_move_idx = idx
                                break
                    
                    # Look for un-retract command after the last move (E-only move with E >= 0)
                    if last_move_idx >= 0:
                        for idx in range(last_move_idx + 1, len(bridge_buffer)):
                            buf_line = bridge_buffer[idx]
                            if buf_line.startswith("G1") and "E" in buf_line and "X" not in buf_line and "Y" not in buf_line:
                                params = parse_gcode_line(buf_line)
                                if params['e'] is not None and params['e'] >= 0:
                                    # This is an un-retract command
                                    unretract_line = buf_line
                                    logging.info(f"[BRIDGE] Found un-retract command: {buf_line.strip()}")
                                    break
                    
                    # Check if densified path ended at a different position than expected
                    if last_x is not None and last_y is not None:
                        dist = math.sqrt((final_pos[0] - last_x)**2 + (final_pos[1] - last_y)**2)
                        if dist > 0.01:  # More than 0.01mm away
                            # Need to travel to expected position
                            write_and_track(output_buffer, 
                                add_inline_comment(f"G0 X{last_x:.3f} Y{last_y:.3f} F8400\n", 
                                                 "[Bridge Densifier] Return to expected position"),
                                recent_output_lines)
                            position['x'] = last_x
                            position['y'] = last_y
                            logging.info(f"[BRIDGE] Added travel to expected position: X={last_x:.3f} Y={last_y:.3f}, distance={dist:.3f}mm")
                        else:
                            # Already at expected position
                            position['x'] = final_pos[0]
                            position['y'] = final_pos[1]
                    else:
                        # No last position found, use final position
                        position['x'] = final_pos[0]
                        position['y'] = final_pos[1]
                    
                    # DON'T output un-retract command when exiting on TYPE change
                    # The unretract belongs to the NEXT section, not the bridge
                    # It will be processed normally by the main loop
                    # (Only preserve unretract when exiting on retraction, which happens below)
                    
                    # Update E position after bridge
                    position['e'] = final_e
                    current_e = final_e
                    
                    # Clear buffer but keep TYPE marker if present
                    bridge_buffer = []
                    in_bridge_section = False
                    logging.info(f"[BRIDGE] Bridge section processed, E={final_e:.5f}")
            
            # BRIDGE DENSIFIER: Start buffering when entering bridge (only "Bridge infill", NOT "Internal bridge infill")
            if enable_bridge_densifier:
                if "Bridge infill" in current_type and "Internal bridge infill" not in current_type:
                    if not in_bridge_section:
                        in_bridge_section = True
                        bridge_buffer = []
                        bridge_start_e = position['e']
                        bridge_start_x = position['x']
                        bridge_start_y = position['y']
                        logging.info(f"[BRIDGE] Entering bridge section at X={bridge_start_x:.3f} Y={bridge_start_y:.3f} E={bridge_start_e:.5f}")
            
            # Track when we're in bridge infill (any kind) for Z-hop logic
            if "Bridge infill" in current_type or "Internal bridge infill" in current_type:
                in_bridge_infill = True
            else:
                in_bridge_infill = False
        
        # BRIDGE DENSIFIER: Buffer lines when in bridge section
        if enable_bridge_densifier and in_bridge_section:
            # Check if we hit a layer boundary - stop buffering immediately!
            if ";LAYER_CHANGE" in line or ";LAYER:" in line:
                # Process what we have so far
                if bridge_buffer:
                    logging.info(f"[BRIDGE] Hit layer boundary, processing {len(bridge_buffer)} buffered lines")
                    densified_lines, final_e, final_pos = process_bridge_section(
                        bridge_buffer, current_z, bridge_start_e, bridge_start_x, bridge_start_y, bridge_connector_max_length, logging, debug, bridge_feedrate_slowdown=0.6
                    )
                    
                    # Output densified bridge lines
                    for densified_line in densified_lines:
                        write_and_track(output_buffer, densified_line, recent_output_lines)
                    
                    # Update position
                    position['x'] = final_pos[0]
                    position['y'] = final_pos[1]
                    position['e'] = final_e
                    current_e = final_e
                    
                    # Clear buffer
                    bridge_buffer = []
                    in_bridge_section = False
                
                # Don't buffer the LAYER_CHANGE line - let it be processed normally
                # DON'T continue - fall through so the line gets processed by other handlers
                # The `if` check below will not match since we cleared in_bridge_section
            else:
                # Check if this line is a retraction (negative E move)
                # If so, we've reached the end of this bridge section
                is_retraction = False
                if line.strip().startswith("G1") and "E" in line:
                    params = parse_gcode_line(line)
                    if params['e'] is not None and params['e'] < 0:
                        is_retraction = True
                        logging.info(f"[BRIDGE] Detected retraction during bridge buffering, will process bridge section")
                
                # Buffer this line BEFORE processing (retraction belongs to bridge section)
                bridge_buffer.append(line)
                
                # If retraction detected, process the bridge section now
                if is_retraction:
                    logging.info(f"[BRIDGE] Processing bridge section due to retraction, {len(bridge_buffer)} buffered lines")
                    densified_lines, final_e, final_pos = process_bridge_section(
                        bridge_buffer, current_z, bridge_start_e, bridge_start_x, bridge_start_y, bridge_connector_max_length, logging, debug, bridge_feedrate_slowdown=0.6
                    )
                    
                    # Output densified bridge lines
                    for densified_line in densified_lines:
                        write_and_track(output_buffer, densified_line, recent_output_lines)
                    
                    # Find where the original G-code expects to continue from (last XY move in bridge buffer)
                    # Also look for un-retract command that needs to be preserved
                    last_x, last_y = None, None
                    unretract_line = None
                    last_move_idx = -1
                    
                    for idx in range(len(bridge_buffer) - 1, -1, -1):
                        buf_line = bridge_buffer[idx]
                        if buf_line.startswith("G1") and "X" in buf_line and "Y" in buf_line:
                            params = parse_gcode_line(buf_line)
                            if params['x'] is not None:
                                last_x = params['x']
                            if params['y'] is not None:
                                last_y = params['y']
                            if last_x is not None and last_y is not None:
                                last_move_idx = idx
                                break
                    
                    # Look for un-retract command after the last move (E-only move with E >= 0)
                    # BUT this won't exist yet since we just saw the retraction!
                    # The un-retract will come AFTER the travel move in subsequent lines
                    
                    # Check if densified path ended at a different position than expected
                    if last_x is not None and last_y is not None:
                        dist = math.sqrt((final_pos[0] - last_x)**2 + (final_pos[1] - last_y)**2)
                        if dist > 0.01:  # More than 0.01mm away
                            # Need to travel to expected position
                            write_and_track(output_buffer, 
                                add_inline_comment(f"G0 X{last_x:.3f} Y{last_y:.3f} F8400\n", "[Bridge Densifier] Return to expected position"),
                                recent_output_lines)
                            position['x'] = last_x
                            position['y'] = last_y
                            logging.info(f"[BRIDGE] Added travel to expected position: X={last_x:.3f} Y={last_y:.3f}, distance={dist:.3f}mm")
                        else:
                            # Already at expected position
                            position['x'] = final_pos[0]
                            position['y'] = final_pos[1]
                    else:
                        # No last position found, use final position
                        position['x'] = final_pos[0]
                        position['y'] = final_pos[1]
                    
                    # Update E position after bridge
                    position['e'] = final_e
                    current_e = final_e
                    
                    # Output the retraction command (this will reduce E)
                    retract_e = extract_e(line)
                    write_and_track(output_buffer, 
                        add_inline_comment(line, "[Bridge Densifier] Original retraction"),
                        recent_output_lines)
                    
                    # Update position tracking after retraction
                    if retract_e is not None:
                        position['e'] = retract_e
                        current_e = retract_e
                    
                    # Clear buffer and exit bridge mode
                    bridge_buffer = []
                    in_bridge_section = False
                    logging.info(f"[BRIDGE] Bridge section processed, E after densification={final_e:.5f}, E after retract={position['e']:.5f}")
                    
                    # Check if we should immediately re-enter bridge mode
                    # (We're still in Bridge infill TYPE, just had a retraction between bridge segments)
                    # But DON'T re-enter if the next section is a TYPE change (bridge is ending)
                    should_reenter = False
                    if "Bridge infill" in current_type and "Internal bridge infill" not in current_type:
                        # Look ahead to see if TYPE is changing soon
                        next_is_type_change = False
                        for j in range(i + 1, min(i + 10, len(lines))):
                            if ";TYPE:" in lines[j]:
                                # Check if it's a different TYPE
                                if "Bridge infill" not in lines[j] or "Internal bridge infill" in lines[j]:
                                    next_is_type_change = True
                                break
                        
                        if not next_is_type_change:
                            should_reenter = True
                    
                    if should_reenter:
                        in_bridge_section = True
                        bridge_buffer = []
                        # CRITICAL: Use current E position (after retraction), not final_e!
                        # The next bridge section will start from wherever we are NOW (after retract/travel/unretract)
                        bridge_start_e = position['e']
                        bridge_start_x = position['x']
                        bridge_start_y = position['y']
                        logging.info(f"[BRIDGE] Re-entering bridge section after retraction at X={bridge_start_x:.3f} Y={bridge_start_y:.3f} E={bridge_start_e:.5f}")
                
                # Only continue for retraction case - for LAYER_CHANGE, fall through
                i += 1
                continue
        
        # Progress indicator every 10,000 lines (less verbose)
        if i > 0 and i % 10000 == 0:
            progress = (i / total_lines) * 100
            print(f"  Progress: {i}/{total_lines} lines ({progress:.1f}%)", end='\r')
        
        # Detect layer changes and get adaptive layer height
        if ";LAYER_CHANGE" in line:
            # Mark that we've started the first layer (enable Z-hop from now on)
            seen_first_layer = True
            
            # Flag that the NEXT standalone Z move must be written (layer base Z)
            # This prevents Smoothificator from skipping the critical layer Z positioning
            force_write_next_z = True
            
            # DON'T increment current_layer yet - we need to read the layer number first
            
            
            # Look ahead to find the ;LAYER: marker and get actual layer number
            layer_found = False
            for j in range(i + 1, min(i + 10, len(lines))):
                if ";LAYER:" in lines[j]:
                    layer_match = re.search(r';LAYER:(\d+)', lines[j])
                    if layer_match:
                        current_layer = int(layer_match.group(1))
                        layer_found = True
                        break
            
            if not layer_found:
                # Fallback: increment as before
                current_layer += 1
            
            # Insert grid visualization at layer 0
            if current_layer == 0 and 'grid_visualization_gcode' in locals():
                for viz_line in grid_visualization_gcode:
                    write_and_track(output_buffer, viz_line, recent_output_lines)
                del grid_visualization_gcode  # Only insert once
            
            perimeter_block_count = 0  # Reset block counter for new layer
            move_history = []  # Clear move history for new layer
            is_hopped = False  # Reset hop state for new layer
            has_extruded_on_layer = False  # Reset extrusion flag for new layer
            
            # Look ahead for HEIGHT and Z markers to update current_z
            for j in range(i + 1, min(i + 10, len(lines))):
                if ";Z:" in lines[j]:
                    z_marker_match = re.search(r';Z:([-\d.]+)', lines[j])
                    if z_marker_match:
                        old_z = current_z
                        current_z = float(z_marker_match.group(1))
                        # Synchronize working/base Z and position tracker with marker value.
                        # This overrides any earlier priming Z (e.g., 0.8) so first layer starts at real base (e.g., 0.2).
                        working_z = current_z
                        current_travel_z = current_z
                        position['z'] = current_z  # Comment markers don't write a line, so force tracker update
                        #logging.info(f"\nLayer {current_layer} Z marker: updated current_z from {old_z:.3f} to {current_z:.3f}")
                if ";HEIGHT:" in lines[j]:
                    height_match = re.search(r';HEIGHT:([\d.]+)', lines[j])
                    if height_match:
                        current_layer_height = float(height_match.group(1))
                        break
            
            # Generate LUT visualization for the current layer
            # We do this at the LAYER_CHANGE event AFTER processing all infill from the previous layer
            if enable_nonplanar and debug >= 1 and current_layer > 0 and noise_lut is not None:
                # Use the PREVIOUS layer number/z since we just finished processing it
                vis_layer = current_layer - 1
                vis_layer_z = current_z - current_layer_height  # Approximate
                
                # Find exact Z for the layer we just finished
                if vis_layer in z_layer_map:
                    vis_layer_z = z_layer_map[vis_layer]
                
                # NOTE: LUT visualization is now generated in the grid debug section above
                # (layer_lut_* images showing solid + infill crossings)
                # The old generate_lut_visualization() function is no longer called here
            
            write_and_track(output_buffer, line, recent_output_lines)
            i += 1
            continue

        # Track G90/G91 (absolute/relative positioning mode)
        if line.startswith("G91"):
            use_relative_e = True
            write_and_track(output_buffer, line, recent_output_lines)
            i += 1
            continue
        elif line.startswith("G90"):
            use_relative_e = False
            write_and_track(output_buffer, line, recent_output_lines)
            i += 1
            continue
        
        # Track G92 E (extruder reset)
        if line.startswith("G92") and "E" in line:
            e_reset_match = re.search(r'E([-\d.]+)', line)
            if e_reset_match:
                current_e = float(e_reset_match.group(1))
            write_and_track(output_buffer, line, recent_output_lines)
            i += 1
            continue

        # Get current Z position (for tracking only, don't recalculate layer height)
        # Match G1 commands that contain Z (with or without X/Y)
        # IMPORTANT: Don't track Z during non-planar infill (to avoid tracking modulated Z values)
        if line.startswith("G1") and "Z" in line and "X" not in line and "Y" not in line and not in_infill:
            #logging.info(f"  [Z-MATCH] Line index {i}: {line.strip()}")
            z_match = re.search(r'Z([-\d.]+)', line)
            if z_match:
                old_z = current_z
                current_z = float(z_match.group(1))
                # DON'T calculate layer height from Z - use the HEIGHT marker instead!
                
                # Update working_z for Z-hop (this is the layer's base Z where extrusion happens)
                working_z = current_z
                current_travel_z = current_z
                is_hopped = False  # Explicit Z move means we're at working height, not hopped
                
                # DON'T update actual_layer_max_z here - it gets set by Smoothificator/Bricklayers/Non-planar
                # when they ACTUALLY raise Z above the base layer height!
            
            # Check if next lines contain external perimeter - if so, don't output Z yet
            # Smoothificator will handle Z for each pass
            # CRITICAL: Only skip Z moves if we're in an actual layer (seen_first_layer is True)
            # Don't skip the initial Z positioning moves before first layer!
            # CRITICAL: ALWAYS write Z moves immediately after LAYER_CHANGE (layer base positioning)
            should_output_z = True
            if enable_smoothificator and seen_first_layer and not force_write_next_z:
                # Look ahead to see if external perimeter is coming
                for j in range(i + 1, min(i + 10, len(lines))):
                    if ";TYPE:External perimeter" in lines[j] or ";TYPE:Outer wall" in lines[j] or ";TYPE:Overhang perimeter" in lines[j]:
                        should_output_z = False
                        #logging.info(f"  [SMOOTHIFICATOR] Skipping Z move - external perimeter follows")
                        break
                    # Stop looking if we hit actual extrusion
                    if "G1" in lines[j] and "E" in lines[j] and ("X" in lines[j] or "Y" in lines[j]):
                        break
            
            if should_output_z:
                write_and_track(output_buffer, line, recent_output_lines)
            
            # Clear the force flag after processing the Z move
            force_write_next_z = False
            
            actual_output_z = current_z  # Update actual output Z tracker
            
            i += 1
            continue

        # ========== GAP FILL REMOVAL: Optionally skip gap fill sections ==========
        if ";TYPE:Gap fill" in line:
            # If gap fill removal is DISABLED: just pass through without any collection/buffering
            # Let write_and_track handle position updates naturally - don't interfere!
            if not remove_gap_fill:
                write_and_track(output_buffer, line, recent_output_lines)
                i += 1
                continue
            
            # If gap fill removal IS ENABLED: replace with travel move to final position
            # CRITICAL: We must actually MOVE the nozzle to where gap fill would have ended!
            i += 1  # Skip the TYPE:Gap fill marker
            
            # Save E and Z state BEFORE gap fill - we'll restore them after
            # Gap fill shouldn't change layer Z or E state
            saved_e = position['e']
            saved_z = position['z']
            saved_current_z = current_z
            saved_working_z = working_z
            
            # Track final XY position through all gap fill moves
            gap_fill_final_x = position['x']
            gap_fill_final_y = position['y']
            
            while i < len(lines):
                current_line = lines[i]
                
                # Stop at layer boundary
                if ";LAYER_CHANGE" in current_line or ";LAYER:" in current_line:
                    break
                
                # Stop at different TYPE marker
                if ";TYPE:" in current_line and ";TYPE:Gap fill" not in current_line:
                    break
                
                # Update position tracker by "writing" this line to the tracker (but not to output)
                # This ensures position tracker knows where gap fill ENDS
                if _update_position_for_output:
                    _update_position_for_output(current_line)
                
                # Track final XY position from moves
                if current_line.startswith("G1") or current_line.startswith("G0"):
                    params = parse_gcode_line(current_line)
                    if params['x'] is not None:
                        gap_fill_final_x = params['x']
                    if params['y'] is not None:
                        gap_fill_final_y = params['y']
                
                i += 1
            
            # Restore E and Z to what they were BEFORE gap fill
            # Gap fill might have retractions/G92/Z-hops that we don't want to apply
            position['e'] = saved_e
            position['z'] = saved_z
            current_z = saved_current_z
            working_z = saved_working_z
            
            # Now replace gap fill with a single travel move to final position
            # Write it directly and let it go through Z-hop in a FUTURE iteration
            # We can't fall through to Z-hop (it's an elif) so we need to inject the travel
            # We do this by writing the travel immediately (Z-hop will process it when written)
            travel_line = f"G1 X{gap_fill_final_x:.3f} Y{gap_fill_final_y:.3f} F8400 ; Travel replacing gap fill\n"
            
            # Process the travel line through Z-hop manually
            # Check if Z-hop should apply
            # IMPORTANT: We might ALREADY be hopped from a travel move before gap fill started!
            # In that case, just write the travel and drop back to current Z afterward
            did_hop_for_gap_fill = False
            if enable_safe_z_hop and seen_first_layer:
                # Check if we need Z-hop
                layer_max_z = 0.0
                if current_layer in actual_layer_max_z:
                    layer_max_z = actual_layer_max_z[current_layer]
                
                if layer_max_z > 0:
                    safe_z = layer_max_z + safe_z_hop_margin
                    
                    # Only hop if we're not already above safe_z
                    if current_travel_z < safe_z:
                        # Hop up
                        write_and_track(output_buffer, f"G0 Z{safe_z:.3f} F8400 ; Safe Z-hop\n", recent_output_lines)
                        current_travel_z = safe_z
                        did_hop_for_gap_fill = True
                
                # Write the travel
                write_and_track(output_buffer, travel_line, recent_output_lines)
                
                # Drop back immediately after gap fill travel if we're hopped (either from us or already hopped)
                # Check is_hopped to see if we were already hopped, or did_hop_for_gap_fill if we just hopped
                if is_hopped or did_hop_for_gap_fill:
                    write_and_track(output_buffer, f"G0 Z{current_z:.3f} F8400 ; Drop back to current Z\n", recent_output_lines)
                    current_travel_z = current_z
                    is_hopped = False  # Clear the hopped state
            else:
                # No Z-hop, just write the travel
                write_and_track(output_buffer, travel_line, recent_output_lines)
            
            # Sync current_e with position tracker
            if not use_relative_e:
                current_e = position['e']
            
            # Continue to process the line that ended gap fill (LAYER_CHANGE or TYPE)
            # The scanning loop left i pointing at this line, so continue will process it fresh
            continue

        # ========== SMOOTHIFICATOR: External Perimeter Processing ==========
        elif enable_smoothificator and (smoothificator_skip_first_layer and current_layer > 0 or not smoothificator_skip_first_layer) and (";TYPE:External perimeter" in line or ";TYPE:Outer wall" in line or ";TYPE:Overhang perimeter" in line):
            
            external_block_lines = [line]
            external_block_indices = [i]
            i += 1
            
            # Collect all lines until next TYPE change OR layer change
            # Include WIPE moves and everything up to the next TYPE marker
            while i < len(lines):
                current_line = lines[i]
                # Stop at layer boundary to prevent crossing layers
                if ";LAYER_CHANGE" in current_line or current_line.startswith(";LAYER:"):
                    break
                
                # Stop at different type marker (this is the real end of external perimeter block)
                if (";TYPE:" in current_line and 
                    ";TYPE:External perimeter" not in current_line and 
                    ";TYPE:Outer wall" not in current_line and
                    ";TYPE:Overhang perimeter" not in current_line):
                    break
                    
                external_block_lines.append(current_line)
                external_block_indices.append(i)
                i += 1
            
            # The slicer may retract, prime, wipe and travel *inside* one
            # outer-wall TYPE section. Replaying these commands in every pass
            # is unsafe; skipping the whole section loses the thin walls.
            # Split it into consecutive positive-extrusion paths instead.
            smooth_paths = 0
            raw_moves = 0
            path_lines = []
            path_indices = []
            path_entry_x = position['x']
            path_entry_y = position['y']

            def flush_outer_path():
                nonlocal path_lines, path_indices, path_entry_x, path_entry_y
                nonlocal smooth_paths
                if not path_lines:
                    return

                effective_height = current_layer_height if current_layer_height > 0.01 else outer_layer_height
                if effective_height > outer_layer_height:
                    above = math.ceil(effective_height / outer_layer_height)
                    below = max(1, math.floor(effective_height / outer_layer_height))
                    options = [(1, effective_height), (above, effective_height / above),
                               (below, effective_height / below)]
                    passes_needed, height_per_pass = min(
                        options, key=lambda item: abs(item[1] - outer_layer_height))
                else:
                    passes_needed, height_per_pass = 1, effective_height

                # A path begins at the nozzle's *actual* previous XY, after
                # any travel/retract commands were applied exactly once.
                origin_x, origin_y = path_entry_x, path_entry_y
                path_z = current_z
                if passes_needed > 1:
                    smooth_paths += 1
                for pass_num in range(passes_needed):
                    pass_z = path_z - (passes_needed - pass_num - 1) * height_per_pass
                    actual_layer_max_z[current_layer] = max(
                        actual_layer_max_z.get(current_layer, pass_z), pass_z)
                    if pass_num == 0:
                        write_and_track(output_buffer,
                            f"; ====== SMOOTHIFICATOR START: {passes_needed} passes at {height_per_pass:.4f}mm each ======\n",
                            recent_output_lines)
                    elif math.hypot(position['x'] - origin_x, position['y'] - origin_y) > 0.001:
                        write_and_track(output_buffer,
                            f"G0 X{origin_x:.3f} Y{origin_y:.3f} F8400 ; Smoothificator return to path start\n",
                            recent_output_lines)
                    write_and_track(output_buffer,
                        f"G0 Z{pass_z:.3f} ; Smoothificator pass {pass_num + 1}/{passes_needed}\n",
                        recent_output_lines)

                    for source_idx, original in zip(path_indices, path_lines):
                        code = original.split(';', 1)[0].strip()
                        if not re.match(r'^G0?[01](?:\s|$)', code):
                            # Comments are metadata, not repeated commands.
                            if pass_num == 0:
                                write_and_track(output_buffer, original, recent_output_lines)
                            continue
                        if extract_e(code) is None:
                            # Repeating a modal F-only instruction is safe.
                            write_and_track(output_buffer, original, recent_output_lines)
                            continue
                        # Distribute 5-decimal E rounding error to the
                        # last pass. Three 0.01333mm segments otherwise
                        # add up to 0.03999 instead of the source's
                        # 0.04000mm, systematically losing filament.
                        if pass_num < passes_needed - 1:
                            delta = round(source_e_deltas[source_idx] / passes_needed, 5)
                        else:
                            delta = (source_e_deltas[source_idx]
                                     - round(source_e_deltas[source_idx] / passes_needed, 5)
                                     * (passes_needed - 1))
                        e_value = delta if source_relative_modes[source_idx] else position['e'] + delta
                        write_and_track(output_buffer,
                            replace_e(original, e_value), recent_output_lines)

                # Rebase to the slicer's absolute E coordinate so the
                # following unmodified retract/prime can use its native E.
                last_idx = path_indices[-1]
                if not source_relative_modes[last_idx]:
                    write_and_track(output_buffer,
                        f"G92 E{source_e_targets[last_idx]:.5f} ; Smoothificator E sync\n",
                        recent_output_lines)
                # Diagnostic provenance marker. It has no effect on the
                # printer and distinguishes transformed vs untouched walls.
                write_and_track(output_buffer,
                    "; ====== SMOOTHIFICATOR END ======\n", recent_output_lines)
                path_lines = []
                path_indices = []

            for source_idx, original in zip(external_block_indices, external_block_lines):
                code = original.split(';', 1)[0].strip()
                if not code:
                    if ';WIPE' in original.upper():
                        # Wipe state must start/end at its original point,
                        # not inside each repeated outer-wall pass.
                        flush_outer_path()
                        write_and_track(output_buffer, original, recent_output_lines)
                    elif path_lines:
                        path_lines.append(original)
                        path_indices.append(source_idx)
                    else:
                        write_and_track(output_buffer, original, recent_output_lines)
                    continue

                is_move = re.match(r'^G0?[01](?:\s|$)', code) is not None
                params = parse_gcode_line(code) if is_move else None
                has_xy = is_move and (params['x'] is not None or params['y'] is not None)
                delta = source_e_deltas[source_idx]
                is_extrusion = (is_move and has_xy and delta > 0
                                and params['z'] is None)
                f_only = (is_move and not has_xy and params['z'] is None
                          and params['e'] is None and params['f'] is not None)
                # Slicers periodically insert status/progress commands in
                # the middle of an otherwise continuous exterior contour.
                # Keep them at their original place in pass 1 instead of
                # splitting a 700+ mm wall into many tiny paths.
                status_only = re.match(r'^M(?:117|73)(?:\s|$)', code) is not None

                if status_only and path_lines:
                    path_lines.append(original)
                    path_indices.append(source_idx)
                elif is_extrusion:
                    if not path_lines:
                        path_entry_x, path_entry_y = position['x'], position['y']
                    path_lines.append(original)
                    path_indices.append(source_idx)
                elif f_only and path_lines:
                    path_lines.append(original)
                    path_indices.append(source_idx)
                else:
                    # A barrier can be a retract/unretract, travel, G92,
                    # mode switch, Z move or other motion/control command.
                    # Flush the previous wall FIRST, then execute it once.
                    flush_outer_path()
                    write_and_track(output_buffer, original, recent_output_lines)
                    if is_move and params['z'] is not None:
                        current_z = params['z']
                        working_z = current_z
                    if is_move and has_xy and params['e'] is not None and delta > 0:
                        raw_moves += 1

            flush_outer_path()
            if debug >= 1:
                logging.info("Smoothificator outer-wall block: %d subdivided paths, %d unsupported XYZ-E paths preserved",
                             smooth_paths, raw_moves)
            continue
        
        # ========== BRICKLAYERS: Internal Perimeter Processing ==========
        elif enable_bricklayers and (";TYPE:Perimeter" in line or ";TYPE:Internal perimeter" in line or ";TYPE:Inner wall" in line):
            if ";TYPE:External perimeter" not in line:
                write_and_track(output_buffer, line, recent_output_lines)
                i += 1
                
                z_shift = current_layer_height * 0.5
                is_last_layer = (current_layer == max_layer)
                
                # Use global position tracker for entry position (where nozzle is NOW)
                entry_x = position['x']
                entry_y = position['y']
                
                # Collect the entire perimeter block
                perimeter_block_lines = []
                perimeter_block_indices = []
                
                while i < len(lines):
                    current_line = lines[i]
                    
                    # Capture standalone Z move (no X/Y) to set correct base Z before modifying block
                    if current_line.startswith("G1") and "Z" in current_line and "X" not in current_line and "Y" not in current_line:
                        z_match = re.search(r'Z([-+]?\d*\.?\d+)', current_line)
                        if z_match:
                            old_z = current_z
                            current_z = float(z_match.group(1))
                            working_z = current_z
                            # We do not write this now; Bricklayers will emit its own Z moves
                    
                    # Stop collecting at next TYPE marker OR layer change
                    if (";TYPE:" in current_line or ";LAYER_CHANGE" in current_line
                            or current_line.startswith(";LAYER:")):
                        break
                    
                    perimeter_block_lines.append(current_line)
                    perimeter_block_indices.append(i)
                    i += 1
                
                # Pressure changes (G92 / M82 / M83 / retracts / primes) may
                # occur between contours in the SAME ;TYPE:Internal perimeter
                # section. They are not a reason to discard the whole block.
                # The contour collector below stops before every pressure or
                # travel barrier; each such command is emitted exactly once.

                # Now process the collected block
                # Split into individual perimeter loops (separated by travel moves)
                j = 0
                while j < len(perimeter_block_lines):
                    current_line = perimeter_block_lines[j]
                    
                    # Detect start of perimeter block (extrusion move)
                    if (re.match(r'^G0?[01](?:\s|$)', current_line) and
                            (extract_x(current_line) is not None or extract_y(current_line) is not None) and
                            source_e_deltas[perimeter_block_indices[j]] > 0 and
                            extract_z(current_line) is None):
                        perimeter_block_count += 1
                        
                        # Look back within perimeter_block_lines to find travel position for THIS block
                        block_travel_x, block_travel_y = None, None
                        for back_j in range(j - 1, max(-1, j - 15), -1):
                            if back_j < 0:
                                break
                            back_line = perimeter_block_lines[back_j]
                            if back_line.startswith("G1") and "X" in back_line and "Y" in back_line and "E" not in back_line and "F" in back_line:
                                # Found travel move for this block
                                x_match = re.search(r'X([-\d.]+)', back_line)
                                y_match = re.search(r'Y([-\d.]+)', back_line)
                                if x_match and y_match:
                                    block_travel_x = float(x_match.group(1))
                                    block_travel_y = float(y_match.group(1))
                                    break
                        
                        # If no block-specific travel found, use the entry position (global position tracker)
                        if block_travel_x is None:
                            block_travel_x = entry_x
                            block_travel_y = entry_y
                        
                        # Collect this perimeter loop first
                        loop_lines = [current_line]
                        loop_indices = [perimeter_block_indices[j]]
                        j += 1
                        
                        # Collect only a positive-extrusion contour. Retractions,
                        # travel, G92, mode changes, Z moves, fan/pressure
                        # commands are barriers executed ONCE in original order.
                        # Status messages can be carried through unchanged.
                        while j < len(perimeter_block_lines):
                            part = perimeter_block_lines[j]
                            part_idx = perimeter_block_indices[j]
                            code = part.split(';', 1)[0].strip()
                            # Non-motion status/fan commands do not end
                            # a contour. The two-pass base writes these once;
                            # shifted/ordinary paths retain original order.
                            # A wipe marker, however, is a real path barrier.
                            harmless_control = re.match(
                                r'^M(?:117|73|106|107)(?:\s|$)', code)
                            if (not code and ';WIPE' not in part.upper()) or harmless_control:
                                loop_lines.append(part)
                                loop_indices.append(part_idx)
                                j += 1
                                continue
                            is_xy_extrusion = (
                                re.match(r'^G0?[01](?:\s|$)', code) and
                                (extract_x(code) is not None or extract_y(code) is not None) and
                                extract_z(code) is None and
                                source_e_deltas[part_idx] > 0
                            )
                            if not is_xy_extrusion:
                                break
                            loop_lines.append(part)
                            loop_indices.append(part_idx)
                            j += 1

                        # Detect if this layer is a base or top of a solid region
                        # Sample along the actual perimeter path to check what's above
                        is_base_layer = False
                        is_top_layer = False
                        
                        # Collect all XY positions along this perimeter loop for sampling
                        sample_positions = []
                        for loop_line in loop_lines:
                            if loop_line.startswith("G1") and ("X" in loop_line or "Y" in loop_line):
                                x = extract_x(loop_line)
                                y = extract_y(loop_line)
                                if x is not None and y is not None:
                                    sample_positions.append((x, y))
                        
                        # Check if stackable: only apply bricklayers if TYPE_INTERNAL_PERIMETER above
                        # (otherwise just output as regular internal perimeter)
                        has_perimeter_above = False
                        if len(sample_positions) > 0:
                            for x, y in sample_positions[::5]:  # Sample every 5th position
                                gx = int(x / grid_resolution)
                                gy = int(y / grid_resolution)
                                next_layer_key = (gx, gy, current_layer + 1)
                                if next_layer_key in solid_at_grid:
                                    next_type = solid_at_grid[next_layer_key].get('type', TYPE_NONE)
                                    if next_type == TYPE_INTERNAL_PERIMETER:
                                        has_perimeter_above = True
                                        break
                        
                        # Skip bricklayers entirely if no stackable perimeter above
                        if not has_perimeter_above:
                            # Output as regular internal perimeter (no bricklayers modification)
                            for loop_line in loop_lines:
                                write_and_track(output_buffer, loop_line, recent_output_lines)
                            bricklayers_unstackable_count += 1
                            # Count this contour only once, at its start.
                            # Don't use continue here - it would loop forever!
                            # Just move to next j and let the loop continue naturally
                        else:
                            # Base layer logic:
                            # - Two-pass at 0.75h: (nothing below OR solid below) AND NO solid above
                            # - This is the START of a stackable brick region
                            
                            # Check what's below - differentiate between regular internal perimeter and bricklayer
                            has_bricklayer_below = False  # shifted or base bricklayer (stackable)
                            has_regular_perimeter_below = False  # non-bricklayer internal perimeter
                            has_non_perimeter_below = False  # solid/infill/etc
                            if current_layer > 0 and len(sample_positions) > 0:
                                for x, y in sample_positions[::5]:
                                    gx = int(x / grid_resolution)
                                    gy = int(y / grid_resolution)
                                    prev_layer_key = (gx, gy, current_layer - 1)
                                    if prev_layer_key in solid_at_grid:
                                        prev_type = solid_at_grid[prev_layer_key].get('type', TYPE_NONE)
                                        if prev_type == TYPE_INTERNAL_PERIMETER:
                                            # Check if it's a bricklayer or regular perimeter
                                            prev_bricklayer = solid_at_grid[prev_layer_key].get('bricklayer_type', None)
                                            if prev_bricklayer in ['base', 'shifted']:
                                                has_bricklayer_below = True
                                            else:
                                                has_regular_perimeter_below = True
                                        elif prev_type != TYPE_NONE:
                                            has_non_perimeter_below = True
                                        # A first sampled grid cell can be a
                                        # regular/untyped perimeter even
                                        # when later cells are bricklayer
                                        # cells. Prefer an actual previous
                                        # brick instead of stopping early.
                                        if has_bricklayer_below:
                                            break
                            
                            # Check what's above (solid = solid infill, NOT internal perimeters)
                            has_solid_above = False
                            if len(sample_positions) > 0:
                                for x, y in sample_positions[::5]:
                                    gx = int(x / grid_resolution)
                                    gy = int(y / grid_resolution)
                                    next_layer_key = (gx, gy, current_layer + 1)
                                    if next_layer_key in solid_at_grid:
                                        next_type = solid_at_grid[next_layer_key].get('type', TYPE_NONE)
                                        # Solid types that block bricklayering
                                        if next_type in [TYPE_SOLID_INFILL, TYPE_TOP_SOLID_INFILL]:
                                            has_solid_above = True
                                            break
                            
                            # Base layer = (nothing OR solid below OR regular perimeter) AND NO solid above AND NO bricklayer below
                            # Only if we have a bricklayer below do we use alternating shift pattern
                            if not has_bricklayer_below and not has_solid_above:
                                is_base_layer = True
                            else:
                                is_base_layer = False
                            
                            # Top layer detection for flush shift (when solid above)
                            is_top_layer = has_solid_above
                            
                            # Re-determine shift status based on updated base layer detection
                            if is_base_layer:
                                is_shifted = False  # Two-pass base layer (no shift)
                            else:
                                is_shifted = perimeter_block_count % 2 == 1  # Alternating shift pattern
                            
                            # Now output this loop with bricklayer pattern
                            if is_shifted:
                                # Shifted block: check if ANY part of loop has solid above
                                # If yes, reduce shift to 0.5x for entire loop (flush with externals)
                                has_solid_above_loop = False
                                for x, y in sample_positions[::5]:  # Sample every 5th position
                                    gx = int(x / grid_resolution)
                                    gy = int(y / grid_resolution)
                                    next_layer_key = (gx, gy, current_layer + 1)
                                    if next_layer_key in solid_at_grid:
                                        above_type = solid_at_grid[next_layer_key].get('type', TYPE_NONE)
                                        # Solid types that require reduced shift
                                        if above_type not in [TYPE_NONE, TYPE_INTERNAL_PERIMETER]:
                                            has_solid_above_loop = True
                                            break
                                
                                # Apply shift for entire loop based on detection
                                if has_solid_above_loop:
                                    # Reduce shift to 0.5x when solid above (flush with external perimeters)
                                    adjusted_z = current_z + (z_shift * 0.5)
                                    extrusion_factor = 0.5
                                else:
                                    # Full shift when no solid above
                                    adjusted_z = current_z + z_shift
                                    extrusion_factor = 1.0
                                
                                # Track actual max Z for this layer (for safe Z-hop)
                                if current_layer not in actual_layer_max_z or adjusted_z > actual_layer_max_z[current_layer]:
                                    actual_layer_max_z[current_layer] = adjusted_z
                                
                                write_and_track(output_buffer, f"G0 Z{adjusted_z:.3f} ; Bricklayers shifted block #{perimeter_block_count}\n", recent_output_lines)
                                
                                for source_idx, loop_line in zip(loop_indices, loop_lines):
                                    if loop_line.startswith(('G0 ', 'G1 ')) and extract_e(loop_line) is not None:
                                        delta = source_e_deltas[source_idx]
                                        if delta > 0:
                                            delta *= extrusion_factor * bricklayers_extrusion_multiplier
                                        e = delta if source_relative_modes[source_idx] else position['e'] + delta
                                        loop_line = replace_e(loop_line, e)
                                    write_and_track(output_buffer, loop_line, recent_output_lines)

                                # Reset Z
                                write_and_track(output_buffer, f"G1 Z{current_z:.3f} ; Reset Z\n", recent_output_lines)
                                
                                # Mark grid cells as shifted bricklayer
                                for x, y in sample_positions:
                                    gx = int(x / grid_resolution)
                                    gy = int(y / grid_resolution)
                                    cell_key = (gx, gy, current_layer)
                                    if cell_key in solid_at_grid:
                                        solid_at_grid[cell_key]['bricklayer_type'] = 'shifted'
                        
                            else:
                                # Base block (non-shifted)
                                # Use two-pass on base layers (start of solid regions), single pass otherwise
                                if is_base_layer:
                                    # Base layer: TWO passes at 0.75h each (total 1.5h)
                                    pass1_z = current_z  # Start at normal layer Z
                                    pass2_z = current_z + (current_layer_height * 0.75)  # Raise by 0.75h for second pass
                                    
                                    # Track actual max Z for this layer (for safe Z-hop)
                                    if current_layer not in actual_layer_max_z or pass2_z > actual_layer_max_z[current_layer]:
                                        actual_layer_max_z[current_layer] = pass2_z
                                    
                                    #logging.info(f"  [BRICKLAYERS] Layer {current_layer}, Block #{perimeter_block_count}: Base in 2 passes at Z={pass1_z:.3f} and Z={pass2_z:.3f}")
                                    
                                    # Replay positive source E deltas at 0.75x per pass.
                                    extrusion_moves = [
                                        (source_idx, loop_line)
                                        for source_idx, loop_line in zip(loop_indices, loop_lines)
                                        if loop_line.startswith("G1")
                                        and (extract_x(loop_line) is not None or extract_y(loop_line) is not None)
                                        and source_e_deltas[source_idx] > 0
                                    ]
                                    if not extrusion_moves:
                                        logging.warning("Bricklayers base loop has no extrusion; preserving original")
                                        for original in loop_lines:
                                            write_and_track(output_buffer, original, recent_output_lines)
                                    else:
                                        start_x, start_y = position['x'], position['y']
                                        for pass_num, pass_z in enumerate((pass1_z, pass2_z)):
                                            if pass_num:
                                                write_and_track(output_buffer,
                                                    f"G0 X{start_x:.3f} Y{start_y:.3f} F8400 ; Return for Bricklayers pass 2\n",
                                                    recent_output_lines)
                                            write_and_track(output_buffer,
                                                f"G0 Z{pass_z:.3f} ; Bricklayers base pass {pass_num + 1}/2\n",
                                                recent_output_lines)
                                            for source_idx, original in extrusion_moves:
                                                delta = source_e_deltas[source_idx] * 0.75 * bricklayers_extrusion_multiplier
                                                target_e = delta if source_relative_modes[source_idx] else position['e'] + delta
                                                write_and_track(output_buffer, replace_e(original, target_e), recent_output_lines)

                                        # Do not repeat travel, fan or pressure controls.
                                        used_indices = {source_idx for source_idx, _ in extrusion_moves}
                                        for source_idx, original in zip(loop_indices, loop_lines):
                                            if source_idx not in used_indices:
                                                write_and_track(output_buffer, original, recent_output_lines)
                                        write_and_track(output_buffer,
                                            f"G1 Z{current_z:.3f} ; Reset Z\n", recent_output_lines)

                                    # Mark grid cells as base bricklayer
                                    for x, y in sample_positions:
                                        gx = int(x / grid_resolution)
                                        gy = int(y / grid_resolution)
                                        cell_key = (gx, gy, current_layer)
                                        if cell_key in solid_at_grid:
                                            solid_at_grid[cell_key]['bricklayer_type'] = 'base'
                                else:
                                    # Normal (unshifted) half of the brick bond:
                                    # keep this inner wall at the slicer's layer Z.
                                    # The adjacent shifted wall is already
                                    # extruded at Z + 0.5h (or +0.25h next
                                    # to a solid roof). Using Z + 0.5h here
                                    # too made BOTH roles coincide and erased
                                    # the visible stagger despite correct
                                    # Bricklayers marker counts.
                                    adjusted_z = current_z
                                    extrusion_factor = 1.0
                                    
                                    # Track actual max Z for this layer (for safe Z-hop)
                                    if current_layer not in actual_layer_max_z or adjusted_z > actual_layer_max_z[current_layer]:
                                        actual_layer_max_z[current_layer] = adjusted_z
                                    
                                    write_and_track(output_buffer, f"G0 Z{adjusted_z:.3f} ; Bricklayers base block #{perimeter_block_count}\n", recent_output_lines)
                                    #logging.info(f"  [BRICKLAYERS] Layer {current_layer}, Block #{perimeter_block_count}: Base at Z={adjusted_z:.3f} (extrusion: {extrusion_factor}x)")
                                    
                                    for source_idx, loop_line in zip(loop_indices, loop_lines):
                                        if loop_line.startswith(('G0 ', 'G1 ')) and extract_e(loop_line) is not None:
                                            delta = source_e_deltas[source_idx]
                                            if delta > 0:
                                                delta *= extrusion_factor * bricklayers_extrusion_multiplier
                                            e = delta if source_relative_modes[source_idx] else position['e'] + delta
                                            loop_line = replace_e(loop_line, e)
                                        write_and_track(output_buffer, loop_line, recent_output_lines)
                                    
                                    # Reset Z
                                    write_and_track(output_buffer, f"G1 Z{current_z:.3f} ; Reset Z\n", recent_output_lines)
                                    
                                    # Mark grid cells as base bricklayer (single pass)
                                    for x, y in sample_positions:
                                        gx = int(x / grid_resolution)
                                        gy = int(y / grid_resolution)
                                        cell_key = (gx, gy, current_layer)
                                        if cell_key in solid_at_grid:
                                            solid_at_grid[cell_key]['bricklayer_type'] = 'base'
                            
                            # Rebase to the slicer's E coordinates BEFORE
                            # any original retract/prime is executed. Base
                            # contours can intentionally extrude 1.5x, and
                            # shifted contours may use reduced E on top
                            # faces. Neither must corrupt the next E-only
                            # command in absolute mode.
                            last_idx = loop_indices[-1]
                            if not source_relative_modes[last_idx]:
                                write_and_track(output_buffer,
                                    f"G92 E{source_e_targets[last_idx]:.5f} ; Bricklayers contour E sync\n",
                                    recent_output_lines)
                            # No second increment: alternating shifted and
                            # base contours depends on correct 1,2,3 parity.

                    else:
                        # Travel, retract, prime, G92 and modal commands
                        # must remain in source order and execute once.
                        write_and_track(output_buffer, current_line, recent_output_lines)
                        code = current_line.split(';', 1)[0].strip()
                        if re.match(r'^G0?[01](?:\s|$)', code):
                            z = parse_gcode_line(code)['z']
                            if z is not None:
                                current_z = z
                                working_z = z
                        j += 1
                
                if perimeter_block_indices and not source_relative_modes[perimeter_block_indices[-1]]:
                    write_and_track(output_buffer,
                        f"G92 E{source_e_targets[perimeter_block_indices[-1]]:.5f} ; Bricklayers E sync\n",
                        recent_output_lines)
                continue
        
        # ========== NON-PLANAR INFILL: Process infill with Z modulation ==========
        elif enable_nonplanar and (";TYPE:Internal infill" in line):
            if debug >= 3:
                logging.info(f"[INFILL] Entering infill section at line {i}, line content: {line.strip()}")
            in_infill = True
            write_and_track(output_buffer, line, recent_output_lines)
            i += 1
            
            # Save the current layer Z before applying non-planar modulation
            layer_z = current_z
            last_infill_z = layer_z  # Track last Z used in infill
            adaptive_comment_added = False  # Track if we've added the adaptive E comment for this layer
            
            # Initialize infill position tracking from global position tracker
            # The global tracker is updated with EVERY line, so it knows exactly where the nozzle is NOW
            infill_current_x = position['x']
            infill_current_y = position['y']
            infill_current_e = position['e']
            
            # Valley filling tracking
            # Valley filling is applied PER-CELL based on infill_at_grid metadata
            # (not layer-wide, since different cells may have different needs)
            in_valley = False
            valley_segments = []
            valley_start_e = None
            prev_z = None

            def emitted_e(absolute_target, relative_delta, source_index):
                return relative_delta if source_relative_modes[source_index] else absolute_target

            def flush_pending_valley(source_index):
                nonlocal in_valley, valley_segments, valley_start_e, current_e
                if not in_valley:
                    return
                # An open valley must never silently discard buffered extrusion.
                for pending in valley_segments:
                    valley_start_e += pending['e_delta']
                    target_e = emitted_e(valley_start_e, pending['e_delta'], source_index)
                    pending_f = pending['feedrate']
                    f_cmd = f" F{int(pending_f)}" if pending_f is not None else ""
                    write_and_track(output_buffer,
                        f"G1 X{pending['x']:.3f} Y{pending['y']:.3f} Z{pending['z']:.3f} E{target_e:.5f}{f_cmd}\n",
                        recent_output_lines)
                current_e = valley_start_e
                in_valley = False
                valley_segments = []

            # Process infill lines
            while i < len(lines):
                current_line = lines[i]
                
                # Flush a pending valley before travel/retraction/mode changes.
                if in_valley and not (
                    current_line.startswith(('G0 ', 'G1 ')) and
                    (extract_x(current_line) is not None or extract_y(current_line) is not None) and
                    source_e_deltas[i] > 0
                ):
                    flush_pending_valley(i)

                # CRITICAL: Check for layer change FIRST - restore Z before new layer starts!
                if current_line.startswith(";LAYER_CHANGE") or current_line.startswith(";LAYER:"):
                    in_infill = False
                    # Restore Z before the new layer begins
                    if last_infill_z != layer_z:
                        write_and_track(output_buffer, f"G1 Z{layer_z:.3f} F8400 ; Restore layer Z after non-planar infill\n", recent_output_lines)
                        current_z = float(f"{layer_z:.3f}")
                        actual_output_z = current_z
                        old_z = current_z - current_layer_height
                        #logging.info(f"  [NON-PLANAR INFILL] Restoring Z from {last_infill_z:.3f} to {layer_z:.3f} at layer boundary")
                    # The next iteration of the main loop will process this marker.
                    # i already points at it; decrementing would replay the prior line.
                    break
                
                if ";TYPE:" in current_line:
                    in_infill = False
                    if debug >= 3:
                        logging.info(f"[INFILL] Exiting infill section at line {i}, line content: {current_line.strip()}")
                    # CRITICAL: Restore proper Z height after infill with non-planar modulation
                    if last_infill_z != layer_z:
                        write_and_track(output_buffer, f"G1 Z{layer_z:.3f} F8400 ; Restore layer Z after non-planar infill\n", recent_output_lines)
                        # Update runtime Z trackers so subsequent processing uses the restored layer Z
                        current_z = float(f"{layer_z:.3f}")
                        actual_output_z = current_z
                        # CRITICAL: Also update old_z to maintain proper layer bottom for smoothificator
                        old_z = current_z - current_layer_height
                        #logging.info(f"  [NON-PLANAR INFILL] Restoring Z from {last_infill_z:.3f} to {layer_z:.3f} at TYPE change")
                    break
                
                # Process infill extrusion moves: G1 with X, Y, and E all present
                # CRITICAL: Use SAME logic as grid building for detecting extrusions
                if i not in processed_infill_indices and current_line.startswith('G1'):
                    
                    # Accept X-only/Y-only extrusion moves and any G-code parameter order.
                    move_params = parse_gcode_line(current_line)
                    if (move_params['x'] is not None or move_params['y'] is not None) and move_params['e'] is not None:
                        x2 = move_params['x'] if move_params['x'] is not None else infill_current_x
                        y2 = move_params['y'] if move_params['y'] is not None else infill_current_y
                        e_end = move_params['e']
                        
                        # Extrusion is a *positive delta*, not a positive E
                        # coordinate. Relative E commands may also be negative.
                        e_delta = source_e_deltas[i]
                        x1, y1 = infill_current_x, infill_current_y
                        e_start = infill_current_e

                        # Only subdivide if delta is positive (actual extrusion, not travel or retraction)
                        if e_delta > 0 and math.hypot(x2 - x1, y2 - y1) > 1e-7:
                            # Extract feedrate from current line if present
                            feedrate = None
                            f_match = re.search(r'F(\d+\.?\d*)', current_line)
                            if f_match:
                                feedrate = float(f_match.group(1)) * nonplanar_feedrate_multiplier
                            
                            if debug >= 3:
                                logging.info(f"[INFILL] Line {i}: SUBDIVIDING from ({x1:.2f},{y1:.2f}) to ({x2:.2f},{y2:.2f}), e_delta={e_delta:.5f}")
                            # Mark as processed ONLY when we actually process it
                            processed_infill_indices.add(i)
                            # Simple subdivision: from where we are (x1, y1) to where we're going (x2, y2)
                            segments = segment_line(x1, y1, x2, y2, segment_length)
                            if debug >= 3:
                                logging.info(f"[INFILL] Created {len(segments)} segments")
                            
                            # Calculate total XY distance for the move
                            total_xy_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            
                            # Calculate base E per mm of XY distance
                            # This ensures consistent extrusion regardless of segment count
                            e_per_mm = e_delta / total_xy_distance if total_xy_distance > 0 else 0
                            
                            current_e = e_start
                            prev_segment = None
                            
                            # STEP 2: Add Z modulation using LUT with wall-proximity tapering
                            # Reduce modulation near walls/perimeters to prevent visible artifacts
                            
                            # Process all segments starting from the first
                            for idx, (sx, sy) in enumerate(segments):
                                # Calculate XY distance for THIS segment from previous segment
                                if idx == 0:
                                    # First segment - distance from start point (x1, y1) to first segment point
                                    # This is where we start extrusion (distance > 0 from entry point to first segment)
                                    seg_distance = math.sqrt((sx - x1)**2 + (sy - y1)**2)
                                else:
                                    # Subsequent segments - distance from previous segment point
                                    seg_distance = math.sqrt((sx - prev_segment[0])**2 + (sy - prev_segment[1])**2)
                                
                                # Base extrusion for this segment based on XY distance
                                base_e_for_segment = seg_distance * e_per_mm
                                
                                # Calculate distance to nearest perimeter/solid to taper modulation
                                # Check surrounding grid cells for solid material at current layer
                                # Use floor division to match grid building method
                                gx = int(sx / grid_resolution)
                                gy = int(sy / grid_resolution)
                                
                                # Find minimum distance to any solid cell at this layer
                                min_dist_to_solid = float('inf')
                                search_radius = 5  # Check cells within 5mm
                                for dx in range(-search_radius, search_radius + 1):
                                    for dy in range(-search_radius, search_radius + 1):
                                        check_gx = gx + dx
                                        check_gy = gy + dy
                                        # Check if this cell has solid at current layer
                                        cell_key = (check_gx, check_gy, current_layer)
                                        if cell_key in solid_at_grid and solid_at_grid[cell_key].get('solid', False):
                                            # Calculate distance to this solid cell center
                                            solid_x = check_gx * grid_resolution
                                            solid_y = check_gy * grid_resolution
                                            dist = ((sx - solid_x)**2 + (sy - solid_y)**2)**0.5
                                            min_dist_to_solid = min(min_dist_to_solid, dist)
                                
                                # Calculate tapering factor based on distance to walls
                                # Within 2mm of wall: taper to 0
                                # Beyond 3mm from wall: full modulation
                                taper_distance_start = 2.0  # Start tapering at 2mm from wall
                                taper_distance_full = 3.0   # Full modulation beyond 3mm
                                
                                if min_dist_to_solid < taper_distance_start:
                                    # Very close to wall - no modulation
                                    taper_factor = 0.0
                                elif min_dist_to_solid > taper_distance_full:
                                    # Far from wall - full modulation
                                    taper_factor = 1.0
                                else:
                                    # Transition zone - smooth interpolation
                                    # Linear interpolation between start and full distances
                                    t = (min_dist_to_solid - taper_distance_start) / (taper_distance_full - taper_distance_start)
                                    # Smooth using cosine for gentler transition
                                    taper_factor = (1.0 - math.cos(t * math.pi)) / 2.0
                                
                                # Calculate non-planar Z using helper function
                                z_mod = calculate_nonplanar_z(noise_lut, sx, sy, layer_z, amplitude, taper_factor)
                                
                                # Get safezone bounds for this grid cell
                                local_z_min, local_z_max, layers_until_ceiling, height_until_ceiling = get_safezone_bounds(
                                    gx, gy, current_layer, grid_cell_solid_regions, base_layer_height
                                )
                                
                                # Clamp Z to safe range
                                z_mod_original = z_mod
                                if local_z_min > -999:  # Valid z_min
                                    z_mod = max(local_z_min, z_mod)
                                if local_z_max < 999:  # Valid z_max
                                    #z_mod = min(local_z_max - (layers_until_ceiling * base_layer_height), z_mod)
                                    z_mod = min(local_z_max, z_mod)

                                last_infill_z = z_mod
                                
                                # Track actual max Z for this layer (for safe Z-hop)
                                if current_layer not in actual_layer_max_z or z_mod > actual_layer_max_z[current_layer]:
                                    actual_layer_max_z[current_layer] = z_mod
                                
                                # Calculate E multiplier based on Z lift (only when going UP and if enabled)
                                # This adds extra material that droops down to bond with layer below
                                # ONLY apply on first infill layer (when solid is directly below)
                                # CRITICAL: Start with base extrusion (XY distance), ADD extra for Z lift
                                adjusted_e_for_segment = base_e_for_segment  # Always extrude for XY distance!
                                applied_adaptive_extrusion = False  # Track if we actually apply it
                                segment_feedrate = feedrate  # Default to original feedrate
                                
                                if enable_adaptive_extrusion:
                                    # Check if this CELL is marked as 'first of safezone' (benefits from adaptive extrusion)
                                    is_first_infill_layer = False
                                    
                                    if (gx, gy, current_layer) in infill_at_grid:
                                        cell_data = infill_at_grid[(gx, gy, current_layer)]
                                        if isinstance(cell_data, dict):
                                            is_first_infill_layer = cell_data.get('is_first_of_safezone', False)
                                    
                                    if is_first_infill_layer:
                                        z_lift = z_mod - layer_z  # How much above base layer
                                        
                                        if z_lift > 0:  # Only when lifting UP
                                            # ADD extra material proportional to lift
                                            # Formula: base_e + (base_e * (z_lift / layer_height) * multiplier)
                                            lift_in_layers = z_lift / base_layer_height
                                            extra_e = base_e_for_segment * lift_in_layers * adaptive_extrusion_multiplier
                                            adjusted_e_for_segment += extra_e  # ADD to base!
                                            applied_adaptive_extrusion = True
                                            
                                            # CRITICAL: Reduce feedrate proportionally to maintain even distribution
                                            # If extruding 1.5x material, move at ~67% speed (1/1.5 = 0.67)
                                            # Add extra slowdown factor (0.5) for safety margin on heavy extrusion
                                            # This ensures the extra filament is distributed evenly along the path
                                            if base_e_for_segment > 0 and feedrate is not None:
                                                extrusion_ratio = adjusted_e_for_segment / base_e_for_segment
                                                segment_feedrate = (feedrate / extrusion_ratio) * 0.5  # Extra 50% slowdown
                                
                                # Update current E position
                                current_e += adjusted_e_for_segment
                                
                                # Add a comment once per layer when adaptive extrusion is being applied
                                if applied_adaptive_extrusion and not adaptive_comment_added:
                                    total_multiplier = adjusted_e_for_segment / base_e_for_segment if base_e_for_segment > 0 else 1.0
                                    write_and_track(output_buffer, f"; Adaptive E: {total_multiplier:.2f}x (z_lift={z_lift:.3f}mm, local_z_min={local_z_min:.2f}, layer_z={layer_z:.2f})\n", recent_output_lines)
                                    adaptive_comment_added = True
                                
                                # Save current segment position for next iteration
                                prev_segment = (sx, sy)
                                
                                # ========== VALLEY FILLING ==========
                                # Check if this CELL is marked as 'last of safezone' (needs valley filling)
                                # This is per-cell, not per-layer!
                                cell_needs_valley_fill = False
                                if (gx, gy, current_layer) in infill_at_grid:
                                    cell_data = infill_at_grid[(gx, gy, current_layer)]
                                    if isinstance(cell_data, dict):
                                        cell_needs_valley_fill = cell_data.get('is_last_of_safezone', False)
                                
                                # If Z drops below layer_z, collect segments and fill when valley ends
                                valley_threshold = 0.05  # 0.05mm below layer_z to trigger valley filling
                                
                                # Detect valley entry (only if this CELL needs it)
                                if cell_needs_valley_fill and not in_valley and z_mod < layer_z - valley_threshold:
                                    in_valley = True
                                    valley_segments = []
                                    valley_start_e = current_e - adjusted_e_for_segment
                                    if debug >= 2:
                                        write_and_track(output_buffer, f"; Valley ENTER at segment {idx} (cell {gx},{gy} is last of safezone)\n", recent_output_lines)
                                
                                # Collect segments while in valley
                                if in_valley:
                                    valley_segments.append({
                                        'x': sx,
                                        'y': sy,
                                        'z': z_mod,
                                        'e_delta': adjusted_e_for_segment,
                                        'feedrate': segment_feedrate,  # Use adaptive feedrate
                                        'relative': source_relative_modes[i]
                                    })
                                
                                # Detect valley exit
                                valley_exit = False
                                if in_valley and z_mod >= layer_z - valley_threshold:
                                    valley_exit = True
                                
                                # Process valley exit
                                if valley_exit:
                                    if debug >= 2:
                                        write_and_track(output_buffer, f"; Valley EXIT - filling {len(valley_segments)} segments\n", recent_output_lines)
                                    
                                    # Collect all unique crossing cells touched by this valley (for later decrement)
                                    cells_touched_by_valley = set()
                                    for seg in valley_segments:
                                        seg_gx = int(seg['x'] / grid_resolution)
                                        seg_gy = int(seg['y'] / grid_resolution)
                                        cell_key = (seg_gx, seg_gy, current_layer)
                                        
                                        # Track cells with crossings
                                        if cell_key in solid_at_grid and solid_at_grid[cell_key].get('infill_crossings', 0) > 0:
                                            cells_touched_by_valley.add(cell_key)
                                    
                                    # Output all valley segments at their original Z (the valley path)
                                    for seg in valley_segments:
                                        if seg['feedrate'] is not None:
                                            write_and_track(output_buffer,
                                                f"G1 X{seg['x']:.3f} Y{seg['y']:.3f} Z{seg['z']:.3f} E{(seg['e_delta'] if seg['relative'] else valley_start_e + seg['e_delta']):.5f} F{int(seg['feedrate'])}\n", recent_output_lines
                                            )
                                        else:
                                            write_and_track(output_buffer,
                                                f"G1 X{seg['x']:.3f} Y{seg['y']:.3f} Z{seg['z']:.3f} E{(seg['e_delta'] if seg['relative'] else valley_start_e + seg['e_delta']):.5f}\n", recent_output_lines
                                            )
                                        valley_start_e += seg['e_delta']
                                    
                                    # FILL THE VALLEY - go back and forth to build up to layer_z
                                    min_z = min(seg['z'] for seg in valley_segments)
                                    valley_depth = layer_z - min_z
                                    num_fill_passes = max(1, int(valley_depth / 0.1))  # 0.1mm increments
                                    
                                    for fill_pass in range(num_fill_passes):
                                        fill_z_offset = (fill_pass + 1) * (valley_depth / num_fill_passes)
                                        current_fill_z = min_z + fill_z_offset
                                        
                                        # Filter segments that need filling at this height
                                        segments_to_fill = [seg for seg in valley_segments if seg['z'] < current_fill_z - 0.01]
                                        
                                        if len(segments_to_fill) == 0:
                                            break
                                        
                                        # Alternate direction: odd passes go forward, even passes go reverse
                                        if fill_pass % 2 == 0:
                                            # Even passes: REVERSE direction
                                            prev_point = None
                                            for seg in reversed(segments_to_fill):
                                                # For REVERSE direction, segment goes FROM seg TO prev_point (or end)
                                                # Check if segment crosses a crossing cell
                                                seg_gx = int(seg['x'] / grid_resolution)
                                                seg_gy = int(seg['y'] / grid_resolution)
                                                end_cell = (seg_gx, seg_gy, current_layer)
                                                
                                                # Check if we should skip this segment
                                                should_skip = False
                                                if prev_point is not None:
                                                    prev_gx = int(prev_point[0] / grid_resolution)
                                                    prev_gy = int(prev_point[1] / grid_resolution)
                                                    start_cell = (prev_gx, prev_gy, current_layer)
                                                    
                                                    # Check if EITHER endpoint is in a crossing cell with count > 1
                                                    if start_cell in solid_at_grid:
                                                        crossing_count = solid_at_grid[start_cell].get('infill_crossings', 0)
                                                        if crossing_count > 1:
                                                            should_skip = True
                                                    if end_cell in solid_at_grid:
                                                        crossing_count = solid_at_grid[end_cell].get('infill_crossings', 0)
                                                        if crossing_count > 1:
                                                            should_skip = True
                                                
                                                if should_skip:
                                                    # Skip extrusion (travel only)
                                                    if debug >= 2:
                                                        write_and_track(output_buffer,
                                                            f"G1 X{seg['x']:.3f} Y{seg['y']:.3f} Z{min(current_fill_z, layer_z):.3f} ; Skip fill (crossing)\n", recent_output_lines
                                                        )
                                                    else:
                                                        write_and_track(output_buffer,
                                                            f"G1 X{seg['x']:.3f} Y{seg['y']:.3f} Z{min(current_fill_z, layer_z):.3f}\n", recent_output_lines
                                                        )
                                                else:
                                                    # Extrude normally
                                                    valley_start_e += seg['e_delta'] * 0.5
                                                    write_and_track(output_buffer,
                                                        f"G1 X{seg['x']:.3f} Y{seg['y']:.3f} Z{min(current_fill_z, layer_z):.3f} E{(seg['e_delta'] * 0.5 if seg['relative'] else valley_start_e):.5f}\n", recent_output_lines
                                                    )
                                                prev_point = (seg['x'], seg['y'])
                                        else:
                                            # Odd passes: FORWARD direction
                                            prev_point = None
                                            for seg in segments_to_fill:
                                                # For FORWARD direction, segment goes FROM prev_point TO seg
                                                # Check if segment crosses a crossing cell
                                                seg_gx = int(seg['x'] / grid_resolution)
                                                seg_gy = int(seg['y'] / grid_resolution)
                                                end_cell = (seg_gx, seg_gy, current_layer)
                                                
                                                # Check if we should skip this segment
                                                should_skip = False
                                                if prev_point is not None:
                                                    prev_gx = int(prev_point[0] / grid_resolution)
                                                    prev_gy = int(prev_point[1] / grid_resolution)
                                                    start_cell = (prev_gx, prev_gy, current_layer)
                                                    
                                                    # Check if EITHER endpoint is in a crossing cell with count > 1
                                                    if start_cell in solid_at_grid:
                                                        crossing_count = solid_at_grid[start_cell].get('infill_crossings', 0)
                                                        if crossing_count > 1:
                                                            should_skip = True
                                                    if end_cell in solid_at_grid:
                                                        crossing_count = solid_at_grid[end_cell].get('infill_crossings', 0)
                                                        if crossing_count > 1:
                                                            should_skip = True
                                                
                                                if should_skip:
                                                    # Skip extrusion (travel only)
                                                    if debug >= 2:
                                                        write_and_track(output_buffer,
                                                            f"G1 X{seg['x']:.3f} Y{seg['y']:.3f} Z{min(current_fill_z, layer_z):.3f} ; Skip fill (crossing)\n", recent_output_lines
                                                        )
                                                    else:
                                                        write_and_track(output_buffer,
                                                            f"G1 X{seg['x']:.3f} Y{seg['y']:.3f} Z{min(current_fill_z, layer_z):.3f}\n", recent_output_lines
                                                        )
                                                else:
                                                    # Extrude normally
                                                    valley_start_e += seg['e_delta'] * 0.5
                                                    write_and_track(output_buffer,
                                                        f"G1 X{seg['x']:.3f} Y{seg['y']:.3f} Z{min(current_fill_z, layer_z):.3f} E{(seg['e_delta'] * 0.5 if seg['relative'] else valley_start_e):.5f}\n", recent_output_lines
                                                    )
                                                prev_point = (seg['x'], seg['y'])
                                    
                                    # DECREMENT crossing count for each unique cell touched by this valley
                                    # This ensures next valley will have one less crossing to skip
                                    for cell_key in cells_touched_by_valley:
                                        if cell_key in solid_at_grid and solid_at_grid[cell_key].get('infill_crossings', 0) > 0:
                                            solid_at_grid[cell_key]['infill_crossings'] -= 1
                                    
                                    # Reset valley tracking
                                    in_valley = False
                                    valley_segments = []
                                    current_e = valley_start_e  # Sync current_e with valley fill
                                
                                # Update prev_z for next iteration
                                prev_z = z_mod
                                
                                # Output segment only if NOT in valley (valley segments are output during fill)
                                if not in_valley and not valley_exit:
                                    # Output with Z modulation, adjusted E, and adaptive feedrate
                                    if segment_feedrate is not None:
                                        write_and_track(output_buffer, 
                                            f"G1 X{sx:.3f} Y{sy:.3f} Z{z_mod:.3f} E{emitted_e(current_e, adjusted_e_for_segment, i):.5f} F{int(segment_feedrate)}\n", recent_output_lines
                                        )
                                    else:
                                        write_and_track(output_buffer, 
                                            f"G1 X{sx:.3f} Y{sy:.3f} Z{z_mod:.3f} E{emitted_e(current_e, adjusted_e_for_segment, i):.5f}\n", recent_output_lines
                                        )
                            
                            # CRITICAL: Update tracking positions to END of this move!
                            # Use the ACTUAL final position after all segments were output
                            # current_e might differ from e_end due to adaptive extrusion/valley filling
                            infill_current_x = x2
                            infill_current_y = y2
                            infill_current_e = current_e  # Actual E after segmented output
                            
                            i += 1
                            continue
                # Boost feedrate for standalone F commands (e.g., "G1 F3600")
                if current_line.startswith('G1') and 'F' in current_line and 'X' not in current_line and 'Y' not in current_line and 'E' not in current_line:
                    original_feedrate = extract_f(current_line)
                    if original_feedrate is not None:
                        boosted_feedrate = int(original_feedrate * nonplanar_feedrate_multiplier)
                        boosted_line = replace_f(current_line, boosted_feedrate)
                        write_and_track(output_buffer, boosted_line, recent_output_lines)
                        i += 1
                        continue
                
                # If we get here, preserve movement but rebase absolute E
                # after any added material from previous non-planar segments.
                if current_line.startswith(('G0 ', 'G1 ')) and extract_e(current_line) is not None:
                    if not source_relative_modes[i]:
                        current_line = replace_e(current_line, position['e'] + source_e_deltas[i])

                # If we get here, the line wasn't processed - append as-is  
                # Update position tracking for ANY unprocessed G1 line
                if current_line.startswith('G1') and i not in processed_infill_indices:
                    # Update X position if present
                    x_match = re.search(r'X([-+]?\d*\.?\d+)', current_line)
                    if x_match:
                        infill_current_x = float(x_match.group(1))
                    
                    # Update Y position if present
                    y_match = re.search(r'Y([-+]?\d*\.?\d+)', current_line)
                    if y_match:
                        infill_current_y = float(y_match.group(1))
                    
                    # Update E position if present
                    e_match = re.search(r'E([-+]?\d*\.?\d+)', current_line)
                    if e_match:
                        infill_current_e = float(e_match.group(1))
                
                write_and_track(output_buffer, current_line, recent_output_lines)
                infill_current_e = position['e']
                i += 1

            if in_valley:
                flush_pending_valley(min(i, len(lines) - 1))

            # Return to the slicer's original E coordinate before other
            # sections are passed through unchanged.
            if i > 0 and not source_relative_modes[i - 1]:
                write_and_track(output_buffer,
                    f"G92 E{source_e_targets[i - 1]:.5f} ; Non-planar E sync\n",
                    recent_output_lines)
            continue
        
        else:
            write_and_track(output_buffer, line, recent_output_lines)
            i += 1

    # Write the modified G-code
    print(f"\n[OK] Processed {current_layer} layers")
    
    # ========================================================================
    # FINAL PASS: Apply Z-hop to all travel moves
    # ========================================================================
    # This final pass processes the complete output G-code to insert Z-hop
    # (retract + lift) before travel moves and drop (lower + unretract) before
    # the next extrusion. This approach is cleaner than trying to inject Z-hop
    # logic during feature processing, which can interfere with carefully crafted
    # feature output (Smoothificator passes, Bricklayers Z-shifts, etc.).
    
    if enable_safe_z_hop:
        print("Applying Safe Z-hop to travel moves...")
        logging.info("\n" + "="*70)
        logging.info("FINAL PASS: Applying Safe Z-hop")
        logging.info("="*70)
        
        # Get the processed G-code lines
        modified_gcode = output_buffer.getvalue()
        output_buffer.close()
        gcode_lines = modified_gcode.splitlines(keepends=True)
        
        # State tracking for Z-hop pass
        zhop_current_layer = 0
        zhop_seen_first_layer = False
        zhop_current_z = 0.0
        zhop_working_z = 0.0  # Base Z for current layer (where extrusion happens)
        zhop_has_extruded_on_layer = False
        in_bridge = False
        in_wipe = False
        
        # Build final output with Z-hop insertions
        final_output = StringIO()
        
        # Simple state: are we currently hopped?
        is_hopped = False
        last_x, last_y = 0.0, 0.0
        
        # Statistics
        zhop_lift_count = 0
        zhop_drop_count = 0
        zhop_skipped_micro_travel = 0
        zhop_skipped_already_safe = 0
        
        for line_idx, line in enumerate(gcode_lines):
            # Track layer changes
            if ";LAYER_CHANGE" in line or ";LAYER:" in line:
                if ";LAYER:" in line:
                    layer_match = re.search(r';LAYER:(\d+)', line)
                    if layer_match:
                        zhop_current_layer = int(layer_match.group(1))
                else:
                    zhop_current_layer += 1
                
                zhop_seen_first_layer = True
                is_hopped = False  # Reset hop state on layer change
                zhop_has_extruded_on_layer = False
                final_output.write(line)
                continue
            
            # Track Z markers to update working_z
            if ";Z:" in line:
                z_marker_match = re.search(r';Z:([-\d.]+)', line)
                if z_marker_match:
                    zhop_current_z = float(z_marker_match.group(1))
                    zhop_working_z = zhop_current_z
                final_output.write(line)
                continue
            
            # Track bridge infill (affects lifting but we still allow dropping)
            if ";TYPE:" in line:
                if "Bridge infill" in line or "Internal bridge infill" in line:
                    in_bridge = True
                else:
                    in_bridge = False
                final_output.write(line)
                continue

            # Track WIPE sequences to suppress Z-hop lifts inside wipes
            if ";WIPE" in line:
                lu = line.upper()
                if "WIPE_START" in lu or "WIPE START" in lu:
                    in_wipe = True
                elif "WIPE_END" in lu or "WIPE END" in lu:
                    in_wipe = False
                # Always pass through wipe comments
                final_output.write(line)
                continue
            
            # Track standalone Z moves (update working Z)
            # Also handles G0 Z moves (e.g., from Smoothificator)
            if (line.startswith("G1") or line.startswith("G0")) and "Z" in line and "X" not in line and "Y" not in line and "E" not in line:
                z_match = re.search(r'Z([-\d.]+)', line)
                if z_match:
                    zhop_current_z = float(z_match.group(1))
                    zhop_working_z = zhop_current_z
                    is_hopped = False  # Explicit Z move = at working height, ready for extrusion
                final_output.write(line)
                continue
            
            # DETECT TRAVEL MOVES: G0 or (G1 with X/Y but NO E)
            # This is the KEY fix - don't try to track E values, just check if E parameter exists
            if zhop_seen_first_layer and (line.startswith("G0") or line.startswith("G1")):
                params = parse_gcode_line(line)
                has_xy = params['x'] is not None or params['y'] is not None
                has_e = params['e'] is not None
                has_z = params['z'] is not None
                
                # Preserve previous position for travel path sampling BEFORE updating
                prev_x, prev_y = last_x, last_y
                if params['x'] is not None:
                    last_x = params['x']
                if params['y'] is not None:
                    last_y = params['y']
                
                # A G1 X/Y/Z/E is a *simultaneous* motion: its Z is the END
                # height, not proof that a pending Z-hop has already dropped.
                # Keep the pre-hop printing height until we restore it below.
                z_extrusion_from_hop = is_hopped and has_xy and has_e
                if has_z and not z_extrusion_from_hop:
                    zhop_current_z = params['z']
                    zhop_working_z = params['z']
                    is_hopped = False  # An explicit non-extruding Z move sets the physical height
                
                # TRAVEL MOVE = has X/Y but NO E parameter (and no Z)
                is_travel = has_xy and not has_e and not has_z
                
                # EXTRUSION = has X/Y AND has E parameter
                is_extrusion = has_xy and has_e
                
                if is_travel and not is_hopped:
                    # Calculate safe Z for THIS SPECIFIC TRAVEL PATH
                    # Sample noise LUT along travel line to find maximum non-planar infill height
                    start_x, start_y = prev_x, prev_y
                    end_x = last_x
                    end_y = last_y
                    
                    # Calculate travel distance first
                    travel_dist = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
                    if travel_dist < 0.01:
                        # Ignore micro-travel; no hop needed
                        zhop_skipped_micro_travel += 1
                        final_output.write(line)
                        continue
                    
                    # Find maximum Z along the travel path by sampling noise LUT
                    path_max_z = 0.0
                    if 'grid_resolution' in locals() and 'noise_lut' in locals() and noise_lut is not None and 'amplitude' in locals():
                        # Cache layer base Z lookup
                        layer_base_z = z_layer_map.get(zhop_current_layer, zhop_working_z)
                        
                        # Sample points along the travel line
                        num_samples = max(5, int(travel_dist / grid_resolution) + 1)
                            
                        for i in range(num_samples):
                            t = i / max(1, num_samples - 1)
                            sample_x = start_x + t * (end_x - start_x)
                            sample_y = start_y + t * (end_y - start_y)
                            
                            # Calculate non-planar Z using helper function (no taper for travel path)
                            actual_z = calculate_nonplanar_z(noise_lut, sample_x, sample_y, layer_base_z, amplitude, taper_factor=1.0)
                            
                            # Track maximum Z encountered
                            path_max_z = max(path_max_z, actual_z)
                    
                    # Fallback to layer-wide max if LUT not available
                    if path_max_z == 0.0:
                        path_max_z = actual_layer_max_z.get(zhop_current_layer, layer_max_z.get(zhop_current_layer, 0.0))
                    
                    if path_max_z > 0 and not in_bridge and not in_wipe:  # never lift during bridge or wipe
                        safe_z = path_max_z + safe_z_hop_margin
                        # Only hop if difference is significant (> 0.1mm threshold)
                        hop_distance = safe_z - zhop_working_z
                        if hop_distance > 0.1:
                            final_output.write(f"G0 Z{safe_z:.3f} F8400 ; Z-hop lift\n")
                            is_hopped = True
                            zhop_lift_count += 1
                        else:
                            zhop_skipped_already_safe += 1
                    
                    # Write the travel line
                    final_output.write(line)
                    continue
                
                if is_extrusion and is_hopped:
                    # ALWAYS restore the pre-hop height *before* extrusion.
                    # In non-planar infill each segment carries its own Z target.
                    # Without this separate drop the nozzle extrudes diagonally
                    # from the elevated hop height, producing vertical spikes.
                    final_output.write(f"G0 Z{zhop_working_z:.3f} F8400 ; Z-hop drop\n")
                    zhop_drop_count += 1
                    is_hopped = False
                    zhop_has_extruded_on_layer = True
                    # Only after the drop may a Z-bearing extrusion update
                    # the printing-height tracker for subsequent travels.
                    if has_z:
                        zhop_current_z = params['z']
                        zhop_working_z = params['z']
                    final_output.write(line)
                    continue
                
                if is_extrusion:
                    zhop_has_extruded_on_layer = True
                
                # All other G0/G1: pass through
                final_output.write(line)
                continue
            
            # All other lines: pass through
            final_output.write(line)
        
        # Get the final output with Z-hop applied
        modified_gcode = final_output.getvalue()
        final_output.close()
        logging.info(f"Z-hop pass complete: {zhop_lift_count} lifts, {zhop_drop_count} drops")
        logging.info(f"  Skipped: {zhop_skipped_micro_travel} micro-travels, {zhop_skipped_already_safe} already safe")
    else:
        # Z-hop disabled, use output as-is
        modified_gcode = output_buffer.getvalue()
        output_buffer.close()
    
    modified_gcode = restore_extrusion_feedrates(modified_gcode)
    print(f"Writing modified G-code to: {os.path.basename(output_file)}...")
    
    # Write to file
    with open(output_file, 'w') as outfile:
        outfile.write(modified_gcode)

    logging.info("\n" + "="*70)
    logging.info("G-code processing completed successfully")
    logging.info("="*70)
    # Diagnostic summary for reclassified bridge TYPE comments (only logged when debug enabled)
    try:
        if debug >= 1:
            logging.info(f"Bridge TYPE comments reclassified: {reclassified_bridge_count}")
    except NameError:
        # If debug or counter not defined (shouldn't happen), skip
        pass
    
    if enable_bricklayers:
        logging.info("Bricklayers: %d contours skipped because no matching internal perimeter above",
                     bricklayers_unstackable_count)

    # Print summary to console
    print("\n" + "=" * 85)
    print("  [OK] SILKSTEEL POST-PROCESSING COMPLETE")
    print("=" * 85)
    print(f"  Total layers: {current_layer}")
    print(f"  Output size: {len(modified_gcode)} bytes")
    print(f"  Output file: {output_file}")
    print("=" * 85 + "\n")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SilkSteel - Advanced G-code Post-Processor\n'
                    '"Smooth on the outside, strong on the inside"\n\n'
                    'Smoothificator: Splits external perimeters into multiple thin passes for silk-smooth surfaces.\n'
                    'Bricklayers: Offsets alternating internal perimeters for steel-strong layer bonding.\n'
                    'Non-planar Infill: Modulates Z during infill for improved layer adhesion.\n'
                    'Safe Z-hop: Lifts nozzle to safe height before travel moves to prevent collisions.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('input_file', help='Input G-code file')
    parser.add_argument('-o', '--output', dest='output_file', 
                       help='Output G-code file. If not specified, modifies input file IN-PLACE (required for slicer usage). '
                            'Use -o for manual testing to preserve the original file.')
    
    parser.add_argument('-full', '--enable-all', action='store_true', default=False,
                       help='Enable ALL features: Bricklayers and Non-planar Infill. '
                            'Smoothificator and Safe Z-hop are already enabled by default. '
                            'Individual feature flags can still override this setting.')
    
    parser.add_argument('-outerLayerHeight', '--outer-layer-height', type=parse_outer_layer_height, default=DEFAULT_OUTER_LAYER_HEIGHT,
                       help='Outer wall height: "Auto" (min of first/base layer * 0.5), "Min" (uses min_layer_height), or float value in mm (default: Auto)')
    
    # Feature toggles
    parser.add_argument('-enableSmoothificator', '--enable-smoothificator', action='store_true', default=True,
                       help='Enable Smoothificator (external perimeter smoothing) (default: enabled)')
    parser.add_argument('-disableSmoothificator', '--disable-smoothificator', action='store_false', dest='enable_smoothificator',
                       help='Disable Smoothificator functionality')
    parser.add_argument('-smoothificatorSkipFirstLayer', '--smoothificator-skip-first-layer', 
                       action='store_true', default=True, dest='smoothificator_skip_first_layer',
                       help='Skip first layer in smoothificator (default: enabled - preserves first layer tuning)')
    parser.add_argument('-smoothificatorProcessFirstLayer', '--smoothificator-process-first-layer',
                       action='store_false', dest='smoothificator_skip_first_layer',
                       help='Process first layer with smoothificator')
    
    parser.add_argument('-enableBricklayers', '--enable-bricklayers', action='store_const', const=True, dest='enable_bricklayers', default=None,
                       help='Enable Bricklayers Z-shifting (default: disabled, enabled with -full)')
    parser.add_argument('-disableBricklayers', '--disable-bricklayers', action='store_const', const=False, dest='enable_bricklayers',
                       help='Disable Bricklayers (overrides -full)')
    parser.add_argument('-bricklayersExtrusion', '--bricklayers-extrusion', type=float, default=1.0,
                       help='Extrusion multiplier for Bricklayers shifted blocks (default: 1.0)')
    
    parser.add_argument('-enableNonPlanar', '--enable-non-planar', action='store_const', const=True, dest='enable_non_planar', default=None,
                       help='Enable non-planar infill modulation (default: disabled, enabled with -full)')
    parser.add_argument('-disableNonPlanar', '--disable-non-planar', action='store_const', const=False, dest='enable_non_planar',
                       help='Disable non-planar infill (overrides -full)')
    parser.add_argument('-deformType', '--deform-type', type=str, default='sine', choices=['sine', 'noise'],
                       help='Type of deformation pattern: sine (smooth waves) or noise (Perlin noise) (default: sine)')
    parser.add_argument('-segmentLength', '--segment-length', type=float, default=DEFAULT_SEGMENT_LENGTH,
                       help=f'Length of subdivided segments for non-planar infill in mm (default: {DEFAULT_SEGMENT_LENGTH})')
    parser.add_argument('-nonplanarFeedrateMultiplier', '--nonplanar-feedrate-multiplier', type=float, default=DEFAULT_NONPLANAR_FEEDRATE_MULTIPLIER,
                       help=f'Feedrate multiplier for non-planar infill to compensate for 3D motion planning overhead (default: {DEFAULT_NONPLANAR_FEEDRATE_MULTIPLIER})')
    parser.add_argument('-amplitude', '--amplitude', type=float, default=DEFAULT_AMPLITUDE,
                       help=f'Amplitude of Z modulation for non-planar infill in mm when float, layers when integer (default: {DEFAULT_AMPLITUDE})')
    parser.add_argument('-frequency', '--frequency', type=float, default=DEFAULT_FREQUENCY,
                       help=f'Frequency of Z modulation for non-planar infill (default: {DEFAULT_FREQUENCY})')
    parser.add_argument('-disableAdaptiveExtrusion', '--disable-adaptive-extrusion', action='store_false', dest='enable_adaptive_extrusion',
                       help='Disable adaptive extrusion multiplier for Z-lift (adds extra material when lifting to bond with layer below, default: enabled)')
    parser.add_argument('-adaptiveExtrusionMultiplier', '--adaptive-extrusion-multiplier', type=float, default=DEFAULT_ADAPTIVE_EXTRUSION_MULTIPLIER,
                       help=f'Extrusion multiplier for adaptive extrusion per layer height of Z-lift (default: {DEFAULT_ADAPTIVE_EXTRUSION_MULTIPLIER}x, try 1.0-2.0)')
    
    parser.add_argument('-disableSafeZHop', '--disable-safe-z-hop', action='store_false', dest='enable_safe_z_hop',
                       help='Disable safe Z-hop during travel moves (default: enabled)')
    parser.add_argument('-safeZHopMargin', '--safe-z-hop-margin', type=float, default=DEFAULT_SAFE_Z_HOP_MARGIN,
                       help=f'Safety margin in mm to add above max Z during travel (default: {DEFAULT_SAFE_Z_HOP_MARGIN})')
    
    parser.add_argument('-enableBridgeDensifier', '--enable-bridge-densifier', action='store_const', const=True, dest='enable_bridge_densifier', default=None,
                       help='Enable Bridge Densifier to add intermediate lines between bridge extrusions for better bridging (experimental, requires explicit opt-in; not enabled by -full)')
    parser.add_argument('-disableBridgeDensifier', '--disable-bridge-densifier', action='store_const', const=False, dest='enable_bridge_densifier',
                       help='Disable Bridge Densifier (overrides -full)')
    
    parser.add_argument('-enableRemoveGapFill', '--enable-remove-gap-fill', action='store_const', const=True, dest='enable_remove_gap_fill', default=None,
                       help='Enable gap fill removal (default: disabled, enabled with -full)')
    parser.add_argument('-disableRemoveGapFill', '--disable-remove-gap-fill', action='store_const', const=False, dest='enable_remove_gap_fill',
                       help='Disable gap fill removal (overrides -full)')
    
    # Debug mode arguments
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument('-debug', '--debug', dest='debug_level', action='store_const', const=1, default=0,
                       help='Enable basic debug mode with standard logging (INFO level)')
    debug_group.add_argument('-debug-full', '--debug-full', dest='debug_level', action='store_const', const=2,
                       help='Enable full debug mode: INFO logging + PNG layer images + debug G-code visualization')
    
    args = parser.parse_args()
    
    # Set logging level based on debug argument
    debug = args.debug_level
    if debug >= 1:
        logging.getLogger().setLevel(logging.INFO)
    
    # Log all received arguments for debugging
    if debug >= 1:
        logging.info("Debug mode enabled - log level set to INFO")
        logging.info("=" * 85)
        logging.info("Command-line arguments received:")
        logging.info(f"  Raw sys.argv: {sys.argv}")
        logging.info(f"  Parsed input_file: {args.input_file}")
    logging.info(f"  Parsed output_file: {args.output_file}")
    logging.info(f"  Enable all: {args.enable_all}")
    logging.info(f"  Enable bricklayers: {args.enable_bricklayers}")
    logging.info(f"  Enable non-planar: {args.enable_non_planar}")
    logging.info("=" * 85)
    
    # Validate that we have an input file
    if not args.input_file:
        logging.error("ERROR: No input file provided!")
        sys.exit(1)
    
    logging.info(f"Starting processing of: {args.input_file}")
    
    # Handle -full flag: enable all optional features
    # Individual feature flags specified after -full can still override
    if args.enable_all:
        logging.info("Full mode enabled - activating all features")
        # Only enable features if not explicitly set by user (None = not specified)
        if args.enable_bricklayers is None:
            args.enable_bricklayers = True
        if args.enable_non_planar is None:
            args.enable_non_planar = True
        # Experimental bridge reconstruction has unresolved E-mode bugs.
        # Require an explicit opt-in rather than enabling it in -full.
        if args.enable_bridge_densifier is None:
            args.enable_bridge_densifier = False
        # Gap fill removal is too buggy, don't enable it with -full
        # Smoothificator and Safe Z-hop are already enabled by default
    
    # Convert None to False for features that default to disabled (if user never specified them)
    if args.enable_bricklayers is None:
        args.enable_bricklayers = False
    if args.enable_non_planar is None:
        args.enable_non_planar = False
    if args.enable_bridge_densifier is None:
        args.enable_bridge_densifier = False
    if args.enable_remove_gap_fill is None:
        args.enable_remove_gap_fill = False
    
    try:
        logging.info("Calling process_gcode()...")
        process_gcode(
            input_file=args.input_file,
            output_file=args.output_file,
            outer_layer_height=args.outer_layer_height,
            enable_smoothificator=args.enable_smoothificator,
            smoothificator_skip_first_layer=args.smoothificator_skip_first_layer,
            enable_bricklayers=args.enable_bricklayers,
            bricklayers_extrusion_multiplier=args.bricklayers_extrusion,
            enable_nonplanar=args.enable_non_planar,
            deform_type=args.deform_type,
            segment_length=args.segment_length,
            nonplanar_feedrate_multiplier=args.nonplanar_feedrate_multiplier,
            enable_adaptive_extrusion=args.enable_adaptive_extrusion,
            adaptive_extrusion_multiplier=args.adaptive_extrusion_multiplier,
            amplitude=args.amplitude,
            frequency=args.frequency,
            enable_safe_z_hop=args.enable_safe_z_hop,
            safe_z_hop_margin=args.safe_z_hop_margin,
            enable_bridge_densifier=args.enable_bridge_densifier,
            remove_gap_fill=args.enable_remove_gap_fill,
            debug=debug
        )
        
    except Exception as e:
        logging.error(f"\n{'='*70}")
        logging.error(f"FATAL ERROR: {str(e)}")
        logging.error(f"{'='*70}")
        import traceback
        logging.error(traceback.format_exc())
        
        # Print error to console
        print("\n" + "=" * 85, file=sys.stderr)
        print("  ✗ ERROR: POST-PROCESSING FAILED", file=sys.stderr)
        print("=" * 85, file=sys.stderr)
        print(f"  {str(e)}", file=sys.stderr)
        print(f"\n  📄 Check the log file for details: {log_file}", file=sys.stderr)
        print("=" * 85, file=sys.stderr)
        # Never block slicer post-processing, even when stdin is a console.
        # The traceback is in SilkSteel_log.txt and exit 2 signals failure.
        sys.exit(2)
    
    # Check for warnings/errors and pause if any occurred (after successful completion)
    if _warning_count > 0 or _error_count > 0:
        print("\n" + "=" * 85)
        print("  ⚠️  PROCESSING COMPLETED WITH ISSUES")
        print("=" * 85)
        if _error_count > 0:
            print(f"  ✗ Errors: {_error_count}")
        if _warning_count > 0:
            print(f"  ⚠️  Warnings: {_warning_count}")
        print(f"\n  📄 Check the log file for details: {log_file}")
        print("=" * 85)
        # Warnings are non-fatal; leave a console/log summary and exit normally.

