
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
    # Try to auto-install Pillow if not present
    print("PIL/Pillow not found. Attempting to install...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "--quiet"])
        print("✓ Pillow installed successfully!")
        # Try importing again
        from PIL import Image, ImageDraw
        HAS_PIL = True
    except Exception as e:
        print(f"⚠ Could not auto-install Pillow: {e}")
        print(f"  Debug PNG generation will be disabled.")
        print(f"  To enable, manually install with: pip install Pillow")
        pass  # PIL is optional - only needed for debug PNG generation

    # Counters for diagnostics
    # Incremented when we reclassify bridge TYPE comments during grid building
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
REGEX_F = re.compile(r'F(\d+)')
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
    return int(match.group(1)) if match else None

def replace_e(line, new_e):
    """Replace E value in G-code line"""
    return REGEX_E_SUB.sub(f'E{new_e:.5f}', line)

def replace_f(line, new_f):
    """Replace F (feedrate) value in G-code line"""
    return re.sub(r'F\d+\.?\d*', f'F{new_f}', line)

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
        result['f'] = int(f_match.group(1))
    
    return result

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
    num_segments = max(1, int(total_length // segment_length))
    
    for i in range(num_segments + 1):
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

def process_bridge_section(buffered_lines, current_z, current_e, start_x, start_y, connector_max_length, logging, debug=False, bridge_feedrate_slowdown=0.6):
    """
    Process a buffered bridge section and densify it by inserting intermediate lines between parallel bridges.
    
    Algorithm:
    1. Parse all extrusion moves, filter by length (keep only long bridge lines)
    2. For each consecutive pair of long lines, check if parallel
    3. Calculate perpendicular spacing between them
    4. If spacing < threshold, insert ONE intermediate line between them
    5. Use average length and extrusion rate for intermediate
    
    Args:
        buffered_lines: List of G-code lines from the bridge section
        current_z: Current Z height
        current_e: Current E position at start of bridge
        start_x: Current X position at start of bridge (BEFORE first extrusion)
        start_y: Current Y position at start of bridge (BEFORE first extrusion)
        connector_max_length: Maximum length of connectors to skip (typically 2× extrusion width)
        logging: Logger instance
        debug: If True, add inline comments to output lines
        bridge_feedrate_slowdown: Factor to slow down bridge extrusion (0.6 = 60% of original speed)
    
    Returns:
        Tuple of (processed_lines, final_e, final_pos) where processed_lines is the densified G-code, 
        final_e is the updated E position, and final_pos is (x, y) tuple of final position
    """
    
    # Extract feedrates from buffered lines
    extrusion_feedrate = None
    travel_feedrate = None
    
    for line in buffered_lines:
        if line.startswith("G1") and "F" in line:
            f_val = re.search(r'F(\d+\.?\d*)', line)
            if f_val:
                feedrate = float(f_val.group(1))
                if "E" in line and ("X" in line or "Y" in line):
                    # Extrusion move with feedrate
                    if extrusion_feedrate is None:
                        extrusion_feedrate = feedrate
                elif "E" not in line and ("X" in line or "Y" in line):
                    # Travel move with feedrate
                    if travel_feedrate is None:
                        travel_feedrate = feedrate
        elif line.startswith("G0") and "F" in line:
            f_val = re.search(r'F(\d+\.?\d*)', line)
            if f_val and travel_feedrate is None:
                travel_feedrate = float(f_val.group(1))
    
    # Apply slowdown to extrusion feedrate, use defaults if not found
    if extrusion_feedrate is not None:
        bridge_extrusion_feedrate = int(extrusion_feedrate * bridge_feedrate_slowdown)
    else:
        bridge_extrusion_feedrate = int(1800 * bridge_feedrate_slowdown)  # Fallback: 1800mm/min
    
    if travel_feedrate is None:
        travel_feedrate = 8400  # Fallback travel speed
    
    # Helper function to conditionally add comments based on debug mode
    def comment_if_debug(gcode_line, comment):
        """Add inline comment only if debug mode is enabled."""
        if debug:
            return add_inline_comment(gcode_line, comment)
        else:
            return gcode_line
    
    # Track E mode to avoid redundant M82/M83 commands
    # Assume we start in absolute mode (M82)
    current_e_mode = 'absolute'
    
    def set_e_mode(target_mode):
        """Helper to output E mode changes only when needed."""
        nonlocal current_e_mode
        if current_e_mode != target_mode:
            if target_mode == 'absolute':
                output.append(f"M82 ; Absolute E mode\n")
                # logging.debug(f"[BRIDGE] Switched to ABSOLUTE E mode")
            else:
                output.append(f"M83 ; Relative E mode\n")
                # logging.debug(f"[BRIDGE] Switched to RELATIVE E mode")
            current_e_mode = target_mode
    
    # Step 1: Parse all extrusion moves
    # IMPORTANT: prev_x and prev_y start at the position BEFORE the bridge section started!
    moves = []
    prev_x, prev_y, prev_e = start_x, start_y, current_e
    
    for i, line in enumerate(buffered_lines):
        if line.startswith("G1") and "X" in line and "Y" in line:
            x = extract_x(line)
            y = extract_y(line)
            e = extract_e(line)
            
            if x is not None and y is not None and prev_x is not None and prev_y is not None:
                dx = x - prev_x
                dy = y - prev_y
                length = math.sqrt(dx*dx + dy*dy)
                
                e_delta = 0.0
                if e is not None:
                    e_delta = e - prev_e
                
                # Only keep extrusion moves (positive E)
                if e_delta > 0:
                    moves.append({
                        'x1': prev_x, 'y1': prev_y,
                        'x2': x, 'y2': y,
                        'e_start': prev_e,
                        'e_end': e,
                        'e_delta': e_delta,
                        'length': length,
                        'line_index': i,
                        'dx': dx,
                        'dy': dy
                    })
                
                if e is not None:
                    prev_e = e
            
            if x is not None:
                prev_x = x
            if y is not None:
                prev_y = y
    
    if len(moves) < 2:
        # Single move - create edges on both sides and output as serpentine
        if len(moves) == 1:
            move = moves[0]
            
            # Calculate perpendicular vector
            perp_x = -move['dy'] / move['length']
            perp_y = move['dx'] / move['length']
            
            # Use extrusion width (0.4mm) as spacing
            spacing = 0.4
            offset = spacing / 2.0
            
            # Create three parallel lines: LEFT, MIDDLE (original), RIGHT
            left_start = (move['x1'] - perp_x * offset, move['y1'] - perp_y * offset)
            left_end = (move['x2'] - perp_x * offset, move['y2'] - perp_y * offset)
            
            right_start = (move['x1'] + perp_x * offset, move['y1'] + perp_y * offset)
            right_end = (move['x2'] + perp_x * offset, move['y2'] + perp_y * offset)
            
            # Calculate E values for edges (same extrusion rate as original)
            e_per_mm = move['e_delta'] / move['length']
            edge_e_delta = move['length'] * e_per_mm
            
            # Build output with serpentine pattern: LEFT → MIDDLE → RIGHT
            output = []
            
            # Find first G1 line in buffered_lines
            first_g1_index = 0
            for idx, line in enumerate(buffered_lines):
                if line.startswith("G1") and "X" in line and "Y" in line:
                    first_g1_index = idx
                    break
            
            # Output metadata lines (TYPE, WIDTH, HEIGHT, etc.)
            for i in range(first_g1_index):
                output.append(buffered_lines[i])
            
            # Start from initial position (start_x, start_y)
            current_pos = (start_x, start_y)
            
            # Calculate distances to LEFT endpoints
            dist_to_left_start = math.sqrt((left_start[0] - current_pos[0])**2 + (left_start[1] - current_pos[1])**2)
            dist_to_left_end = math.sqrt((left_end[0] - current_pos[0])**2 + (left_end[1] - current_pos[1])**2)
            
            if debug:
                output.append(f"; [Bridge Densifier] Single line densification (3 parallel lines, spacing={spacing:.3f}mm)\n")
            set_e_mode('relative')
            
            # Draw LEFT edge (choose closest endpoint)
            if dist_to_left_start < dist_to_left_end:
                if dist_to_left_start > 0.001:
                    output.append(comment_if_debug(
                        f"G0 X{left_start[0]:.3f} Y{left_start[1]:.3f} F{int(travel_feedrate)}\n",
                        f"Connector to LEFT edge {dist_to_left_start:.2f}mm"
                    ))
                output.append(comment_if_debug(
                    f"G1 X{left_end[0]:.3f} Y{left_end[1]:.3f} E{edge_e_delta:.5f} F{bridge_extrusion_feedrate}\n",
                    f"LEFT edge fwd, offset={offset:.2f}mm"
                ))
                current_pos = left_end
            else:
                if dist_to_left_end > 0.001:
                    output.append(comment_if_debug(
                        f"G0 X{left_end[0]:.3f} Y{left_end[1]:.3f} F{int(travel_feedrate)}\n",
                        f"Connector to LEFT edge {dist_to_left_end:.2f}mm"
                    ))
                output.append(comment_if_debug(
                    f"G1 X{left_start[0]:.3f} Y{left_start[1]:.3f} E{edge_e_delta:.5f} F{bridge_extrusion_feedrate}\n",
                    f"LEFT edge rev, offset={offset:.2f}mm"
                ))
                current_pos = left_start
            
            # Draw MIDDLE (original line) - choose closest endpoint
            dist_to_mid_start = math.sqrt((move['x1'] - current_pos[0])**2 + (move['y1'] - current_pos[1])**2)
            dist_to_mid_end = math.sqrt((move['x2'] - current_pos[0])**2 + (move['y2'] - current_pos[1])**2)
            
            if dist_to_mid_start < dist_to_mid_end:
                if dist_to_mid_start > 0.001:
                    output.append(add_inline_comment(
                        f"G0 X{move['x1']:.3f} Y{move['y1']:.3f} F{int(travel_feedrate)}\n",
                        f"Connector to MIDDLE {dist_to_mid_start:.2f}mm"
                    ))
                output.append(add_inline_comment(
                    f"G1 X{move['x2']:.3f} Y{move['y2']:.3f} E{move['e_delta']:.5f} F{bridge_extrusion_feedrate}\n",
                    f"MIDDLE (original) fwd"
                ))
                current_pos = (move['x2'], move['y2'])
            else:
                if dist_to_mid_end > 0.001:
                    output.append(add_inline_comment(
                        f"G0 X{move['x2']:.3f} Y{move['y2']:.3f} F{int(travel_feedrate)}\n",
                        f"Connector to MIDDLE {dist_to_mid_end:.2f}mm"
                    ))
                output.append(add_inline_comment(
                    f"G1 X{move['x1']:.3f} Y{move['y1']:.3f} E{move['e_delta']:.5f} F{bridge_extrusion_feedrate}\n",
                    f"MIDDLE (original) rev"
                ))
                current_pos = (move['x1'], move['y1'])
            
            # Draw RIGHT edge - choose closest endpoint
            dist_to_right_start = math.sqrt((right_start[0] - current_pos[0])**2 + (right_start[1] - current_pos[1])**2)
            dist_to_right_end = math.sqrt((right_end[0] - current_pos[0])**2 + (right_end[1] - current_pos[1])**2)
            
            if dist_to_right_start < dist_to_right_end:
                if dist_to_right_start > 0.001:
                    output.append(add_inline_comment(
                        f"G0 X{right_start[0]:.3f} Y{right_start[1]:.3f} F{int(travel_feedrate)}\n",
                        f"Connector to RIGHT edge {dist_to_right_start:.2f}mm"
                    ))
                output.append(add_inline_comment(
                    f"G1 X{right_end[0]:.3f} Y{right_end[1]:.3f} E{edge_e_delta:.5f} F{bridge_extrusion_feedrate}\n",
                    f"RIGHT edge fwd, offset={offset:.2f}mm"
                ))
                current_pos = right_end
            else:
                if dist_to_right_end > 0.001:
                    output.append(add_inline_comment(
                        f"G0 X{right_end[0]:.3f} Y{right_end[1]:.3f} F{int(travel_feedrate)}\n",
                        f"Connector to RIGHT edge {dist_to_right_end:.2f}mm"
                    ))
                output.append(add_inline_comment(
                    f"G1 X{right_start[0]:.3f} Y{right_start[1]:.3f} E{edge_e_delta:.5f} F{bridge_extrusion_feedrate}\n",
                    f"RIGHT edge rev, offset={offset:.2f}mm"
                ))
                current_pos = right_start
            
            # Switch back to absolute mode at the end
            set_e_mode('absolute')
            
            # Calculate final E (original E + 2 edges)
            final_e = move['e_end'] + (2 * edge_e_delta)
            
            # if debug >= 2:
            #     logging.info(f"[BRIDGE] Single-move densification complete: E={current_e:.5f} -> {final_e:.5f} (added {2*edge_e_delta:.5f})")
            
            return output, final_e, current_pos
        else:
            # Zero moves - just return buffered lines
            # This typically happens when buffer contains only travel moves, retracts, or metadata
            type_line = next((line for line in buffered_lines if ";TYPE:" in line), None)
            type_str = type_line.strip() if type_line else "unknown"
            
            # Count actual extrusion moves in buffer
            extrusion_lines = [line for line in buffered_lines 
                             if line.startswith("G1") and "E" in line and "X" in line and "Y" in line]
            
            if len(extrusion_lines) == 0:
                # This is normal - buffer only has travel moves or retracts, not actual bridge extrusion
                logging.debug(f"[BRIDGE] Skipping non-extrusion buffer at Z={current_z:.2f}mm ({len(buffered_lines)} lines)")
            elif len(extrusion_lines) == 1:
                # Single extrusion move - this is a tail segment, just pass through
                # This happens at the end of bridge sequences after main processing
                logging.debug(f"[BRIDGE] Skipping single-move buffer at Z={current_z:.2f}mm (tail segment)")
            else:
                # Multiple extrusion moves but couldn't parse them - this IS unexpected
                logging.warning(f"[BRIDGE] Could not parse {len(extrusion_lines)} extrusion moves at Z={current_z:.2f}mm")
                logging.warning(f"[BRIDGE]   TYPE: {type_str}, buffer size: {len(buffered_lines)} lines")
                # Log first few lines for debugging
                for idx, line in enumerate(buffered_lines[:5]):
                    logging.warning(f"[BRIDGE]   Line {idx+1}: {line.rstrip()}")
                if len(buffered_lines) > 5:
                    logging.warning(f"[BRIDGE]   ... ({len(buffered_lines) - 5} more lines)")
            
            final_e = current_e
            return buffered_lines, final_e, (start_x, start_y)
    
    # Step 2: Filter by length - keep only long bridge lines (> minimum length threshold)
    long_moves = [m for m in moves if m['length'] >= DEFAULT_BRIDGE_MIN_LENGTH]
    
    # Even if there are no "long" lines, we still want to process short bridge sections
    # So don't return early - continue with densification logic
    
    # Step 3: Mark each move with whether it's a "long" line and calculate intermediates
    # We'll build a list of what to insert after each line
    for move in moves:
        move['is_long'] = move['length'] >= DEFAULT_BRIDGE_MIN_LENGTH
        move['intermediate_after'] = None  # Will store intermediate data if needed
        move['is_connector_to_skip'] = False  # Mark short connectors between parallel lines
    
    # Step 4: Find pairs of parallel long lines and mark intermediates
    long_move_indices = [i for i, m in enumerate(moves) if m['is_long']]
    
    for i, long_idx in enumerate(long_move_indices):
        long_move = moves[long_idx]
        
        # Look ahead for parallel lines (within gap limit)
        for lookahead in range(1, min(DEFAULT_BRIDGE_MAX_GAP + 1, len(long_move_indices) - i)):
            next_long_idx = long_move_indices[i + lookahead]
            next_move = moves[next_long_idx]
            
            # Skip if we already created an intermediate for this long_move
            if long_move['intermediate_after'] is not None:
                break
            
            # Check if there are ANY moves between these two long lines
            # If the indices are consecutive in the moves array AND close in space,
            # they're definitely part of the same pattern (even if no explicit connector)
            moves_between = next_long_idx - long_idx - 1
            
            # If they're far apart in the buffer (many moves between), require a connector
            # This prevents pairing lines from different bridge sections that overlap spatially
            if moves_between > 3:
                # Check if there's at least ONE short connector between them
                has_connector = False
                for conn_idx in range(long_idx + 1, next_long_idx):
                    if moves[conn_idx]['length'] <= connector_max_length:
                        has_connector = True
                        break
                
                # If no connector found and many moves apart, skip (separate sections)
                if not has_connector:
                    continue
            
            # Check parallel
            len1 = long_move['length']
            len2 = next_move['length']
            
            if len1 > 0.01 and len2 > 0.01:
                dot_product = (long_move['dx'] * next_move['dx'] + long_move['dy'] * next_move['dy']) / (len1 * len2)
                
                if abs(dot_product) > 0.8:  # Parallel
                    # Calculate spacing
                    perp_x = -long_move['dy'] / len1
                    perp_y = long_move['dx'] / len1
                    
                    vec_x = next_move['x1'] - long_move['x1']
                    vec_y = next_move['y1'] - long_move['y1']
                    
                    spacing = abs(vec_x * perp_x + vec_y * perp_y)
                    
                    if spacing < DEFAULT_BRIDGE_MAX_SPACING:
                        # Mark short connectors between THESE parallel lines to skip (we'll create optimized connections)
                        # Only skip connectors that are very short (about 2× extrusion width)
                        for conn_idx in range(long_idx + 1, next_long_idx):
                            conn_move = moves[conn_idx]
                            if conn_move['length'] <= connector_max_length:
                                conn_move['is_connector_to_skip'] = True
                        
                        # Calculate intermediate
                        dist1_start = math.sqrt((long_move['x2'] - (long_move['x2'] + next_move['x1'])/2.0)**2 + 
                                               (long_move['y2'] - (long_move['y2'] + next_move['y1'])/2.0)**2)
                        dist2_start = math.sqrt((long_move['x2'] - (long_move['x1'] + next_move['x1'])/2.0)**2 + 
                                               (long_move['y2'] - (long_move['y1'] + next_move['y1'])/2.0)**2)
                        
                        if dist1_start <= dist2_start:
                            inter_x1 = (long_move['x2'] + next_move['x1']) / 2.0
                            inter_y1 = (long_move['y2'] + next_move['y1']) / 2.0
                            inter_x2 = (long_move['x1'] + next_move['x2']) / 2.0
                            inter_y2 = (long_move['y1'] + next_move['y2']) / 2.0
                        else:
                            inter_x1 = (long_move['x1'] + next_move['x1']) / 2.0
                            inter_y1 = (long_move['y1'] + next_move['y1']) / 2.0
                            inter_x2 = (long_move['x2'] + next_move['x2']) / 2.0
                            inter_y2 = (long_move['y2'] + next_move['y2']) / 2.0
                        
                        inter_dx = inter_x2 - inter_x1
                        inter_dy = inter_y2 - inter_y1
                        inter_length = math.sqrt(inter_dx*inter_dx + inter_dy*inter_dy)
                        
                        # Calculate E with compensation: bridge extrusions are round (~nozzle diameter)
                        # not squished (extrusion width), so intermediate only fills gap
                        e_per_mm = (long_move['e_delta']/len1 + next_move['e_delta']/len2) / 2.0
                        inter_e_delta = inter_length * e_per_mm * DEFAULT_BRIDGE_EXTRUSION_COMPENSATION
                        
                        # Store intermediate to insert AFTER long_move
                        long_move['intermediate_after'] = {
                            'start': (inter_x1, inter_y1),
                            'end': (inter_x2, inter_y2),
                            'e_delta': inter_e_delta,
                            'spacing': spacing,
                            'length': inter_length,
                            'next_start': (next_move['x1'], next_move['y1']),
                            'paired_idx': next_long_idx  # Remember which line we paired with
                        }
                        
                        # Mark the NEXT line to know it was paired with a previous line
                        if 'paired_with_prev' not in next_move:
                            next_move['paired_with_prev'] = long_idx
                        
                        # Found a pair, stop looking
                        break
    
    # Step 4.5: Handle single unpaired long lines - add edge intermediates on BOTH sides
    for long_idx in long_move_indices:
        long_move = moves[long_idx]
        
        # Check if this line is unpaired (no intermediate after it, and not paired with previous)
        is_unpaired = (long_move.get('intermediate_after') is None and 
                      long_move.get('paired_with_prev') is None)
        
        if is_unpaired:
            # Unpaired line - add edges on both sides
            
            # Use the default spacing (half the max spacing threshold)
            spacing = DEFAULT_BRIDGE_MAX_SPACING / 2.0
            
            # Calculate perpendicular vector
            perp_x = -long_move['dy'] / long_move['length']
            perp_y = long_move['dx'] / long_move['length']
            
            # Create edge intermediates on BOTH sides
            e_per_mm = long_move['e_delta'] / long_move['length']
            edge_e_delta = long_move['length'] * e_per_mm
            
            # Store both edges in a special field
            long_move['single_line_edges'] = {
                'left': {
                    'start': (long_move['x1'] + perp_x * (spacing / 2.0), long_move['y1'] + perp_y * (spacing / 2.0)),
                    'end': (long_move['x2'] + perp_x * (spacing / 2.0), long_move['y2'] + perp_y * (spacing / 2.0)),
                    'e_delta': edge_e_delta,
                    'spacing': spacing / 2.0
                },
                'right': {
                    'start': (long_move['x1'] - perp_x * (spacing / 2.0), long_move['y1'] - perp_y * (spacing / 2.0)),
                    'end': (long_move['x2'] - perp_x * (spacing / 2.0), long_move['y2'] - perp_y * (spacing / 2.0)),
                    'e_delta': edge_e_delta,
                    'spacing': spacing / 2.0
                }
            }
            
            # Created edge intermediates on both sides
    
    # Special case: If there are NO long lines but there IS at least one move, 
    # treat the longest move as a "single line" and add edges to it
    if len(long_move_indices) == 0 and len(moves) > 0:
        if debug >= 2:
            logging.info(f"[BRIDGE] No long lines found, but {len(moves)} moves exist - finding longest move")
        
        # Find the longest move
        longest_move = max(moves, key=lambda m: m['length'])
        longest_idx = moves.index(longest_move)
        
        if longest_move['length'] > 1.0:  # At least 1mm to be worth densifying
            if debug >= 2:
                logging.info(f"[BRIDGE] Treating longest move as single line: index {longest_idx}, length {longest_move['length']:.2f}mm")
            
            spacing = DEFAULT_BRIDGE_MAX_SPACING / 2.0
            perp_x = -longest_move['dy'] / longest_move['length']
            perp_y = longest_move['dx'] / longest_move['length']
            
            e_per_mm = longest_move['e_delta'] / longest_move['length']
            edge_e_delta = longest_move['length'] * e_per_mm
            
            longest_move['single_line_edges'] = {
                'left': {
                    'start': (longest_move['x1'] + perp_x * (spacing / 2.0), longest_move['y1'] + perp_y * (spacing / 2.0)),
                    'end': (longest_move['x2'] + perp_x * (spacing / 2.0), longest_move['y2'] + perp_y * (spacing / 2.0)),
                    'e_delta': edge_e_delta,
                    'spacing': spacing / 2.0
                },
                'right': {
                    'start': (longest_move['x1'] - perp_x * (spacing / 2.0), longest_move['y1'] - perp_y * (spacing / 2.0)),
                    'end': (longest_move['x2'] - perp_x * (spacing / 2.0), longest_move['y2'] - perp_y * (spacing / 2.0)),
                    'e_delta': edge_e_delta,
                    'spacing': spacing / 2.0
                }
            }
            
            if debug >= 2:
                logging.info(f"[BRIDGE] Created edge intermediates for longest move (spacing={spacing:.3f}mm)")
    
    # Step 5: Calculate edge intermediates for first and last long lines
    # ONLY if they actually have intermediates paired with them
    edge_before_first = None
    edge_after_last = None
    
    if len(long_move_indices) >= 1:
        first_long = moves[long_move_indices[0]]
        last_long = moves[long_move_indices[-1]]
        
        # Only add edge BEFORE first line if the first line HAS an intermediate after it
        # OR if it has single_line_edges (unpaired single line)
        # AND there's no travel move (G0) between first and second line (continuous section)
        if first_long.get('intermediate_after') is not None or first_long.get('single_line_edges') is not None:
            # Check for travel moves between first line and its paired line
            paired_idx = None
            if first_long.get('intermediate_after') is not None:
                paired_idx = first_long['intermediate_after'].get('paired_idx')
            has_travel = False
            if paired_idx is not None:
                # Check all lines in buffer between the two long lines
                first_line_idx = first_long['line_index']
                paired_line_idx = moves[paired_idx]['line_index']
                for buf_idx in range(first_line_idx, paired_line_idx):
                    if buffered_lines[buf_idx].startswith("G0"):
                        has_travel = True
                        break
            
            # Only create edge intermediate if no travel move (continuous section)
            if not has_travel:
                # Check if this is a paired line (has intermediate_after) or single line (has single_line_edges)
                if first_long.get('intermediate_after') is not None:
                    # Use the intermediate's data to calculate the edge
                    inter = first_long['intermediate_after']
                    spacing = inter['spacing']
                    paired_idx = inter.get('paired_idx')
                    
                    # logging.info(f"[BRIDGE] ===== CALCULATING EDGE BEFORE (paired line) =====")
                    # logging.info(f"[BRIDGE] Starting position (where we're coming from): ({start_x:.3f}, {start_y:.3f})")
                    # logging.info(f"[BRIDGE] First long line: ({first_long['x1']:.3f},{first_long['y1']:.3f}) → ({first_long['x2']:.3f},{first_long['y2']:.3f})")
                    # logging.info(f"[BRIDGE] Intermediate start: {inter['start']}, end: {inter['end']}")
                    # logging.info(f"[BRIDGE] Intermediate spacing: {spacing:.3f}mm")
                    
                    # Get the second long line (the one paired with the first)
                    if paired_idx is not None:
                        second_long = moves[paired_idx]
                        logging.info(f"[BRIDGE] Second long line (paired): ({second_long['x1']:.3f},{second_long['y1']:.3f}) → ({second_long['x2']:.3f},{second_long['y2']:.3f})")
                    else:
                        second_long = None
                        logging.warning(f"[BRIDGE] No paired second line found!")
                    
                    # Calculate perpendicular vector to the first long line
                    perp_x = -first_long['dy'] / first_long['length']
                    perp_y = first_long['dx'] / first_long['length']
                    
                    logging.info(f"[BRIDGE] Perpendicular vector: ({perp_x:.3f}, {perp_y:.3f})")
                    
                    # Determine which side of the first long line the SECOND line is on
                    # The intermediate goes BETWEEN them, so edge BEFORE should be on the OPPOSITE side
                    if second_long is not None:
                        # Use the start point of the second line to determine which side it's on
                        cross_second = first_long['dx'] * (second_long['y1'] - first_long['y1']) - first_long['dy'] * (second_long['x1'] - first_long['x1'])
                        
                        # The edge BEFORE should be on the OPPOSITE side from the second line
                        if cross_second > 0:
                            # Second line is on the "left" side, so edge BEFORE goes on the "right" side
                            edge_offset_x = -perp_x * (spacing / 2.0)
                            edge_offset_y = -perp_y * (spacing / 2.0)
                            logging.info(f"[BRIDGE] Second line on LEFT (cross={cross_second:.3f}), edge BEFORE on RIGHT (-perp)")
                        else:
                            # Second line is on the "right" side, so edge BEFORE goes on the "left" side
                            edge_offset_x = perp_x * (spacing / 2.0)
                            edge_offset_y = perp_y * (spacing / 2.0)
                            logging.info(f"[BRIDGE] Second line on RIGHT (cross={cross_second:.3f}), edge BEFORE on LEFT (+perp)")
                    else:
                        # Fallback: use starting position
                        cross_start = first_long['dx'] * (start_y - first_long['y1']) - first_long['dy'] * (start_x - first_long['x1'])
                        if cross_start > 0:
                            edge_offset_x = perp_x * (spacing / 2.0)
                            edge_offset_y = perp_y * (spacing / 2.0)
                        else:
                            edge_offset_x = -perp_x * (spacing / 2.0)
                            edge_offset_y = -perp_y * (spacing / 2.0)
                        logging.info(f"[BRIDGE] Fallback: using start_pos (cross={cross_start:.3f})")
                    
                    logging.info(f"[BRIDGE] Edge offset: ({edge_offset_x:.3f}, {edge_offset_y:.3f})")
                    
                    e_per_mm = first_long['e_delta'] / first_long['length']
                    edge_e_delta = first_long['length'] * e_per_mm
                    
                    edge_before_first = {
                        'start': (first_long['x1'] + edge_offset_x, first_long['y1'] + edge_offset_y),
                        'end': (first_long['x2'] + edge_offset_x, first_long['y2'] + edge_offset_y),
                        'e_delta': edge_e_delta,
                        'spacing': spacing / 2.0
                    }
                    
                    logging.info(f"[BRIDGE] Edge BEFORE calculated: start={edge_before_first['start']}, end={edge_before_first['end']}")
                
                elif first_long.get('single_line_edges') is not None:
                    # Single unpaired line - use the 'left' edge as edge BEFORE
                    edges = first_long['single_line_edges']
                    spacing = edges['left']['spacing']
                    
                    logging.info(f"[BRIDGE] ===== CALCULATING EDGE BEFORE (single line) =====")
                    logging.info(f"[BRIDGE] Using LEFT edge from single_line_edges")
                    
                    edge_before_first = {
                        'start': edges['left']['start'],
                        'end': edges['left']['end'],
                        'e_delta': edges['left']['e_delta'],
                        'spacing': spacing
                    }
                    
                    logging.info(f"[BRIDGE] Edge BEFORE calculated: start={edge_before_first['start']}, end={edge_before_first['end']}")
        
        # Only add edge AFTER last line if any line before it has an intermediate
        # (meaning there's a densified section ending with last_long)
        has_any_intermediate = any(m.get('intermediate_after') is not None for m in moves)
        
        if has_any_intermediate:
            # Find the last line that has an intermediate (might be second-to-last or earlier)
            last_with_intermediate = None
            for idx in reversed(long_move_indices):
                if moves[idx].get('intermediate_after') is not None:
                    last_with_intermediate = moves[idx]
                    break
            
            if last_with_intermediate is not None:
                inter = last_with_intermediate['intermediate_after']
                spacing = inter['spacing']
                
                # Calculate perpendicular offset in opposite direction
                perp_x = -last_long['dy'] / last_long['length']
                perp_y = last_long['dx'] / last_long['length']
                
                # Determine offset direction (same side as intermediate)
                cross = last_long['dx'] * (inter['start'][1] - last_long['y1']) - last_long['dy'] * (inter['start'][0] - last_long['x1'])
                edge_offset_x = -perp_x * (spacing / 2.0) if cross > 0 else perp_x * (spacing / 2.0)
                edge_offset_y = -perp_y * (spacing / 2.0) if cross > 0 else perp_y * (spacing / 2.0)
                
                e_per_mm = last_long['e_delta'] / last_long['length']
                edge_e_delta = last_long['length'] * e_per_mm
                
                edge_after_last = {
                    'start': (last_long['x1'] + edge_offset_x, last_long['y1'] + edge_offset_y),
                    'end': (last_long['x2'] + edge_offset_x, last_long['y2'] + edge_offset_y),
                    'e_delta': edge_e_delta,
                    'spacing': spacing / 2.0
                }
    
    # Step 6: Output - iterate through all moves and insert intermediates where marked
    output = []
    
    # Find first G1 line in buffered_lines
    first_g1_index = 0
    for idx, line in enumerate(buffered_lines):
        if line.startswith("G1") and "X" in line and "Y" in line:
            first_g1_index = idx
            break
    
    # Output metadata lines (TYPE, WIDTH, HEIGHT, etc.)
    for i in range(first_g1_index):
        output.append(buffered_lines[i])
    
    # Initialize current position at the start of the first move in this section
    current_pos = (moves[0]['x1'], moves[0]['y1'])
    if debug >= 2:
        logging.info(f"[BRIDGE] Initial current_pos: {current_pos}")
    
    # Output all moves in continuous serpentine loop
    # BUT: detect when buffer order doesn't match spatial order and add travel moves
    first_long_drawn = False  # Track if we've drawn the first long line
    edge_before_drawn = False  # Track if we've drawn the edge BEFORE
    
    for move_idx, move in enumerate(moves):
        # Skip old connectors - we create our own continuous path
        if move['is_connector_to_skip']:
            continue
        
        # Insert edge BEFORE right before we draw the first LONG line
        if not edge_before_drawn and edge_before_first and move.get('is_long') and not first_long_drawn:
            if debug >= 2:
                logging.info(f"[BRIDGE] ===== INSERTING EDGE BEFORE =====")
                logging.info(f"[BRIDGE] Current position: {current_pos}")
                logging.info(f"[BRIDGE] About to draw first long line: ({move['x1']:.3f},{move['y1']:.3f}) → ({move['x2']:.3f},{move['y2']:.3f})")
                logging.info(f"[BRIDGE] Edge BEFORE start: {edge_before_first['start']}")
                logging.info(f"[BRIDGE] Edge BEFORE end: {edge_before_first['end']}")
            # Find which endpoint of edge BEFORE is closer to current position
            dist_to_start = math.sqrt((edge_before_first['start'][0] - current_pos[0])**2 + 
                                     (edge_before_first['start'][1] - current_pos[1])**2)
            dist_to_end = math.sqrt((edge_before_first['end'][0] - current_pos[0])**2 + 
                                   (edge_before_first['end'][1] - current_pos[1])**2)
            if debug >= 2:
                logging.info(f"[BRIDGE] Distance from current_pos to edge start: {dist_to_start:.3f}mm")
                logging.info(f"[BRIDGE] Distance from current_pos to edge end: {dist_to_end:.3f}mm")
            if debug:
                output.append(f"; [Bridge Densifier] Edge intermediate BEFORE first long line (spacing={edge_before_first['spacing']:.3f}mm)\n")
            set_e_mode('relative')
            if dist_to_start < dist_to_end:
                # Closer to start - travel to start, then draw start->end
                if dist_to_start > 0.001:  # Only if we're not already there
                    output.append(add_inline_comment(
                        f"G0 X{edge_before_first['start'][0]:.3f} Y{edge_before_first['start'][1]:.3f} F{int(travel_feedrate)}\n",
                        f"Edge BEFORE connector {dist_to_start:.2f}mm"
                    ))
                if debug >= 2:
                    logging.info(f"[BRIDGE] Drawing edge BEFORE: start->end")
                output.append(add_inline_comment(
                    f"G1 X{edge_before_first['end'][0]:.3f} Y{edge_before_first['end'][1]:.3f} E{edge_before_first['e_delta']:.5f} F{bridge_extrusion_feedrate}\n",
                    f"Edge BEFORE start->end, spacing={edge_before_first['spacing']:.3f}mm"
                ))
                current_pos = (edge_before_first['end'][0], edge_before_first['end'][1])
            else:
                # Closer to end - travel to end, then draw end->start
                if dist_to_end > 0.001:  # Only if we're not already there
                    output.append(add_inline_comment(
                        f"G0 X{edge_before_first['end'][0]:.3f} Y{edge_before_first['end'][1]:.3f} F{int(travel_feedrate)}\n",
                        f"Edge BEFORE connector {dist_to_end:.2f}mm"
                    ))
                logging.info(f"[BRIDGE] Drawing edge BEFORE: end->start")
                output.append(add_inline_comment(
                    f"G1 X{edge_before_first['start'][0]:.3f} Y{edge_before_first['start'][1]:.3f} E{edge_before_first['e_delta']:.5f} F{bridge_extrusion_feedrate}\n",
                    f"Edge BEFORE end->start, spacing={edge_before_first['spacing']:.3f}mm"
                ))
                current_pos = (edge_before_first['start'][0], edge_before_first['start'][1])
            
            set_e_mode('absolute')
            if debug >= 2:
                logging.info(f"[BRIDGE] After edge BEFORE, current_pos: {current_pos}")
            edge_before_drawn = True
        
        # CRITICAL: Check if this line is "out of sequence"
        # If the PREVIOUS long line has an intermediate paired with THIS line,
        # we can continue serpentine. Otherwise, we need to travel/restart.
        is_continuation = False
        is_first_long = False
        
        if move.get('is_long'):
            # Check if this is the very first long line
            if not first_long_drawn and move_idx == long_move_indices[0]:
                is_first_long = True
                first_long_drawn = True
            
            # Check continuation for non-first lines
            if not is_first_long and move_idx > 0:
                # Look back to find the previous long line
                for prev_idx in range(move_idx - 1, -1, -1):
                    if moves[prev_idx].get('is_long'):
                        prev_long = moves[prev_idx]
                        # Check if previous line's intermediate was paired with us
                        if prev_long.get('intermediate_after') and prev_long['intermediate_after'].get('paired_idx') == move_idx:
                            is_continuation = True
                        break
        
        # If not a continuation AND not the first line, we need to travel to the original position
        if not is_continuation and not is_first_long and move_idx > 0 and move.get('is_long'):
            # Find where the original G-code expects us to be (from buffered_lines)
            expected_x, expected_y = move['x1'], move['y1']
            dist_to_expected = math.sqrt((expected_x - current_pos[0])**2 + (expected_y - current_pos[1])**2)
            
            if dist_to_expected > 0.5:  # If we're more than 0.5mm away, add travel
                output.append(f"G0 X{expected_x:.3f} Y{expected_y:.3f} F{int(travel_feedrate)} ; [Bridge Densifier] Restart serpentine\n")
                current_pos = (expected_x, expected_y)
        
        # Choose optimal direction for this line (minimize distance from current position)
        # EXCEPT for the first long line with edge BEFORE - use natural connection
        if is_first_long and edge_before_first:
            # Force direction that connects naturally from edge BEFORE
            # current_pos is already where edge ended, so use whichever endpoint is closer
            dist_to_start = math.sqrt((move['x1'] - current_pos[0])**2 + (move['y1'] - current_pos[1])**2)
            dist_to_end = math.sqrt((move['x2'] - current_pos[0])**2 + (move['y2'] - current_pos[1])**2)
        else:
            # Normal direction optimization
            dist_to_start = math.sqrt((move['x1'] - current_pos[0])**2 + (move['y1'] - current_pos[1])**2)
            dist_to_end = math.sqrt((move['x2'] - current_pos[0])**2 + (move['y2'] - current_pos[1])**2)
        
        set_e_mode('relative')
        
        # Define a threshold for "long jump" - if connector is longer than this, use travel instead
        LONG_JUMP_THRESHOLD = 5.0  # mm - anything longer is a separate bridge section
        
        # Determine optimal direction (choose closest endpoint)
        # This is the core of serpentine optimization
        if dist_to_start <= dist_to_end:
            # Draw connector to start, then start → end
            if dist_to_start > 0.001:  # Only if not already at start
                if dist_to_start > LONG_JUMP_THRESHOLD:
                    # Long connector - use travel instead of extrusion
                    set_e_mode('absolute')
                    output.append(add_inline_comment(
                        f"G0 X{move['x1']:.3f} Y{move['y1']:.3f} F{int(travel_feedrate)}\n",
                        f"Long jump {dist_to_start:.2f}mm - new section"
                    ))
                    set_e_mode('relative')
                else:
                    # Normal short connector - extrude
                    output.append(add_inline_comment(
                        f"G1 X{move['x1']:.3f} Y{move['y1']:.3f} E{dist_to_start * 0.04187:.5f} F{bridge_extrusion_feedrate}\n",
                        f"Connector {dist_to_start:.2f}mm"
                    ))
            
            # Determine what type of bridge line this is for the comment
            line_type = "isolated"  # Default
            if move.get('intermediate_after') is not None:
                line_type = "paired-has-intermediate"
            elif move.get('paired_with_prev') is not None:
                line_type = "paired-no-intermediate"
            elif move.get('single_line_edges') is not None:
                line_type = "single-with-edges"
            elif move.get('is_long'):
                line_type = "long-isolated-NO-DENSIFICATION"
            
            # Now draw the line itself (start → end)
            output.append(add_inline_comment(
                f"G1 X{move['x2']:.3f} Y{move['y2']:.3f} E{move['e_delta']:.5f} F{bridge_extrusion_feedrate}\n",
                f"Bridge line #{move_idx} fwd, len={move['length']:.2f}mm [{line_type}]"
            ))
            current_pos = (move['x2'], move['y2'])
        else:
            # Draw connector to end, then end → start
            if dist_to_end > 0.001:  # Only if not already at end
                if dist_to_end > LONG_JUMP_THRESHOLD:
                    # Long connector - use travel instead of extrusion
                    set_e_mode('absolute')
                    output.append(add_inline_comment(
                        f"G0 X{move['x2']:.3f} Y{move['y2']:.3f} F{int(travel_feedrate)}\n",
                        f"Long jump {dist_to_end:.2f}mm - new section"
                    ))
                    set_e_mode('relative')
                else:
                    # Normal short connector - extrude
                    output.append(add_inline_comment(
                        f"G1 X{move['x2']:.3f} Y{move['y2']:.3f} E{dist_to_end * 0.04187:.5f} F{bridge_extrusion_feedrate}\n",
                        f"Connector {dist_to_end:.2f}mm"
                    ))
            
            # Determine what type of bridge line this is for the comment
            line_type = "isolated"  # Default
            if move.get('intermediate_after') is not None:
                line_type = "paired-has-intermediate"
            elif move.get('paired_with_prev') is not None:
                line_type = "paired-no-intermediate"
            elif move.get('single_line_edges') is not None:
                line_type = "single-with-edges"
            elif move.get('is_long'):
                line_type = "long-isolated-NO-DENSIFICATION"
            
            # Now draw the line itself (end → start)
            output.append(add_inline_comment(
                f"G1 X{move['x1']:.3f} Y{move['y1']:.3f} E{move['e_delta']:.5f} F{bridge_extrusion_feedrate}\n",
                f"Bridge line #{move_idx} rev, len={move['length']:.2f}mm [{line_type}]"
            ))
            current_pos = (move['x1'], move['y1'])
        
        # Insert intermediate AFTER this line if marked
        if move['intermediate_after']:
            inter = move['intermediate_after']
            
            # Choose optimal direction for intermediate (minimize distance)
            dist_to_inter_start = math.sqrt((inter['start'][0] - current_pos[0])**2 + (inter['start'][1] - current_pos[1])**2)
            dist_to_inter_end = math.sqrt((inter['end'][0] - current_pos[0])**2 + (inter['end'][1] - current_pos[1])**2)
            
            if debug:
                output.append(f"; [Bridge Densifier] Intermediate (spacing={inter['spacing']:.3f}mm, length={inter['length']:.3f}mm)\n")
            set_e_mode('relative')
            
            LONG_JUMP_THRESHOLD = 5.0  # mm
            
            # Determine optimal direction (choose closest endpoint)
            if dist_to_inter_start <= dist_to_inter_end:
                # Draw connector to start, then start → end
                if dist_to_inter_start > 0.001:
                    if dist_to_inter_start > LONG_JUMP_THRESHOLD:
                        # Long connector - use travel
                        set_e_mode('absolute')
                        output.append(add_inline_comment(
                            f"G0 X{inter['start'][0]:.3f} Y{inter['start'][1]:.3f} F{int(travel_feedrate)}\n",
                            f"Inter long jump {dist_to_inter_start:.2f}mm"
                        ))
                        set_e_mode('relative')
                    else:
                        # Normal connector - extrude
                        output.append(add_inline_comment(
                            f"G1 X{inter['start'][0]:.3f} Y{inter['start'][1]:.3f} E{dist_to_inter_start * 0.04187:.5f} F{bridge_extrusion_feedrate}\n",
                            f"Inter connector {dist_to_inter_start:.2f}mm"
                        ))
                # Draw the intermediate line itself (start → end)
                output.append(add_inline_comment(
                    f"G1 X{inter['end'][0]:.3f} Y{inter['end'][1]:.3f} E{inter['e_delta']:.5f} F{bridge_extrusion_feedrate}\n",
                    f"Intermediate fwd, spacing={inter['spacing']:.2f}mm"
                ))
                current_pos = (inter['end'][0], inter['end'][1])
            else:
                # Draw connector to end, then end → start
                if dist_to_inter_end > 0.001:
                    if dist_to_inter_end > LONG_JUMP_THRESHOLD:
                        # Long connector - use travel
                        set_e_mode('absolute')
                        output.append(add_inline_comment(
                            f"G0 X{inter['end'][0]:.3f} Y{inter['end'][1]:.3f} F{int(travel_feedrate)}\n",
                            f"Inter long jump {dist_to_inter_end:.2f}mm"
                        ))
                        set_e_mode('relative')
                    else:
                        # Normal connector - extrude
                        output.append(add_inline_comment(
                            f"G1 X{inter['end'][0]:.3f} Y{inter['end'][1]:.3f} E{dist_to_inter_end * 0.04187:.5f} F{bridge_extrusion_feedrate}\n",
                            f"Inter connector {dist_to_inter_end:.2f}mm"
                        ))
                # Draw the intermediate line itself (end → start)
                output.append(add_inline_comment(
                    f"G1 X{inter['start'][0]:.3f} Y{inter['start'][1]:.3f} E{inter['e_delta']:.5f} F{bridge_extrusion_feedrate}\n",
                    f"Intermediate rev, spacing={inter['spacing']:.2f}mm"
                ))
                current_pos = (inter['start'][0], inter['start'][1])
        
        # Handle single unpaired line - add RIGHT edge (LEFT was already output as edge BEFORE)
        if move.get('single_line_edges'):
            edges = move['single_line_edges']
            
            logging.info(f"[BRIDGE] Outputting RIGHT edge for single unpaired line at index {move_idx}")
            
            # Only draw RIGHT edge (LEFT edge was already drawn as edge BEFORE)
            right_edge = edges['right']
            dist_to_right_start = math.sqrt((right_edge['start'][0] - current_pos[0])**2 + (right_edge['start'][1] - current_pos[1])**2)
            dist_to_right_end = math.sqrt((right_edge['end'][0] - current_pos[0])**2 + (right_edge['end'][1] - current_pos[1])**2)
            
            if debug:
                output.append(f"; [Bridge Densifier] Single line RIGHT edge (spacing={right_edge['spacing']:.3f}mm)\n")
            set_e_mode('relative')
            
            if dist_to_right_start < dist_to_right_end:
                if dist_to_right_start > 0.001:
                    output.append(add_inline_comment(
                        f"G1 X{right_edge['start'][0]:.3f} Y{right_edge['start'][1]:.3f} E{dist_to_right_start * 0.04187:.5f} F{bridge_extrusion_feedrate}\n",
                        f"Right edge connector {dist_to_right_start:.2f}mm"
                    ))
                output.append(add_inline_comment(
                    f"G1 X{right_edge['end'][0]:.3f} Y{right_edge['end'][1]:.3f} E{right_edge['e_delta']:.5f} F{bridge_extrusion_feedrate}\n",
                    f"Single line RIGHT edge, spacing={right_edge['spacing']:.2f}mm"
                ))
                current_pos = (right_edge['end'][0], right_edge['end'][1])
            else:
                if dist_to_right_end > 0.001:
                    output.append(add_inline_comment(
                        f"G1 X{right_edge['end'][0]:.3f} Y{right_edge['end'][1]:.3f} E{dist_to_right_end * 0.04187:.5f} F{bridge_extrusion_feedrate}\n",
                        f"Right edge connector {dist_to_right_end:.2f}mm"
                    ))
                output.append(add_inline_comment(
                    f"G1 X{right_edge['start'][0]:.3f} Y{right_edge['start'][1]:.3f} E{right_edge['e_delta']:.5f} F{bridge_extrusion_feedrate}\n",
                    f"Single line RIGHT edge, spacing={right_edge['spacing']:.2f}mm"
                ))
                current_pos = (right_edge['start'][0], right_edge['start'][1])
        
        # Insert edge intermediate AFTER last bridge line
        if long_move_indices and move_idx == long_move_indices[-1] and edge_after_last:
            dist_to_edge_start = math.sqrt((edge_after_last['start'][0] - current_pos[0])**2 + (edge_after_last['start'][1] - current_pos[1])**2)
            dist_to_edge_end = math.sqrt((edge_after_last['end'][0] - current_pos[0])**2 + (edge_after_last['end'][1] - current_pos[1])**2)
            
            if debug:
                output.append(f"; [Bridge Densifier] Edge intermediate AFTER last line (spacing={edge_after_last['spacing']:.3f}mm)\n")
            set_e_mode('relative')
            
            if dist_to_edge_start <= dist_to_edge_end:
                # Draw connector to start, then start → end
                output.append(add_inline_comment(
                    f"G1 X{edge_after_last['start'][0]:.3f} Y{edge_after_last['start'][1]:.3f} E{dist_to_edge_start * 0.04187:.5f} F{bridge_extrusion_feedrate}\n",
                    f"Edge AFTER connector {dist_to_edge_start:.2f}mm"
                ))
                output.append(add_inline_comment(
                    f"G1 X{edge_after_last['end'][0]:.3f} Y{edge_after_last['end'][1]:.3f} E{edge_after_last['e_delta']:.5f} F{bridge_extrusion_feedrate}\n",
                    f"Edge AFTER fwd, spacing={edge_after_last['spacing']:.2f}mm"
                ))
                current_pos = (edge_after_last['end'][0], edge_after_last['end'][1])
            else:
                # Draw connector to end, then end → start
                output.append(add_inline_comment(
                    f"G1 X{edge_after_last['end'][0]:.3f} Y{edge_after_last['end'][1]:.3f} E{dist_to_edge_end * 0.04187:.5f} F{bridge_extrusion_feedrate}\n",
                    f"Edge AFTER connector {dist_to_edge_end:.2f}mm"
                ))
                output.append(add_inline_comment(
                    f"G1 X{edge_after_last['start'][0]:.3f} Y{edge_after_last['start'][1]:.3f} E{edge_after_last['e_delta']:.5f} F{bridge_extrusion_feedrate}\n",
                    f"Edge AFTER rev, spacing={edge_after_last['spacing']:.2f}mm"
                ))
                current_pos = (edge_after_last['start'][0], edge_after_last['start'][1])
    
    # Switch back to absolute mode at the end
    set_e_mode('absolute')
    
    # Calculate final E from original buffer (for comparison)
    original_final_e = current_e
    for line in buffered_lines:
        e_match = REGEX_E.search(line)
        if e_match:
            original_final_e = float(e_match.group(1))
    
    # Count total E extruded in densified output (for verification)
    densified_total_e = 0.0
    for line in output:
        if line.startswith("G1") and "E" in line and "X" in line:
            e_match = REGEX_E.search(line)
            if e_match:
                e_val = float(e_match.group(1))
                if e_val > 0:  # Relative mode, positive = extrusion
                    densified_total_e += e_val
    
    if debug >= 2:
        logging.info(f"[BRIDGE] E tracking: start={current_e:.5f}, original_end={original_final_e:.5f}, densified_total={densified_total_e:.5f}")
    
    # Return output, final E, and final XY position (where serpentine path ended)
    final_e = current_e + densified_total_e
    return output, final_e, current_pos

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
    
    # Validate outer layer height
    if outer_layer_height <= 0:
        logging.error(f"Outer layer height ({outer_layer_height}mm) must be greater than 0")
        sys.exit(1)
    
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
    
    # Pass 1: Heuristic-based orphan detection
    # External perimeters are typically continuous extrusion paths that form loops
    annotated_lines = []
    current_type = None
    orphans_found = 0
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
        if (current_type != ";TYPE:External perimeter" and 
            current_type != ";TYPE:Outer wall" and
            current_type != ";TYPE:Overhang perimeter" and
            current_type != ";TYPE:Solid infill" and  # EXCLUDE solid infill!
            current_type != ";TYPE:Bridge infill" and  # EXCLUDE bridge infill!
            current_type != ";TYPE:Internal bridge infill" and  # EXCLUDE internal bridge infill!
            line.startswith("G1") and 
            "X" in line and "Y" in line and "E" in line):
            
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
                e_values = []
                j = i
                while j < len(lines) and len(candidate_path) < 100:
                    check_line = lines[j]
                    
                    if ";TYPE:" in check_line:
                        break
                    if ";LAYER_CHANGE" in check_line:
                        break
                    
                    if "G1" in check_line and "X" in check_line and "Y" in check_line and "E" in check_line:
                        params = parse_gcode_line(check_line)
                        if params['x'] is not None and params['y'] is not None and params['e'] is not None:
                            candidate_path.append((params['x'], params['y']))
                            e_values.append(params['e'])
                    else:
                        # Non-extrusion move, stop collecting
                        break
                    
                    j += 1
                
                # Analyze if this looks like an external perimeter:
                # 1. Has at least 10 points (substantial path)
                # 2. E values continuously increase (no retractions)
                # 3. Forms a closed or nearly-closed loop (first/last distance < 10mm)
                # 4. Has sufficient direction changes (not a straight line)
                #    Note: Some perimeters are open paths, so we use a generous threshold
                is_likely_perimeter = False
                if len(candidate_path) >= 10 and len(e_values) >= 10:
                    # Check E continuously increases
                    e_increasing = all(e_values[k+1] >= e_values[k] for k in range(len(e_values)-1))
                    
                    # Check if closed or nearly-closed loop
                    first_xy = candidate_path[0]
                    last_xy = candidate_path[-1]
                    distance = ((first_xy[0] - last_xy[0])**2 + (first_xy[1] - last_xy[1])**2)**0.5
                    is_closed = distance < 10.0  # Within 10mm (generous for open perimeters)
                    
                    # Check for direction changes to filter out straight infill lines
                    # Count significant angle changes (> 10 degrees)
                    direction_changes = 0
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
                                
                                if angle_deg > 10:  # Significant direction change
                                    direction_changes += 1
                    
                    # Perimeters should have at least 3 direction changes
                    # Straight infill lines will have 0-2
                    has_curvature = direction_changes >= 3
                    
                    if e_increasing and is_closed and has_curvature:
                        is_likely_perimeter = True
                
                if is_likely_perimeter:
                    # This is an orphan external perimeter!
                    annotated_lines.append(";TYPE:External perimeter ; AUTO-ADDED by Smoothificator (heuristic)\n")
                    current_type = ";TYPE:External perimeter"
                    orphans_found += 1
                    print(f"  [ORPHAN] Found at line {i}: {len(candidate_path)} points, closed loop")
        
        annotated_lines.append(line)
        i += 1
    
    print(f"Pass 1 complete: Found and marked {orphans_found} orphan external perimeter segments")
    logging.info(f"Pass 1 complete: Found and marked {orphans_found} orphan external perimeter segments")
    
    # Use annotated lines for Pass 2
    lines = annotated_lines
    
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
    
    def update_position(line_str):
        """Update global position tracker from a G-code line.
        Call this EVERY time you read a line from the input, regardless of processing mode."""
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
            
            external_block_lines = []
            external_block_lines.append(line)
            i += 1
            
            # Collect all lines until next TYPE change OR layer change
            # Include WIPE moves and everything up to the next TYPE marker
            while i < len(lines):
                current_line = lines[i]
                # OUTPUT-DRIVEN POSITION TRACKING: We normally don't update during collection.
                # HOWEVER: If we encounter a standalone Z move (no X/Y) we must capture it as the
                # base Z before Smoothificator replaces Z handling. Otherwise we lose the true
                # layer Z (e.g., initial drop from priming height 0.8 -> 0.2).
                if current_line.startswith("G1") and "Z" in current_line and "X" not in current_line and "Y" not in current_line:
                    z_match = re.search(r'Z([-+]?\d*\.?\d+)', current_line)
                    if z_match:
                        old_z = current_z
                        current_z = float(z_match.group(1))
                        working_z = current_z  # Update working/base Z for passes
                        # We do NOT write this original Z move; passes will emit their own
                        # But we now have the correct base Z (e.g., 0.2 instead of prior 0.8)
                
                # Stop at layer boundary to prevent crossing layers
                if ";LAYER_CHANGE" in current_line:
                    break
                
                # Stop at different type marker (this is the real end of external perimeter block)
                if (";TYPE:" in current_line and 
                    ";TYPE:External perimeter" not in current_line and 
                    ";TYPE:Outer wall" not in current_line and
                    ";TYPE:Overhang perimeter" not in current_line):
                    break
                    
                external_block_lines.append(current_line)
                i += 1
            
            #logging.info(f"  [SMOOTHIFICATOR] Collected external perimeter block with {len(external_block_lines)} lines")
            
            # Calculate effective layer height
            if current_layer_height > 0.01:
                effective_layer_height = current_layer_height
            elif current_z > old_z + 0.001:
                effective_layer_height = current_z - old_z
            else:
                effective_layer_height = outer_layer_height
            
            # Calculate how many passes we need
            if effective_layer_height > outer_layer_height:
                passes_ceil = math.ceil(effective_layer_height / outer_layer_height)
                passes_floor = math.floor(effective_layer_height / outer_layer_height)
                
                height_per_pass_ceil = effective_layer_height / passes_ceil
                height_per_pass_floor = effective_layer_height / passes_floor if passes_floor > 0 else float('inf')
                
                diff_ceil = abs(height_per_pass_ceil - outer_layer_height)
                diff_floor = abs(height_per_pass_floor - outer_layer_height)
                diff_original = abs(effective_layer_height - outer_layer_height)
                
                if diff_original <= diff_ceil and diff_original <= diff_floor:
                    passes_needed = 1
                    height_per_pass = effective_layer_height
                elif diff_ceil < diff_floor:
                    passes_needed = passes_ceil
                    height_per_pass = height_per_pass_ceil
                else:
                    passes_needed = passes_floor
                    height_per_pass = height_per_pass_floor
                
                extrusion_multiplier = 1.0 / passes_needed
                #logging.info(f"  [SMOOTHIFICATOR] Layer {current_layer}: {passes_needed} passes at {height_per_pass:.4f}mm each")
            else:
                passes_needed = 1
                height_per_pass = effective_layer_height
                extrusion_multiplier = 1.0
            
            # SIMPLE APPROACH: Just duplicate the block N times with adjusted Z and E
            current_e = 0.0
            
            # Use the global position tracker - it's always accurate!
            start_pos = (position['x'], position['y'])
            
            for pass_num in range(passes_needed):
                # Calculate Z for this pass (work DOWN from current_z)
                if passes_needed == 1:
                    pass_z = current_z
                else:
                    pass_z = current_z - ((passes_needed - pass_num - 1) * height_per_pass)
                
                # Track actual max Z for this layer (for safe Z-hop)
                if current_layer not in actual_layer_max_z or pass_z > actual_layer_max_z[current_layer]:
                    actual_layer_max_z[current_layer] = pass_z
                
                if pass_num == 0:
                    write_and_track(output_buffer, f"; ====== SMOOTHIFICATOR START: {passes_needed} passes at {height_per_pass:.4f}mm each ======\n", recent_output_lines)
                
                # Output Z move
                write_and_track(output_buffer, f"G0 Z{pass_z:.3f} ; Pass {pass_num + 1} of {passes_needed}\n", recent_output_lines)
                
                # For pass 2+, travel back to TRUE start position (where we were before the block)
                if pass_num > 0 and start_pos:
                    write_and_track(output_buffer, f"G1 X{start_pos[0]:.3f} Y{start_pos[1]:.3f} F8400 ; Travel to start\n", recent_output_lines)
                
                # Now copy ALL lines from the block, adjusting E values
                previous_e = None
                for block_line in external_block_lines:
                    # Skip TYPE markers on subsequent passes
                    if pass_num > 0 and ";TYPE:" in block_line:
                        continue
                    
                    # Skip Z moves in the block (we already set Z above)
                    if "G1 Z" in block_line and "X" not in block_line and "Y" not in block_line:
                        continue
                    
                    # If line has Z coordinate with X/Y, remove the Z part
                    if "G1" in block_line and "Z" in block_line:
                        block_line = REGEX_Z_SUB.sub('', block_line)
                    
                    # Adjust E values
                    if "G1" in block_line and "E" in block_line:
                        original_e = extract_e(block_line)
                        if original_e is not None:
                            
                            if previous_e is None:
                                # First E in this pass
                                if pass_num == 0:
                                    current_e = original_e * extrusion_multiplier
                                else:
                                    delta = original_e * extrusion_multiplier
                                    current_e += delta
                            else:
                                # Calculate delta from previous
                                delta = (original_e - previous_e) * extrusion_multiplier
                                current_e += delta
                            
                            previous_e = original_e
                            
                            # Replace E value
                            block_line = replace_e(block_line, current_e)
                    
                    write_and_track(output_buffer, block_line, recent_output_lines)
            
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
                
                # Look back to find the last E value before this TYPE comment (not a retraction)
                pre_block_e_value = None
                for back_idx in range(i - 2, max(0, i - 10), -1):
                    back_line = lines[back_idx]
                    # Track last E value before the block (not a retraction)
                    if pre_block_e_value is None and back_line.startswith("G1") and "E" in back_line and "E-" not in back_line:
                        e_match = re.search(r'E([-\d.]+)', back_line)
                        if e_match:
                            pre_block_e_value = float(e_match.group(1))
                
                # Collect the entire perimeter block
                perimeter_block_lines = []
                
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
                    if ";TYPE:" in current_line or ";LAYER_CHANGE" in current_line:
                        break
                    
                    perimeter_block_lines.append(current_line)
                    i += 1
                
                # Now process the collected block
                # Split into individual perimeter loops (separated by travel moves)
                j = 0
                while j < len(perimeter_block_lines):
                    current_line = perimeter_block_lines[j]
                    
                    # Detect start of perimeter block (extrusion move)
                    if current_line.startswith("G1") and "X" in current_line and "Y" in current_line and "E" in current_line:
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
                        loop_lines = []
                        loop_lines.append(current_line)
                        j += 1
                        
                        # Continue until travel move (no E) or end
                        while j < len(perimeter_block_lines):
                            line = perimeter_block_lines[j]
                            if line.startswith("G1") and "X" in line and "Y" in line and "F" in line and "E" not in line:
                                # Travel move - end of this loop
                                loop_lines.append(line)
                                j += 1
                                break
                            loop_lines.append(line)
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
                            perimeter_block_count += 1
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
                                        if has_bricklayer_below or has_regular_perimeter_below or has_non_perimeter_below:
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
                                
                                # Output all lines with adjusted extrusion
                                for loop_line in loop_lines:
                                    if loop_line.startswith("G1") and ("X" in loop_line or "Y" in loop_line) and "E" in loop_line:
                                        e_value = extract_e(loop_line)
                                        if e_value is not None:
                                            prev_e = position.get('e', 0.0)
                                            e_delta = e_value - prev_e
                                            if e_delta > 0:
                                                new_e_value = prev_e + (e_delta * extrusion_factor * bricklayers_extrusion_multiplier)
                                                loop_line = replace_e(loop_line, new_e_value)
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
                                    
                                    # Separate extrusion moves from non-extrusion commands
                                    # Extrusion moves: G1 with X, Y, and E (the actual printing)
                                    # Non-extrusion: WIDTH comments, G92, retractions, travel, etc.
                                    # Fan commands (M106/M107): Keep only first and last
                                    extrusion_moves = []
                                    non_extrusion_commands = []
                                    fan_commands = []  # Collect all fan commands separately
                                    
                                    # Use global position tracker for loop start position
                                    start_x, start_y = position['x'], position['y']
                                    
                                    for loop_line in loop_lines:
                                        # Check if this is an extrusion move
                                        is_extrusion = loop_line.startswith("G1") and "X" in loop_line and "Y" in loop_line and "E" in loop_line and "E-" not in loop_line
                                        # Check if this is a fan command
                                        is_fan_command = loop_line.startswith("M106") or loop_line.startswith("M107")
                                        
                                        if is_extrusion:
                                            extrusion_moves.append(loop_line)
                                        elif is_fan_command:
                                            # Collect fan commands separately
                                            fan_commands.append(loop_line)
                                        else:
                                            # Everything else (WIDTH comments, G92, retraction, travel)
                                            non_extrusion_commands.append(loop_line)
                                    
                                    # Keep only first and last fan command
                                    if fan_commands:
                                        if len(fan_commands) == 1:
                                            non_extrusion_commands.insert(0, fan_commands[0])
                                        else:
                                            non_extrusion_commands.insert(0, fan_commands[0])  # First at beginning
                                            non_extrusion_commands.append(fan_commands[-1])    # Last at end
                                    
                                    # Validation: make sure we found extrusion moves
                                    if not extrusion_moves:
                                        logging.warning(f"  [BRICKLAYERS WARNING] Layer {current_layer}, Block #{perimeter_block_count}: No extrusion moves found in loop!")
                                    
                                    # Pass 1: Print at base layer Z
                                    if pre_block_e_value is not None:
                                        write_and_track(output_buffer, f"G1 Z{pass1_z:.3f} E{pre_block_e_value:.5f} ; Bricklayers base block #{perimeter_block_count}, pass 1/2\n", recent_output_lines)
                                    else:
                                        write_and_track(output_buffer, f"G1 Z{pass1_z:.3f} ; Bricklayers base block #{perimeter_block_count}, pass 1/2\n", recent_output_lines)
                                    
                                    # Reset E after Z move so extrusion values start fresh
                                    write_and_track(output_buffer, "G92 E0\n", recent_output_lines)
                                    
                                    # Output all extrusion moves with adjusted E values (0.75x height)
                                    for line in extrusion_moves:
                                        e_value = extract_e(line)
                                        if e_value is not None:
                                            new_e_value = e_value * 0.75 * bricklayers_extrusion_multiplier
                                            line = replace_e(line, new_e_value)
                                        write_and_track(output_buffer, line, recent_output_lines)
                                    
                                    # Pass 2: Travel to start, raise Z, print same moves
                                    write_and_track(output_buffer, "G92 E0 ; Reset extruder for pass 2\n", recent_output_lines)
                                    write_and_track(output_buffer, f"G1 X{start_x:.3f} Y{start_y:.3f} F8400 ; Travel to start for pass 2\n", recent_output_lines)
                                    write_and_track(output_buffer, f"G1 Z{pass2_z:.3f} ; Bricklayers base block #{perimeter_block_count}, pass 2/2\n", recent_output_lines)
                                    
                                    # Output same extrusion moves again at higher Z
                                    for line in extrusion_moves:
                                        e_value = extract_e(line)
                                        if e_value is not None:
                                            new_e_value = e_value * 0.75 * bricklayers_extrusion_multiplier
                                            line = replace_e(line, new_e_value)
                                        write_and_track(output_buffer, line, recent_output_lines)
                                    
                                    # Output non-extrusion commands once (M107, WIDTH, G92, retraction, travel, etc.)
                                    for cmd in non_extrusion_commands:
                                        write_and_track(output_buffer, cmd, recent_output_lines)
                                    
                                    # Reset Z back to layer height
                                    write_and_track(output_buffer, f"G1 Z{current_z:.3f} ; Reset Z\n", recent_output_lines)
                                    
                                    # Mark grid cells as base bricklayer
                                    for x, y in sample_positions:
                                        gx = int(x / grid_resolution)
                                        gy = int(y / grid_resolution)
                                        cell_key = (gx, gy, current_layer)
                                        if cell_key in solid_at_grid:
                                            solid_at_grid[cell_key]['bricklayer_type'] = 'base'
                                else:
                                    # Non-base layers: Single pass at Z + 0.5h (sits on previous layer's shifted block)
                                    # On top layers, use 0.75x height for flat top
                                    z_shift_adjusted = z_shift * 0.75 if is_top_layer else z_shift
                                    adjusted_z = current_z + z_shift_adjusted
                                    extrusion_factor = 1.0
                                    
                                    # Track actual max Z for this layer (for safe Z-hop)
                                    if current_layer not in actual_layer_max_z or adjusted_z > actual_layer_max_z[current_layer]:
                                        actual_layer_max_z[current_layer] = adjusted_z
                                    
                                    write_and_track(output_buffer, f"G0 Z{adjusted_z:.3f} ; Bricklayers base block #{perimeter_block_count}\n", recent_output_lines)
                                    #logging.info(f"  [BRICKLAYERS] Layer {current_layer}, Block #{perimeter_block_count}: Base at Z={adjusted_z:.3f} (extrusion: {extrusion_factor}x)")
                                    
                                    for loop_line in loop_lines:
                                        if "E" in loop_line:
                                            e_value = extract_e(loop_line)
                                            if e_value is not None:
                                                new_e_value = e_value * extrusion_factor * bricklayers_extrusion_multiplier
                                                loop_line = replace_e(loop_line, new_e_value)
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
                            
                            perimeter_block_count += 1
                    
                    else:
                        # Non-extrusion line (comments, etc)
                        write_and_track(output_buffer, current_line, recent_output_lines)
                        j += 1
                
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

            # Process infill lines
            while i < len(lines):
                current_line = lines[i]
                
                # NO position update during collection (output-driven tracking)
                
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
                    # CRITICAL: Decrement i so main loop will process this LAYER_CHANGE line
                    i -= 1
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
                    
                    # Parse X, Y, E from current line
                    match = re.search(r'X([-+]?\d*\.?\d+)\s*Y([-+]?\d*\.?\d+)\s*E([-+]?\d*\.?\d+)', current_line)
                    if match:
                        x2, y2, e_end = map(float, match.groups())  # End point is THIS line
                        
                        # BULLETPROOF extrusion detection: Check if E value is non-negative
                        # This matches the grid building logic and correctly identifies:
                        # - Extrusions: E >= 0
                        # - Retractions: E < 0 (skip these)
                        # Travel moves without E won't match the regex above
                        if e_end < 0:
                            # This is a retraction - skip it, update E tracking, output as-is
                            infill_current_e = e_end
                            # Don't update X/Y for retractions
                            # Fall through to output as-is (don't continue, let it reach the bottom)
                        else:
                            # This is an extrusion or unretraction
                            # Calculate delta for this move
                            x1 = infill_current_x
                            y1 = infill_current_y
                            e_start = infill_current_e
                            e_delta = e_end - e_start
                            
                            # CRITICAL: Detect G92 E0 resets (negative delta but positive e_end)
                            # When E resets (e.g., 7.12 -> 1.75), e_delta is negative but e_end is positive
                            # Grid building would INCLUDE this (checks e_end >= 0)
                            # Processing must do the same!
                            if e_delta < 0 and e_end >= 0:
                                # G92 E0 reset detected - reset tracking and treat as new extrusion start
                                if debug >= 3:
                                    logging.info(f"[INFILL] Line {i}: G92 E0 reset detected (e_delta={e_delta:.5f}, e_end={e_end:.5f}), resetting tracking")
                                infill_current_e = 0  # Reset to 0 to match G92 E0
                                e_start = 0  # Recalculate from 0
                                e_delta = e_end - e_start  # Now positive!
                            
                            # Only subdivide if delta is positive (actual extrusion, not travel or retraction)
                            if e_delta > 0:
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
                                            'feedrate': segment_feedrate  # Use adaptive feedrate
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
                                                    f"G1 X{seg['x']:.3f} Y{seg['y']:.3f} Z{seg['z']:.3f} E{valley_start_e + seg['e_delta']:.5f} F{int(seg['feedrate'])}\n", recent_output_lines
                                                )
                                            else:
                                                write_and_track(output_buffer,
                                                    f"G1 X{seg['x']:.3f} Y{seg['y']:.3f} Z{seg['z']:.3f} E{valley_start_e + seg['e_delta']:.5f}\n", recent_output_lines
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
                                                            f"G1 X{seg['x']:.3f} Y{seg['y']:.3f} Z{min(current_fill_z, layer_z):.3f} E{valley_start_e:.5f}\n", recent_output_lines
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
                                                            f"G1 X{seg['x']:.3f} Y{seg['y']:.3f} Z{min(current_fill_z, layer_z):.3f} E{valley_start_e:.5f}\n", recent_output_lines
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
                                    if not in_valley:
                                        # Output with Z modulation, adjusted E, and adaptive feedrate
                                        if segment_feedrate is not None:
                                            write_and_track(output_buffer, 
                                                f"G1 X{sx:.3f} Y{sy:.3f} Z{z_mod:.3f} E{current_e:.5f} F{int(segment_feedrate)}\n", recent_output_lines
                                            )
                                        else:
                                            write_and_track(output_buffer, 
                                                f"G1 X{sx:.3f} Y{sy:.3f} Z{z_mod:.3f} E{current_e:.5f}\n", recent_output_lines
                                            )
                                
                                # CRITICAL: Update tracking positions to END of this move!
                                # Use the ACTUAL final position after all segments were output
                                # current_e might differ from e_end due to adaptive extrusion/valley filling
                                infill_current_x = x2
                                infill_current_y = y2
                                infill_current_e = current_e  # Use actual E after segments, not original e_end
                                
                                i += 1
                                continue
                            else:
                                # e_delta <= 0: could be travel (e_delta==0) or unretraction
                                # Update E tracking but don't subdivide
                                logging.info(f"[INFILL-SKIP] Line {i}: e_delta={e_delta:.5f} (e_start={e_start:.5f}, e_end={e_end:.5f})")
                                infill_current_x = x2
                                infill_current_y = y2
                                infill_current_e = e_end
                
                # Boost feedrate for standalone F commands (e.g., "G1 F3600")
                if current_line.startswith('G1') and 'F' in current_line and 'X' not in current_line and 'Y' not in current_line and 'E' not in current_line:
                    original_feedrate = extract_f(current_line)
                    if original_feedrate is not None:
                        boosted_feedrate = int(original_feedrate * nonplanar_feedrate_multiplier)
                        boosted_line = replace_f(current_line, boosted_feedrate)
                        write_and_track(output_buffer, boosted_line, recent_output_lines)
                        i += 1
                        continue
                
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
                i += 1
            
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
                
                # If this move has Z parameter, update working Z (e.g., Smoothificator, Bricklayers)
                if has_z:
                    zhop_current_z = params['z']
                    zhop_working_z = params['z']
                    is_hopped = False  # Explicit Z in move = at working height
                
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
                    if 'grid_resolution' in locals() and 'noise_lut' in locals() and 'amplitude' in locals():
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
                    # Drop before extrusion - but DON'T drop if this move already has Z parameter
                    # (Smoothificator, Bricklayers, etc. set their own Z)
                    if not has_z:
                        final_output.write(f"G0 Z{zhop_working_z:.3f} F8400 ; Z-hop drop\n")
                        zhop_drop_count += 1
                    is_hopped = False
                    zhop_has_extruded_on_layer = True
                    # Write the extrusion line
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
                       help='Enable Bridge Densifier to add intermediate lines between bridge extrusions for better bridging (default: disabled, enabled with -full, experimental)')
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
        if args.enable_bridge_densifier is None:
            args.enable_bridge_densifier = True
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
        input("\n  Press ENTER to close this window...")
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
        input("\n  Press ENTER to close this window...")

