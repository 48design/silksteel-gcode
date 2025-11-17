
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

# Pre-compile regex patterns for performance (compiled once, used thousands of times)
REGEX_X = re.compile(r'X([-\d.]+)')
REGEX_Y = re.compile(r'Y([-\d.]+)')
REGEX_Z = re.compile(r'Z([-\d.]+)')
REGEX_E = re.compile(r'E([-\d.]+)')
REGEX_F = re.compile(r'F(\d+)')
REGEX_E_SUB = re.compile(r'E[-\d.]+')
REGEX_Z_SUB = re.compile(r'Z[-\d.]+\s*')

# Helper functions for fast regex operations
def extract_x(line):
    """Extract X coordinate from G-code line"""
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

def write_line(buffer, line):
    """Write a line to output buffer, ensuring it has a newline"""
    if line and not line.endswith('\n'):
        buffer.write(line + '\n')
    else:
        buffer.write(line)

def write_and_track(buffer, line, recent_buffer, max_size=20):
    """Write line to buffer and add to rolling buffer for lookback operations"""
    write_line(buffer, line)
    recent_buffer.append(line)
    if len(recent_buffer) > max_size:
        recent_buffer.pop(0)  # Remove oldest line

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Configure logging to both console and file
log_file = os.path.join(script_dir, "SilkSteel_log.txt")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),  # UTF-8 for Unicode emojis
        logging.StreamHandler(sys.stdout)          # Also print to console
    ]
)

logging.info("=" * 70)
logging.info("SilkSteel started")
logging.info(f"Script directory: {script_dir}")
logging.info(f"Log file: {log_file}")
logging.info(f"Command line args: {sys.argv}")
logging.info("=" * 70)

# Smoothificator constants
DEFAULT_OUTER_LAYER_HEIGHT = "Auto"  # "Auto" = min(first_layer, base_layer) * 0.5, "Min" = min_layer_height from G-code, or float value (mm)

# Non-planar infill constants
DEFAULT_AMPLITUDE = 4  # Default Z variation in mm [float] or layerheight [int] (reduced for smoother look)
DEFAULT_FREQUENCY = 8  # Default frequency of the sine wave (reduced for longer waves)
DEFAULT_SEGMENT_LENGTH = 0.64  # Split infill lines into segments of this length (mm) - LARGER = fewer segments, smoother motion
DEFAULT_NONPLANAR_FEEDRATE_MULTIPLIER = 1.1  # Boost feedrate by this factor for non-planar 3D moves (2.0 = double speed)
DEFAULT_ENABLE_ADAPTIVE_EXTRUSION = True  # Enable adaptive extrusion multiplier for Z-lift (adds material to droop down and bond)
DEFAULT_ADAPTIVE_EXTRUSION_MULTIPLIER = 1.5  # Base multiplier for adaptive extrusion (e.g., 1.33 = 33% extra material per layer height of lift)

# Grid resolution for solid occupancy detection
# Will be read from G-code if available, otherwise use default
DEFAULT_EXTRUSION_WIDTH = 0.45  # Default extrusion width in mm (typical value)

# Safe Z-hop constants
DEFAULT_ENABLE_SAFE_Z_HOP = True  # Enabled by default
DEFAULT_SAFE_Z_HOP_MARGIN = 0.5  # mm - safety margin above max Z in layer
DEFAULT_Z_HOP_RETRACTION = 3.0  # mm - retraction distance during Z-hop to prevent stringing

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
            
            # Calculate actual Z for this noise value
            z_offset = amplitude * noise_val
            z_mod = layer_z + z_offset
            
            # Check if this cell has infill extrusions
            cell_key = (gx, gy, layer_num)
            infill_crossings = 0
            if cell_key in solid_at_grid:
                infill_crossings = solid_at_grid[cell_key].get('infill_crossings', 0)
            
            # Color scheme:
            # - GREEN channel only: valley (z < layer_z) AND single infill crossing
            # - RED channel only: valley (z < layer_z) AND multiple infill crossings
            # - Grayscale: everything else (no infill, or not a valley)
            is_valley = z_mod < layer_z
            
            if is_valley and infill_crossings > 1:
                # Valley with multiple crossings: red channel only
                color = (intensity, 0, 0)
            elif is_valley and infill_crossings == 1:
                # Valley with single infill extrusion: green channel only
                color = (0, intensity, 0)
            else:
                # Everything else: grayscale
                color = (intensity, intensity, intensity)
            
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

def process_gcode(input_file, output_file=None, outer_layer_height=None,
                 enable_smoothificator=True,
                 enable_bricklayers=False, bricklayers_extrusion_multiplier=1.0,
                 enable_nonplanar=False, deform_type='sine',
                 segment_length=DEFAULT_SEGMENT_LENGTH, amplitude=DEFAULT_AMPLITUDE, frequency=DEFAULT_FREQUENCY,
                 nonplanar_feedrate_multiplier=DEFAULT_NONPLANAR_FEEDRATE_MULTIPLIER,
                 enable_adaptive_extrusion=DEFAULT_ENABLE_ADAPTIVE_EXTRUSION,
                 adaptive_extrusion_multiplier=DEFAULT_ADAPTIVE_EXTRUSION_MULTIPLIER,
                 enable_safe_z_hop=DEFAULT_ENABLE_SAFE_Z_HOP, safe_z_hop_margin=DEFAULT_SAFE_Z_HOP_MARGIN,
                 z_hop_retraction=DEFAULT_Z_HOP_RETRACTION,
                 debug=False):
    
    # Determine output filename
    # If no output specified, modify in-place (for slicer compatibility)
    # If output specified, write to that file (for manual testing)
    if output_file is None:
        output_file = input_file  # Modify in-place for slicer
        in_place_mode = True
    else:
        in_place_mode = False
    
    logging.info("=" * 70)
    logging.info("SMOOTHIFICATOR ADVANCED - Starting G-code processing")
    logging.info("=" * 70)
    logging.info(f"Input file: {input_file}")
    if in_place_mode:
        logging.info(f"Output mode: IN-PLACE (for slicer compatibility)")
    else:
        logging.info(f"Output file: {output_file}")
    logging.info(f"Features enabled:")
    logging.info(f"  - Smoothificator (External perimeters): {enable_smoothificator}")
    logging.info(f"  - Bricklayers (Internal perimeters): {enable_bricklayers}")
    logging.info(f"  - Non-planar Infill: {enable_nonplanar}")
    logging.info(f"  - Safe Z-hop: {enable_safe_z_hop}")
    
    # Print to console for user visibility
    print("\n" + "=" * 70)
    print("  SILKSTEEL - Advanced G-code Post-Processor")
    print("  \"Smooth on the outside, strong on the inside\"")
    print("=" * 70)
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
    print(", ".join(features) if features else "(None)")
    print("=" * 70)
    
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
        print(f"  • Smoothificator: target layer height = {outer_layer_height:.3f}mm")
    if enable_bricklayers:
        print(f"  • Bricklayers: extrusion multiplier = {bricklayers_extrusion_multiplier:.2f}x")
    if enable_nonplanar:
        print(f"  • Non-planar Infill: amplitude = {amplitude:.2f}mm, frequency = {frequency:.2f}, type = {deform_type}")
        print(f"                       segment length = {segment_length:.2f}mm, feedrate boost = {nonplanar_feedrate_multiplier:.1f}x")
        print(f"                       adaptive extrusion = {'ON' if enable_adaptive_extrusion else 'OFF'}, multiplier = {adaptive_extrusion_multiplier:.2f}x")
    if enable_safe_z_hop:
        print(f"  • Safe Z-hop: margin = {safe_z_hop_margin:.2f}mm, retraction = {z_hop_retraction:.2f}mm")
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
    
    # First pass: Build 3D grid showing which Z layers have solid at each XY position
    # Then for each layer, determine the safe Z range (space between solid layers)
    
    # Get extrusion width from G-code or use default
    extrusion_width = get_extrusion_width(lines)
    if extrusion_width is None:
        extrusion_width = DEFAULT_EXTRUSION_WIDTH
        logging.info(f"No extrusion_width found in G-code, using default: {extrusion_width}mm")
    else:
        logging.info(f"Detected extrusion_width from G-code: {extrusion_width}mm")
    
    # Set grid resolution to match extrusion width (no diagonal compensation)
    grid_resolution = extrusion_width * 1.4444  # 44% larger ensures diagonal coverage
    
    solid_at_grid = {}  # (grid_x, grid_y, layer_num) -> True if solid exists
    z_layer_map = {}    # layer_num -> Z height
    layer_z_map = {}    # Z height -> layer_num
    
    if enable_nonplanar:
        logging.info("\n" + "="*70)
        logging.info("PASS 1: Building 3D solid occupancy grid")
        logging.info("="*70)
        logging.info(f"Grid resolution: {grid_resolution:.3f}mm (matches extrusion width)")
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
                z_match = re.search(r'Z([-+]?\d*\.?\d+)', line)
                if z_match:
                    temp_z = float(z_match.group(1))
                    if current_layer_num not in z_layer_map:
                        z_layer_map[current_layer_num] = temp_z
                        layer_z_map[temp_z] = current_layer_num
            if line.startswith('G1'):
                x_match = re.search(r'X([-+]?\d*\.?\d+)', line)
                y_match = re.search(r'Y([-+]?\d*\.?\d+)', line)
                if x_match:
                    x_coords.append(float(x_match.group(1)))
                if y_match:
                    y_coords.append(float(y_match.group(1)))
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        total_layers = max(z_layer_map.keys()) if z_layer_map else 0
        logging.info(f"  Print: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}], {total_layers} layers")
        logging.info(f"  Found {len(z_layer_map)} unique Z heights")
        if debug >= 1:
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
        last_solid_pos = None  # Track last position to mark all cells along line
        last_solid_coords = None  # Track actual X,Y coordinates for DDA
        last_infill_pos = None  # Track last position for infill
        last_infill_coords = None  # Track actual X,Y coordinates for infill
        debug_line_count = 0
        debug_cells_marked = 0
        type_markers_seen = set()  # Track all TYPE markers we encounter
        prev_line = ""  # Track previous line for debugging
        line_number = 0  # Track line number for debugging
        
        for line in lines:
            line_number += 1
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
                z_match = re.search(r'Z([-+]?\d*\.?\d+)', line)
                if z_match:
                    temp_z = float(z_match.group(1))
            
            # Detect solid infill AND perimeters (both block infill from below)
            if ';TYPE:Solid infill' in line or ';TYPE:Top solid infill' in line or ';TYPE:Bridge infill' in line or \
               ';TYPE:Internal bridge infill' in line or ';TYPE:Overhang perimeter' in line or \
               ';TYPE:External perimeter' in line or ';TYPE:Internal perimeter' in line or ';TYPE:Perimeter' in line:

                in_solid_infill = True
                in_internal_infill = False
                last_solid_pos = None  # Reset when entering solid infill or perimeter
                last_solid_coords = None
                last_infill_pos = None
                last_infill_coords = None
                if temp_z not in solid_infill_heights:
                    solid_infill_heights.append(temp_z)
            elif ';TYPE:Internal infill' in line:
                in_internal_infill = True
                in_solid_infill = False
                last_solid_pos = None
                last_solid_coords = None
                last_infill_pos = None
                last_infill_coords = None
            elif ';TYPE:' in line:
                # Track all TYPE markers for debugging
                type_match = re.search(r';TYPE:([^\n]+)', line)
                if type_match:
                    type_markers_seen.add(type_match.group(1))
                in_solid_infill = False
                in_internal_infill = False
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
                                    # Mark current cell as solid
                                    cell_key = (current_cell_x, current_cell_y, current_layer_num)
                                    if cell_key not in solid_at_grid:
                                        solid_at_grid[cell_key] = {'solid': True, 'infill_crossings': 0}
                                    else:
                                        solid_at_grid[cell_key]['solid'] = True
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
                                if debug >= 1 and debug_line_count <= 10:
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
                                    # Increment crossing count for this cell
                                    if cell_key not in solid_at_grid:
                                        solid_at_grid[cell_key] = {'solid': False, 'infill_crossings': 1}
                                    else:
                                        solid_at_grid[cell_key]['infill_crossings'] += 1
                                    
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
                                    solid_at_grid[cell_key] = {'solid': False, 'infill_crossings': 1}
                                else:
                                    solid_at_grid[cell_key]['infill_crossings'] += 1
                            
                            # Save coordinates for next iteration
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
        
        if debug >= 1:
            print(f"[DEBUG] Marked {len(solid_at_grid)} grid cells with solid")
            print(f"[DEBUG] Processed {debug_line_count} line segments, avg {debug_cells_marked/max(1,debug_line_count):.1f} cells per line")
            print(f"[DEBUG] Built infill grid with {len(infill_at_grid)} cells ({first_of_safezone_count} first + {last_of_safezone_count} last)")
            print(f"\n[DEBUG] All TYPE markers seen in G-code: {sorted(type_markers_seen)}")
            
            # Show which layers have solid infill
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
        
        if debug >= 1:
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
            all_layer_nums = sorted(set(layer for _, _, layer in solid_at_grid.keys()))
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
                        # Create image (black background = air)
                        img = Image.new('RGB', (img_width, img_height), color='black')
                        draw = ImageDraw.Draw(img)
                        
                        # Draw grid cells with solid material (white only)
                        for cell_key, cell_data in solid_at_grid.items():
                            gx, gy, lay = cell_key
                            if lay == layer and cell_data.get('solid', False):
                                # Convert to image coordinates (flip Y axis)
                                img_x = (gx - grid_x_min) * scale
                                img_y = (grid_y_max - gy) * scale  # Flip Y
                                
                                # Draw filled rectangle for this grid cell (white = solid)
                                draw.rectangle(
                                    [img_x, img_y, img_x + scale - 1, img_y + scale - 1],
                                    fill='white'
                                )
                        
                        # Save image
                        layer_z = z_layer_map.get(layer, 0)
                        img_filename = os.path.join(script_dir, f"layer_{layer:03d}_z{layer_z:.2f}.png")
                        img.save(img_filename)
                        print(f"  Saved: {os.path.basename(img_filename)}")
                    
                    print(f"[DEBUG] Generated {len(layers_to_visualize)} layer visualization PNGs")
            else:
                print(f"[DEBUG] PIL/Pillow not available - skipping PNG generation")
                print(f"[DEBUG] Install with: pip install Pillow")
        
        # Get all layers that have solid or infill material anywhere (for visualization)
        all_solid_layers = sorted(set(layer for (gx, gy, layer) in solid_at_grid.keys()))
        if debug >= 1:
            print(f"[DEBUG] Found solid infill on {len(all_solid_layers)} layers: {all_solid_layers[:10]}..." if len(all_solid_layers) > 10 else f"[DEBUG] Found solid infill on {len(all_solid_layers)} layers: {all_solid_layers}")
        
        # Cache grid bounds (optimization: calculate once, reuse everywhere)
        grid_bounds_cached = calculate_grid_bounds(solid_at_grid)
        
        # Prepare grid visualization G-code to insert at layer 0 (only if debug enabled)
        grid_visualization_gcode = []
        if debug >= 1 and grid_bounds_cached:
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
            
            if debug >= 1:
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
            grid_visualization_gcode.append(f"\n; Cross-section YZ plane (looking from +X, through X={center_gx})\n")
            grid_visualization_gcode.append(f"; Each dot = solid material at that Y,Z position\n")
            
            # Draw solid cells as individual markers
            yz_cells_count = 0
            for (gx, gy, layer) in solid_at_grid.keys():
                if gx == center_gx:  # Only cells on the center slice
                    yz_cells_count += 1
                    if layer in z_layer_map:
                        layer_z = z_layer_map[layer]
                        y_draw = gy * grid_resolution
                        x_draw = x_section_x_base + (layer_z * z_scale)
                        
                        # Draw a small marker (short line)
                        grid_visualization_gcode.append(f"G0 X{x_draw:.2f} Y{y_draw:.2f} Z{first_z:.2f} F6000\n")
                        grid_visualization_gcode.append(f"G1 X{x_draw + 0.2:.2f} Y{y_draw:.2f} Z{first_z:.2f} E{grid_e:.5f} F300\n")
                        grid_e += 0.001
            
            if debug >= 1:
                msg = f"YZ cross-section: found {yz_cells_count} solid cells at gx={center_gx}"
                print(f"[DEBUG] {msg}")
                logging.info(f"[DEBUG] {msg}")
                # Show which Y positions have cells on layer 0
                layer0_cells = [(gy, gx, layer) for (gx, gy, layer) in solid_at_grid.keys() if gx == center_gx and layer == 0]
                if layer0_cells:
                    y_positions = sorted(set([gy for gy, _, _ in layer0_cells]))
                    msg = f"Layer 0 Y positions at gx={center_gx}: {y_positions}"
                    print(f"[DEBUG] {msg}")
                    logging.info(f"[DEBUG] {msg}")
            
            # Cross-section 2: XZ plane (view from +Y direction) - shows X vs Z
            y_section_y_base = (grid_y_max + 3) * grid_resolution  # Base Y position below grid
            grid_visualization_gcode.append(f"\n; Cross-section XZ plane (looking from +Y, through Y={center_gy})\n")
            grid_visualization_gcode.append(f"; Each dot = solid material at that X,Z position\n")
            
            # Draw solid cells as individual markers
            xz_cells_count = 0
            for (gx, gy, layer) in solid_at_grid.keys():
                if gy == center_gy:  # Only cells on the center slice
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
                    f_match = re.search(r'F(\d+)', check_line)
                    if f_match and float(f_match.group(1)) >= 7200:  # High speed travel
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
                        x_match = re.search(r'X([-\d.]+)', check_line)
                        y_match = re.search(r'Y([-\d.]+)', check_line)
                        e_match = re.search(r'E([-\d.]+)', check_line)
                        if x_match and y_match and e_match:
                            x = float(x_match.group(1))
                            y = float(y_match.group(1))
                            e = float(e_match.group(1))
                            candidate_path.append((x, y))
                            e_values.append(e)
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
                z_match = re.search(r'Z([-\d.]+)', line)
                if z_match:
                    z_value = float(z_match.group(1))
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
    
    # Rolling buffer for lookback operations on OUTPUT (keep last 50 lines for Smoothificator)
    recent_output_lines = []
    max_recent_output = 50
    
    while i < total_lines:
        line = lines[i]
        line_count_processed += 1
        
        # Track TYPE markers for Z-hop exclusion logic
        if ";TYPE:" in line:
            current_type = line.strip()
            # Track when we're in bridge infill (any kind)
            if "Bridge infill" in current_type or "Internal bridge infill" in current_type:
                in_bridge_infill = True
            else:
                in_bridge_infill = False
        
        # Progress logging every 10,000 lines
        if line_count_processed % 10000 == 0:
            logging.info(f"  Processed {line_count_processed}/{len(lines)} lines ({100*line_count_processed/len(lines):.1f}%)")
        
        # Detect layer changes and get adaptive layer height
        if ";LAYER_CHANGE" in line:
            # Mark that we've started the first layer (enable Z-hop from now on)
            seen_first_layer = True
            
            # DON'T increment current_layer yet - we need to read the layer number first
            
            
            # Look ahead to find the ;LAYER: marker and get actual layer number
            layer_found = False
            for j in range(i + 1, min(i + 10, len(lines))):
                if ";LAYER:" in lines[j]:
                    layer_match = re.search(r';LAYER:(\d+)', lines[j])
                    if layer_match:
                        current_layer = int(layer_match.group(1))
                        layer_found = True
                        if current_layer % 5 == 0:
                            print(f"  Processing layer {current_layer}...")
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
            
            # Look ahead for HEIGHT and Z markers to update current_z
            for j in range(i + 1, min(i + 10, len(lines))):
                if ";Z:" in lines[j]:
                    z_marker_match = re.search(r';Z:([-\d.]+)', lines[j])
                    if z_marker_match:
                        old_z = current_z
                        current_z = float(z_marker_match.group(1))
                        #logging.info(f"\nLayer {current_layer} Z marker: updated current_z from {old_z:.3f} to {current_z:.3f}")
                if ";HEIGHT:" in lines[j]:
                    height_match = re.search(r';HEIGHT:([\d.]+)', lines[j])
                    if height_match:
                        current_layer_height = float(height_match.group(1))
                        logging.info(f"Layer {current_layer} HEIGHT marker: layer_height={current_layer_height:.3f}")
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
                
                # Generate visualization (uses cached PIL check and grid bounds)
                success = generate_lut_visualization(
                    vis_layer, vis_layer_z, noise_lut, amplitude, grid_resolution,
                    solid_at_grid, grid_cell_solid_regions, script_dir, logging
                )
                
                # Warn once if PIL not available (on first layer only)
                if not success and not HAS_PIL and current_layer == 1:
                    logging.warning("PIL/Pillow not available - skipping LUT visualization")
                    logging.warning("Install with: pip install Pillow")
            
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
            should_output_z = True
            if enable_smoothificator:
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
            actual_output_z = current_z  # Update actual output Z tracker
            
            i += 1
            continue

        # ========== SMOOTHIFICATOR: External Perimeter Processing ==========
        if enable_smoothificator and (";TYPE:External perimeter" in line or ";TYPE:Outer wall" in line or ";TYPE:Overhang perimeter" in line):
            
            external_block_lines = []
            external_block_lines.append(line)
            i += 1
            
            # Collect all lines until next TYPE change OR layer change
            # Include WIPE moves and everything up to the next TYPE marker
            while i < len(lines):
                current_line = lines[i]
                
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
            
            # Find the TRUE start position: the last XY position BEFORE this smoothificator block
            # This is where the nozzle was when it encountered the TYPE marker
            # Look backwards through recent output lines to find last XY coordinate
            start_pos = None
            for j in range(len(recent_output_lines) - 1, -1, -1):
                prev_line = recent_output_lines[j]
                if "G1" in prev_line and ("X" in prev_line or "Y" in prev_line):
                    x_val = extract_x(prev_line)
                    y_val = extract_y(prev_line)
                    if x_val is not None and y_val is not None:
                        start_pos = (x_val, y_val)
                        #logging.info(f"  [SMOOTHIFICATOR] Found TRUE start position from previous lines: X{start_pos[0]:.3f} Y{start_pos[1]:.3f}")
                        break
                    elif x_val is not None:
                        # Only X found, need to find last Y
                        for k in range(j - 1, -1, -1):
                            y_val2 = extract_y(recent_output_lines[k])
                            if y_val2 is not None:
                                start_pos = (x_val, y_val2)
                                #logging.info(f"  [SMOOTHIFICATOR] Found TRUE start position (X from recent line {j}, Y from recent line {k}): X{start_pos[0]:.3f} Y{start_pos[1]:.3f}")
                                break
                        break
                    elif y_val is not None:
                        # Only Y found, need to find last X
                        for k in range(j - 1, -1, -1):
                            x_val2 = extract_x(recent_output_lines[k])
                            if x_val2 is not None:
                                start_pos = (x_val2, y_val)
                                #logging.info(f"  [SMOOTHIFICATOR] Found TRUE start position (X from recent line {k}, Y from recent line {j}): X{start_pos[0]:.3f} Y{start_pos[1]:.3f}")
                                break
                        break
            
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
                
                # Look back to find the travel position and last E value before this TYPE comment
                travel_start_x, travel_start_y = None, None
                pre_block_e_value = None
                for back_idx in range(i - 2, max(0, i - 10), -1):
                    back_line = lines[back_idx]
                    # Look for travel move
                    if travel_start_x is None and back_line.startswith("G1") and "X" in back_line and "Y" in back_line and "E" not in back_line:
                        x_match = re.search(r'X([-\d.]+)', back_line)
                        y_match = re.search(r'Y([-\d.]+)', back_line)
                        if x_match and y_match:
                            travel_start_x = float(x_match.group(1))
                            travel_start_y = float(y_match.group(1))
                    # Also track last E value before the block (not a retraction)
                    if pre_block_e_value is None and back_line.startswith("G1") and "E" in back_line and "E-" not in back_line:
                        e_match = re.search(r'E([-\d.]+)', back_line)
                        if e_match:
                            pre_block_e_value = float(e_match.group(1))
                
                # Collect the entire perimeter block
                perimeter_block_lines = []
                
                while i < len(lines):
                    current_line = lines[i]
                    
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
                        
                        # If no block-specific travel found, use the TYPE comment travel (for first block)
                        if block_travel_x is None:
                            block_travel_x = travel_start_x
                            block_travel_y = travel_start_y
                        
                        # Detect if this layer is a base or top of a solid region
                        # Use grid position of the perimeter block to check solid regions
                        is_base_layer = False
                        is_top_layer = False
                        
                        if block_travel_x is not None and block_travel_y is not None:
                            gx = int(round(block_travel_x / grid_resolution))
                            gy = int(round(block_travel_y / grid_resolution))
                            
                            if (gx, gy) in grid_cell_solid_regions:
                                for region_start, region_end, z_bottom, z_top in grid_cell_solid_regions[(gx, gy)]:
                                    if current_layer == region_start:
                                        is_base_layer = True
                                        #logging.info(f"  [BRICKLAYERS DEBUG] Layer {current_layer}, Block #{perimeter_block_count}: Detected BASE layer (region_start={region_start}, region_end={region_end})")
                                    if current_layer == region_end:
                                        is_top_layer = True
                                        #logging.info(f"  [BRICKLAYERS DEBUG] Layer {current_layer}, Block #{perimeter_block_count}: Detected TOP layer (region_start={region_start}, region_end={region_end})")
                        
                        # On base layers: all blocks are base blocks (no shifting)
                        # On other layers: alternate between shifted and base
                        if is_base_layer:
                            is_shifted = False  # All base blocks on base layers
                        else:
                            is_shifted = perimeter_block_count % 2 == 1
                        
                        # Collect this perimeter loop
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
                        
                        # Now output this loop with bricklayer pattern
                        if is_shifted:
                            # Shifted block: single pass at Z + 0.5h
                            # On top layers, use 0.75x height (0.5 → 0.375 shift) for flat top
                            z_shift_adjusted = z_shift * 0.75 if is_top_layer else z_shift
                            adjusted_z = current_z + z_shift_adjusted
                            extrusion_factor = 1.0
                            
                            # Track actual max Z for this layer (for safe Z-hop)
                            if current_layer not in actual_layer_max_z or adjusted_z > actual_layer_max_z[current_layer]:
                                actual_layer_max_z[current_layer] = adjusted_z
                            
                            write_and_track(output_buffer, f"G0 Z{adjusted_z:.3f} ; Bricklayers shifted block #{perimeter_block_count}\n", recent_output_lines)
                            #logging.info(f"  [BRICKLAYERS] Layer {current_layer}, Block #{perimeter_block_count}: Shifted at Z={adjusted_z:.3f} (extrusion: {extrusion_factor}x)")
                            
                            # Adjust extrusion based on whether it's the last layer
                            for loop_line in loop_lines:
                                if "E" in loop_line:
                                    e_value = extract_e(loop_line)
                                    if e_value is not None:
                                        new_e_value = e_value * extrusion_factor * bricklayers_extrusion_multiplier
                                        loop_line = replace_e(loop_line, new_e_value)
                                write_and_track(output_buffer, loop_line, recent_output_lines)
                            
                            # Reset Z
                            write_and_track(output_buffer, f"G1 Z{current_z:.3f} ; Reset Z\n", recent_output_lines)
                        
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
                                start_x, start_y = None, None
                                
                                for loop_line in loop_lines:
                                    # Check if this is an extrusion move
                                    is_extrusion = loop_line.startswith("G1") and "X" in loop_line and "Y" in loop_line and "E" in loop_line and "E-" not in loop_line
                                    # Check if this is a fan command
                                    is_fan_command = loop_line.startswith("M106") or loop_line.startswith("M107")
                                    
                                    if is_extrusion:
                                        extrusion_moves.append(loop_line)
                                        # Extract start position from first extrusion move
                                        if start_x is None:
                                            x_match = re.search(r'X([-\d.]+)', loop_line)
                                            y_match = re.search(r'Y([-\d.]+)', loop_line)
                                            if x_match and y_match:
                                                start_x = float(x_match.group(1))
                                                start_y = float(y_match.group(1))
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
                                
                                # Validation: make sure we found extrusion moves and start position
                                if not extrusion_moves:
                                    logging.warning(f"  [BRICKLAYERS WARNING] Layer {current_layer}, Block #{perimeter_block_count}: No extrusion moves found in loop!")
                                if start_x is None or start_y is None:
                                    logging.warning(f"  [BRICKLAYERS WARNING] Layer {current_layer}, Block #{perimeter_block_count}: Could not extract start position!")
                                
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
                                if start_x is not None and start_y is not None:
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
                    
                    else:
                        # Non-extrusion line (comments, etc)
                        write_and_track(output_buffer, current_line, recent_output_lines)
                        j += 1
                
                continue
        
        # ========== NON-PLANAR INFILL: Process infill with Z modulation ==========
        elif enable_nonplanar and (";TYPE:Internal infill" in line):
            in_infill = True
            write_and_track(output_buffer, line, recent_output_lines)
            i += 1
            
            # Save the current layer Z before applying non-planar modulation
            layer_z = current_z
            last_infill_z = layer_z  # Track last Z used in infill
            adaptive_comment_added = False  # Track if we've added the adaptive E comment for this layer
            
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
                
                # Process infill extrusion moves
                if i not in processed_infill_indices and current_line.startswith('G1') and 'E' in current_line:
                    processed_infill_indices.add(i)
                    match = re.search(r'X([-+]?\d*\.?\d+)\s*Y([-+]?\d*\.?\d+)\s*E([-+]?\d*\.?\d+)', current_line)
                    if match:
                        x1, y1, e_end = map(float, match.groups())
                        
                        # Extract feedrate from current line if present
                        feedrate = None
                        f_match = re.search(r'F(\d+\.?\d*)', current_line)
                        if f_match:
                            feedrate = float(f_match.group(1)) * nonplanar_feedrate_multiplier
                        
                        # Get the next X,Y from the NEXT line with X,Y,E (extrusion move)
                        # CRITICAL: Only connect points in same continuous extrusion sequence
                        # Stop at retractions or travel moves (G0) to avoid connecting separate segments
                        x2, y2 = None, None
                        for j in range(i + 1, min(i + 20, len(lines))):  # Look ahead up to 20 lines
                            next_line = lines[j]
                            # STOP CONDITIONS: Don't cross these boundaries
                            if next_line.startswith(';TYPE:') or next_line.startswith(';LAYER'):
                                break  # Different feature or layer
                            if next_line.startswith('G0'):
                                break  # Travel move = end of continuous extrusion
                            # Check for retraction (negative E or E less than current)
                            if next_line.startswith('G1') and 'E' in next_line:
                                e_check = re.search(r'E([-+]?\d*\.?\d+)', next_line)
                                if e_check:
                                    next_e = float(e_check.group(1))
                                    if next_e < e_end:  # Retraction detected
                                        break  # End of continuous extrusion
                            # Skip non-movement commands
                            if next_line.startswith('M') or next_line.startswith('G92'):
                                continue
                            # Only match if line is an extrusion move with XY (has X or Y AND E)
                            if next_line.startswith('G1') and 'E' in next_line and ('X' in next_line or 'Y' in next_line):
                                next_match = re.search(r'X([-+]?\d*\.?\d+)\s*Y([-+]?\d*\.?\d+)', next_line)
                                if next_match:
                                    x2, y2 = map(float, next_match.groups())
                                    break  # Found next point, stop looking
                        
                        if x2 is not None and y2 is not None:
                            # Get the starting E value from previous line
                            e_start = 0.0
                            for j in range(i - 1, max(0, i - 10), -1):
                                prev_e_match = re.search(r'E([-+]?\d*\.?\d+)', lines[j])
                                if prev_e_match:
                                    e_start = float(prev_e_match.group(1))
                                    break
                            
                            # Calculate total extrusion for this move
                            e_delta = e_end - e_start
                            
                            segments = segment_line(x1, y1, x2, y2, segment_length)
                            
                            # Calculate total XY distance for the move
                            total_xy_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                            
                            # Calculate base E per mm of XY distance
                            # This ensures consistent extrusion regardless of segment count
                            e_per_mm = e_delta / total_xy_distance if total_xy_distance > 0 else 0
                            
                            current_e = e_start
                            prev_segment = None
                            
                            # STEP 2: Add Z modulation using LUT with wall-proximity tapering
                            # Reduce modulation near walls/perimeters to prevent visible artifacts
                            
                            for idx, (sx, sy) in enumerate(segments):
                                # Calculate XY distance for THIS segment
                                if prev_segment is not None:
                                    seg_distance = math.sqrt((sx - prev_segment[0])**2 + (sy - prev_segment[1])**2)
                                else:
                                    # First segment - distance from original start point
                                    seg_distance = math.sqrt((sx - x1)**2 + (sy - y1)**2)
                                
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
                                
                                # Sample 3D noise at this point
                                noise_value = sample_3d_noise_lut(noise_lut, sx, sy, layer_z)
                                
                                # Apply amplitude with wall-proximity tapering
                                z_offset = amplitude * taper_factor * noise_value
                                z_mod = layer_z + z_offset
                                
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
                                    write_and_track(output_buffer, f"; Valley ENTER at segment {idx} (cell {gx},{gy} is last of safezone)\n", recent_output_lines)
                                
                                # Collect segments while in valley
                                if in_valley:
                                    valley_segments.append({
                                        'x': sx,
                                        'y': sy,
                                        'z': z_mod,
                                        'e_delta': adjusted_e_for_segment,
                                        'feedrate': feedrate
                                    })
                                
                                # Detect valley exit
                                valley_exit = False
                                if in_valley and z_mod >= layer_z - valley_threshold:
                                    valley_exit = True
                                
                                # Process valley exit
                                if valley_exit:
                                    write_and_track(output_buffer, f"; Valley EXIT - filling {len(valley_segments)} segments\n", recent_output_lines)
                                    
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
                                            for seg in reversed(segments_to_fill):
                                                valley_start_e += seg['e_delta'] * 0.5  # 50% extrusion for fill
                                                write_and_track(output_buffer,
                                                    f"G1 X{seg['x']:.3f} Y{seg['y']:.3f} Z{min(current_fill_z, layer_z):.3f} E{valley_start_e:.5f}\n", recent_output_lines
                                                )
                                        else:
                                            # Odd passes: FORWARD direction
                                            for seg in segments_to_fill:
                                                valley_start_e += seg['e_delta'] * 0.5  # 50% extrusion for fill
                                                write_and_track(output_buffer,
                                                    f"G1 X{seg['x']:.3f} Y{seg['y']:.3f} Z{min(current_fill_z, layer_z):.3f} E{valley_start_e:.5f}\n", recent_output_lines
                                                )
                                    
                                    # Reset valley tracking
                                    in_valley = False
                                    valley_segments = []
                                    current_e = valley_start_e  # Sync current_e with valley fill
                                
                                # Update prev_z for next iteration
                                prev_z = z_mod
                                
                                # Output segment only if NOT in valley (valley segments are output during fill)
                                if not in_valley:
                                    # Output with Z modulation, adjusted E, and boosted feedrate
                                    if feedrate is not None:
                                        write_and_track(output_buffer, 
                                            f"G1 X{sx:.3f} Y{sy:.3f} Z{z_mod:.3f} E{current_e:.5f} F{int(feedrate)}\n", recent_output_lines
                                        )
                                    else:
                                        write_and_track(output_buffer, 
                                            f"G1 X{sx:.3f} Y{sy:.3f} Z{z_mod:.3f} E{current_e:.5f}\n", recent_output_lines
                                        )
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
                
                # If we get here, the line wasn't processed - append as-is
                write_and_track(output_buffer, current_line, recent_output_lines)
                i += 1
            
            continue
        
        # ========== SAFE Z-HOP: Apply Z-hop to ALL TRAVEL MOVES ==========
        # CRITICAL: Travel moves don't extrude, so they can collide with non-planar geometry from any layer below
        elif enable_safe_z_hop and seen_first_layer and line.startswith("G1"):
            
            # Track current E position for retraction management
            e_match = re.search(r'E([-\d.]+)', line)
            if e_match and not use_relative_e:
                current_e = float(e_match.group(1))
            
            # Track current Z if this line has Z (but don't update working_z - only layer Z changes do that)
            z_match = re.search(r'Z([-\d.]+)', line)
            if z_match:
                new_z = float(z_match.group(1))
                current_travel_z = new_z
                # Don't update working_z here - it's only updated by explicit layer Z changes
                # If this is a non-planar Z, we don't want to use it as working_z
            
            # Check if this is an extrusion move (has E parameter)
            is_extrusion = 'E' in line
            
            # If we're hopped up and about to extrude, drop back down first AND unretract if needed
            if is_hopped and is_extrusion:
                # Look ahead to see if slicer will add unretraction after drop-down
                slicer_will_unretract = False
                for j in range(i, min(i + 5, len(lines))):  # Look ahead a few lines
                    check_line = lines[j]
                    # Check for positive E move (unretraction) coming up
                    if check_line.startswith("G1") and "E" in check_line and "X" not in check_line and "Y" not in check_line and "Z" not in check_line:
                        e_check = re.search(r'E([-\d.]+)', check_line)
                        if e_check:
                            e_val = float(e_check.group(1))
                            if e_val >= 0:  # Positive E = unretraction
                                slicer_will_unretract = True
                                break
                    # Stop if we hit actual extrusion with X/Y
                    if check_line.startswith("G1") and ("X" in check_line or "Y" in check_line) and "E" in check_line:
                        break
                
                if not slicer_will_unretract:
                    # Detect retraction feedrate from recent retractions
                    retract_feedrate = 3900  # Default from slicer
                    for j in range(max(0, i - 20), i):
                        check_line = lines[j]
                        if "G1" in check_line and "E-" in check_line and "F" in check_line:
                            f_match = re.search(r'F(\d+)', check_line)
                            if f_match:
                                retract_feedrate = int(f_match.group(1))
                                break
                    
                    write_and_track(output_buffer, f"G1 E{current_e:.5f} F{retract_feedrate} ; Unretract after Z-hop\n", recent_output_lines)
                
                write_and_track(output_buffer, f"G0 Z{working_z:.3f} F8400 ; Drop back to working Z\n", recent_output_lines)
                current_travel_z = working_z
                is_hopped = False
            
            # Detect travel moves: G1 with X/Y, F (feedrate), but no E (extrusion)
            if ('X' in line or 'Y' in line) and 'F' in line and not is_extrusion:
                # Skip Z-hops for travel moves within bridge infill (tracked via TYPE markers)
                if in_bridge_infill:
                    write_and_track(output_buffer, line, recent_output_lines)
                    i += 1
                    continue
                
                # This is a travel move - apply Z-hop if needed
                # SMART: Only check CURRENT layer's max Z (all deformations build on top of current layer)
                # No need to check ancient layers below - they're already covered by new layers!
                layer_max_z = 0.0
                if current_layer in actual_layer_max_z:
                    layer_max_z = actual_layer_max_z[current_layer]
                
                if layer_max_z > 0:
                    safe_z = layer_max_z + safe_z_hop_margin
                    
                    # Only hop if we're not already above safe_z
                    if current_travel_z < safe_z:
                        # Check if slicer already retracted right before this travel
                        slicer_already_retracted = False
                        retract_feedrate = 3900  # Default from slicer
                        
                        for j in range(max(0, i - 5), i):  # Look back a few lines
                            check_line = lines[j]
                            # Check for retraction (negative E move)
                            if check_line.startswith("G1") and "E-" in check_line:
                                slicer_already_retracted = True
                                # Extract the feedrate the slicer used
                                f_match = re.search(r'F(\d+)', check_line)
                                if f_match:
                                    retract_feedrate = int(f_match.group(1))
                                break
                        
                        # Only add retraction if slicer didn't already do it
                        if not slicer_already_retracted:
                            retracted_e = current_e - z_hop_retraction
                            write_and_track(output_buffer, f"G1 E{retracted_e:.5f} F{retract_feedrate} ; Retract before Z-hop\n", recent_output_lines)
                            current_e = retracted_e  # Update E position after retraction
                        
                        write_and_track(output_buffer, f"G0 Z{safe_z:.3f} F8400 ; Safe Z-hop\n", recent_output_lines)
                        # Do the travel move (without Z since we just set it)
                        # Remove Z from the travel line if it exists
                        travel_line = REGEX_Z_SUB.sub('', line)
                        write_and_track(output_buffer, travel_line, recent_output_lines)
                        current_travel_z = safe_z
                        is_hopped = True  # Mark that we're hopped up
                        i += 1
                        continue
            
            # Not a travel move or no Z-hop needed - just output as is
            write_and_track(output_buffer, line, recent_output_lines)
            i += 1
            continue
        
        else:
            write_and_track(output_buffer, line, recent_output_lines)
            i += 1

    # Write the modified G-code
    print(f"\n[OK] Processed {current_layer} layers")
    print(f"Writing modified G-code to: {os.path.basename(output_file)}...")
    
    # Get the complete output from StringIO buffer
    modified_gcode = output_buffer.getvalue()
    output_buffer.close()
    
    # Write to file
    with open(output_file, 'w') as outfile:
        outfile.write(modified_gcode)

    logging.info("\n" + "="*70)
    logging.info("G-code processing completed successfully")
    logging.info("="*70)
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("  [OK] SILKSTEEL POST-PROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Total layers: {current_layer}")
    print(f"  Output size: {len(modified_gcode)} bytes")
    print(f"  Output file: {output_file}")
    print("=" * 70 + "\n")

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
    
    parser.add_argument('-enableBricklayers', '--enable-bricklayers', action='store_true', default=False,
                       help='Enable Bricklayers Z-shifting (default: disabled)')
    parser.add_argument('-bricklayersExtrusion', '--bricklayers-extrusion', type=float, default=1.0,
                       help='Extrusion multiplier for Bricklayers shifted blocks (default: 1.0)')
    
    parser.add_argument('-enableNonPlanar', '--enable-non-planar', action='store_true', default=False,
                       help='Enable non-planar infill modulation (default: disabled)')
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
    
    # Debug mode arguments
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument('-debug', '--debug', dest='debug_level', action='store_const', const=0, default=-1,
                       help='Enable basic debug mode with standard logging (WARNING level)')
    debug_group.add_argument('-debug-full', '--debug-full', dest='debug_level', action='store_const', const=1,
                       help='Enable full debug mode: INFO logging + PNG layer images + debug G-code visualization')
    
    args = parser.parse_args()
    
    # Set logging level based on debug argument
    debug = args.debug_level
    if debug >= 1:
        logging.getLogger().setLevel(logging.INFO)
    
    # Log all received arguments for debugging
    if debug >= 1:
        logging.info("Debug mode enabled - log level set to INFO")
        logging.info("=" * 70)
        logging.info("Command-line arguments received:")
        logging.info(f"  Raw sys.argv: {sys.argv}")
        logging.info(f"  Parsed input_file: {args.input_file}")
    logging.info(f"  Parsed output_file: {args.output_file}")
    logging.info(f"  Enable all: {args.enable_all}")
    logging.info(f"  Enable bricklayers: {args.enable_bricklayers}")
    logging.info(f"  Enable non-planar: {args.enable_non_planar}")
    logging.info("=" * 70)
    
    # Validate that we have an input file
    if not args.input_file:
        logging.error("ERROR: No input file provided!")
        sys.exit(1)
    
    logging.info(f"Starting processing of: {args.input_file}")
    
    # Handle -full flag: enable all optional features
    # Individual feature flags specified after -full can still override
    if args.enable_all:
        logging.info("Full mode enabled - activating all features")
        if not args.enable_bricklayers:
            args.enable_bricklayers = True
        if not args.enable_non_planar:
            args.enable_non_planar = True
        # Smoothificator and Safe Z-hop are already enabled by default
    
    try:
        logging.info("Calling process_gcode()...")
        process_gcode(
            input_file=args.input_file,
            output_file=args.output_file,
            outer_layer_height=args.outer_layer_height,
            enable_smoothificator=args.enable_smoothificator,
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
            debug=debug
        )
    except Exception as e:
        logging.error(f"\n{'='*70}")
        logging.error(f"FATAL ERROR: {str(e)}")
        logging.error(f"{'='*70}")
        import traceback
        logging.error(traceback.format_exc())
        
        # Print error to console
        print("\n" + "=" * 70, file=sys.stderr)
        print("  ✗ ERROR: POST-PROCESSING FAILED", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"  {str(e)}", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(2)

