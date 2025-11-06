
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

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Configure logging to both console and file
log_file = os.path.join(script_dir, "SilkSteel_log.txt")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Overwrite log each time
        logging.StreamHandler(sys.stdout)          # Also print to console
    ]
)

logging.info("=" * 70)
logging.info("SilkSteel started")
logging.info(f"Script directory: {script_dir}")
logging.info(f"Log file: {log_file}")
logging.info(f"Command line args: {sys.argv}")
logging.info("=" * 70)

# Non-planar infill constants
DEFAULT_AMPLITUDE = 3  # Default Z variation in mm [float] or layerheight [int] (reduced for smoother look)
DEFAULT_FREQUENCY = 8  # Default frequency of the sine wave (reduced for longer waves)
SEGMENT_LENGTH = 0.2  # Split infill lines into segments of this length (mm) - smaller for smoother curves

# Safe Z-hop constants
DEFAULT_ENABLE_SAFE_Z_HOP = True  # Enabled by default
DEFAULT_SAFE_Z_HOP_MARGIN = 0.5  # mm - safety margin above max Z in layer

def get_layer_height(gcode_lines):
    """Extract layer height from G-code header comments"""
    for line in gcode_lines:
        if "layer_height =" in line.lower():
            match = re.search(r'; layer_height = (\d*\.?\d+)', line, re.IGNORECASE)
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
    Generate 3D Perlin noise using numpy - simplified robust version.
    Based on: https://pvigier.github.io/2018/11/02/3d-perlin-noise-numpy.html
    
    Args:
        shape: Tuple of (width, height, depth) for output array
        res: Tuple of (res_x, res_y, res_z) - resolution of grid
        seed: Random seed for reproducibility
    
    Returns:
        3D numpy array with Perlin noise values in range [-1, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    def f(t):
        """Smoothstep function"""
        return 6*t**5 - 15*t**4 + 10*t**3
    
    # Generate coordinate grid
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1], 0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    
    # The grid might be slightly larger or smaller than expected due to rounding
    # Trim or pad to exact shape
    actual_grid = np.zeros(shape + (3,))
    min_x = min(grid.shape[0], shape[0])
    min_y = min(grid.shape[1], shape[1])
    min_z = min(grid.shape[2], shape[2])
    actual_grid[:min_x, :min_y, :min_z, :] = grid[:min_x, :min_y, :min_z, :]
    grid = actual_grid
    
    # Generate random gradients
    theta = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    phi = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    gradients = np.stack((np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)), axis=3)
    
    # Make tileable
    gradients[-1] = gradients[0]
    
    # Create gradient arrays by repeating, then trim/pad to exact shape
    def resize_gradient(g_slice):
        g_repeated = g_slice.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
        g_out = np.zeros(shape + (3,))
        min_x = min(g_repeated.shape[0], shape[0])
        min_y = min(g_repeated.shape[1], shape[1])
        min_z = min(g_repeated.shape[2], shape[2])
        g_out[:min_x, :min_y, :min_z, :] = g_repeated[:min_x, :min_y, :min_z, :]
        return g_out
    
    g000 = resize_gradient(gradients[0:-1, 0:-1, 0:-1])
    g100 = resize_gradient(gradients[1:  , 0:-1, 0:-1])
    g010 = resize_gradient(gradients[0:-1, 1:  , 0:-1])
    g110 = resize_gradient(gradients[1:  , 1:  , 0:-1])
    g001 = resize_gradient(gradients[0:-1, 0:-1, 1:  ])
    g101 = resize_gradient(gradients[1:  , 0:-1, 1:  ])
    g011 = resize_gradient(gradients[0:-1, 1:  , 1:  ])
    g111 = resize_gradient(gradients[1:  , 1:  , 1:  ])
    
    # Calculate dot products
    n000 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g000, 3)
    n100 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g100, 3)
    n010 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g010, 3)
    n110 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g110, 3)
    n001 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g001, 3)
    n101 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g101, 3)
    n011 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g011, 3)
    n111 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g111, 3)
    
    # Interpolate
    t = f(grid)
    n00 = n000*(1-t[:,:,:,0]) + t[:,:,:,0]*n100
    n10 = n010*(1-t[:,:,:,0]) + t[:,:,:,0]*n110
    n01 = n001*(1-t[:,:,:,0]) + t[:,:,:,0]*n101
    n11 = n011*(1-t[:,:,:,0]) + t[:,:,:,0]*n111
    n0 = (1-t[:,:,:,1])*n00 + t[:,:,:,1]*n10
    n1 = (1-t[:,:,:,1])*n01 + t[:,:,:,1]*n11
    
    return (1-t[:,:,:,2])*n0 + t[:,:,:,2]*n1

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

def process_gcode(input_file, output_file=None, outer_layer_height=None,
                 enable_smoothificator=True,
                 enable_bricklayers=False, bricklayers_extrusion_multiplier=1.0,
                 enable_nonplanar=False, deform_type='sine',
                 segment_length=SEGMENT_LENGTH, amplitude=DEFAULT_AMPLITUDE, frequency=DEFAULT_FREQUENCY,
                 enable_safe_z_hop=DEFAULT_ENABLE_SAFE_Z_HOP, safe_z_hop_margin=DEFAULT_SAFE_Z_HOP_MARGIN,
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
    
    if outer_layer_height is None:
        outer_layer_height = get_min_layer_height(lines)
        if outer_layer_height is None:
            outer_layer_height = base_layer_height / 2
            logging.warning(f"Could not find min_layer_height, using half of base: {outer_layer_height}mm")
    
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
        logging.info(f"Non-planar infill - Deform type: {deform_type}, Amplitude: {amplitude}mm, Frequency: {frequency}")
    
    # Print feature settings summary to console
    print("\nFeature Settings:")
    if enable_smoothificator:
        print(f"  • Smoothificator: target layer height = {outer_layer_height:.3f}mm")
    if enable_bricklayers:
        print(f"  • Bricklayers: extrusion multiplier = {bricklayers_extrusion_multiplier:.2f}x")
    if enable_nonplanar:
        print(f"  • Non-planar Infill: amplitude = {amplitude:.2f}mm, frequency = {frequency:.2f}, type = {deform_type}")
    if enable_safe_z_hop:
        print(f"  • Safe Z-hop: margin = {safe_z_hop_margin:.2f}mm")
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
    layer_max_z = {}  # layer_num -> maximum Z value seen in that layer
    current_travel_z = 0.0  # Track current Z during travel moves
    seen_first_layer = False  # Don't apply Z-hop until we've started printing
    
    # Bricklayers variables
    perimeter_block_count = 0
    inside_perimeter_block = False
    is_shifted = False
    previous_g1_movement = None
    previous_f_speed = None
    
    # Non-planar infill variables
    solid_infill_heights = []
    in_infill = False
    processed_infill_indices = set()
    
    # First pass: Build 3D grid showing which Z layers have solid at each XY position
    # Then for each layer, determine the safe Z range (space between solid layers)
    grid_resolution = 1.0  # 1mm grid spacing
    solid_at_grid = {}  # (grid_x, grid_y, layer_num) -> True if solid exists
    z_layer_map = {}    # layer_num -> Z height
    layer_z_map = {}    # Z height -> layer_num
    
    if enable_nonplanar:
        logging.info("\n" + "="*70)
        logging.info("PASS 1: Building 3D solid occupancy grid")
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
        if debug:
            print(f"[DEBUG] Found {total_layers} layers")
        
        # Second pass: Mark which grid cells have solid infill at each layer
        logging.info("\nScanning solid infill to build occupancy grid...")
        logging.info(f"  Processing {len(lines)} lines...")
        temp_z = 0.0
        prev_layer_z = 0.0
        current_layer_height = base_layer_height  # Default
        current_layer_num = -1  # Will be set from ;LAYER: marker
        in_solid_infill = False
        last_solid_pos = None  # Track last position to mark all cells along line
        debug_line_count = 0
        debug_cells_marked = 0
        
        for line in lines:
            if ';LAYER:' in line:
                layer_match = re.search(r';LAYER:(\d+)', line)
                if layer_match:
                    current_layer_num = int(layer_match.group(1))
                    last_solid_pos = None  # Reset on layer change
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
               ';TYPE:Internal bridge infill' in line or \
               ';TYPE:External perimeter' in line or ';TYPE:Internal perimeter' in line or ';TYPE:Perimeter' in line or \
               ';TYPE:Overhang perimeter' in line:
                in_solid_infill = True
                last_solid_pos = None  # Reset when entering solid infill or perimeter
                if temp_z not in solid_infill_heights:
                    solid_infill_heights.append(temp_z)
            elif in_solid_infill and ';TYPE:' in line:
                in_solid_infill = False
                last_solid_pos = None  # Reset when exiting solid infill or perimeter
            
            # Track position during solid infill (both G0 and G1 moves)
            if in_solid_infill and (line.startswith('G1') or line.startswith('G0')):
                xy_match = re.search(r'X([-+]?\d*\.?\d+)\s*Y([-+]?\d*\.?\d+)', line)
                if xy_match:
                    x, y = map(float, xy_match.groups())
                    gx = int(round(x / grid_resolution))
                    gy = int(round(y / grid_resolution))
                    
                    # Only mark grid cells for G1 moves WITH extrusion (has E value)
                    has_extrusion = line.startswith('G1') and 'E' in line
                    if has_extrusion:
                        # Mark all grid cells from last position to current position
                        if last_solid_pos is not None:
                            last_gx, last_gy = last_solid_pos
                            # Use Bresenham-like algorithm to mark all cells along the line
                            dx = abs(gx - last_gx)
                            dy = abs(gy - last_gy)
                            sx = 1 if gx > last_gx else -1
                            sy = 1 if gy > last_gy else -1
                            err = dx - dy
                            
                            cells_on_this_line = 0
                            curr_gx, curr_gy = last_gx, last_gy
                            while True:
                                solid_at_grid[(curr_gx, curr_gy, current_layer_num)] = True
                                cells_on_this_line += 1
                                if curr_gx == gx and curr_gy == gy:
                                    break
                                e2 = 2 * err
                                if e2 > -dy:
                                    err -= dy
                                    curr_gx += sx
                                if e2 < dx:
                                    err += dx
                                    curr_gy += sy
                            
                            debug_line_count += 1
                            debug_cells_marked += cells_on_this_line
                            
                            # Debug output for first few lines
                            if debug and debug_line_count <= 10:
                                print(f"[DEBUG] Line {debug_line_count}: ({last_gx},{last_gy}) -> ({gx},{gy}) marked {cells_on_this_line} cells")
                        else:
                            # First point - just mark it
                            solid_at_grid[(gx, gy, current_layer_num)] = True
                    
                    # Update position for BOTH G0 and G1
                    last_solid_pos = (gx, gy)
        
        logging.info(f"Total solid infill layers: {len(solid_infill_heights)}")
        logging.info(f"Grid cells marked with solid: {len(solid_at_grid)}")
        if debug:
            print(f"[DEBUG] Marked {len(solid_at_grid)} grid cells with solid infill")
            print(f"[DEBUG] Processed {debug_line_count} line segments, avg {debug_cells_marked/max(1,debug_line_count):.1f} cells per line")
        
        # Third pass: Calculate safe Z range PER GRID CELL
        # For each grid cell (gx, gy), track the safe Z range between solid layers
        # z_min = Z of last solid layer seen at this cell (bottom of safe range)
        # z_max = Z of next solid layer seen at this cell (top of safe range)
        logging.info("\nCalculating safe Z ranges per grid cell...")
        logging.info(f"  Processing {len(set((gx, gy) for gx, gy, _ in solid_at_grid.keys()))} unique grid positions...")
        grid_cell_safe_z = {}  # (gx, gy) -> list of (layer_num, z_min, z_max) tuples
        
        # Get all unique grid positions
        all_grid_positions = sorted(set((gx, gy) for gx, gy, _ in solid_at_grid.keys()))
        
        # For each grid cell, scan through layers and find safe ranges
        for gx, gy in all_grid_positions:
            # Get all layers where this grid cell has solid infill, sorted
            solid_layers_at_cell = sorted([layer for (g_x, g_y, layer) in solid_at_grid.keys() 
                                          if g_x == gx and g_y == gy])
            
            if not solid_layers_at_cell:
                continue
            
            # Build safe ranges for this cell
            safe_ranges = []
            
            # For each pair of consecutive solid layers, the space between is safe
            for i in range(len(solid_layers_at_cell) - 1):
                layer_bottom = solid_layers_at_cell[i]
                layer_top = solid_layers_at_cell[i + 1]
                
                z_bottom = z_layer_map[layer_bottom]
                z_top = z_layer_map[layer_top]
                
                # The solid at layer_bottom occupies space from z_bottom to (z_bottom + base_layer_height)
                # So the safe range starts AFTER the solid layer ends
                z_min_safe = z_bottom + base_layer_height
                z_max_safe = z_top  # Top boundary is where the next solid layer starts
                
                # For all layers between these two solid layers, the safe range is [z_min_safe, z_max_safe]
                for layer_num in range(layer_bottom + 1, layer_top):
                    safe_ranges.append((layer_num, z_min_safe, z_max_safe))
            
            # Store all safe ranges for this grid cell
            if safe_ranges:
                grid_cell_safe_z[(gx, gy)] = safe_ranges
        
        logging.info(f"  Calculated safe Z ranges for {len(all_grid_positions)} unique grid cells")
        total_safe_ranges = sum(len(ranges) for ranges in grid_cell_safe_z.values())
        logging.info(f"  Total safe range entries: {total_safe_ranges}")
        
        if debug:
            print(f"[DEBUG] Calculated safe Z ranges for {len(all_grid_positions)} unique grid cells")
            total_safe_ranges = sum(len(ranges) for ranges in grid_cell_safe_z.values())
            print(f"[DEBUG] Total safe range entries: {total_safe_ranges}")
            
            # Show example safe ranges for a few grid cells
            print(f"\n[DEBUG VISUALIZATION] Example safe Z ranges for first 5 grid cells:")
            for idx, ((gx, gy), ranges) in enumerate(list(grid_cell_safe_z.items())[:5]):
                print(f"  Grid cell ({gx}, {gy}) at X={gx*grid_resolution:.1f}, Y={gy*grid_resolution:.1f}:")
                for layer_num, z_min, z_max in ranges[:3]:  # Show first 3 ranges
                    layer_z = z_layer_map.get(layer_num, 0)
                    print(f"    Layer {layer_num} (Z={layer_z:.2f}): safe range [{z_min:.2f}, {z_max:.2f}]")
                if len(ranges) > 3:
                    print(f"    ... and {len(ranges) - 3} more ranges")
        
        # Get all layers that have solid infill anywhere (for visualization)
        all_solid_layers = sorted(set(layer for _, _, layer in solid_at_grid.keys()))
        if debug:
            print(f"[DEBUG] Found solid infill on {len(all_solid_layers)} layers: {all_solid_layers[:10]}..." if len(all_solid_layers) > 10 else f"[DEBUG] Found solid infill on {len(all_solid_layers)} layers: {all_solid_layers}")
        
        # Prepare grid visualization G-code to insert at layer 0 (only if debug enabled)
        grid_visualization_gcode = []
        if debug:
            grid_visualization_gcode.append("; ========================================\n")
            grid_visualization_gcode.append("; GRID VISUALIZATION - Solid Infill Detection & Safe Z Ranges PER CELL\n")
            grid_visualization_gcode.append("; Grid: horizontal and vertical lines showing grid structure\n")
            grid_visualization_gcode.append("; + markers: safe Z range boundaries for each grid cell\n")
            grid_visualization_gcode.append("; ========================================\n")
            
            # Calculate overall grid bounds from solid cells
            all_gx = [gx for gx, gy, lay in solid_at_grid.keys()]
            all_gy = [gy for gx, gy, lay in solid_at_grid.keys()]
            grid_x_min, grid_x_max = min(all_gx), max(all_gx)
            grid_y_min, grid_y_max = min(all_gy), max(all_gy)
            
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
            # AND also visualize cells with solid material (even if no safe ranges)
            grid_visualization_gcode.append(f"\n; === SAFE Z RANGE VISUALIZATION (per grid cell) ===\n")
            
            # First, visualize safe ranges (gaps between solid layers)
            for (gx, gy), ranges in grid_cell_safe_z.items():
                if not ranges:
                    continue
                
                # Find overall min and max Z for this cell across all layers
                overall_z_min = min(z_min for _, z_min, _ in ranges)
                overall_z_max = max(z_max for _, _, z_max in ranges)
                
                # Skip sentinel values
                if overall_z_min < -100 or overall_z_max > 100:
                    continue
                
                # Markers at exact grid coordinates (where LUT data is)
                x = gx * grid_resolution
                y = gy * grid_resolution
                
                # Draw + marker at z_min (bottom of safe range)
                grid_visualization_gcode.append(f"; Cell ({gx},{gy}): z_min={overall_z_min:.2f}\n")
                grid_visualization_gcode.append(f"G0 Z{overall_z_min:.2f} F3000\n")
                grid_visualization_gcode.append(f"G0 X{x-0.3:.2f} Y{y:.2f} Z{overall_z_min:.2f} F6000\n")
                grid_visualization_gcode.append(f"G1 X{x+0.3:.2f} Y{y:.2f} Z{overall_z_min:.2f} E{grid_e:.5f} F1200\n")
                grid_e += 0.002
                grid_visualization_gcode.append(f"G0 X{x:.2f} Y{y-0.3:.2f} Z{overall_z_min:.2f} F6000\n")
                grid_visualization_gcode.append(f"G1 X{x:.2f} Y{y+0.3:.2f} Z{overall_z_min:.2f} E{grid_e:.5f} F1200\n")
                grid_e += 0.002
                
                # Draw + marker at z_max (top of safe range)
                grid_visualization_gcode.append(f"; Cell ({gx},{gy}): z_max={overall_z_max:.2f}\n")
                grid_visualization_gcode.append(f"G0 Z{overall_z_max:.2f} F3000\n")
                grid_visualization_gcode.append(f"G0 X{x-0.3:.2f} Y{y:.2f} Z{overall_z_max:.2f} F6000\n")
                grid_visualization_gcode.append(f"G1 X{x+0.3:.2f} Y{y:.2f} Z{overall_z_max:.2f} E{grid_e:.5f} F1200\n")
                grid_e += 0.002
                grid_visualization_gcode.append(f"G0 X{x:.2f} Y{y-0.3:.2f} Z{overall_z_max:.2f} F6000\n")
                grid_visualization_gcode.append(f"G1 X{x:.2f} Y{y+0.3:.2f} Z{overall_z_max:.2f} E{grid_e:.5f} F1200\n")
                grid_e += 0.002
            
            # Second, visualize ALL solid cells (show min/max Z where solid exists)
            grid_visualization_gcode.append(f"\n; === SOLID MATERIAL VISUALIZATION (cells with continuous solid) ===\n")
            for gx, gy in all_grid_positions:
                # Skip if already visualized in safe ranges
                if (gx, gy) in grid_cell_safe_z:
                    continue
                
                # Get all layers where this cell has solid material
                solid_layers = sorted([layer for (g_x, g_y, layer) in solid_at_grid.keys() 
                                      if g_x == gx and g_y == gy])
                
                if len(solid_layers) < 2:
                    continue  # Need at least 2 layers to show a range
                
                # Show min and max Z where solid exists
                z_min = z_layer_map[solid_layers[0]]
                z_max = z_layer_map[solid_layers[-1]]
                
                x = gx * grid_resolution
                y = gy * grid_resolution
                
                # Draw + marker at z_min (first solid layer)
                grid_visualization_gcode.append(f"; Cell ({gx},{gy}) SOLID: z_min={z_min:.2f}\n")
                grid_visualization_gcode.append(f"G0 Z{z_min:.2f} F3000\n")
                grid_visualization_gcode.append(f"G0 X{x-0.3:.2f} Y{y:.2f} Z{z_min:.2f} F6000\n")
                grid_visualization_gcode.append(f"G1 X{x+0.3:.2f} Y{y:.2f} Z{z_min:.2f} E{grid_e:.5f} F1200\n")
                grid_e += 0.002
                grid_visualization_gcode.append(f"G0 X{x:.2f} Y{y-0.3:.2f} Z{z_min:.2f} F6000\n")
                grid_visualization_gcode.append(f"G1 X{x:.2f} Y{y+0.3:.2f} Z{z_min:.2f} E{grid_e:.5f} F1200\n")
                grid_e += 0.002
                
                # Draw + marker at z_max (last solid layer)
                grid_visualization_gcode.append(f"; Cell ({gx},{gy}) SOLID: z_max={z_max:.2f}\n")
                grid_visualization_gcode.append(f"G0 Z{z_max:.2f} F3000\n")
                grid_visualization_gcode.append(f"G0 X{x-0.3:.2f} Y{y:.2f} Z{z_max:.2f} F6000\n")
                grid_visualization_gcode.append(f"G1 X{x+0.3:.2f} Y{y:.2f} Z{z_max:.2f} E{grid_e:.5f} F1200\n")
                grid_e += 0.002
                grid_visualization_gcode.append(f"G0 X{x:.2f} Y{y-0.3:.2f} Z{z_max:.2f} F6000\n")
                grid_visualization_gcode.append(f"G1 X{x:.2f} Y{y+0.3:.2f} Z{z_max:.2f} E{grid_e:.5f} F1200\n")
                grid_e += 0.002
            
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
    
    modified_lines = []
    i = 0
    last_type_seen = None  # Track the last TYPE: marker we encountered
    line_count_processed = 0
    
    while i < len(lines):
        line = lines[i]
        line_count_processed += 1
        
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
                if debug:
                    print(f"[DEBUG] Inserting grid visualization at layer 0 ({len(grid_visualization_gcode)} lines)")
                modified_lines.extend(grid_visualization_gcode)
                del grid_visualization_gcode  # Only insert once
            
            perimeter_block_count = 0  # Reset block counter for new layer
            move_history = []  # Clear move history for new layer
            last_type_seen = None  # Reset TYPE tracking for new layer
            
            # Look ahead for HEIGHT and Z markers to update current_z
            for j in range(i + 1, min(i + 10, len(lines))):
                if ";Z:" in lines[j]:
                    z_marker_match = re.search(r';Z:([-\d.]+)', lines[j])
                    if z_marker_match:
                        old_z = current_z
                        current_z = float(z_marker_match.group(1))
                        logging.info(f"\nLayer {current_layer} Z marker: updated current_z from {old_z:.3f} to {current_z:.3f}")
                if ";HEIGHT:" in lines[j]:
                    height_match = re.search(r';HEIGHT:([\d.]+)', lines[j])
                    if height_match:
                        current_layer_height = float(height_match.group(1))
                        logging.info(f"Layer {current_layer} HEIGHT marker: layer_height={current_layer_height:.3f}")
                        break
            modified_lines.append(line)
            i += 1
            continue

        # Get current Z position (for tracking only, don't recalculate layer height)
        # Match G1 commands that contain Z (with or without X/Y)
        # IMPORTANT: Don't track Z during non-planar infill (to avoid tracking modulated Z values)
        if line.startswith("G1") and "Z" in line and "X" not in line and "Y" not in line and not in_infill:
            logging.info(f"  [Z-MATCH] Line index {i}: {line.strip()}")
            z_match = re.search(r'Z([-\d.]+)', line)
            if z_match:
                old_z = current_z
                current_z = float(z_match.group(1))
                # DON'T calculate layer height from Z - use the HEIGHT marker instead!
            
            # Check if next lines contain external perimeter - if so, don't output Z yet
            # Smoothificator will handle Z for each pass
            should_output_z = True
            if enable_smoothificator:
                # Look ahead to see if external perimeter is coming
                for j in range(i + 1, min(i + 10, len(lines))):
                    if ";TYPE:External perimeter" in lines[j] or ";TYPE:Outer wall" in lines[j] or ";TYPE:Overhang perimeter" in lines[j]:
                        should_output_z = False
                        logging.info(f"  [SMOOTHIFICATOR] Skipping Z move - external perimeter follows")
                        break
                    # Stop looking if we hit actual extrusion
                    if "G1" in lines[j] and "E" in lines[j] and ("X" in lines[j] or "Y" in lines[j]):
                        break
            
            if should_output_z:
                modified_lines.append(line)
            actual_output_z = current_z  # Update actual output Z tracker
            
            i += 1
            continue

        # ========== SMOOTHIFICATOR: External Perimeter Processing ==========
        if enable_smoothificator and (";TYPE:External perimeter" in line or ";TYPE:Outer wall" in line or ";TYPE:Overhang perimeter" in line):
            last_type_seen = "external_perimeter"  # Mark current TYPE
            
            
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
            
            logging.info(f"  [SMOOTHIFICATOR] Collected external perimeter block with {len(external_block_lines)} lines")
            
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
                logging.info(f"  [SMOOTHIFICATOR] Layer {current_layer}: {passes_needed} passes at {height_per_pass:.4f}mm each")
            else:
                passes_needed = 1
                height_per_pass = effective_layer_height
                extrusion_multiplier = 1.0
            
            # SIMPLE APPROACH: Just duplicate the block N times with adjusted Z and E
            current_e = 0.0
            
            # Find the TRUE start position: the last XY position BEFORE this smoothificator block
            # This is where the nozzle was when it encountered the TYPE marker
            # Look backwards from current position to find last XY coordinate
            start_pos = None
            for j in range(len(modified_lines) - 1, max(0, len(modified_lines) - 50), -1):
                prev_line = modified_lines[j]
                if "G1" in prev_line and ("X" in prev_line or "Y" in prev_line):
                    x_match = re.search(r'X([-\d.]+)', prev_line)
                    y_match = re.search(r'Y([-\d.]+)', prev_line)
                    if x_match and y_match:
                        start_pos = (float(x_match.group(1)), float(y_match.group(1)))
                        logging.info(f"  [SMOOTHIFICATOR] Found TRUE start position from previous lines: X{start_pos[0]:.3f} Y{start_pos[1]:.3f}")
                        break
                    elif x_match:
                        # Only X found, need to find last Y
                        x_val = float(x_match.group(1))
                        for k in range(j - 1, max(0, len(modified_lines) - 50), -1):
                            if "Y" in modified_lines[k]:
                                y_match2 = re.search(r'Y([-\d.]+)', modified_lines[k])
                                if y_match2:
                                    start_pos = (x_val, float(y_match2.group(1)))
                                    logging.info(f"  [SMOOTHIFICATOR] Found TRUE start position (X from line {j}, Y from line {k}): X{start_pos[0]:.3f} Y{start_pos[1]:.3f}")
                                    break
                        break
                    elif y_match:
                        # Only Y found, need to find last X
                        y_val = float(y_match.group(1))
                        for k in range(j - 1, max(0, len(modified_lines) - 50), -1):
                            if "X" in modified_lines[k]:
                                x_match2 = re.search(r'X([-\d.]+)', modified_lines[k])
                                if x_match2:
                                    start_pos = (float(x_match2.group(1)), y_val)
                                    logging.info(f"  [SMOOTHIFICATOR] Found TRUE start position (X from line {k}, Y from line {j}): X{start_pos[0]:.3f} Y{start_pos[1]:.3f}")
                                    break
                        break
            
            for pass_num in range(passes_needed):
                # Calculate Z for this pass (work DOWN from current_z)
                if passes_needed == 1:
                    pass_z = current_z
                else:
                    pass_z = current_z - ((passes_needed - pass_num - 1) * height_per_pass)
                
                if pass_num == 0:
                    modified_lines.append(f"; ====== SMOOTHIFICATOR START: {passes_needed} passes at {height_per_pass:.4f}mm each ======\n")
                
                # Output Z move
                modified_lines.append(f"G1 Z{pass_z:.3f} ; Pass {pass_num + 1} of {passes_needed}\n")
                
                # For pass 2+, travel back to TRUE start position (where we were before the block)
                if pass_num > 0 and start_pos:
                    modified_lines.append(f"G1 X{start_pos[0]:.3f} Y{start_pos[1]:.3f} F8400 ; Travel to start\n")
                
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
                        block_line = re.sub(r'\s*Z[-\d.]+', '', block_line)
                    
                    # Adjust E values
                    if "G1" in block_line and "E" in block_line:
                        e_match = re.search(r'E([-\d.]+)', block_line)
                        if e_match:
                            original_e = float(e_match.group(1))
                            
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
                            block_line = re.sub(r'E[-\d.]+', f'E{current_e:.5f}', block_line)
                    
                    modified_lines.append(block_line)
            
            continue
        
        # ========== BRICKLAYERS: Internal Perimeter Processing ==========
        elif enable_bricklayers and (";TYPE:Perimeter" in line or ";TYPE:Internal perimeter" in line or ";TYPE:Inner wall" in line):
            if ";TYPE:External perimeter" not in line:
                modified_lines.append(line)
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
                    
                    if ";TYPE:" in current_line:
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
                        
                        # On layer 0: all blocks are base blocks (no shifting)
                        # On layer 1+: alternate between shifted and base
                        if current_layer == 0:
                            is_shifted = False  # All base blocks on layer 0
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
                            # On last layer, use 0.5x extrusion to end flat
                            adjusted_z = current_z + z_shift
                            extrusion_factor = 0.5 if is_last_layer else 1.0
                            
                            modified_lines.append(f"G1 Z{adjusted_z:.3f} ; Bricklayers shifted block #{perimeter_block_count}\n")
                            logging.info(f"  [BRICKLAYERS] Layer {current_layer}, Block #{perimeter_block_count}: Shifted at Z={adjusted_z:.3f} (extrusion: {extrusion_factor}x)")
                            
                            # Adjust extrusion based on whether it's the last layer
                            for loop_line in loop_lines:
                                if "E" in loop_line:
                                    e_match = re.search(r'E([-\d.]+)', loop_line)
                                    if e_match:
                                        e_value = float(e_match.group(1))
                                        new_e_value = e_value * extrusion_factor * bricklayers_extrusion_multiplier
                                        loop_line = re.sub(r'E[-\d.]+', f'E{new_e_value:.5f}', loop_line)
                                modified_lines.append(loop_line)
                            
                            # Reset Z
                            modified_lines.append(f"G1 Z{current_z:.3f} ; Reset Z\n")
                        
                        else:
                            # Base block (non-shifted)
                            # Only use two-pass on layer 0, single pass thereafter
                            if current_layer == 0:
                                # Layer 0: TWO passes at 0.75h each (total 1.5h)
                                pass1_z = current_z  # Start at normal layer Z
                                pass2_z = current_z + (current_layer_height * 0.75)  # Raise by 0.75h for second pass
                                
                                logging.info(f"  [BRICKLAYERS] Layer {current_layer}, Block #{perimeter_block_count}: Base in 2 passes at Z={pass1_z:.3f} and Z={pass2_z:.3f}")
                                
                                # Extract starting position from first extrusion move for pass 2 travel
                                start_x, start_y = None, None
                                # Separate all lines before ending from ending commands
                                main_section = []
                                ending_commands = []
                                in_ending = False
                                seen_extrusion = False  # Track if we've seen any extrusion moves yet
                                
                                for loop_line in loop_lines:
                                    # Check if we've reached the ending sequence
                                    if not in_ending:
                                        # Mark that we've seen extrusion
                                        if loop_line.startswith("G1") and "X" in loop_line and "Y" in loop_line and "E" in loop_line:
                                            seen_extrusion = True
                                        
                                        # Only consider M107/G92 as ending if we've already seen extrusion moves
                                        if seen_extrusion and (loop_line.startswith("M107") or loop_line.startswith("G92") or (loop_line.startswith("G1") and "E-" in loop_line and "X" not in loop_line)):
                                            in_ending = True
                                            ending_commands.append(loop_line)
                                        else:
                                            main_section.append(loop_line)
                                            # Get start position from first extrusion move
                                            if start_x is None and loop_line.startswith("G1") and "X" in loop_line and "Y" in loop_line and "E" in loop_line:
                                                x_match = re.search(r'X([-\d.]+)', loop_line)
                                                y_match = re.search(r'Y([-\d.]+)', loop_line)
                                                if x_match and y_match:
                                                    start_x = float(x_match.group(1))
                                                    start_y = float(y_match.group(1))
                                    else:
                                        ending_commands.append(loop_line)
                                
                                # Pass 1: First 0.75h - all main section (extrusion moves + M117 messages)
                                if pre_block_e_value is not None:
                                    modified_lines.append(f"G1 Z{pass1_z:.3f} E{pre_block_e_value:.5f} ; Bricklayers base block #{perimeter_block_count}, pass 1/2\n")
                                else:
                                    modified_lines.append(f"G1 Z{pass1_z:.3f} ; Bricklayers base block #{perimeter_block_count}, pass 1/2\n")
                                # Reset E after Z lift so extrusion values start fresh
                                modified_lines.append("G92 E0\n")
                                last_e_value = None
                                for line in main_section:
                                    if line.startswith("G1") and "E" in line:
                                        e_match = re.search(r'E([-\d.]+)', line)
                                        if e_match:
                                            e_value = float(e_match.group(1))
                                            new_e_value = e_value * 0.75 * bricklayers_extrusion_multiplier
                                            line = re.sub(r'E[-\d.]+', f'E{new_e_value:.5f}', line)
                                            last_e_value = new_e_value  # Track last E value for travel move
                                    modified_lines.append(line)
                                
                                # Pass 2: Second 0.75h - travel to start, then only extrusion moves
                                # Reset extruder position before pass 2
                                modified_lines.append("G92 E0 ; Reset extruder for pass 2\n")
                                # Use block-specific travel position if found, otherwise fall back to first extrusion point
                                use_x = block_travel_x if block_travel_x is not None else start_x
                                use_y = block_travel_y if block_travel_y is not None else start_y
                                if use_x is not None and use_y is not None:
                                    modified_lines.append(f"G1 X{use_x:.3f} Y{use_y:.3f} F8400 ; Travel to start for pass 2\n")
                                # Raise Z for pass 2
                                modified_lines.append(f"G1 Z{pass2_z:.3f} ; Bricklayers base block #{perimeter_block_count}, pass 2/2\n")
                                for line in main_section:
                                    # Only output actual extrusion G1 moves (skip M117, M107, etc. on second pass)
                                    if line.startswith("G1") and "X" in line and "Y" in line and "E" in line:
                                        e_match = re.search(r'E([-\d.]+)', line)
                                        if e_match:
                                            e_value = float(e_match.group(1))
                                            new_e_value = e_value * 0.75 * bricklayers_extrusion_multiplier
                                            line = re.sub(r'E[-\d.]+', f'E{new_e_value:.5f}', line)
                                        modified_lines.append(line)
                                
                                # Output ending commands once (M107, G92, retraction, travel)
                                for cmd in ending_commands:
                                    modified_lines.append(cmd)
                                
                                # Reset Z
                                modified_lines.append(f"G1 Z{current_z:.3f} ; Reset Z\n")
                            else:
                                # Layer 1+: Single pass at Z + 0.5h (sits on previous layer's shifted block)
                                # On last layer, use 0.5x extrusion to end flat
                                adjusted_z = current_z + z_shift
                                extrusion_factor = 0.5 if is_last_layer else 1.0
                                
                                modified_lines.append(f"G1 Z{adjusted_z:.3f} ; Bricklayers base block #{perimeter_block_count}\n")
                                logging.info(f"  [BRICKLAYERS] Layer {current_layer}, Block #{perimeter_block_count}: Base at Z={adjusted_z:.3f} (extrusion: {extrusion_factor}x)")
                                
                                for loop_line in loop_lines:
                                    if "E" in loop_line:
                                        e_match = re.search(r'E([-\d.]+)', loop_line)
                                        if e_match:
                                            e_value = float(e_match.group(1))
                                            new_e_value = e_value * extrusion_factor * bricklayers_extrusion_multiplier
                                            loop_line = re.sub(r'E[-\d.]+', f'E{new_e_value:.5f}', loop_line)
                                    modified_lines.append(loop_line)
                                
                                # Reset Z
                                modified_lines.append(f"G1 Z{current_z:.3f} ; Reset Z\n")
                    
                    else:
                        # Non-extrusion line (comments, etc)
                        modified_lines.append(current_line)
                        j += 1
                
                continue
        
        # ========== NON-PLANAR INFILL: Process infill with Z modulation ==========
        elif enable_nonplanar and (";TYPE:Internal infill" in line):
            in_infill = True
            modified_lines.append(line)
            i += 1
            
            # Save the current layer Z before applying non-planar modulation
            layer_z = current_z
            last_infill_z = layer_z  # Track last Z used in infill
            
            # Z bounds are now in the grid maps (grid_z_min_map, grid_z_max_map)
            # No need to calculate per-layer bounds
            
            # Process infill lines
            while i < len(lines):
                current_line = lines[i]
                
                # CRITICAL: Check for layer change FIRST - restore Z before new layer starts!
                if current_line.startswith(";LAYER_CHANGE") or current_line.startswith(";LAYER:"):
                    in_infill = False
                    # Restore Z before the new layer begins
                    if last_infill_z != layer_z:
                        modified_lines.append(f"G1 Z{layer_z:.3f} F8400 ; Restore layer Z after non-planar infill\n")
                        current_z = float(f"{layer_z:.3f}")
                        actual_output_z = current_z
                        old_z = current_z - current_layer_height
                        logging.info(f"  [NON-PLANAR INFILL] Restoring Z from {last_infill_z:.3f} to {layer_z:.3f} at layer boundary")
                    # CRITICAL: Decrement i so main loop will process this LAYER_CHANGE line
                    i -= 1
                    break
                
                if ";TYPE:" in current_line:
                    in_infill = False
                    # CRITICAL: Restore proper Z height after infill with non-planar modulation
                    if last_infill_z != layer_z:
                        modified_lines.append(f"G1 Z{layer_z:.3f} F8400 ; Restore layer Z after non-planar infill\n")
                        # Update runtime Z trackers so subsequent processing uses the restored layer Z
                        current_z = float(f"{layer_z:.3f}")
                        actual_output_z = current_z
                        # CRITICAL: Also update old_z to maintain proper layer bottom for smoothificator
                        old_z = current_z - current_layer_height
                        logging.info(f"  [NON-PLANAR INFILL] Restoring Z from {last_infill_z:.3f} to {layer_z:.3f} at TYPE change")
                    break
                
                # Process infill extrusion moves
                if i not in processed_infill_indices and current_line.startswith('G1') and 'E' in current_line:
                    processed_infill_indices.add(i)
                    match = re.search(r'X([-+]?\d*\.?\d+)\s*Y([-+]?\d*\.?\d+)\s*E([-+]?\d*\.?\d+)', current_line)
                    if match:
                        x1, y1, e_end = map(float, match.groups())
                        
                        # Get the next X,Y from the NEXT line with X,Y,E (extrusion move)
                        # Skip over M117 commands that might be in between
                        x2, y2 = None, None
                        for j in range(i + 1, min(i + 5, len(lines))):  # Look ahead up to 5 lines
                            next_line = lines[j]
                            # Skip M117 (display messages) and other non-movement commands
                            if next_line.startswith('M117'):
                                continue
                            # Only match if line is an extrusion move (has E)
                            if next_line.startswith('G1') and 'E' in next_line:
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
                            
                            # Calculate extrusion per segment
                            num_segments = len(segments)
                            e_per_segment = e_delta / num_segments
                            current_e = e_start
                            
                            # STEP 2: Add Z modulation using 3D noise LUT with wall-proximity tapering
                            # Reduce modulation near walls/perimeters to prevent visible artifacts
                            
                            for idx, (sx, sy) in enumerate(segments):
                                current_e += e_per_segment
                                
                                # Calculate distance to nearest perimeter/solid to taper modulation
                                # Check surrounding grid cells for solid material at current layer
                                gx = int(round(sx / grid_resolution))
                                gy = int(round(sy / grid_resolution))
                                
                                # Find minimum distance to any solid cell at this layer
                                min_dist_to_solid = float('inf')
                                search_radius = 5  # Check cells within 5mm
                                for dx in range(-search_radius, search_radius + 1):
                                    for dy in range(-search_radius, search_radius + 1):
                                        check_gx = gx + dx
                                        check_gy = gy + dy
                                        # Check if this cell has solid at current layer
                                        if (check_gx, check_gy, current_layer) in solid_at_grid:
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
                                
                                # PER-CELL CLAMPING using grid-based safe Z range
                                # (gx, gy already calculated above for wall proximity)
                                
                                # Find safe range for this cell at this layer
                                local_z_min = -999
                                local_z_max = 999
                                if (gx, gy) in grid_cell_safe_z:
                                    # Search for this layer in the cell's safe ranges
                                    for range_layer, z_min, z_max in grid_cell_safe_z[(gx, gy)]:
                                        if range_layer == current_layer:
                                            # Add layer_height margin to safe range
                                            # z_min is the top of solid below, so add layer_height for clearance
                                            local_z_min = z_min + base_layer_height
                                            # z_max is the bottom of solid above
                                            # Subtract 2x layer_height to stay well below it
                                            local_z_max = z_max - (2 * base_layer_height)
                                            break
                                
                                # ADDITIONAL CONSTRAINT: Don't let infill rise/fall more than amplitude from current layer
                                # This prevents infill from going way too high/low when there's a large gap
                                max_rise_from_layer = layer_z + amplitude
                                max_fall_from_layer = layer_z - amplitude
                                if local_z_max < 100:  # Only constrain if we have a valid z_max
                                    local_z_max = min(local_z_max, max_rise_from_layer)
                                if local_z_min > -100:  # Only constrain if we have a valid z_min
                                    local_z_min = max(local_z_min, max_fall_from_layer)
                                
                                # DEBUG output for first segment only (when debug flag is enabled)
                                if debug and idx == 1 and current_layer <= 3 and current_layer % 1 == 0:
                                    print(f"[Layer {current_layer}] Z={layer_z:.2f}, cell=({gx},{gy}), range=[{local_z_min:.2f}, {local_z_max:.2f}], z_mod: {z_mod:.3f} -> ", end='')
                                
                                # Clamp Z to safe range (with layer_height margins)
                                z_mod_original = z_mod
                                if local_z_min > -100:  # Valid z_min
                                    z_mod = max(local_z_min, z_mod)
                                if local_z_max < 100:  # Valid z_max
                                    z_mod = min(local_z_max, z_mod)
                                
                                if debug and idx == 1 and current_layer <= 3 and current_layer % 1 == 0:
                                    clamped = z_mod != z_mod_original
                                    print(f"{z_mod:.3f} {'(CLAMPED)' if clamped else ''}")
                                
                                last_infill_z = z_mod
                                
                                # Output with Z modulation (X and Y unchanged!)
                                modified_lines.append(
                                    f"G1 X{sx:.3f} Y{sy:.3f} Z{z_mod:.3f} E{current_e:.5f}\n"
                                )
                            i += 1
                            continue
                
                modified_lines.append(current_line)
                i += 1
            
            continue
        
        # ========== SAFE Z-HOP: Apply Z-hop to travel moves ==========
        elif enable_safe_z_hop and seen_first_layer and line.startswith("G1"):
            # Update current Z if this line has Z
            z_match = re.search(r'Z([-\d.]+)', line)
            if z_match:
                current_travel_z = float(z_match.group(1))
            
            # Detect travel moves: G1 with X/Y, F (feedrate), but no E (extrusion)
            if ('X' in line or 'Y' in line) and 'F' in line and 'E' not in line:
                # This is a travel move - apply Z-hop if needed
                if current_layer in layer_max_z:
                    safe_z = layer_max_z[current_layer] + safe_z_hop_margin
                    
                    # Only hop if we're not already above safe_z
                    if current_travel_z < safe_z:
                        # Lift to safe Z before travel
                        modified_lines.append(f"G1 Z{safe_z:.3f} F8400 ; Safe Z-hop\n")
                        # Do the travel move (without Z since we just set it)
                        # Remove Z from the travel line if it exists
                        travel_line = re.sub(r'Z[-\d.]+\s*', '', line)
                        modified_lines.append(travel_line)
                        current_travel_z = safe_z
                        i += 1
                        continue
            
            # Not a travel move or no Z-hop needed - just output as is
            modified_lines.append(line)
            i += 1
            continue
        
        else:
            # Track ANY TYPE marker to prevent mis-detecting as orphan
            if ";TYPE:" in line:
                last_type_seen = line.strip()
            
            modified_lines.append(line)
            i += 1

    # Write the modified G-code
    print(f"\n[OK] Processed {current_layer} layers")
    print(f"Writing modified G-code to: {os.path.basename(output_file)}...")
    with open(output_file, 'w') as outfile:
        outfile.writelines(modified_lines)

    logging.info("\n" + "="*70)
    logging.info("G-code processing completed successfully")
    logging.info("="*70)
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("  [OK] SILKSTEEL POST-PROCESSING COMPLETE")
    print("=" * 70)
    print(f"  Total layers: {current_layer}")
    print(f"  Output lines: {len(modified_lines)}")
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
    
    parser.add_argument('-outerLayerHeight', '--outer-layer-height', type=float,
                       help='Desired height for outer walls (mm). If not provided, uses min_layer_height from G-code')
    
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
    parser.add_argument('-segmentLength', '--segment-length', type=float, default=SEGMENT_LENGTH,
                       help=f'Length of subdivided segments for non-planar infill in mm (default: {SEGMENT_LENGTH})')
    parser.add_argument('-amplitude', '--amplitude', type=float, default=DEFAULT_AMPLITUDE,
                       help=f'Amplitude of Z modulation for non-planar infill in mm when float, layers when integer (default: {DEFAULT_AMPLITUDE})')
    parser.add_argument('-frequency', '--frequency', type=float, default=DEFAULT_FREQUENCY,
                       help=f'Frequency of Z modulation for non-planar infill (default: {DEFAULT_FREQUENCY})')
    
    parser.add_argument('-disableSafeZHop', '--disable-safe-z-hop', action='store_false', dest='enable_safe_z_hop',
                       help='Disable safe Z-hop during travel moves (default: enabled)')
    parser.add_argument('-safeZHopMargin', '--safe-z-hop-margin', type=float, default=DEFAULT_SAFE_Z_HOP_MARGIN,
                       help=f'Safety margin in mm to add above max Z during travel (default: {DEFAULT_SAFE_Z_HOP_MARGIN})')
    
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Enable debug visualization: draws grid showing detected solid infill and safe Z ranges (default: disabled)')
    
    args = parser.parse_args()
    
    # Log all received arguments for debugging
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
            amplitude=args.amplitude,
            frequency=args.frequency,
            enable_safe_z_hop=args.enable_safe_z_hop,
            safe_z_hop_margin=args.safe_z_hop_margin,
            debug=args.debug
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
