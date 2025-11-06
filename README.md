# SilkSteel G-Code Post-Processor

**SilkSteel** — *Smooth on the outside, strong on the inside.*

A powerful G-code post-processor that transforms your 3D prints by creating silk-smooth exterior surfaces while building steel-strong internal structures through intelligent layer manipulation.

---

## 🎯 What is SilkSteel?

SilkSteel is an advanced G-code post-processing tool that applies multiple optimization techniques to your sliced files:

- **Smoothificator**: Splits thick external perimeters into multiple ultra-thin passes for glass-smooth surfaces
- **Bricklayers**: Z-shifts alternating internal perimeters to create interlocking "brick pattern" for superior layer adhesion
- **Non-planar Infill**: Modulates Z height during infill printing to mechanically interlock with adjacent layers
- **Safe Z-hop**: Intelligently lifts the nozzle during travel moves to prevent collisions with already-printed geometry

The result? Prints with **silk-smooth exteriors** and **steel-strong interiors**.

---

## ✨ Features

### 🧵 Smoothificator (Enabled by Default)
Creates ultra-smooth external surfaces by splitting thick perimeter walls into multiple thin passes:
- Converts single thick wall (e.g., 0.4mm) into 2-3 ultra-thin passes (e.g., 0.15mm each)
- Each pass is printed at a lower Z height, building up gradually
- Eliminates visible layer lines on vertical surfaces
- Perfect for aesthetic parts, mechanical surfaces, and light-diffusing prints

**Best for:** Visible surfaces, vases, enclosures, decorative parts

### 🧱 Bricklayers (Optional, `-enableBricklayers`)
Strengthens parts by creating a brick-like interlocking pattern with internal perimeters:
- **Layer 0**: All perimeter blocks printed in 2 passes at 0.75h each (total 1.5h)
- **Layer 1+**: Alternating blocks are Z-shifted by +0.5h (shifted) or printed normally (base)
- Creates mechanical interlocking between layers
- Dramatically improves layer adhesion and structural strength
- Properly handles E values and travel moves to prevent artifacts

**Best for:** Functional parts, mechanical components, parts under stress

### 🌊 Non-planar Infill (Optional, `-enableNonPlanar`)
Modulates Z height during infill printing to create 3D wave patterns:
- Infill lines undulate up and down, creating mechanical anchors
- Automatically detects solid layers above/below and respects safe Z boundaries
- Pre-calculates 3D occupancy grid to prevent nozzle collisions
- Two pattern modes: sine waves or Perlin noise
- Significantly improves infill-to-perimeter bonding

**Best for:** Large flat surfaces, parts with sparse infill, structural components

### 🛡️ Safe Z-hop (Enabled by Default)
Prevents nozzle collisions during travel moves:
- Pre-calculates maximum Z height per layer
- Lifts nozzle to `max_z + margin` before travel moves
- Only activates after first layer to avoid startup issues
- Configurable safety margin (default: 0.5mm)
- Essential when using Bricklayers or Non-planar features

**Best for:** All prints, especially with Z-shifting features enabled

---

## 📦 Installation

### Requirements
- Python 3.7+
- NumPy (for non-planar noise generation)

### Setup
```bash
# Clone or download this repository
git clone https://github.com/48design/silksteel-gcode.git
cd silksteel-gcode

# Install dependencies
pip install numpy
```

---

## 🚀 Usage

### Basic Usage (Smoothificator only)
```bash
python SilkSteel.py input.gcode -o output.gcode
```

### Enable All Features
```bash
python SilkSteel.py input.gcode -o output.gcode -full
```

### Custom Feature Combinations
```bash
# Smoothificator + Bricklayers only
python SilkSteel.py input.gcode -o output.gcode -enableBricklayers

# All features with custom settings
python SilkSteel.py input.gcode -o output.gcode -full -amplitude 1.5 -frequency 8

# Bricklayers with reduced extrusion on shifted blocks
python SilkSteel.py input.gcode -o output.gcode -enableBricklayers -bricklayersExtrusion 0.9
```

### In-Place Mode (For Slicer Integration)
```bash
# Modifies the input file directly (required for slicer post-processing scripts, gcode automagically 🧙 added by slider)
"{fielpath-to-your-python.exe}" "{path-to-silksteel}\SilkSteel.py" -full
```

---

## 🛠️ Command-Line Options

### Input/Output
```
input_file              Input G-code file (required)
-o, --output           Output file (if omitted, modifies input file in-place)
```

### Feature Toggles
```
-full, --enable-all                Enable all features (Bricklayers + Non-planar)
-enableBricklayers                 Enable Bricklayers Z-shifting (default: disabled)
-enableNonPlanar                   Enable non-planar infill (default: disabled)
-disableSmoothificator             Disable Smoothificator (default: enabled)
-disableSafeZHop                   Disable safe Z-hop (default: enabled)
```

### Smoothificator Settings
```
-outerLayerHeight FLOAT            Height for outer wall passes in mm
                                   (default: uses min_layer_height from G-code)
```

### Bricklayers Settings
```
-bricklayersExtrusion FLOAT        Extrusion multiplier for shifted blocks
                                   (default: 1.0, try 0.9-1.1 for tuning)
```

### Non-planar Infill Settings
```
-deformType {sine,noise}           Pattern type (default: sine)
                                   sine: smooth wave patterns
                                   noise: Perlin noise for organic variation
-segmentLength FLOAT               Line subdivision length in mm (default: 0.2)
-amplitude FLOAT                   Z modulation amplitude in mm (default: 2)
                                   Higher = more pronounced waves
-frequency FLOAT                   Pattern frequency (default: 6)
                                   Higher = tighter waves
```

### Safe Z-hop Settings
```
-safeZHopMargin FLOAT              Safety margin above max Z in mm (default: 0.5)
```

### Debug Options
```
--debug                            Enable visualization: draws grid showing
                                   detected solid infill and safe Z ranges
```

---

## 📋 Workflow & Best Practices

### Slicer Settings
1. **Slice your model normally** with your preferred settings
2. Use **variable layer heights** if desired (SilkSteel handles them correctly)
3. For Smoothificator: Set external perimeter extrusion width to 0.4-0.5mm
4. For Bricklayers: Enable at least 2 perimeters
5. For Non-planar: Use 10-20% infill with rectilinear or grid pattern

### Recommended Feature Combinations

**Display Parts (Focus: Appearance)**
```bash
python SilkSteel.py model.gcode -o output.gcode
# Smoothificator only - perfect surface finish
```

**Functional Parts (Focus: Strength)**
```bash
python SilkSteel.py model.gcode -o output.gcode -enableBricklayers
# Smoothificator + Bricklayers - strong and smooth
```

**Large Structural Parts (Focus: Maximum Strength)**
```bash
python SilkSteel.py model.gcode -o output.gcode -full
# All features - ultimate strength with good finish
```

### Processing Time
- Small prints (< 50 layers): < 5 seconds
- Medium prints (50-200 layers): 10-30 seconds  
- Large prints (200+ layers): 30-90 seconds
- Non-planar infill adds 20-50% processing time due to 3D grid calculations

---

## 🔧 Integration with Slicers

### PrusaSlicer / SuperSlicer
1. Go to: **Print Settings → Output options → Post-processing scripts**
2. Add: `python "C:\path\to\SilkSteel.py"`
3. The script will modify the G-code file in-place after slicing

### Cura
1. Go to: **Extensions → Post Processing → Modify G-Code**
2. Add script: **Run a script after slicing**
3. Set path: `python "C:\path\to\SilkSteel.py" "[output_file]"`

### OrcaSlicer
1. Go to: **Printer Settings → Machine G-code → Post-process script**
2. Add: `python "C:\path\to\SilkSteel.py"`

---

## 📊 Technical Details

### How Smoothificator Works
1. Detects external perimeter blocks via TYPE comments
2. Calculates number of passes needed based on outer_layer_height
3. Splits perimeter into multiple passes, each at incrementing Z
4. Adjusts extrusion per pass to maintain proper wall thickness
5. Handles wipe moves and travel moves correctly

### How Bricklayers Works
1. Detects internal perimeter blocks
2. On layer 0: All blocks printed in 2 passes (0.75h + 0.75h)
3. On layer 1+: Alternates between shifted (+0.5h) and base (normal Z)
4. Uses G92 E0 resets to maintain proper E value tracking
5. Reduces extrusion to 0.5x on shifted blocks of the last layer (to end flat)

### How Non-planar Infill Works
1. **Pass 1**: Builds 3D occupancy grid by scanning all solid layers
2. **Pass 2**: Calculates safe Z ranges per grid cell (between solid layers)
3. **Pass 3**: Processes infill, subdividing lines and modulating Z
4. Applies sine wave or Perlin noise pattern
5. Clamps Z values to safe range to prevent collisions
6. Updates E values proportionally to actual line length in 3D space

### How Safe Z-hop Works
1. Pre-calculates maximum Z per layer during first scan pass
2. Before each travel move (G1 with X/Y but no E), lifts to max_z + margin
3. Only activates after first layer starts
4. Returns to normal Z after travel completes

---

## 🐛 Troubleshooting

### Issue: Rough surfaces after Smoothificator
**Solution**: Decrease outer_layer_height (try 0.1mm or 0.12mm)

### Issue: Weak Bricklayers shifting
**Solution**: Increase bricklayers_extrusion multiplier to 1.05-1.1

### Issue: Non-planar infill colliding with walls
**Solution**: Reduce amplitude or check that solid layers are detected (use --debug)

### Issue: Stringing or blobs on travels
**Solution**: Safe Z-hop should prevent this - check that it's enabled

### Issue: First layer problems
**Solution**: Safe Z-hop intentionally skips first layer to avoid issues

### Issue: Print time significantly increased
**Solution**: Non-planar adds many small moves - reduce frequency or disable for faster prints

---

## 🙏 Credits & Acknowledgments

**SilkSteel** is developed by **48DESIGN GmbH** (Fabian Groß).

Special thanks to **Roman Tenger** for the original concepts and inspiration:
- Smoothificator technique for multi-pass external perimeters
- Bricklayers Z-shifting concept for improved layer bonding  
- Non-planar infill modulation for mechanical interlocking

SilkSteel represents a complete rewrite and optimization of these concepts, with significant enhancements:
- Proper E value tracking and G92 E0 resets for Bricklayers
- 3D occupancy grid system for safe non-planar infill
- Safe Z-hop collision avoidance
- Robust TYPE marker detection and travel move handling
- Performance optimizations for large files

---

## 📄 License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

**Copyright (c) [2025] [48DESIGN GmbH - Fabian Groß]**

---

## 🔗 Links & Resources

- GitHub Repository: [48design/silksteel-gcode]
- Issues & Feature Requests: [GitHub Issues]
- Original Smoothificator concept: [github.com/TengerTechnologies]

---

## 🌟 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**Made with ❤️ and 🧠 and 🤖 for the 3D printing community**
