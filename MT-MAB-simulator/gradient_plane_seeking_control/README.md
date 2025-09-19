# Gradient Plane Seeking Control for Blimp Simulator

A map-based gradient following system for autonomous blimp navigation to light sources in Webots robotics simulator.

## 🚁 Overview

This project implements an intelligent gradient plane fitting algorithm to guide a blimp to a light source using spatial mapping and least squares optimization. The system builds a spatial map of light intensity measurements, using only improved readings to estimate gradients through plane fitting. The controller features a smart EXPLORING/SEEKING state machine with proximity-based mission completion and comprehensive trajectory logging.

## 📁 Project Structure

```
gradient_plane_seeking_control/
├── asset/                      # 3D models and assets
│   └── simple_MAB.stl         # Blimp 3D model
├── controllers/               # Webots controllers
│   └── blimp/                # Gradient plane seeking controller
│       ├── blimp.cpp         # Map-based gradient plane fitting implementation
│       ├── js.h              # Joystick interface (legacy)
│       ├── logs/             # Trajectory debug logs
│       ├── analysis/         # Trajectory analysis tools
│       │   ├── trajectory_analyzer.py  # Python analysis script
│       │   └── requirements.txt        # Python dependencies
│       └── Makefile          # Build configuration
├── plugins/                   # Physics plugins
│   └── physics/
│       └── blimp_physics/    # Custom physics implementation
│           ├── *.c/*.h       # Physics source files
│           └── Makefile      # Physics build config
├── protos/                   # Webots PROTO definitions
│   └── Blimp.proto          # Blimp robot definition
└── worlds/                   # Simulation worlds
    └── blimp.wbt            # Main simulation world
```

## 🎯 Features

### Map-Based Gradient Control System
- **Spatial Light Mapping**: Builds a grid-based map of light intensity measurements
- **Selective Map Updates**: Only stores measurements that improve upon previous readings
- **Least Squares Plane Fitting**: Estimates gradients using spatial distribution of measurements
- **Quality-Based Gradient**: Uses R² coefficient to validate gradient reliability
- **Intelligent Memory**: Retains spatial knowledge throughout the mission

### Multi-State Control Architecture
- **EXPLORING Mode**: Spiral search pattern while building spatial map
- **SEEKING Mode**: Active gradient following based on plane fitting
- **Smart State Transitions**: Switches based on gradient magnitude and fit quality
- **Map-Driven Navigation**: Uses accumulated spatial knowledge for decision making

### Robust Mission Management
- **Proximity Detection**: Automatic mission completion when close to light source
- **Signal Loss Recovery**: Exploration mode with configurable light loss timeout
- **Target Tracking**: Real-time distance monitoring to predefined target location
- **Performance Metrics**: Mission time and accuracy tracking

### Debug Logging System
- **Cross-Platform Compatibility**: Windows, Linux, macOS support
- **CSV Format**: Structured data for easy analysis
- **Conditional Compilation**: Debug-only logging with `DEBUG_LOGGING` flag
- **Multiple Log Files**: Separate logs for different components
- **Automatic Path Detection**: Runtime DLL/SO path resolution

### Data Analysis Tools
- **Trajectory Visualization**: 3D flight path analysis with mode tracking
- **Real-time Performance**: Controller state and signal strength monitoring
- **Statistical Analysis**: RMS error calculations and tracking accuracy metrics
- **Export Capabilities**: High-quality plots for research documentation

## 🔬 Algorithm Details

### Gradient Plane Fitting Theory

The controller implements a map-based gradient estimation algorithm:

1. **Spatial Discretization**: Space is divided into grid cells with resolution:
   ```
   grid_x = floor(x / GRID_RESOLUTION)
   grid_y = floor(y / GRID_RESOLUTION)
   ```

2. **Selective Map Updates**: Only updates map when light intensity improves:
   ```
   if (new_intensity > previous_intensity + MIN_IMPROVEMENT):
       map[grid_cell] = new_measurement
   ```

3. **Least Squares Plane Fitting**: Fits plane z = ax + by + c to nearby points:
   ```
   [∂z/∂x, ∂z/∂y] = [a, b]  // Gradient components
   magnitude = sqrt(a² + b²)
   direction = atan2(b, a)
   ```

4. **Quality Assessment**: Uses R² coefficient to validate gradient reliability:
   ```
   R² = 1 - (SS_residual / SS_total)
   accept_gradient = (R² > threshold)
   ```

### State Machine Logic

- **Map Building**: Continuously updates spatial map during exploration
- **Gradient Detection**: Switches to seeking when gradient magnitude exceeds threshold
- **Quality Control**: Only follows gradients with sufficient fit quality (R² > 0.2)
- **Spatial Memory**: Uses accumulated map knowledge for persistent navigation
- **Proximity Completion**: Mission ends when within specified distance of target

### Key Parameters

```cpp
// Mapping parameters
const double GRID_RESOLUTION = 0.2;         // Grid cell size (meters)
const double MIN_LIGHT_IMPROVEMENT = 0.1;   // Minimum improvement to update map
const double GRADIENT_THRESHOLD = 2.0;      // Minimum gradient magnitude for seeking

// Mission parameters
const double SEEKING_SPEED = 0.3;           // Forward speed during seeking (m/s)
const double EXPLORING_SPEED = 0.15;        // Speed during exploration (m/s)
const double EXPLORING_RADIUS = 1.0;        // Radius for spiral exploration (m)
const double PROXIMITY_THRESHOLD = 0.5;     // Mission completion distance (m)

// Target location
const double TARGET_X = 5.1988;            // Target X coordinate (meters)
const double TARGET_Y = 5.329;             // Target Y coordinate (meters)

// Plane fitting parameters
const double MAX_DISTANCE = 3.0;           // Maximum distance for gradient fitting
const double MIN_R_SQUARED = 0.2;          // Minimum R² for gradient acceptance
```

## 🚀 Quick Start

### Prerequisites
- **Webots**: R2023b or later
- **Compiler**: GCC/Clang (Linux/Mac) or MSVC (Windows)
- **Python**: 3.7+ (for analysis tools)

### Installation

1. **Clone the project** into your Webots projects directory:
   ```bash
   git clone <repository-url> gradient_plane_seeking_control
   ```

2. **Open Webots** and load the world file:
   ```
   gradient_plane_seeking_control/worlds/blimp.wbt
   ```

3. **Add a PointLight** to your Webots world to act as the light source

4. **Build the physics plugin**:
   ```bash
   cd gradient_plane_seeking_control/plugins/physics/blimp_physics
   make
   ```

5. **Build the controller**:
   ```bash
   cd gradient_plane_seeking_control/controllers/blimp
   make
   ```

### Running the Simulation

1. **Start Webots simulation** with the `blimp.wbt` world
2. **Press 'S' key** to activate autonomous gradient plane seeking
3. **Press 'Q' key** to quit the simulation
4. **Monitor progress** via console output showing:
   - Current control state (EXPLORING/SEEKING)
   - Light signal intensity
   - Distance to target
   - Map size and gradient magnitude

### Mission Sequence

1. **Initialization**: Blimp starts in EXPLORING mode
2. **Activation**: Press 'S' to start map-based gradient seeking
3. **Exploration**: Blimp spirals while building spatial light intensity map
4. **Map Building**: Only improved measurements are stored in spatial grid
5. **Gradient Detection**: Switches to SEEKING when reliable gradient found
6. **Seeking**: Follows gradient direction from plane fitting
7. **Completion**: Mission ends when within proximity threshold of light source

## 🔧 Configuration

### Algorithm Parameters

Modify gradient plane seeking parameters in `blimp.cpp`:

```cpp
// === MAPPING PARAMETERS ===
const double GRID_RESOLUTION = 0.2;         // Grid cell size for spatial mapping
const double MIN_LIGHT_IMPROVEMENT = 0.1;   // Required improvement to update map
const double GRADIENT_THRESHOLD = 2.0;      // Minimum gradient magnitude for seeking

// === MISSION PARAMETERS ===
const double SEEKING_SPEED = 0.3;           // m/s during gradient following
const double EXPLORING_SPEED = 0.15;        // m/s during spiral search  
const double EXPLORING_RADIUS = 1.0;        // m, radius for exploration spiral
const double PROXIMITY_THRESHOLD = 0.5;     // m, mission completion distance

// === TARGET LOCATION ===
const double TARGET_X = 5.1988; // meters
const double TARGET_Y = 5.329;  // meters

// === PLANE FITTING PARAMETERS ===
const double MAX_DISTANCE = 3.0;           // Maximum distance for gradient points
const double MIN_R_SQUARED = 0.2;          // Minimum fit quality for gradient
```

### Light Source Setup

In your Webots world file, add a PointLight:

```vrml
PointLight {
  attenuation 0 0 1
  color 1 1 1
  intensity 1
  location 5.2 5.3 2
  on TRUE
  radius 100
}
```

### Trajectory Logging

The controller automatically logs comprehensive trajectory data to CSV:

- **Log Location**: `logs/history_gradient_trajectory.csv` or current directory
- **Data Columns**: 16 columns including position, orientation, map data, and gradient info
- **Real-time Updates**: Logs every timestep during active operation

## 📊 Data Analysis

### Trajectory Analysis Tools

The project includes comprehensive Python analysis tools for trajectory visualization:

```bash
cd controllers/blimp/analysis

# Install dependencies
pip install -r requirements.txt

# Run trajectory analysis
python trajectory_analyzer.py
```

### Generated Analysis
- **3D Trajectory Plot**: Complete flight path with mode visualization
- **2D Top-Down View**: Navigation analysis with direction arrows  
- **Position vs Time**: Individual X, Y, Z, and yaw tracking
- **Speed Analysis**: Speed controller performance and setpoint tracking
- **Tracking Error Analysis**: RMS error calculations for all axes
- **Mode Analysis**: State transitions and time spent in each mode

### Log File Format

Trajectory CSV logs contain comprehensive data with 16 columns:
```csv
time,x,y,z,yaw,setpoint_x,setpoint_y,setpoint_z,setpoint_yaw,setpoint_speed,current_speed,mode,distance_to_target,light_intensity,gradient_angle,gradient_magnitude
146.500,1.234,2.456,2.000,-1.138,1.234,2.456,2.000,-1.138,0.150,0.148,EXPLORING,16.75,8.2,45.3,2.8
```

### Performance Metrics

The analysis automatically calculates:
- **Mission Completion Time**: Total time to reach target
- **Path Efficiency**: Direct distance vs actual path length
- **Map Quality**: Number of grid cells explored and gradient fit quality
- **State Distribution**: Time spent in EXPLORING vs SEEKING modes
- **Gradient Analysis**: Direction accuracy and magnitude tracking over time

## 🛠️ Development

### Adding New Mapping Algorithms

1. **Implement algorithm** in `blimp.cpp` within the gradient estimation functions
2. **Add new map update strategies** to the `update_measurement_map` function
3. **Add comprehensive logging**:
   ```cpp
   fprintf(trajectory_log, "%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%s,%.3f,%.1f,%.1f,%.4f\n",
           current_time, est_x, est_y, position[2], current_yaw_rad,
           setpoint_x, setpoint_y, setpoint_z, setpoint_yaw_rad,
           forward_speed, current_speed, mode, distance_to_target,
           current_signal, grad_angle, grad_magnitude);
   ```
4. **Update analysis scripts** to handle new mapping approaches

### Algorithm Tuning Guidelines

1. **Grid Resolution**: Smaller cells provide higher resolution but require more exploration
2. **Improvement Threshold**: Higher thresholds create sparser but higher-quality maps
3. **Gradient Threshold**: Lower values trigger seeking earlier but may follow noise
4. **R² Threshold**: Higher values ensure reliable gradients but may be too restrictive

### Common Tuning Issues
- **Slow Map Building**: Reduce improvement threshold or grid resolution
- **Noisy Gradients**: Increase R² threshold or reduce maximum fitting distance
- **Stuck in Exploration**: Lower gradient threshold or improve light source intensity
- **Poor Convergence**: Adjust seeking speed or plane fitting parameters

## 📈 Performance Optimization

### Gradient Plane Tuning Guidelines

1. **Start with Moderate Grid Resolution**: Balance between detail and exploration time
2. **Adjust Improvement Threshold**: Higher values create cleaner maps but slower convergence
3. **Tune Gradient Detection**: Lower thresholds enable earlier seeking but may follow noise
4. **Optimize Fit Quality**: Balance between gradient reliability and responsiveness

### Algorithm Performance Factors
- **Map Quality**: Density and distribution of improvement-based measurements
- **Light Source Characteristics**: Point vs distributed sources affect gradient clarity
- **Exploration Pattern**: Spiral radius affects map coverage and gradient detection
- **Environmental Disturbances**: Noise affects plane fitting quality and R² values

### Mission Success Criteria
- **Map Coverage**: Sufficient spatial distribution for reliable gradient estimation
- **Gradient Quality**: High R² values indicating good plane fits
- **Convergence Efficiency**: Smooth transition from exploration to seeking
- **Target Acquisition**: Successful navigation to proximity threshold

## 🔍 Troubleshooting

### Common Problems

**No Response to 'S' Key**:
```bash
# Check if keyboard is properly enabled
grep -r "wb_keyboard_enable" blimp.cpp

# Verify map initialization
echo "Check measurement_map.clear() in activation"
```

**Stuck in Exploring Mode**:
```bash
# Check gradient threshold
echo "Verify GRADIENT_THRESHOLD value (try lowering to 1.0)"

# Monitor map building
echo "Check console output for map updates and gradient calculations"
```

**Poor Gradient Quality**:
```bash
# Check R² threshold
echo "Verify MIN_R_SQUARED value (try lowering to 0.1)"

# Monitor plane fitting
echo "Check 'Map gradient' debug output for fit quality"
```

**Compilation Errors**:
```bash
# Check Webots includes
make clean && make DEBUG_LOGGING=1 -v

# Verify controller name matches directory
ls -la controllers/blimp/
```

**Missing Trajectory Logs**:
```bash
# Check if logs directory exists
mkdir -p logs/

# Verify file creation in current directory as fallback
ls -la *.csv
```

### Debug Output Interpretation

**Console Output Examples**:
```
NEW MAP ENTRY: Grid(25,26) at (5.123,5.456) with light=12.5
MAP UPDATE: Grid(25,26) improved from 10.2 to 12.5 (Δ=2.3)
Map gradient: points=8, grad=(2.1,1.8), mag=2.8, R²=0.85, map_size=15
[T:45.1s] State:SEEKING | Light:12.5 | Dist2Target:2.1m | MapSize:15 | GradMag:2.8 | Yaw:42.3°
```

- **NEW MAP ENTRY**: First measurement at a grid location
- **MAP UPDATE**: Improved measurement replacing previous value  
- **Map gradient**: Plane fitting results with quality metrics
- **State**: Current control mode (EXPLORING/SEEKING)
- **MapSize**: Number of grid locations with measurements
- **GradMag**: Current gradient magnitude from plane fitting

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-mapping-algorithm`
3. **Add comprehensive logging** for new components
4. **Update documentation** and analysis tools
5. **Submit pull request**

### Code Style
- Follow existing naming conventions for mapping parameters
- Add comprehensive comments explaining plane fitting theory
- Include map update logging for new strategies
- Update trajectory analysis tools for new map data

## 📚 References

- **Least Squares Methods**: "Numerical Methods for Least Squares Problems" - Björck, Å.
- **Gradient Estimation**: "Gradient estimation using stochastic approximation" - Spall, J. C.  
- **Plane Fitting**: "Least-squares fitting of planes to 3D data" - mathematical optimization literature
- **Webots Documentation**: [cyberbotics.com](https://cyberbotics.com/doc/guide/index)
- **Spatial Mapping**: Grid-based mapping and localization techniques

## 📄 License

This project is based on the original blimp physics model from EPFL's Laboratory of Intelligent Systems. Please see the original license headers in source files.

**Original Authors**: Alexis Guanella, Antoine Beyeler, Jean-Christophe Zufferey, Dario Floreano

**Enhanced Version**: Extended with map-based gradient plane fitting and comprehensive trajectory analysis capabilities.

---

## 🎉 Acknowledgments

- **EPFL LIS**: Original blimp physics implementation
- **Webots Team**: Excellent robotics simulation platform  
- **Optimization Community**: Theoretical foundations for gradient estimation and plane fitting
- **Contributors**: Developers and researchers advancing spatial mapping and source seeking methods

For questions or support, please open an issue in the repository.
