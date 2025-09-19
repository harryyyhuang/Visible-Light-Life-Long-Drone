# Dither Source Seeking Control for Blimp Simulator

A position-free extremum seeking control system for autonomous blimp navigation to light sources in Webots robotics simulator.

## 🚁 Overview

This project implements a classic dither-and-demodulation extremum seeking (ES) algorithm to guide a blimp to a light source using only yaw angle control. The system operates without global position feedback, relying on light sensor measurements, velocity estimation, and attitude sensing. The controller features intelligent state management with automatic exploration when the light signal is lost and proximity-based mission completion.

## 📁 Project Structure

```
dither_source_seeking_control/
├── asset/                      # 3D models and assets
│   └── simple_MAB.stl         # Blimp 3D model
├── controllers/               # Webots controllers
│   └── blimp/                # Source seeking controller
│       ├── blimp.cpp         # Extremum seeking implementation
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

### Extremum Seeking Control System
- **Position-Free Navigation**: No global position feedback required
- **Dither-and-Demodulation**: Classic ES algorithm with sinusoidal perturbation
- **Yaw Angle Control**: Steering-based navigation with constant forward speed
- **Light Signal Processing**: Real-time signal gradient estimation with high-pass filtering
- **Adaptive Base Heading**: Continuous integration of gradient estimates

### Multi-State Control Architecture
- **FORWARD Mode**: Initial straight-line movement before activation
- **SEEKING Mode**: Active extremum seeking with dithering and demodulation
- **EXPLORING Mode**: Spiral search pattern when light signal is lost
- **Intelligent State Transitions**: Automatic switching based on light intensity thresholds

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

### Extremum Seeking Theory

The controller implements a classic extremum seeking algorithm based on:

1. **Signal Perturbation**: A sinusoidal dither is added to the base heading:
   ```
   yaw_setpoint = base_heading + A*sin(ω*t)
   ```

2. **Gradient Estimation**: The light signal derivative is calculated with high-pass filtering:
   ```
   dJ/dt = highpass_filter(J(t) - J(t-dt))
   ```

3. **Demodulation**: The gradient is multiplied by the dither derivative:
   ```
   gradient_estimate = (dJ/dt) * A*ω*cos(ω*t)
   ```

4. **Integration**: The base heading is updated using the gradient estimate:
   ```
   base_heading += k * gradient_estimate * dt
   ```

### State Machine Logic

- **Light Loss Detection**: Monitors signal intensity below threshold for configurable time
- **Signal Recovery**: Automatic return to seeking when light intensity increases
- **Spiral Exploration**: Constant angular velocity search pattern when signal is lost
- **Proximity Completion**: Mission ends when within specified distance of target

### Key Parameters

```cpp
// Extremum seeking parameters
const double ES_DITHER_FREQ = 0.2;      // Perturbation frequency (rad/s)
const double ES_DITHER_AMP_YAW = 20;    // Perturbation amplitude (degrees)
const double ES_LEARNING_RATE = 1.0;    // Integration gain
const double ES_WASHOUT_FREQ = 0.5;     // High-pass filter cutoff (rad/s)

// Mission parameters
const double CONSTANT_FORWARD_SPEED = 0.2;  // Forward speed (m/s)
const double EXPLORING_SPEED = 0.15;        // Speed during exploration (m/s)
const double PROXIMITY_THRESHOLD = 0.5;     // Mission completion distance (m)

// Signal thresholds
const double LIGHT_LOSS_THRESHOLD = 1.0;    // Signal level for exploration trigger
const double LIGHT_FOUND_THRESHOLD = 5.0;   // Signal level for seeking return
const double LIGHT_LOSS_TIME = 3.0;         // Time before exploration (seconds)
```

## 🚀 Quick Start

### Prerequisites
- **Webots**: R2023b or later
- **Compiler**: GCC/Clang (Linux/Mac) or MSVC (Windows)
- **Python**: 3.7+ (for analysis tools)

### Installation

1. **Clone the project** into your Webots projects directory:
   ```bash
   git clone <repository-url> dither_source_seeking_control
   ```

2. **Open Webots** and load the world file:
   ```
   dither_source_seeking_control/worlds/blimp.wbt
   ```

3. **Add a PointLight** to your Webots world to act as the light source

4. **Build the physics plugin**:
   ```bash
   cd dither_source_seeking_control/plugins/physics/blimp_physics
   make
   ```

5. **Build the controller** (enable debug mode for trajectory logging):
   ```bash
   cd dither_source_seeking_control/controllers/blimp
   make DEBUG_LOGGING=1
   ```

### Running the Simulation

1. **Start Webots simulation** with the `blimp.wbt` world
2. **Press 'S' key** to activate autonomous source seeking
3. **Press 'Q' key** to quit the simulation
4. **Monitor progress** via console output showing:
   - Current control state (FORWARD/SEEKING/EXPLORING)
   - Light signal intensity
   - Distance to target
   - Yaw angle tracking

### Mission Sequence

1. **Initialization**: Blimp moves forward at constant speed
2. **Activation**: Press 'S' to start extremum seeking
3. **Seeking**: Blimp follows gradient using dither-and-demodulation
4. **Exploration**: If signal lost, switches to spiral search
5. **Completion**: Mission ends when within proximity threshold of light source

## 🔧 Configuration

### Algorithm Parameters

Modify extremum seeking parameters in `blimp.cpp`:

```cpp
// === EXTREMUM SEEKING PARAMETERS ===
const double ES_DITHER_FREQ = 0.2;      // ω (rad/s): Frequency of heading wiggle
const double ES_DITHER_AMP_YAW = 20;    // A (deg): Amplitude of heading wiggle  
const double ES_LEARNING_RATE = 1.0;    // k: Integration gain, steering speed
const double ES_WASHOUT_FREQ = 0.5;     // ω_h (rad/s): High-pass filter cutoff

// === MISSION PARAMETERS ===
const double CONSTANT_FORWARD_SPEED = 0.2; // m/s
const double EXPLORING_SPEED = 0.15;       // m/s during spiral search
const double PROXIMITY_THRESHOLD = 0.5;    // m, mission completion distance

// === TARGET LOCATION ===
const double TARGET_X = 5.1988; // meters
const double TARGET_Y = 5.329;  // meters
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

### Debug Logging

Enable trajectory logging by setting the `DEBUG_LOGGING` flag:

```bash
# Enable debug logging
make DEBUG_LOGGING=1

# Disable debug logging (production)
make DEBUG_LOGGING=0
```

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

Trajectory CSV logs contain the following columns:
```csv
time,x,y,z,yaw,setpoint_x,setpoint_y,setpoint_z,setpoint_yaw,setpoint_speed,current_speed,mode,distance_to_target
146.500,1.234,2.456,2.000,-65.200,1.234,2.456,2.000,-1.138,0.150,0.148,EXPLORING,16.75
```

### Performance Metrics

The analysis automatically calculates:
- **Mission Completion Time**: Total time to reach target
- **Path Efficiency**: Direct distance vs actual path length
- **Tracking Accuracy**: RMS errors for position and orientation
- **State Distribution**: Time spent in each control mode
- **Signal Quality**: Light sensor response analysis

## 🛠️ Development

### Adding New Search Algorithms

1. **Implement algorithm** in `blimp.cpp` within the state machine
2. **Add new control state** to the `ControlState` enum
3. **Add debug logging**:
   ```cpp
   DEBUG_LOG_TRAJECTORY("%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%s,%.3f\n",
                        current_time, gps_pos[0], gps_pos[1], gps_pos[2], rpy[2],
                        setpoint_x, setpoint_y, setpoint_z, setpoint_yaw_rad,
                        forward_speed, current_speed, mode, distance_to_target);
   ```
4. **Update analysis scripts** to handle new control modes

### Algorithm Tuning Guidelines

1. **Dither Frequency**: Higher frequency improves gradient estimation but may cause oscillations
2. **Dither Amplitude**: Larger amplitude improves signal-to-noise ratio but increases path deviation  
3. **Learning Rate**: Higher rate speeds up convergence but may cause instability
4. **Washout Filter**: Higher cutoff frequency reduces noise but may filter gradient information

### Common Tuning Issues
- **Slow Convergence**: Increase learning rate or dither amplitude
- **Oscillation**: Reduce learning rate or dither frequency
- **Poor Gradient Estimation**: Adjust washout filter frequency
- **Premature Exploration**: Increase light loss threshold or timeout

## 📈 Performance Optimization

### Extremum Seeking Tuning Guidelines

1. **Start with Conservative Parameters**: Use low learning rate and moderate dither amplitude
2. **Adjust Dither Frequency**: Balance between gradient estimation accuracy and response time
3. **Tune Learning Rate**: Increase gradually until oscillations appear, then reduce
4. **Optimize Filter Settings**: Match washout frequency to signal characteristics

### Algorithm Performance Factors
- **Signal-to-Noise Ratio**: Affects gradient estimation quality
- **Light Source Characteristics**: Point vs distributed sources require different tuning
- **Vehicle Dynamics**: Blimp inertia affects achievable dither frequency
- **Environmental Disturbances**: Wind and turbulence impact controller stability

### Mission Success Criteria
- **Convergence Time**: Time to reach proximity threshold
- **Path Efficiency**: Ratio of direct path to actual path length  
- **Signal Tracking**: Ability to follow gradient despite noise
- **Robustness**: Recovery capability when signal is temporarily lost

## 🔍 Troubleshooting

### Common Problems

**No Response to 'S' Key**:
```bash
# Check if keyboard is properly enabled
grep -r "wb_keyboard_enable" blimp.cpp

# Verify control state initialization
echo "Check current_state variable initialization"
```

**Blimp Not Following Light**:
```bash
# Verify light sensor is detected
echo "Check light sensor device name in Webots"

# Check light sensor readings
echo "Monitor light signal values in console output"
```

**Constant Exploring Mode**:
```bash
# Check light thresholds
echo "Verify LIGHT_LOSS_THRESHOLD and LIGHT_FOUND_THRESHOLD values"

# Verify light source intensity
echo "Ensure PointLight intensity is sufficient"
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
# Check debug flag is enabled  
grep -r "DEBUG_LOGGING" Makefile

# Verify logs directory exists
mkdir -p logs/
```

### Debug Output Interpretation

**Console Output Example**:
```
[T:146.9s] State:EXPLORING | Light:0.0 | Dist2Target:16.68m | Speed:0.15 | Yaw: -65.2->-65.2 (err:-0.0)
```

- **State**: Current control mode (FORWARD/SEEKING/EXPLORING)
- **Light**: Current light sensor reading
- **Dist2Target**: Distance to predefined target location
- **Speed**: Current forward speed setpoint
- **Yaw**: Current->Setpoint yaw angle with error

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-search-algorithm`
3. **Add trajectory logging** for new components
4. **Update documentation** and analysis tools
5. **Submit pull request**

### Code Style
- Follow existing naming conventions for extremum seeking parameters
- Add comprehensive comments explaining algorithm theory
- Include debug logging for new control states
- Update trajectory analysis tools for new modes

## 📚 References

- **Extremum Seeking Control**: Krstić, M., & Wang, H. H. (2000). "Stability of extremum seeking feedback for general nonlinear dynamic systems"
- **Webots Documentation**: [cyberbotics.com](https://cyberbotics.com/doc/guide/index)
- **Source Seeking Literature**: "Real-time optimization by extremum-seeking control" - Ariyur & Krstić
- **Dither-and-Demodulation**: Classical perturbation-based optimization methods

## 📄 License

This project is based on the original blimp physics model from EPFL's Laboratory of Intelligent Systems. Please see the original license headers in source files.

**Original Authors**: Alexis Guanella, Antoine Beyeler, Jean-Christophe Zufferey, Dario Floreano

**Enhanced Version**: Extended with extremum seeking control and comprehensive trajectory analysis capabilities.

---

## 🎉 Acknowledgments

- **EPFL LIS**: Original blimp physics implementation
- **Webots Team**: Excellent robotics simulation platform  
- **Extremum Seeking Community**: Theoretical foundations and algorithm development
- **Contributors**: Developers and researchers advancing source seeking methods

For questions or support, please open an issue in the repository.
