# 16-Sensor Light Array Bearing Control for Blimp Simulator

A direct bearing calculation system for autonomous blimp navigation to light sources using a circular array of 16 light sensors in Webots robotics simulator.

## 🚁 Overview

This project implements a multi-sensor light array approach to guide a blimp to a light source using direct bearing calculation and yaw angle control. The system uses 16 light sensors arranged in a circular pattern to calculate the weighted bearing angle to the light source, eliminating the need for dithering or exploration. The controller provides instant directional response and omnidirectional light detection.

## 📁 Project Structure

```
bearing_angle_control/
├── asset/                      # 3D models and assets
│   └── simple_MAB.stl         # Blimp 3D model
├── controllers/               # Webots controllers
│   └── blimp/                # 16-sensor bearing controller
│       ├── blimp.cpp         # Multi-sensor bearing implementation
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
│   └── Blimp.proto          # Blimp robot with 16-sensor array
└── worlds/                   # Simulation worlds
    └── blimp.wbt            # Main simulation world
```

## 🎯 Features

### 16-Sensor Light Array System
- **Omnidirectional Detection**: 16 sensors in circular pattern (22.5° separation)
- **Direct Bearing Calculation**: Weighted vector addition for instant direction finding
- **No Dithering Required**: Eliminates oscillations and response delays
- **360° Coverage**: No blind spots or exploration needed
- **Real-time Response**: Immediate reaction to light source changes

### Simplified Control Architecture
- **IDLE Mode**: Stationary until user activation
- **SEEKING Mode**: Direct bearing-based navigation to light source
- **No Exploration Needed**: Sensors cover all directions simultaneously
- **Instant State Response**: Immediate start/stop based on light detection

### Robust Sensor Array Features
- **Visual Indicators**: Red lines show each sensor's pointing direction
- **Sensor Diagnostics**: Real-time display of strongest sensor and angle
- **Noise Filtering**: Smoothed bearing calculation with configurable filter
- **Minimum Light Threshold**: Robust operation in low-light conditions

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

### Multi-Sensor Bearing Calculation

The controller implements direct bearing calculation using weighted vector addition:

1. **Sensor Reading**: Read all 16 light sensors simultaneously:
   ```cpp
   for (int i = 0; i < 16; i++) {
     light_values[i] = wb_light_sensor_get_value(light_sensors[i]);
   }
   ```

2. **Weighted Vector Sum**: Calculate bearing using light intensities as weights:
   ```cpp
   sum_x += light_values[i] * cos(SENSOR_ANGLES[i] * PI/180.0);
   sum_y += light_values[i] * sin(SENSOR_ANGLES[i] * PI/180.0);
   ```

3. **Bearing Angle**: Calculate direction to light source:
   ```cpp
   bearing_angle = atan2(sum_y, sum_x) * 180.0/PI;
   ```

4. **Yaw Setpoint**: Set heading directly toward calculated bearing:
   ```cpp
   setpoint_yaw = current_yaw + smoothed_bearing;
   ```

### Sensor Array Configuration

- **16 Sensors**: Positioned in circle at 0.35m radius from robot center
- **Angular Separation**: 22.5° between adjacent sensors for full 360° coverage
- **Sensor Angles**: 0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°, 180°, 202.5°, 225°, 247.5°, 270°, 292.5°, 315°, 337.5°
- **Visual Indicators**: Red lines show sensor pointing directions in simulation

### State Machine Logic

- **Simple Two-State**: IDLE (stationary) and SEEKING (active navigation)
- **Immediate Response**: No exploration delays since all directions are monitored
- **Light Threshold**: Stops when total light intensity drops below minimum
- **Proximity Completion**: Mission ends when within specified distance of target

### Key Parameters

```cpp
// 16-sensor array configuration
const int NUM_LIGHT_SENSORS = 16;
const double SENSOR_ANGLES[16] = {0.0, 22.5, 45.0, ..., 337.5}; // degrees

// Bearing calculation parameters
const double MIN_LIGHT_INTENSITY = 0.1;     // Minimum light per sensor (threshold)
const double MIN_TOTAL_LIGHT = 1.0;         // Minimum total light to proceed
const double BEARING_SMOOTHING_FACTOR = 0.3; // Low-pass filter for bearing

// Mission parameters
const double CONSTANT_FORWARD_SPEED = 0.2;  // Forward speed (m/s)
const double ALTITUDE_SETPOINT = 2.0;       // Altitude setpoint (m)
const double PROXIMITY_THRESHOLD = 2.0;     // Mission completion distance (m)

// Target location
const double TARGET_X = 5.1988;  // Target X coordinate (m)
const double TARGET_Y = 5.329;   // Target Y coordinate (m)
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

4. **Verify sensor array**: The Blimp.proto file includes 16 light sensors named `"light_sensor_0"` through `"light_sensor_15"`

5. **Build the physics plugin**:
   ```bash
   cd bearing_angle_control/plugins/physics/blimp_physics
   make
   ```

6. **Build the controller** (enable debug mode for trajectory logging):
   ```bash
   cd bearing_angle_control/controllers/blimp
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

1. **Initialization**: Blimp remains stationary, reading all 16 sensors
2. **Activation**: Press 'S' to start autonomous light seeking
3. **Bearing Calculation**: Sensors calculate weighted bearing to light source
4. **Direct Navigation**: Blimp turns toward calculated bearing and moves forward
5. **Completion**: Mission ends when within proximity threshold of light source

## 🔧 Configuration

### Algorithm Parameters

Modify bearing calculation parameters in `blimp.cpp`:

```cpp
// === SENSOR ARRAY PARAMETERS ===
const int NUM_LIGHT_SENSORS = 16;
const double SENSOR_ANGLES[16] = {0.0, 22.5, 45.0, ...}; // Full 360° coverage

// === BEARING CALCULATION PARAMETERS ===
const double MIN_LIGHT_INTENSITY = 0.1;     // Minimum light per sensor
const double MIN_TOTAL_LIGHT = 1.0;         // Minimum total light to proceed
const double BEARING_SMOOTHING_FACTOR = 0.3; // Noise filtering (0.0-1.0)

// === MISSION PARAMETERS ===
const double CONSTANT_FORWARD_SPEED = 0.2; // m/s forward speed
const double ALTITUDE_SETPOINT = 2.0;      // m altitude setpoint
const double PROXIMITY_THRESHOLD = 2.0;    // m, mission completion distance

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

1. **Sensor Sensitivity**: Adjust `MIN_LIGHT_INTENSITY` to filter noise while maintaining detection
2. **Bearing Smoothing**: Increase `BEARING_SMOOTHING_FACTOR` for stability, decrease for responsiveness
3. **Total Light Threshold**: Set `MIN_TOTAL_LIGHT` based on your light source intensity
4. **Sensor Placement**: Ensure sensors are properly positioned and oriented outward

### Common Tuning Issues
- **Erratic Behavior**: Increase bearing smoothing factor or minimum light thresholds
- **Slow Response**: Decrease bearing smoothing factor or check sensor alignment
- **No Movement**: Verify light source intensity and sensor detection thresholds
- **Wrong Direction**: Check sensor angle definitions match physical placement

## 📈 Performance Optimization

### Multi-Sensor Array Tuning Guidelines

1. **Start with Default Values**: Use provided parameters as baseline
2. **Adjust Smoothing**: Increase for stability in noisy environments
3. **Calibrate Thresholds**: Match minimum light levels to your simulation environment
4. **Verify Sensor Alignment**: Ensure visual indicators point in correct directions

### Algorithm Performance Factors
- **Sensor Precision**: Higher resolution sensors improve bearing accuracy
- **Light Source Characteristics**: Point sources work best for directional sensing
- **Sensor Array Geometry**: 16 sensors provide good resolution for most applications
- **Environmental Conditions**: Ambient light and reflections can affect bearing calculation

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
# Verify all 16 sensors are detected
echo "Check sensor names: light_sensor_0 through light_sensor_15"

# Check sensor readings in console
echo "Monitor TotalLight and StrongestSensor values"

# Verify sensor orientations
echo "Check red visual indicators point outward in simulation"
```

**Blimp Not Moving**:
```bash
# Check total light threshold
echo "Verify MIN_TOTAL_LIGHT matches your light source intensity"

# Check individual sensor readings
echo "Ensure at least one sensor detects light above MIN_LIGHT_INTENSITY"
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
[T:146.9s] SEEKING | TotalLight:45.8 | Bearing:23.4° | StrongestSensor:2(45.0°) | Dist:16.68m | Yaw:-65.2->-41.8
```

- **State**: Current control mode (IDLE/SEEKING)
- **TotalLight**: Sum of all 16 sensor readings
- **Bearing**: Calculated bearing angle to light source (relative to robot)
- **StrongestSensor**: ID and angle of sensor with highest reading
- **Dist**: Distance to predefined target location
- **Yaw**: Current->Setpoint yaw angle

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

- **Multi-Sensor Arrays**: "Sensor Arrays and Multi-channel Signal Processing" - various robotics literature
- **Webots Documentation**: [cyberbotics.com](https://cyberbotics.com/doc/guide/index)
- **Bearing Calculation Methods**: Weighted vector addition and circular statistics
- **Light Sensor Applications**: Robotic navigation and source seeking using photodiodes

## 📄 License

This project is based on the original blimp physics model from EPFL's Laboratory of Intelligent Systems. Please see the original license headers in source files.

**Original Authors**: Alexis Guanella, Antoine Beyeler, Jean-Christophe Zufferey, Dario Floreano

**Enhanced Version**: Extended with 16-sensor light array for direct bearing calculation and omnidirectional light source detection.

---

## 🎉 Acknowledgments

- **EPFL LIS**: Original blimp physics implementation
- **Webots Team**: Excellent robotics simulation platform  
- **Multi-Sensor Community**: Research in sensor arrays and bearing calculation methods
- **Contributors**: Developers and researchers advancing direct sensing approaches

For questions or support, please open an issue in the repository.
