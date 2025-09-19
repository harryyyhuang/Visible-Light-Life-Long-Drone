# PID Control Tuning for Blimp Simulator

A comprehensive PID controller tuning and debugging system for autonomous blimp simulation in Webots robotics simulator.

## 🚁 Overview

This project provides a sophisticated multi-loop PID control system for autonomous blimp navigation with comprehensive debugging capabilities, cross-platform logging, and data analysis tools. The system implements cascaded PID controllers for position, velocity, and attitude control with real-time performance monitoring.

## 📁 Project Structure

```
pid_control_tunning/
├── asset/                      # 3D models and assets
│   └── simple_MAB.stl         # Blimp 3D model
├── controllers/               # Webots controllers
│   └── blimp/                # Main blimp controller
│       ├── blimp.cpp         # Controller implementation
│       ├── js.h              # Joystick interface
│       ├── logs/             # Controller debug logs
│       ├── analysis/         # Trajectory analysis tools
│       └── Makefile          # Build configuration
├── plugins/                   # Physics plugins
│   └── physics/
│       └── blimp_physics/    # Custom physics implementation
│           ├── *.c/*.h       # Physics and PID source files
│           ├── logs/         # Debug CSV logs
│           ├── analysis/     # Python analysis tools
│           └── utils.c/h     # Cross-platform utilities
├── protos/                   # Webots PROTO definitions
│   └── Blimp.proto          # Blimp robot definition
└── worlds/                   # Simulation worlds
    └── blimp.wbt            # Main simulation world
```

## 🎯 Features

### Multi-Loop PID Control System
- **Position Controllers**: X/Y/Z position control with velocity cascading
- **Attitude Controllers**: Yaw angle and rate control
- **Velocity Controllers**: Speed and altitude velocity control
- **Motor Allocation**: 4-motor thrust distribution system

### Debug Logging System
- **Cross-Platform Compatibility**: Windows, Linux, macOS support
- **CSV Format**: Structured data for easy analysis
- **Conditional Compilation**: Debug-only logging with `DEBUG_LOGGING` flag
- **Multiple Log Files**: Separate logs for different components
- **Automatic Path Detection**: Runtime DLL/SO path resolution

### Data Analysis Tools
- **Python Analysis Suite**: Automated plot generation and performance analysis
- **Real-time Plotting**: Controller performance visualization
- **Statistical Analysis**: PID parameter effectiveness metrics
- **Export Capabilities**: High-quality plots for documentation

## 🚀 Quick Start

### Prerequisites
- **Webots**: R2023b or later
- **Compiler**: GCC/Clang (Linux/Mac) or MSVC (Windows)
- **Python**: 3.7+ (for analysis tools)

### Installation

1. **Clone the project** into your Webots projects directory:
   ```bash
   git clone <repository-url> pid_control_tunning
   ```

2. **Open Webots** and load the world file:
   ```
   pid_control_tunning/worlds/blimp.wbt
   ```

3. **Build the physics plugin** (enable debug mode):
   ```bash
   cd pid_control_tunning/plugins/physics/blimp_physics
   make DEBUG_LOGGING=1
   ```

4. **Build the controller**:
   ```bash
   cd pid_control_tunning/controllers/blimp
   make DEBUG_LOGGING=1
   ```

### Running the Simulation

1. **Start Webots simulation** with the `blimp.wbt` world
2. **Debug logs** will be automatically generated in:
   - `plugins/physics/blimp_physics/logs/`
   - `controllers/blimp/logs/`

## 🔧 Configuration

### Debug Logging

Enable comprehensive debug logging by setting the `DEBUG_LOGGING` flag:

```bash
# Enable debug logging
make DEBUG_LOGGING=1

# Disable debug logging (production)
make DEBUG_LOGGING=0
```

### PID Parameters

Modify PID gains in `controller_pid.c`:

```c
// Speed controller gains
static float Kp_speed = 0.8f;
static float Ki_speed = 0.1f;
static float Kd_speed = 0.05f;

// Yaw angle controller gains
static float Kp_yaw_angle = 2.0f;
static float Ki_yaw_angle = 0.0f;
static float Kd_yaw_angle = 0.1f;

// Altitude controllers...
```

## 📊 Data Analysis

### Python Analysis Tools

The project includes comprehensive Python analysis tools:

```bash
cd plugins/physics/blimp_physics/analysis

# Install dependencies
pip install -r requirements.txt

# Run analysis
python pid_analyzer.py
```

### Generated Analysis
- **Controller Performance Plots**: Response time, overshoot, settling time
- **Error Analysis**: Tracking accuracy and stability metrics
- **Comparative Analysis**: Before/after tuning comparisons
- **Export Options**: PNG/PDF plots for documentation

### Trajectory Analysis Tools

The project also includes comprehensive trajectory visualization and analysis:

```bash
cd controllers/blimp/analysis

# Install dependencies
pip install -r requirements.txt

# Run trajectory analysis
python trajectory_analyzer.py
```

#### Trajectory Analysis Features
- **3D Trajectory Plot**: Full 3D path with setpoint reference
- **2D Top-Down View**: Bird's eye view with direction arrows
- **Position vs Time**: Individual X, Y, Z, and yaw tracking
- **Speed Analysis**: Speed tracking and error analysis
- **Tracking Error Analysis**: RMS error calculations for all axes
- **Performance Metrics**: Tracking accuracy, distance calculations, mode analysis

#### Generated Trajectory Plots
- `blimp_trajectory_3d.png` - Complete flight path visualization
- `blimp_trajectory_2d.png` - Top-down navigation analysis
- `blimp_position_vs_time.png` - Time-series position tracking
- `blimp_speed_analysis.png` - Speed controller performance
- `blimp_tracking_errors.png` - Comprehensive error analysis

### Log File Format

CSV logs contain the following columns:
```csv
time,controller,desired,current,error,output,integral,derivative
0.100,speed,1.500,0.000,1.500,1200.000,0.150,15.000
0.110,yaw_angle,45.000,0.000,45.000,900.000,4.500,450.000
...
```

## 🛠️ Development

### Adding New Controllers

1. **Implement controller** in `controller_pid.c`
2. **Add debug logging**:
   ```c
   DEBUG_LOG("controller_name_debug.csv", "%.3f,controller_type,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", 
             time, desired, current, error, output, integral, derivative);
   ```
3. **Update Makefile** if needed
4. **Create analysis script** in `analysis/` directory

### Cross-Platform Support

The system automatically detects the platform and adjusts file paths:
- **Windows**: Uses `GetModuleFileName()` for DLL path detection
- **Linux/Mac**: Uses `dladdr()` for shared library path detection
- **Fallback**: Relative paths if platform detection fails

### Build System

Each component has its own Makefile with debug support:
```makefile
# Enable debug mode
CFLAGS += -DDEBUG_LOGGING=1

# Include cross-platform utilities
SOURCES += utils.c
```

## 📈 Performance Tuning

### PID Tuning Guidelines

1. **Start with P-only**: Set I=0, D=0, adjust P for desired response
2. **Add Integral**: Increase I to eliminate steady-state error
3. **Add Derivative**: Increase D to reduce overshoot and improve stability
4. **Iterate**: Use analysis tools to quantify improvements

### Common Issues
- **Oscillation**: Reduce P gain or increase D gain
- **Slow Response**: Increase P gain
- **Steady-State Error**: Increase I gain
- **Instability**: Check for sensor noise, reduce D gain

## 🔍 Troubleshooting

### Common Problems

**Compilation Errors**:
```bash
# Check include paths
make clean && make DEBUG_LOGGING=1 -v

# Verify utils.h is found
ls -la utils.h
```

**Missing Log Files**:
```bash
# Check debug flag is enabled
grep -r "DEBUG_LOGGING" Makefile

# Verify logs directory exists
ls -la logs/
```

**Analysis Script Issues**:
```bash
# Install dependencies
pip install pandas matplotlib numpy

# Check log file path
python -c "import os; print(os.path.abspath('logs/pid_tuning_debug.csv'))"
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-controller`
3. **Add debug logging** for new components
4. **Update documentation**
5. **Submit pull request**

### Code Style
- Follow existing naming conventions
- Add comprehensive comments
- Include debug logging for new features
- Update analysis tools as needed

## 📚 References

- **Webots Documentation**: [cyberbotics.com](https://cyberbotics.com/doc/guide/index)
- **PID Control Theory**: Classical control systems literature
- **Blimp Dynamics**: Autonomous aerial vehicle control references

## 📄 License

This project is based on the original blimp physics model from EPFL's Laboratory of Intelligent Systems. Please see the original license headers in source files.

**Original Authors**: Alexis Guanella, Antoine Beyeler, Jean-Christophe Zufferey, Dario Floreano

**Enhanced Version**: Extended with comprehensive PID tuning and debugging capabilities.

---

## 🎉 Acknowledgments

- **EPFL LIS**: Original blimp physics implementation
- **Webots Team**: Excellent robotics simulation platform
- **Community**: Contributors and testers

For questions or support, please open an issue in the repository.
