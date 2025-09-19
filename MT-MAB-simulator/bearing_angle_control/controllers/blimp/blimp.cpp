/****************************************************************************

 blimp_16_sensor_bearing_control.cpp -- 16-Sensor Light Array Bearing Control

 This controller implements a multi-sensor light array for direct bearing
 calculation to guide a blimp to a light source. Uses 16 light sensors
 arranged in a circular pattern to calculate the weighted bearing angle
 to the light source.

 ALGORITHM:
 1. Read light intensity from all 16 sensors arranged in a circle
 2. Calculate weighted bearing angle using vector addition method
 3. Set yaw angle setpoint directly toward the calculated bearing
 4. Move forward at constant speed toward the light source

 SENSOR ARRAY:
 - 16 light sensors in circular pattern (22.5° separation)
 - Each sensor has a known angular position
 - Weighted vector sum gives bearing to strongest light

 USAGE:
 - Add a PointLight to your Webots world to act as the source.
 - The blimp uses sensors "light_sensor_0" through "light_sensor_15".
 - Press the 'S' key to start the autonomous source-seeking steering.
 - Press the 'Q' key to quit the simulation.

******************************************************************************/
#include <webots/emitter.h>
#include <webots/gps.h>
#include <webots/gyro.h>
#include <webots/inertial_unit.h>
#include <webots/keyboard.h>
#include <webots/light_sensor.h>
#include <webots/robot.h>

#include <cmath>
#include <cstdio>

// Debug logging includes
#if DEBUG_LOGGING
  #include <cstdio>
  #include <cstring>
  
  // Debug logging macro for trajectory data
  #define DEBUG_LOG_TRAJECTORY(fmt, ...) do { \
    FILE *debug_fp = fopen("logs/trajectory_debug.csv", "a"); \
    if (debug_fp) { \
      fprintf(debug_fp, fmt, ##__VA_ARGS__); \
      fclose(debug_fp); \
    } \
  } while(0)
#else
  #define DEBUG_LOG_TRAJECTORY(fmt, ...) do { } while(0)
#endif

#define TIMESTEP 32  // ms
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// === 16-SENSOR LIGHT ARRAY CONFIGURATION ===
const int NUM_LIGHT_SENSORS = 16;

// Sensor angles in degrees (matches the sensor positions in Blimp.proto)
const double SENSOR_ANGLES[NUM_LIGHT_SENSORS] = {
    0.0,     // Sensor 0
    22.5,    // Sensor 1
    45.0,    // Sensor 2
    67.5,    // Sensor 3
    90.0,    // Sensor 4
    112.5,   // Sensor 5
    135.0,   // Sensor 6
    157.5,   // Sensor 7
    180.0,   // Sensor 8
    202.5,   // Sensor 9
    225.0,   // Sensor 10
    247.5,   // Sensor 11
    270.0,   // Sensor 12
    292.5,   // Sensor 13
    315.0,   // Sensor 14
    337.5    // Sensor 15
};

// === SEARCH PARAMETERS ===
const double CONSTANT_FORWARD_SPEED = 0.2; // m/s
const double ALTITUDE_SETPOINT = 2.0;      // meters

// === TARGET LIGHT SOURCE LOCATION ===
const double TARGET_X = 5.1988; // meters
const double TARGET_Y = 5.329;  // meters
const double PROXIMITY_THRESHOLD = 2.0; // meters - stop when within this distance

// === LIGHT SIGNAL THRESHOLDS ===
const double MIN_TOTAL_LIGHT = 1.0;        // Minimum total light intensity to proceed

// === BEARING CALCULATION PARAMETERS ===
const double MIN_LIGHT_INTENSITY = 0.1;    // Minimum light to consider sensor reading valid
const double BEARING_SMOOTHING_FACTOR = 0.3; // Low-pass filter for bearing angle

// === STATE MACHINE ===
enum ControlState { IDLE, SEEKING };

// === BEARING CALCULATION FUNCTIONS ===

/**
 * @brief Calculates weighted bearing angle using light sensor array
 * @param light_values Array of light intensity values from all 16 sensors
 * @param result_angle Pointer to store the calculated bearing angle in degrees
 * @param total_weight Pointer to store the total weight (sum of light values)
 * @return True if calculation was successful
 */
bool calculate_light_bearing(double* light_values, double* result_angle, double* total_weight) {
    double sum_x = 0.0;  // Sum of weighted cosines
    double sum_y = 0.0;  // Sum of weighted sines
    double weight_sum = 0.0;
    
    // Calculate weighted vector sum
    for (int i = 0; i < NUM_LIGHT_SENSORS; i++) {
        if (light_values[i] > MIN_LIGHT_INTENSITY) {  // Only include sensors with sufficient light
            double angle_rad = SENSOR_ANGLES[i] * M_PI / 180.0;  // Convert to radians
            sum_x += light_values[i] * cos(angle_rad);
            sum_y += light_values[i] * sin(angle_rad);
            weight_sum += light_values[i];
        }
    }
    
    *total_weight = weight_sum;
    
    // If total weight is zero, we can't calculate a meaningful angle
    if (weight_sum <= MIN_LIGHT_INTENSITY) {
        *result_angle = 0.0;
        return false;
    }
    
    // Calculate the resulting angle using atan2
    double result_rad = atan2(sum_y, sum_x);
    
    // Convert to degrees and normalize to -180 to +180 range
    *result_angle = result_rad * 180.0 / M_PI;
    
    return true;
}

/**
 * @brief Normalizes angle to [-180, 180] range
 */
double normalize_angle(double angle) {
    while (angle > 180.0) angle -= 360.0;
    while (angle < -180.0) angle += 360.0;
    return angle;
}

// Main loop
int main() {
  printf("=== 16-SENSOR LIGHT ARRAY BLIMP SOURCE SEEKING ===\n");
  printf("Uses circular array of 16 light sensors for bearing calculation.\n");
  printf("Commands: S=START STEERING, Q=QUIT\n");
  printf("====================================================\n\n");

  wb_robot_init();

  // Initialize sensors
  WbDeviceTag imu = wb_robot_get_device("imu");
  wb_inertial_unit_enable(imu, TIMESTEP);
  WbDeviceTag gyro = wb_robot_get_device("gyro");
  wb_gyro_enable(gyro, TIMESTEP);
  WbDeviceTag gps = wb_robot_get_device("gps"); // Used ONLY to simulate onboard velocity estimation
  wb_gps_enable(gps, TIMESTEP);
  
  // Initialize 16 light sensors
  WbDeviceTag light_sensors[NUM_LIGHT_SENSORS];
  char sensor_name[32];
  for (int i = 0; i < NUM_LIGHT_SENSORS; i++) {
    sprintf(sensor_name, "light_sensor_%d", i);
    light_sensors[i] = wb_robot_get_device(sensor_name);
    if (light_sensors[i] == 0) {
      printf("Warning: Could not find sensor %s\n", sensor_name);
    } else {
      wb_light_sensor_enable(light_sensors[i], TIMESTEP);
      printf("Initialized %s at angle %.1f°\n", sensor_name, SENSOR_ANGLES[i]);
    }
  }

  // Initialize emitter
  WbDeviceTag gEmitter = wb_robot_get_device("emitter");

  // Enable keyboard
  wb_keyboard_enable(TIMESTEP);

  // State variables
  double controls[12] = {0};
  double past_time = 0.0;
  bool search_active = false;
  ControlState current_state = IDLE;

  // Light sensor array state variables
  double light_values[NUM_LIGHT_SENSORS] = {0};
  double bearing_angle = 0.0;        // Calculated bearing to light source (relative to robot)
  double smoothed_bearing = 0.0;     // Low-pass filtered bearing angle
  double total_light_intensity = 0.0; // Sum of all sensor readings

  // Wait for the first valid sensor readings
  while (wb_robot_step(TIMESTEP) != -1) {
    if (wb_robot_get_time() > 0.0) {
      past_time = wb_robot_get_time();
      
      // Initialize all light sensor readings
      total_light_intensity = 0.0;
      for (int i = 0; i < NUM_LIGHT_SENSORS; i++) {
        if (light_sensors[i] != 0) {
          light_values[i] = wb_light_sensor_get_value(light_sensors[i]);
          total_light_intensity += light_values[i];
        }
      }
      
      // Calculate initial bearing
      double temp_weight;
      if (calculate_light_bearing(light_values, &bearing_angle, &temp_weight)) {
        smoothed_bearing = bearing_angle;
        printf("Initial light bearing: %.1f° (total intensity: %.2f)\n", bearing_angle, total_light_intensity);
      }
      
      break;
    }
  }

  printf("Blimp initialized and moving forward. Ready for command.\n");

  // Initialize trajectory debug logging - write CSV header
  #if DEBUG_LOGGING
  // Use "w" mode to overwrite file on each simulation run
  FILE *debug_fp = fopen("logs/trajectory_debug.csv", "w");
  if (debug_fp) {
    fprintf(debug_fp, "time,x,y,z,yaw,setpoint_x,setpoint_y,setpoint_z,setpoint_yaw,setpoint_speed,current_speed,mode,distance_to_target\n");
    fclose(debug_fp);
  }
  #endif

  // Main control loop
  while (wb_robot_step(TIMESTEP) != -1) {
    double current_time = wb_robot_get_time();
    const double dt = current_time - past_time;
    past_time = current_time;

    // --- Keyboard Input ---
    int key = wb_keyboard_get_key();
    if (key > 0) {
      if (key == 'S' && !search_active) {
        search_active = true;
        current_state = SEEKING;
        printf(">>> 'S' key pressed. Starting autonomous light seeking...\n");
      } else if (key == 'Q') {
        printf(">>> 'Q' key pressed. Quitting simulation.\n");
        break;
      }
    }

    // --- Sensor Feedback ---
    const double *rpy = wb_inertial_unit_get_roll_pitch_yaw(imu);
    const double *gyro_vals = wb_gyro_get_values(gyro);
    const double *velocity = wb_gps_get_speed_vector(gps);
    double current_yaw = rpy[2] * 180.0 / M_PI;
    
    // Read all 16 light sensors
    total_light_intensity = 0.0;
    for (int i = 0; i < NUM_LIGHT_SENSORS; i++) {
      if (light_sensors[i] != 0) {
        light_values[i] = wb_light_sensor_get_value(light_sensors[i]);
        total_light_intensity += light_values[i];
      }
    }
    
    // Calculate bearing to light source
    double temp_weight;
    bool bearing_valid = calculate_light_bearing(light_values, &bearing_angle, &temp_weight);
    
    // Smooth the bearing angle to reduce noise
    if (bearing_valid) {
      // Handle angle wrapping for smooth filtering
      double angle_diff = bearing_angle - smoothed_bearing;
      if (angle_diff > 180.0) angle_diff -= 360.0;
      if (angle_diff < -180.0) angle_diff += 360.0;
      smoothed_bearing += BEARING_SMOOTHING_FACTOR * angle_diff;
      smoothed_bearing = normalize_angle(smoothed_bearing);
    }

    // --- Check proximity to target light source ---
    const double *gps_pos = wb_gps_get_values(gps);
    double distance_to_target = sqrt(pow(gps_pos[0] - TARGET_X, 2) + pow(gps_pos[1] - TARGET_Y, 2));
    
    if (distance_to_target <= PROXIMITY_THRESHOLD) {
      printf("\n🎯 SUCCESS! Reached light source at (%.3f, %.3f)\n", TARGET_X, TARGET_Y);
      printf("Final position: (%.3f, %.3f, %.3f)\n", gps_pos[0], gps_pos[1], gps_pos[2]);
      printf("Distance to target: %.3f m (threshold: %.3f m)\n", distance_to_target, PROXIMITY_THRESHOLD);
      printf("Total mission time: %.1f seconds\n", current_time);
      printf("Stopping simulation...\n");
      break;
    }

    // Set control state based on search activation
    if (search_active) {
      current_state = SEEKING;
    } else {
      current_state = IDLE;
    }

    // --- Control Algorithm ---
    double setpoint_yaw = current_yaw;
    double forward_speed = 0.0;

    if (current_state == SEEKING) {
      // === BEARING-BASED SEEKING MODE ===
      if (bearing_valid && total_light_intensity > MIN_TOTAL_LIGHT) {
        // Convert relative bearing to absolute yaw setpoint
        // The bearing is relative to the robot's current orientation
        setpoint_yaw = current_yaw + smoothed_bearing;
        setpoint_yaw = normalize_angle(setpoint_yaw);
        forward_speed = CONSTANT_FORWARD_SPEED;
      } else {
        // No valid bearing or insufficient light - stop and maintain heading
        setpoint_yaw = current_yaw;
        forward_speed = 0.0;
        if (total_light_intensity <= MIN_TOTAL_LIGHT) {
          printf(">>> No light detected (total: %.2f). Waiting for light source...\n", total_light_intensity);
        }
      }
    } else {
      // IDLE state - maintain current heading, no movement
      setpoint_yaw = current_yaw;
      forward_speed = 0.0;
    }

    // --- Send Commands to Low-Level Controller ---
    if (gEmitter) {
      controls[0] = ALTITUDE_SETPOINT;
      controls[1] = setpoint_yaw;              // Yaw angle setpoint
      controls[2] = forward_speed;             // Variable speed based on state
      controls[3] = 0.0;                       // Yaw rate setpoint (not used)
      controls[4] = 0.0;                       // Altitude velocity setpoint

      // Current state feedback
      controls[5] = velocity[0];
      controls[6] = velocity[1];
      controls[7] = velocity[2];
      controls[8] = wb_gps_get_values(gps)[2];
      controls[9] = current_yaw;
      controls[10] = gyro_vals[2] * 180.0 / M_PI;
      controls[11] = dt;

      wb_emitter_send(gEmitter, controls, sizeof(controls));
    }

    // --- Debug Trajectory Logging ---
    #if DEBUG_LOGGING
    double current_speed = sqrt(velocity[0]*velocity[0] + velocity[1]*velocity[1]); // Horizontal speed
    const char* mode;
    if (current_state == SEEKING) {
      mode = "SEEKING";
    } else {
      mode = "IDLE";
    }
    
    // For this source seeking controller, we don't have explicit X,Y setpoints
    // But we can log the current position and derived setpoints
    double setpoint_x = gps_pos[0]; // No explicit X setpoint in this controller
    double setpoint_y = gps_pos[1]; // No explicit Y setpoint in this controller
    double setpoint_z = ALTITUDE_SETPOINT;
    double setpoint_yaw_rad = setpoint_yaw * M_PI / 180.0;
    
    DEBUG_LOG_TRAJECTORY("%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%s,%.3f\n",
                        current_time,
                        gps_pos[0], gps_pos[1], gps_pos[2], // Current x,y,z
                        rpy[2], // Current yaw in radians
                        setpoint_x, setpoint_y, setpoint_z, // Setpoint x,y,z
                        setpoint_yaw_rad, // Setpoint yaw in radians
                        forward_speed, // Current setpoint speed (varies by state)
                        current_speed, // Current actual speed
                        mode, // Controller mode (FORWARD/SEEKING/EXPLORING)
                        distance_to_target); // Distance to light source target
    #endif

    // Status printing
    static double last_print_time = 0;
    if (current_time - last_print_time > 0.5) {
       double yaw_error = setpoint_yaw - current_yaw;
       yaw_error = normalize_angle(yaw_error);
      
       const char* state_name;
       if (current_state == SEEKING) {
         state_name = "SEEKING";
       } else {
         state_name = "IDLE";
       }
      
      // Show strongest sensor reading for debugging
      double max_light = 0.0;
      int strongest_sensor = -1;
      for (int i = 0; i < NUM_LIGHT_SENSORS; i++) {
        if (light_values[i] > max_light) {
          max_light = light_values[i];
          strongest_sensor = i;
        }
      }
      
      printf("[T:%.1fs] %s | TotalLight:%.1f | Bearing:%.1f° | StrongestSensor:%d(%.1f°) | Dist:%.2fm | Yaw:%.1f->%.1f\n",
             current_time,
             state_name,
             total_light_intensity,
             bearing_valid ? smoothed_bearing : 999.9,
             strongest_sensor,
             strongest_sensor >= 0 ? SENSOR_ANGLES[strongest_sensor] : 0.0,
             distance_to_target,
             current_yaw, setpoint_yaw);
      last_print_time = current_time;
    }
  }

  wb_robot_cleanup();
  return 0;
}