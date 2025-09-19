/****************************************************************************

 blimp_source_seeking_es_yaw_angle.cpp -- Position-Free Source Seeking (Yaw Angle Control)

 This controller implements a classic dither-and-demodulation Extremum
 Seeking (ES) algorithm to guide a blimp to a light source by controlling
 its YAW ANGLE setpoint. This method is designed for systems WITHOUT
 global position feedback.

 ALGORITHM:
 1. A base heading is maintained, representing the estimated direction to the source.
 2. A small sinusoidal "dither" is added to this base heading to create a
    continuously wiggling YAW ANGLE setpoint.
 3. The rate of change of the light sensor signal (dJ/dt) is calculated.
 4. The dJ/dt is multiplied by the rate-of-change of the dither signal
    (demodulation) to estimate the gradient.
 5. The result is filtered and integrated to update the base heading,
    steering the blimp "uphill".
 6. Forward speed is kept constant.

 USAGE:
 - Add a PointLight to your Webots world to act as the source.
 - Add a LightSensor to your blimp robot and name it "light_sensor".
 - The blimp will start moving forward.
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

// === SEARCH PARAMETERS ===
const double CONSTANT_FORWARD_SPEED = 0.2; // m/s
const double ALTITUDE_SETPOINT = 2.0;      // meters
const double EXPLORING_SPEED = 0.15;       // m/s - slower speed during exploration
const double EXPLORING_RADIUS = 1.0;       // meters - radius for spiral exploration

// === TARGET LIGHT SOURCE LOCATION ===
const double TARGET_X = 5.1988; // meters
const double TARGET_Y = 5.329;  // meters
const double PROXIMITY_THRESHOLD = 2.0; // meters - stop when within this distance

// === LIGHT SIGNAL THRESHOLDS ===
const double LIGHT_LOSS_THRESHOLD = 1.0;   // Light intensity below this triggers exploring
const double LIGHT_FOUND_THRESHOLD = 30.0;  // Light intensity above this returns to seeking
const double LIGHT_LOSS_TIME = 5.0;        // Seconds of low light before switching to exploring

// === CLASSIC EXTREMUM SEEKING (ES) PARAMETERS (FOR YAW ANGLE) ===
const double ES_DITHER_FREQ = 0.2;      // ω (rad/s): Frequency of the heading wiggle.
const double ES_DITHER_AMP_YAW = 20;    // A (deg): Amplitude of the heading wiggle.
const double ES_LEARNING_RATE = 0.5;    // k: Integration gain, how fast it steers.
const double ES_WASHOUT_FREQ = 0.8;     // ω_h (rad/s): High-pass filter cutoff for dJ/dt.

// === STATE MACHINE ===
enum ControlState { FORWARD, SEEKING, EXPLORING };


// Main loop
int main() {
  printf("=== CLASSIC POSITION-FREE BLIMP SOURCE SEEKING (YAW ANGLE) ===\n");
  printf("Relies only on velocity, attitude, and signal measurements.\n");
  printf("Commands: S=START STEERING, Q=QUIT\n");
  printf("=============================================================\n\n");

  wb_robot_init();

  // Initialize sensors
  WbDeviceTag imu = wb_robot_get_device("imu");
  wb_inertial_unit_enable(imu, TIMESTEP);
  WbDeviceTag gyro = wb_robot_get_device("gyro");
  wb_gyro_enable(gyro, TIMESTEP);
  WbDeviceTag gps = wb_robot_get_device("gps"); // Used ONLY to simulate onboard velocity estimation
  wb_gps_enable(gps, TIMESTEP);
  WbDeviceTag light_sensor = wb_robot_get_device("light_sensor");
  wb_light_sensor_enable(light_sensor, TIMESTEP);

  // Initialize emitter
  WbDeviceTag gEmitter = wb_robot_get_device("emitter");

  // Enable keyboard
  wb_keyboard_enable(TIMESTEP);

  // State variables
  double controls[12] = {0};
  double past_time = 0.0;
  bool search_active = false;
  ControlState current_state = FORWARD;

  // ES algorithm state variables
  double past_signal_value = -1;
  double dJ_dt_filtered = 0.0;
  double base_yaw_angle = 0.0; // This is the integrated steering command (the main heading)

  // Light loss detection variables
  double light_loss_start_time = -1.0;

  // Wait for the first valid sensor readings
  while (wb_robot_step(TIMESTEP) != -1) {
    if (wb_robot_get_time() > 0.0) {
      past_time = wb_robot_get_time();
      past_signal_value = wb_light_sensor_get_value(light_sensor);
      const double *rpy = wb_inertial_unit_get_roll_pitch_yaw(imu);
      base_yaw_angle = rpy[2] * 180.0 / M_PI; // Initialize base yaw to starting yaw
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
        current_state = SEEKING; // Start in seeking mode
        printf(">>> 'S' key pressed. Starting autonomous steering in SEEKING mode...\n");
      } else if (key == 'Q') {
        printf(">>> 'Q' key pressed. Quitting simulation.\n");
        break;
      }
    }

    // --- Sensor Feedback ---
    const double *rpy = wb_inertial_unit_get_roll_pitch_yaw(imu);
    const double *gyro_vals = wb_gyro_get_values(gyro);
    const double *velocity = wb_gps_get_speed_vector(gps);
    double current_signal = wb_light_sensor_get_value(light_sensor);
    double current_yaw = rpy[2] * 180.0 / M_PI;

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

    // --- Light Signal Monitoring and State Machine ---
    if (search_active) {
      // Monitor light signal loss
      if (current_signal < LIGHT_LOSS_THRESHOLD) {
        if (light_loss_start_time < 0) {
          light_loss_start_time = current_time; // Start timing light loss
        } else if (current_time - light_loss_start_time > LIGHT_LOSS_TIME && current_state != EXPLORING) {
          printf(">>> LIGHT SIGNAL LOST! Switching to EXPLORING mode after %.1fs\n", LIGHT_LOSS_TIME);
          past_signal_value = 0.0;
          current_state = EXPLORING;
        }
      } else if (current_signal > LIGHT_FOUND_THRESHOLD) {
        // Light signal recovered
        if (current_state == EXPLORING) {
          printf(">>> LIGHT SIGNAL RECOVERED! Returning to SEEKING mode\n");
          current_state = SEEKING;
        }
        light_loss_start_time = -1.0; // Reset light loss timer
      } else {
        // Signal is in middle range - maintain current state but reset loss timer if signal improving
        if (current_signal > past_signal_value) {
          light_loss_start_time = -1.0;
        }
      }
    }

    // --- Multi-State Control Algorithm ---
    double setpoint_yaw = base_yaw_angle;
    double forward_speed = CONSTANT_FORWARD_SPEED;

    if (search_active) {
      if (current_state == SEEKING) {
        // === EXTREMUM SEEKING MODE ===
        forward_speed = CONSTANT_FORWARD_SPEED;
        
        // 1. Calculate rate of change of the signal (dJ/dt) with a high-pass filter
        double dJ = current_signal - past_signal_value;
        double alpha = dt * ES_WASHOUT_FREQ;
        dJ_dt_filtered = (1 - alpha) * dJ_dt_filtered + (1 - alpha) * dJ;
        past_signal_value = current_signal;

        // 2. Create the dither signal for the YAW ANGLE
        double dither_angle = ES_DITHER_AMP_YAW * sin(ES_DITHER_FREQ * current_time);

        // 3. Create the demodulation signal (the derivative of the dither angle)
        double demodulation_signal = ES_DITHER_AMP_YAW * ES_DITHER_FREQ * cos(ES_DITHER_FREQ * current_time);

        // 4. Demodulate: Multiply dJ/dt by the demodulation signal
        double gradient_estimate = dJ_dt_filtered * demodulation_signal;

        // 5. Integrate to update the base heading
        base_yaw_angle += ES_LEARNING_RATE * gradient_estimate * dt;

        // Normalize the base_yaw_angle to keep it within [-180, 180]
        while (base_yaw_angle > 180.0) base_yaw_angle -= 360.0;
        while (base_yaw_angle < -180.0) base_yaw_angle += 360.0;
        
        // The final yaw setpoint is the base heading plus the probing dither
        setpoint_yaw = base_yaw_angle + dither_angle;
        
      } else if (current_state == EXPLORING) {
        // === EXPLORING MODE (Spiral Search) ===
        forward_speed = EXPLORING_SPEED;
        
        // Spiral search pattern: constant angular velocity
        double yaw_rate = (EXPLORING_SPEED / EXPLORING_RADIUS) * (180.0 / M_PI); // deg/s
        base_yaw_angle += yaw_rate * dt;
        
        // Normalize the base yaw angle
        while (base_yaw_angle > 180.0) base_yaw_angle -= 360.0;
        while (base_yaw_angle < -180.0) base_yaw_angle += 360.0;
        
        setpoint_yaw = base_yaw_angle;
      }
    } else {
      // Not active - just maintain current heading
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
    if (!search_active) {
      mode = "FORWARD";
    } else if (current_state == SEEKING) {
      mode = "SEEKING";
    } else if (current_state == EXPLORING) {
      mode = "EXPLORING";
    } else {
      mode = "UNKNOWN";
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
       while (yaw_error > 180.0) yaw_error -= 360.0;
       while (yaw_error < -180.0) yaw_error += 360.0;
      
       const char* state_name;
       if (!search_active) {
         state_name = "FORWARD";
       } else if (current_state == SEEKING) {
         state_name = "SEEKING";
       } else if (current_state == EXPLORING) {
         state_name = "EXPLORING";
       } else {
         state_name = "UNKNOWN";
       }
      
      printf("[T:%.1fs] State:%s | Light:%.1f | Dist2Target:%.2fm | Speed:%.2f | Yaw: %.1f->%.1f (err:%.1f)\n",
             current_time,
             state_name,
             current_signal,
             distance_to_target,
             forward_speed,
             current_yaw, setpoint_yaw, yaw_error);
      last_print_time = current_time;
    }
  }

  wb_robot_cleanup();
  return 0;
}