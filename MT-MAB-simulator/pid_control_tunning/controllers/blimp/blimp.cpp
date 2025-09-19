/****************************************************************************

blimp_three_primitive_test -- Test controller for Georgia Tech blimp approach

This controller tests the three motion primitives separately:
1. YAW CONTROL - Turn left/right in place
2. SPEED CONTROL - Move forward/backward at different speeds  
3. ALTITUDE CONTROL - Move up/down to different heights

******************************************************************************/

#include <webots/camera.h>
#include <webots/emitter.h>
#include <webots/keyboard.h>
#include <webots/robot.h>
#include <webots/inertial_unit.h>
#include <webots/gyro.h>
#include <webots/supervisor.h>
#include <webots/distance_sensor.h>
#include <webots/gps.h>
#include <stdio.h>
#include <fstream>
#include <math.h>

#include "js.h"

// Debug logging macro - only log if DEBUG_LOGGING is enabled
// DEBUG_LOGGING is set via Makefile: -DDEBUG_LOGGING=1 or -DDEBUG_LOGGING=0
#if DEBUG_LOGGING
#define TRAJECTORY_LOG(...) do { \
    FILE *debug_fp = fopen("logs/trajectory_debug.csv", "a"); \
    if (debug_fp) { \
        fprintf(debug_fp, __VA_ARGS__); \
        fclose(debug_fp); \
    } \
} while(0)
#else
#define TRAJECTORY_LOG(...) do {} while(0)
#endif

#define TIMESTEP 32  // ms

// Control parameters
double yaw_step = 5;            // Yaw command step (degree)
double speed_step = 0.1;          // Speed command step (m/s)
double altitude_step = 0.8;       // Altitude command step (meters)
double yaw_rate_step = 0.8;         // Yaw rate step (deg/s) - NEW
double altitude_vel_step = 0.2;      // Altitude velocity step (m/s) - NEW


// Circle parameters
const double circle_radius = 0.5; // meters
const double circle_speed = 0.2; // m/s
const double circle_altitude = 10; // meters
const double angular_velocity = (circle_speed / circle_radius) * 180 / 3.14159; // rad/s

// Setpoints for the three motion primitives
double setpoint_x = 0.0;          // X position (hold constant)
double setpoint_y = 0.0;          // Y position (hold constant)
double setpoint_z = 1.5; // Altitude setpoint (TEST 3)
double setpoint_yaw = 0.0;        // Yaw setpoint (TEST 1)
double setpoint_speed = 0.0;      // Forward speed setpoint (TEST 2)
double setpoint_yaw_rate = 0.0;      // Yaw rate setpoint (deg/s) - NEW
double setpoint_altitude_vel = 0.0;  // Altitude velocity setpoint (m/s) - NEW


// Test modes - EXPANDED to include inner loop tests
typedef enum {
    TEST_YAW = 0,           // Test yaw angle controller (outer loop)
    TEST_YAW_RATE = 1,      // Test yaw rate controller (inner loop) - NEW
    TEST_SPEED = 2,         // Test speed controller
    TEST_ALTITUDE = 3,      // Test altitude position controller (outer loop)  
    TEST_ALTITUDE_VEL = 4,  // Test altitude velocity controller (inner loop) - NEW
    TEST_ALL = 5,           // Test all three together
    TEST_CIRCLE = 6         // Fly in a circle
} test_mode_t;


// Print current test mode
void print_test_mode(test_mode_t current_test_mode) {
  switch(current_test_mode) {
    case TEST_YAW:         printf(">>> YAW ANGLE TEST - Use A/D to change yaw angle setpoint\n"); break;
    case TEST_YAW_RATE:    printf(">>> YAW RATE TEST - Use A/D to change yaw rate setpoint (deg/s)\n"); break;
    case TEST_SPEED:       printf(">>> SPEED TEST - Use W/S to change speed setpoint\n"); break; 
    case TEST_ALTITUDE:    printf(">>> ALTITUDE POSITION TEST - Use R/F to change altitude setpoint\n"); break;
    case TEST_ALTITUDE_VEL:printf(">>> ALTITUDE VELOCITY TEST - Use R/F to change altitude velocity setpoint (m/s)\n"); break;
    case TEST_ALL:         printf(">>> ALL TEST MODE - Use A/D=yaw, W/S=speed, R/F=altitude\n"); break;
    case TEST_CIRCLE:      printf(">>> CIRCLE TEST MODE - Flying in a circle\n"); break;
  }
}

int main() {

  // === GEORGIA TECH THREE-PRIMITIVE BLIMP CONTROLLER TEST ===
  printf("=== GEORGIA TECH BLIMP CONTROLLER TEST (7 MODES) ===\n");
  printf("Testing Speed, Yaw (angle+rate), and Altitude (pos+vel) controllers\n");
  printf("Use keys: 1=YAW_ANGLE, 2=YAW_RATE, 3=SPEED, 4=ALT_POS, 5=ALT_VEL, 6=ALL, 7=CIRCLE\n");
  printf("Commands: A/D=yaw, W/S=speed, R/F=altitude, Q=quit\n");
  printf("==============================================================\n\n");

  // Test mode selection
  test_mode_t current_test_mode = TEST_CIRCLE;  // Start with circle test

  wb_robot_init();

  // Initialize sensors
  WbDeviceTag imu = wb_robot_get_device("imu");
  wb_inertial_unit_enable(imu, TIMESTEP);
  WbDeviceTag gyro = wb_robot_get_device("gyro");
  wb_gyro_enable(gyro, TIMESTEP);
  WbDeviceTag gps = wb_robot_get_device("gps");
  wb_gps_enable(gps, TIMESTEP);
  // Get drone node for position feedback
  WbNodeRef droneNode = wb_supervisor_node_get_from_def("blimp_lis");
  if (droneNode == NULL) {
    printf("Could not find drone node!\n");
    return 1;
  }

  // Initialize emitter
  WbDeviceTag gEmitter = wb_robot_get_device("emitter");
  if (!gEmitter) {
    printf("!!! blimp_three_primitive_test :: emitter is not available.\n");
    return 1;
  }
  // Enable keyboard for commands
  wb_keyboard_enable(TIMESTEP);
  
  // Control buffer and timing
  double controls[12] = {0}; 
  double past_x_global = -1, past_y_global = -1, past_z_global = -1, past_time = wb_robot_get_time();
  
  // Status reporting
  double last_print_time = 0;
  double last_command_time = 0;
  
  // Initialize trajectory logging with CSV header
  #if DEBUG_LOGGING
  // Use "w" mode to overwrite file on each simulation run
  FILE *debug_fp = fopen("logs/trajectory_debug.csv", "w");
  if (debug_fp) {
    fprintf(debug_fp, "time,x,y,z,yaw,setpoint_x,setpoint_y,setpoint_z,setpoint_yaw,setpoint_speed,current_speed,mode\n");
    fclose(debug_fp);
  }
  #endif
  
  bool first_step = true; // Flag for first simulation step
  
  while (wb_robot_step(TIMESTEP) != -1) {
    // On the first simulation step, initialize setpoints to current state
    if (first_step) {

      // Set initial setpoints to drone's starting position and orientation
      const double *initial_position = wb_gps_get_values(gps);
      setpoint_x = initial_position[0];
      setpoint_y = initial_position[1];
      setpoint_z = 0;
      past_x_global = initial_position[0];
      past_y_global = initial_position[1];
      past_z_global = initial_position[2];
      const double *initial_rpy = wb_inertial_unit_get_roll_pitch_yaw(imu);
      setpoint_yaw = initial_rpy[2] * 180.0 / 3.14159;

      // Initialize inner loop setpoints to zero
      setpoint_yaw_rate = 0.0;
      setpoint_altitude_vel = 0.0;

      printf("Initial position: (%.2f, %.2f, %.2f)\n", setpoint_x, setpoint_y, setpoint_z);
      printf("Initial yaw: %.1f degrees\n", setpoint_yaw);
      printf("Initial speed: %.1f m/s\n", 0.f);
      print_test_mode(current_test_mode);
      first_step = false;
      continue;
    }

    // In circle mode, continuously update the yaw setpoint
    if (current_test_mode == TEST_CIRCLE) {
      const double dt = wb_robot_get_time() - past_time;
      setpoint_yaw += angular_velocity * dt;
      setpoint_speed = circle_speed;
      setpoint_z = circle_altitude;
    }

    // Handle keyboard input
    int key = wb_keyboard_get_key();
    if (key > 0) {
      double current_time = wb_robot_get_time();
      
      if (current_time - last_command_time > 0.3) {
        
        // Test mode selection (1-7)
        if (key >= '1' && key <= '7') {
           current_test_mode = (test_mode_t)(key - '1');
           print_test_mode(current_test_mode);
           
           // Reset inner loop setpoints when switching modes
           if (current_test_mode != TEST_YAW_RATE) setpoint_yaw_rate = 0.0;
           if (current_test_mode != TEST_ALTITUDE_VEL) setpoint_altitude_vel = 0.0;
           
           last_command_time = current_time;
           continue;
        }
        
        // Command handling based on test mode
        switch (key) {
          case 'A':  // Yaw left (angle or rate)
            if (current_test_mode == TEST_YAW || current_test_mode == TEST_ALL) {
              setpoint_yaw += yaw_step;
              printf("Yaw ANGLE LEFT  -> %.1f degrees\n", setpoint_yaw);
              last_command_time = current_time;
            }
            else if (current_test_mode == TEST_YAW_RATE) {
              setpoint_yaw_rate += yaw_rate_step;
              setpoint_yaw_rate = (setpoint_yaw_rate > 90.0) ? 90.0 : setpoint_yaw_rate; // Max 90 deg/s
              printf("Yaw RATE LEFT  -> %.1f deg/s\n", setpoint_yaw_rate);
              last_command_time = current_time;
            }
            break;
            
          case 'D':  // Yaw right (angle or rate)
            if (current_test_mode == TEST_YAW || current_test_mode == TEST_ALL) {
              setpoint_yaw -= yaw_step;
              printf("Yaw ANGLE RIGHT -> %.1f degrees\n", setpoint_yaw);
              last_command_time = current_time;
            }
            else if (current_test_mode == TEST_YAW_RATE) {
              setpoint_yaw_rate -= yaw_rate_step;
              setpoint_yaw_rate = (setpoint_yaw_rate < -90.0) ? -90.0 : setpoint_yaw_rate; // Min -90 deg/s
              printf("Yaw RATE RIGHT -> %.1f deg/s\n", setpoint_yaw_rate);
              last_command_time = current_time;
            }
            break;
            
          case 'W':  // Speed forward
            if (current_test_mode == TEST_SPEED || current_test_mode == TEST_ALL) {
              setpoint_speed += speed_step;
              setpoint_speed = (setpoint_speed > 2.0) ? 2.0 : setpoint_speed;
              setpoint_z += altitude_step;
              printf("Speed FORWARD -> %.2f m/s\n", setpoint_speed);
              last_command_time = current_time;
            }
            break;
            
          case 'S':  // Speed backward
            if (current_test_mode == TEST_SPEED || current_test_mode == TEST_ALL) {
              setpoint_speed -= speed_step;
              setpoint_speed = (setpoint_speed < 0.0) ? 0.0 : setpoint_speed;
              printf("Speed BACKWARD -> %.2f m/s\n", setpoint_speed);
              last_command_time = current_time;
            }
            break;
            
          case 'R':  // Altitude up (position or velocity)
            if (current_test_mode == TEST_ALTITUDE || current_test_mode == TEST_ALL) {
              setpoint_z += altitude_step;
              setpoint_z = (setpoint_z > 3.0) ? 3.0 : setpoint_z;
              printf("Altitude POSITION UP -> %.2f meters\n", setpoint_z);
              last_command_time = current_time;
            }
            else if (current_test_mode == TEST_ALTITUDE_VEL) {
              setpoint_altitude_vel += altitude_vel_step;
              setpoint_altitude_vel = (setpoint_altitude_vel > 1.5) ? 1.5 : setpoint_altitude_vel; // Max 1.5 m/s up
              printf("Altitude VELOCITY UP -> %.2f m/s\n", setpoint_altitude_vel);
              last_command_time = current_time;
            }
            break;
            
          case 'F':  // Altitude down (position or velocity)
            if (current_test_mode == TEST_ALTITUDE || current_test_mode == TEST_ALL) {
              setpoint_z -= altitude_step;
              setpoint_z = (setpoint_z < 0.5) ? 0.5 : setpoint_z;
              printf("Altitude POSITION DOWN -> %.2f meters\n", setpoint_z);
              last_command_time = current_time;
            }
            else if (current_test_mode == TEST_ALTITUDE_VEL) {
              setpoint_altitude_vel -= altitude_vel_step;
              setpoint_altitude_vel = (setpoint_altitude_vel < -1.5) ? -1.5 : setpoint_altitude_vel; // Max 1.5 m/s down
              printf("Altitude VELOCITY DOWN -> %.2f m/s\n", setpoint_altitude_vel);
              last_command_time = current_time;
            }
            break;
            
          case 'Q':  // Quit
            printf("Quitting test mode...\n");
            wb_robot_cleanup();
            return 0;
        }
      }
    }
    // Normalize yaw to [-π, π] 
    while (setpoint_yaw > 180.0) setpoint_yaw -= 360.0;
    while (setpoint_yaw < -180.0) setpoint_yaw += 360.0;

    if (gEmitter) {
      // Get current state
      const double *gyro_vals = wb_gyro_get_values(gyro);
      const double *position = wb_gps_get_values(gps);
      const double *rpy = wb_inertial_unit_get_roll_pitch_yaw(imu);
      const double dt = wb_robot_get_time() - past_time;
      past_time = wb_robot_get_time();

      // Calculate global velocity for state feedback
      double x_global = position[0];
      double vx_global = (x_global - past_x_global) / dt;
      past_x_global = x_global;
      
      double y_global = position[1];
      double vy_global = (y_global - past_y_global) / dt;
      past_y_global = y_global;

      double z_global = position[2];
      double vz_global = (z_global - past_z_global) / dt;
      past_z_global = z_global;

      // === EXPANDED CONTROL ARRAY ===
      controls[0] = setpoint_z;           // Z position setpoint
      controls[1] = setpoint_yaw;         // Yaw angle setpoint  
      controls[2] = setpoint_speed;       // Forward speed setpoint
      controls[3] = setpoint_yaw_rate;    // Yaw rate setpoint (NEW)
      controls[4] = setpoint_altitude_vel;// Altitude velocity setpoint (NEW)
      
      // Current state feedback
      controls[5] = vx_global;            // Current global X velocity
      controls[6] = vy_global;            // Current global Y velocity
      controls[7] = vz_global;            // Current global Z velocity
      controls[8] = position[2];          // Current Z position
      controls[9] = rpy[2] * 180 / 3.14159;               // Current Yaw angle
      controls[10] = gyro_vals[2] * 180 / 3.14159;        // Current Yaw rate
      controls[11] = dt;                  // Time step

      wb_emitter_send(gEmitter, controls, sizeof(controls));
      
      // Enhanced trajectory logging
      double cos_yaw = cos(rpy[2]);
      double sin_yaw = sin(rpy[2]);
      double current_speed = vx_global * cos_yaw + vy_global * sin_yaw;
      const char* mode_names[] = {"YAW", "YAW_RATE", "SPEED", "ALTITUDE", "ALT_VEL", "ALL", "CIRCLE"};
      
      TRAJECTORY_LOG("%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s\n",
              wb_robot_get_time(),
              position[0], position[1], position[2], rpy[2],
              setpoint_x, setpoint_y, setpoint_z, setpoint_yaw,
              setpoint_speed, current_speed,
              mode_names[current_test_mode]);
      
      // Enhanced status printing
      if (wb_robot_get_time() - last_print_time > 0.3) {
        double cos_yaw = cos(rpy[2]);
        double sin_yaw = sin(rpy[2]);
        double current_speed = vx_global * cos_yaw + vy_global * sin_yaw;
        double current_yaw_rate = gyro_vals[2] * 180.0 / 3.14159;
        
        printf("position: (%.2f, %.2f, %.2f)\n", position[0], position[1], position[2]);
        
        const char* mode_names[] = {"YAW", "YAW_RATE", "SPEED", "ALTITUDE", "ALT_VEL", "ALL", "CIRCLE"};
        
        if (current_test_mode == TEST_YAW_RATE) {
          printf("[%s] YawRate: %.1f->%.1f deg/s | Speed: %.2f m/s | Alt: %.2f m\n", 
                 mode_names[current_test_mode],
                 current_yaw_rate, setpoint_yaw_rate,
                 current_speed, position[2]);
        }
        else if (current_test_mode == TEST_ALTITUDE_VEL) {
          printf("[%s] AltVel: %.2f->%.2f m/s | Yaw: %.1f° | Speed: %.2f m/s\n", 
                 mode_names[current_test_mode],
                 vz_global, setpoint_altitude_vel,
                 rpy[2] * 180.0 / 3.14159, current_speed);
        }
        else {
          // Original status display for other modes
          double yaw_error = setpoint_yaw - rpy[2] * 180.0 / 3.14159;
          while (yaw_error > 180.0) yaw_error -= 360.0;
          while (yaw_error < -180.0) yaw_error += 360.0;
          double altitude_error = setpoint_z - position[2];
          
          printf("[%s] Yaw: %.1f°->%.1f° (err=%.1f°) | Speed: %.2f->%.2f m/s | Alt: %.2f->%.2f m (err=%.2f)\n", 
                 mode_names[current_test_mode],
                 rpy[2] * 180.0 / 3.14159, setpoint_yaw, yaw_error,
                 current_speed, setpoint_speed,
                 position[2], setpoint_z, altitude_error);
        }
        
        last_print_time = wb_robot_get_time();
      }
    }
  }
  #if DEBUG_LOGGING
  printf("Trajectory data saved to logs/trajectory_debug.csv\n");
  #endif

  wb_robot_cleanup();
  return 0;
}
