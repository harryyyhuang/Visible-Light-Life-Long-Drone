/**
 *    ||          ____  _ __
 * +------+      / __ )(_) /_______________ _____  ___
 * | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *  ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2025 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * controller_pid.c - Georgia Tech Blimp controller implementation
 *
 * Based on "Autopilot design for A class of miniature autonomous blimps" (2017)
 * This controller implements three separate PID controllers for blimp motion primitives:
 * 
 * 1. SPEED CONTROLLER - Controls forward/backward speed (u velocity in body frame)
 *    Input: Desired forward speed (m/s)
 *    Output: Forward thrust force (pitch moment)
 * 
 * 2. YAW CONTROLLER - Controls yaw angle/orientation (ψ)
 *    Input: Desired yaw angle (radians)
 *    Output: Yaw torque
 * 
 * 3. ALTITUDE CONTROLLER - Controls vertical position/height (z)
 *    Input: Desired altitude (meters)
 *    Output: Vertical thrust force
 * 
 * Each controller is independent and can be tuned separately.
 * This matches real blimp behavior: turn to face target, move forward, maintain altitude.
 */

#include "controller_pid.h"
#include "utils.h"  // For cross-platform debug logging
#include <string.h> // For memset
#include <math.h>   // For fabs, etc.
#include <stdio.h>  // For file I/O
#include <stdint.h>

// Blimp control rates (matching Crazyflie structure)
#define BLIMP_CONTROL_RATE    100   // Main control loop rate (Hz)
#define BLIMP_POSITION_RATE   50    // Position control rate (Hz)
#define BLIMP_ATTITUDE_RATE   100   // Attitude control rate (Hz)
#define BLIMP_UPDATE_DT       (1.0f/100.0f) // 100Hz timestep

// Replace with time-based rate control:
static float last_position_time = 0.0f;
static float last_attitude_time = 0.0f;

// Time-based rate execution macro
#define RATE_DO_EXECUTE(RATE_HZ, LAST_TIME, ACCUMELATEDTIME) \
    ((ACCUMELATEDTIME - LAST_TIME) >= (1.0f / RATE_HZ))

// Update time tracking macro
#define RATE_UPDATE_TIME(LAST_TIME, ACCUMELATEDTIME) \
    (LAST_TIME = ACCUMELATEDTIME)


// Helper function to constrain a value between a min and max
static float constrain(float value, float min, float max) {
    if (value > max) return max;
    if (value < min) return min;
    return value;
}

// Static variables for PID controller states (integrals and previous errors)
// Each motion primitive has its own PID state variables

// SPEED CONTROLLER STATE
static float speed_integral = 0.0f;
static float speed_prev_error = 0.0f;

// YAW CONTROLLER STATE (cascaded: angle -> rate)
static float yaw_angle_integral = 0.0f;
static float yaw_angle_prev_error = 0.0f;
static float yaw_rate_integral = 0.0f;
static float yaw_rate_prev_error = 0.0f;
static float yaw_rate_desired = 0.0f;

// ALTITUDE CONTROLLER STATE (cascaded: position -> velocity)
static float altitude_pos_integral = 0.0f;
static float altitude_pos_prev_error = 0.0f;
static float altitude_vel_integral = 0.0f;
static float altitude_vel_prev_error = 0.0f;
static float altitude_vel_desired = 0.0f;

// --- BLIMP PID GAINS (Matching Crazyflie Structure) ---

// SPEED CONTROLLER GAINS (X velocity control)
float Kp_speed = 100.0f;    // X velocity proportional
float Ki_speed = 40.0f;    // X velocity integral  
float Kd_speed = 100.0f;    // X velocity derivative
float speed_integral_limit = 0.0f; // Integral windup limit

// YAW CONTROLLER GAINS (Cascaded: Angle -> Rate)
// Outer loop: Yaw angle control
float Kp_yaw_angle = 5.0f;     // Yaw angle proportional
float Ki_yaw_angle = 0.0f;     // Yaw angle integral
float Kd_yaw_angle = 20.0f;     // Yaw angle derivative
float yaw_angle_integral_limit = 360.0f; // Integral limit (degrees)

// Inner loop: Yaw rate control  
float Kp_yaw_rate = 120.0f;      // Yaw rate proportional
float Ki_yaw_rate = 7.0f;      // Yaw rate integral
float Kd_yaw_rate = 0.0f;      // Yaw rate derivative
float yaw_rate_integral_limit = 166.7f; // Integral limit (deg/s)

// ALTITUDE CONTROLLER GAINS (Cascaded: Position -> Velocity)
// Outer loop: Z position control
float Kp_altitude_pos = 3.0f;    // Z position proportional
float Ki_altitude_pos = 0.005f;  // Z position integral
float Kd_altitude_pos = 3.0f;    // Z position derivative

// Inner loop: Z velocity control
float Kp_altitude_vel = 55.0f;    // Z velocity proportional
float Ki_altitude_vel = 7.0f;    // Z velocity integral
float Kd_altitude_vel = 1.0f;    // Z velocity derivative
float altitude_vel_integral_limit = 32.767f; // PWM-like limit

// Control limits
float max_z_velocity = 2.0f;     // Max Z velocity (m/s)
float max_x_velocity = 2.0f;     // Max X velocity (m/s)

// Controller timestep
float dt = BLIMP_UPDATE_DT;

/**
 * @brief Resets the PID controller's internal states (integrals and derivatives).
 */
void controllerPidReset() {
    // Reset speed controller
    speed_integral = 0.0f;
    speed_prev_error = 0.0f;
    
    // Reset yaw controllers (both angle and rate)
    yaw_angle_integral = 0.0f;
    yaw_angle_prev_error = 0.0f;
    yaw_rate_integral = 0.0f;
    yaw_rate_prev_error = 0.0f;
    yaw_rate_desired = 0.0f;
    
    // Reset altitude controllers (both position and velocity)
    altitude_pos_integral = 0.0f;
    altitude_pos_prev_error = 0.0f;
    altitude_vel_integral = 0.0f;
    altitude_vel_prev_error = 0.0f;
    altitude_vel_desired = 0.0f;
    
}

/**
 * @brief Main Blimp PID Controller - THREE MOTION PRIMITIVES (Crazyflie Style)
 * 
 * This controller implements three independent cascaded PID controllers:
 * 1. SPEED CONTROLLER: Controls forward speed (body-frame u velocity)
 * 2. YAW CONTROLLER: Controls orientation (angle -> rate cascade)  
 * 3. ALTITUDE CONTROLLER: Controls height (position -> velocity cascade)
 * 
 * @param control  Output: Final control signals (thrust, pitch, yaw moments)
 * @param setpoint Input: Desired speed, yaw, and altitude
 * @param sensors  Input: Raw sensor data (gyro for rate feedback)
 * @param state    Input: Current estimated state (position, attitude, rates)
 */
void controllerPid(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors, 
                   const state_t *state, float accumulatedTime) {
    
    
    // Safety check
    if (accumulatedTime <= 0) {
        return;
    }

    // Debug logging - write CSV header only once, then data conditionally
    #if DEBUG_LOGGING
    static int header_written = 0;
    if (!header_written) {
        // Use "w" mode to overwrite file on each simulation run
        FILE *debug_fp = fopen(utils_GetPluginLogPath("pid_tuning_debug.csv"), "w");
        if (debug_fp) {
            fprintf(debug_fp, "time,controller,desired,current,error,output,integral,derivative\n");
            fclose(debug_fp);
        }
        header_written = 1;
    }
    #endif
    // ========================================================================
    // 1. SPEED CONTROLLER - Controls forward/backward movement (50Hz)
    // ========================================================================
    
    if (RATE_DO_EXECUTE(BLIMP_POSITION_RATE, last_position_time, accumulatedTime)) {
        
        float desired_speed_x = setpoint->velocity.v[0]; // Forward speed setpoint
        
        // Calculate body-frame forward velocity
        float cos_yaw = cosf(state->attitude.v[2] * 3.14159f / 180.0f);
        float sin_yaw = sinf(state->attitude.v[2] * 3.14159f / 180.0f);
        
        float current_speed = state->velocity.v[0] * cos_yaw + state->velocity.v[1] * sin_yaw;
        
        // Speed error: desired - current
        float speed_error = desired_speed_x - current_speed;
        
        // Speed PID calculation
        speed_integral += speed_error * (accumulatedTime - last_position_time);
        speed_integral = constrain(speed_integral, -speed_integral_limit, speed_integral_limit);
        
        float speed_derivative = (speed_error - speed_prev_error) / (accumulatedTime - last_position_time);
        speed_prev_error = speed_error;
        
        // Speed controller output (scaled for PWM-like values)
        float speed_output = Kp_speed * speed_error + Ki_speed * speed_integral + Kd_speed * speed_derivative;
        
        // Clamp to positive values and scale
        if (speed_output < 0.0f) speed_output = 0.0f;
        speed_output = constrain(speed_output, 0.0f, 32.767f);
        
        control->pitch = speed_output * 1000.0f; // Scale for internal use
        
        // Log speed controller data
        DEBUG_LOG("pid_tuning_debug.csv", "%.3f,speed,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", 
                    accumulatedTime, desired_speed_x, current_speed, speed_error, 
                    speed_output, speed_integral, speed_derivative);

    }

    // ========================================================================
    // 2. YAW CONTROLLER - Controls turning left/right (100Hz)
    // ========================================================================
    
    if (RATE_DO_EXECUTE(BLIMP_ATTITUDE_RATE, last_attitude_time, accumulatedTime)) {
        
        // Convert setpoint from radians to degrees for consistency
        float desired_yaw_deg = setpoint->attitude.v[2];
        float current_yaw_deg = state->attitude.v[2];
        
        // OUTER LOOP: Yaw angle control (angle -> rate)
        float yaw_angle_error = desired_yaw_deg - current_yaw_deg;
        
        // Normalize yaw error to [-180, 180]
        while (yaw_angle_error > 180.0f) yaw_angle_error -= 360.0f;
        while (yaw_angle_error < -180.0f) yaw_angle_error += 360.0f;
        
        // Yaw angle PID calculation
        yaw_angle_integral += yaw_angle_error * (accumulatedTime - last_attitude_time);
        yaw_angle_integral = constrain(yaw_angle_integral, -yaw_angle_integral_limit, yaw_angle_integral_limit);
        
        float yaw_angle_derivative = (yaw_angle_error - yaw_angle_prev_error) / (accumulatedTime - last_attitude_time);
        yaw_angle_prev_error = yaw_angle_error;
        
        // Desired yaw rate from angle controller
        yaw_rate_desired = Kp_yaw_angle * yaw_angle_error + Ki_yaw_angle * yaw_angle_integral + Kd_yaw_angle * yaw_angle_derivative;
        // Log yaw angle controller data
        DEBUG_LOG("pid_tuning_debug.csv", "%.3f,yaw_angle,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", 
                    accumulatedTime, desired_yaw_deg, current_yaw_deg, yaw_angle_error, 
                    yaw_rate_desired, yaw_angle_integral, yaw_angle_derivative);
        // INNER LOOP: Yaw rate control (rate -> torque)
        float current_yaw_rate = sensors->gyro.v[2] * 180.0f / 3.14159f; // Convert to deg/s
        float yaw_rate_error = yaw_rate_desired - current_yaw_rate;
        
        // Yaw rate PID calculation
        yaw_rate_integral += yaw_rate_error * (accumulatedTime - last_attitude_time);
        yaw_rate_integral = constrain(yaw_rate_integral, -yaw_rate_integral_limit, yaw_rate_integral_limit);
        
        float yaw_rate_derivative = (yaw_rate_error - yaw_rate_prev_error) / (accumulatedTime - last_attitude_time);
        yaw_rate_prev_error = yaw_rate_error;
        
        // Final yaw output (saturated to int16 range like Crazyflie)
        float yaw_output = Kp_yaw_rate * yaw_rate_error + Ki_yaw_rate * yaw_rate_integral + Kd_yaw_rate * yaw_rate_derivative;
        yaw_output = constrain(yaw_output, -32767.0f, 32767.0f);
        
        control->yaw = (int16_t)yaw_output;
        
        DEBUG_LOG("pid_tuning_debug.csv", "%.3f,yaw_rate,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", 
                    accumulatedTime, yaw_rate_desired, current_yaw_rate, yaw_rate_error, 
                    (float)control->yaw, yaw_rate_integral, yaw_rate_derivative);

        RATE_UPDATE_TIME(last_attitude_time, accumulatedTime);
    }

    // ========================================================================
    // 3. ALTITUDE CONTROLLER - Controls up/down movement (50Hz)
    // ========================================================================

    if (RATE_DO_EXECUTE(BLIMP_POSITION_RATE, last_position_time, accumulatedTime)) {
        float desired_altitude = setpoint->position.v[2]; // Z setpoint
        float current_altitude = state->position.v[2];
        
        // OUTER LOOP: Altitude position control (position -> velocity)
        float altitude_pos_error = desired_altitude - current_altitude;
        
        // Altitude position PID calculation
        altitude_pos_integral += altitude_pos_error * (accumulatedTime - last_position_time);
        // altitude_pos_integral = constrain(altitude_pos_integral, -altitude_pos_integral_limit, altitude_pos_integral_limit);
        
        float altitude_pos_derivative = (altitude_pos_error - altitude_pos_prev_error) / (accumulatedTime - last_position_time);
        altitude_pos_prev_error = altitude_pos_error;
        
        // Desired altitude velocity from position controller
        altitude_vel_desired = Kp_altitude_pos * altitude_pos_error + Ki_altitude_pos * altitude_pos_integral + Kd_altitude_pos * altitude_pos_derivative;
        altitude_vel_desired = constrain(altitude_vel_desired, -max_z_velocity * 1.1f, max_z_velocity * 1.1f);
        // Log altitude position controller data
        DEBUG_LOG("pid_tuning_debug.csv", "%.3f,altitude_pos,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", 
                    accumulatedTime, desired_altitude, current_altitude, altitude_pos_error, 
                    altitude_vel_desired, altitude_pos_integral, altitude_pos_derivative);
        // INNER LOOP: Altitude velocity control (velocity -> thrust)
        // altitude_vel_desired = setpoint->velocity.v[2]; // Use setpoint velocity directly for altitude control
        float current_altitude_vel = state->velocity.v[2];
        float altitude_vel_error = altitude_vel_desired - current_altitude_vel;
        
        // Altitude velocity PID calculation
        altitude_vel_integral += altitude_vel_error * (accumulatedTime - last_position_time);
        altitude_vel_integral = constrain(altitude_vel_integral, -altitude_vel_integral_limit, altitude_vel_integral_limit);
        
        float altitude_vel_derivative = (altitude_vel_error - altitude_vel_prev_error) / (accumulatedTime - last_position_time);
        altitude_vel_prev_error = altitude_vel_error;
        
        // Final thrust output (constrained to positive, PWM-like scaling)
        float thrust_raw = Kp_altitude_vel * altitude_vel_error + Ki_altitude_vel * altitude_vel_integral + Kd_altitude_vel * altitude_vel_derivative;
        thrust_raw = constrain(thrust_raw, 0.0f, 65535.0f);
        
        control->thrust = thrust_raw * 1000.0f; // Scale for internal use
        
        // Log altitude velocity controller data
        DEBUG_LOG("pid_tuning_debug.csv", "%.3f,altitude_vel,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", 
                    accumulatedTime, altitude_vel_desired, current_altitude_vel, altitude_vel_error, 
                    thrust_raw, altitude_vel_integral, altitude_vel_derivative);

        RATE_UPDATE_TIME(last_position_time, accumulatedTime);
    }

    // ========================================================================
    // DISABLE UNUSED CONTROLS & SAFETY
    // ========================================================================
    
    control->roll = 0.0f; // Blimps can't roll
    
    // Reset controller if thrust is zero and no setpoints active
    if (control->thrust == 0 && setpoint->position.v[2] == 0 && setpoint->velocity.v[0] == 0) {
        controllerPidReset();
    }
}