/****************************************************************************

 blimp_history_gradient.cpp -- History-Based Gradient Following with Explore/Seek States

 This controller combines the EXPLORING/SEEKING state machine from the map-based
 approach with the simple history-based gradient estimation. Instead of building
 a full map, it uses only recent measurement history to estimate gradients.

 COMPARISON PURPOSES:
 - Same state machine as map-based version
 - Same exploring pattern and seeking behavior  
 - Only difference: gradient calculation method (history vs. map)

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
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>

#define TIMESTEP 32  // ms
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// === CONFIGURATION PARAMETERS ===
const double ALTITUDE_SETPOINT = 2.0;
const double SEEKING_SPEED = 0.3;
const double EXPLORING_SPEED = 0.15;
const double EXPLORING_RADIUS = 1.0;
const double GRADIENT_THRESHOLD = 2;  // Minimum gradient magnitude to trigger seeking
const double GRID_RESOLUTION = 0.2;      // Grid cell size in meters
const double MIN_LIGHT_IMPROVEMENT = 0.1; // Minimum improvement to update map

// === TARGET LIGHT SOURCE LOCATION ===
const double TARGET_X = 5.1988; // meters
const double TARGET_Y = 5.329;  // meters
const double PROXIMITY_THRESHOLD = 0.5; // meters - stop when within this distance

// === STATE MACHINE ===
enum State { EXPLORING, SEEKING };

// === MEASUREMENT MAP STRUCTURE ===
struct MeasurementPoint {
    double x, y;
    double light_intensity;
    double timestamp;
    double yaw;
};

// === SPATIAL MAP FOR MEASUREMENTS ===
#include <map>
#include <utility>

struct GridCell {
    int grid_x, grid_y;
    bool operator<(const GridCell& other) const {
        if (grid_x != other.grid_x) return grid_x < other.grid_x;
        return grid_y < other.grid_y;
    }
};

// Map to store best measurement at each grid location
std::map<GridCell, MeasurementPoint> measurement_map;

// === GRID COORDINATE CONVERSION ===
GridCell position_to_grid(double x, double y) {
    GridCell cell;
    cell.grid_x = (int)floor(x / GRID_RESOLUTION);
    cell.grid_y = (int)floor(y / GRID_RESOLUTION);
    return cell;
}

// === MAP-BASED MEASUREMENT UPDATE ===
bool update_measurement_map(double x, double y, double light_intensity, double timestamp, double yaw) {
    GridCell cell = position_to_grid(x, y);
    
    // Check if this cell already has a measurement
    auto it = measurement_map.find(cell);
    
    if (it == measurement_map.end()) {
        // No previous measurement at this location - add it
        MeasurementPoint new_point;
        new_point.x = x;
        new_point.y = y;
        new_point.light_intensity = light_intensity;
        new_point.timestamp = timestamp;
        new_point.yaw = yaw;
        measurement_map[cell] = new_point;
        printf("NEW MAP ENTRY: Grid(%d,%d) at (%.3f,%.3f) with light=%.3f\n", 
               cell.grid_x, cell.grid_y, x, y, light_intensity);
        return true;
    } else {
        // Check if new measurement is better (higher light intensity)
        if (light_intensity > it->second.light_intensity + MIN_LIGHT_IMPROVEMENT) {
            // Update with better measurement
            MeasurementPoint updated_point;
            updated_point.x = x;
            updated_point.y = y;
            updated_point.light_intensity = light_intensity;
            updated_point.timestamp = timestamp;
            updated_point.yaw = yaw;
            
            printf("MAP UPDATE: Grid(%d,%d) improved from %.3f to %.3f (Δ=%.3f)\n", 
                   cell.grid_x, cell.grid_y, it->second.light_intensity, light_intensity,
                   light_intensity - it->second.light_intensity);
            
            measurement_map[cell] = updated_point;
            return true;
        } else {
            // No improvement - don't update
            return false;
        }
    }
}

// === MAP-BASED GRADIENT ESTIMATION ===
bool estimate_gradient_from_map(double current_x, double current_y, 
                               double& grad_x, double& grad_y, double& magnitude) {
    grad_x = 0.0;
    grad_y = 0.0;
    magnitude = 0.0;
    
    // Need at least 3 points for plane fitting
    if (measurement_map.size() < 3) {
        printf("Not enough map points for gradient estimation: %zu\n", measurement_map.size());
        return false;
    }
    
    const double MAX_DISTANCE = 3.0;   // Maximum distance from current position
    const double MIN_DISTANCE = 0.1;   // Minimum distance to avoid numerical issues
    
    // Collect valid neighboring points for plane fitting
    std::vector<MeasurementPoint> valid_points;
    
    for (const auto& entry : measurement_map) {
        const MeasurementPoint& point = entry.second;
        
        double dx = current_x - point.x;
        double dy = current_y - point.y;
        double distance = sqrt(dx*dx + dy*dy);
        
        // Include points within reasonable distance
        if (distance >= MIN_DISTANCE && distance <= MAX_DISTANCE) {
            valid_points.push_back(point);
        }
    }
    
    // Need at least 3 points for plane fitting
    if (valid_points.size() < 3) {
        printf("Not enough nearby points for plane fitting: %zu (total map: %zu)\n", 
               valid_points.size(), measurement_map.size());
        return false;
    }
    
    // === LEAST SQUARES PLANE FITTING ===
    // We fit the model: z = ax + by + c
    // where z is light intensity, (x,y) is position
    // The gradient is then (∂z/∂x, ∂z/∂y) = (a, b)
    
    int n = valid_points.size();
    double sum_x = 0, sum_y = 0, sum_z = 0;
    double sum_xx = 0, sum_yy = 0, sum_xy = 0;
    double sum_xz = 0, sum_yz = 0;
    
    // Calculate sums for least squares
    for (const auto& p : valid_points) {
        sum_x += p.x;
        sum_y += p.y;
        sum_z += p.light_intensity;
        sum_xx += p.x * p.x;
        sum_yy += p.y * p.y;
        sum_xy += p.x * p.y;
        sum_xz += p.x * p.light_intensity;
        sum_yz += p.y * p.light_intensity;
    }
    
    // Build normal equations matrix (A^T * A) * [a, b, c]^T = A^T * z
    double A[3][3] = {
        {sum_xx, sum_xy, sum_x},
        {sum_xy, sum_yy, sum_y},
        {sum_x,  sum_y,  (double)n}
    };
    
    double B[3] = {sum_xz, sum_yz, sum_z};
    
    // Solve using Cramer's rule (for 3x3 system)
    // Calculate determinant of A
    double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                 A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                 A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    
    if (fabs(det) < 1e-6) {
        printf("Singular matrix in plane fitting\n");
        return false;
    }
    
    // Calculate gradient components (a and b from ax + by + c)
    double det_a = B[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                   A[0][1] * (B[1] * A[2][2] - A[1][2] * B[2]) +
                   A[0][2] * (B[1] * A[2][1] - A[1][1] * B[2]);
    
    double det_b = A[0][0] * (B[1] * A[2][2] - A[1][2] * B[2]) -
                   B[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                   A[0][2] * (A[1][0] * B[2] - B[1] * A[2][0]);
    
    grad_x = det_a / det;  // ∂z/∂x
    grad_y = det_b / det;  // ∂z/∂y
    magnitude = sqrt(grad_x * grad_x + grad_y * grad_y);
    
    // === CALCULATE FIT QUALITY (R-squared) ===
    double mean_z = sum_z / n;
    double ss_tot = 0, ss_res = 0;
    
    for (const auto& p : valid_points) {
        double predicted = grad_x * p.x + grad_y * p.y + (sum_z - grad_x * sum_x - grad_y * sum_y) / n;
        ss_res += pow(p.light_intensity - predicted, 2);
        ss_tot += pow(p.light_intensity - mean_z, 2);
    }
    
    double r_squared = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;
    
    printf("Map gradient: points=%d, grad=(%.4f,%.4f), mag=%.4f, R²=%.3f, map_size=%zu\n", 
           n, grad_x, grad_y, magnitude, r_squared, measurement_map.size());
    
    // Only trust the gradient if fit quality is reasonable
    if (r_squared < 0.2) {  // Lower threshold for map-based approach
        printf("Poor plane fit quality (R²=%.3f), rejecting gradient\n", r_squared);
        return false;
    }
    
    return true;
}

// === CONVERT GRADIENT TO ANGLE ===
double gradient_to_angle(double grad_x, double grad_y) {
    return atan2(grad_y, grad_x) * 180.0 / M_PI;
}

// === WRITE MAP DATA TO FILE ===
void write_map_data(const std::string& filename) {
    FILE *map_file = fopen(filename.c_str(), "w");
    if (!map_file) {
        printf("Failed to create map data file: %s\n", filename.c_str());
        return;
    }
    
    // Write CSV header
    fprintf(map_file, "grid_x,grid_y,x,y,light_intensity,timestamp,yaw\n");
    
    // Write all map entries
    for (const auto& entry : measurement_map) {
        const GridCell& cell = entry.first;
        const MeasurementPoint& point = entry.second;
        
        fprintf(map_file, "%d,%d,%.6f,%.6f,%.3f,%.3f,%.6f\n",
                cell.grid_x, cell.grid_y,
                point.x, point.y, point.light_intensity,
                point.timestamp, point.yaw);
    }
    
    fclose(map_file);
    printf("Map data saved to '%s' with %zu locations\n", filename.c_str(), measurement_map.size());
}

// === WRITE GRADIENT FIELD DATA ===
void write_gradient_field(const std::string& filename, double min_x, double max_x, double min_y, double max_y, double resolution = 0.5) {
    FILE *grad_file = fopen(filename.c_str(), "w");
    if (!grad_file) {
        printf("Failed to create gradient field file: %s\n", filename.c_str());
        return;
    }
    
    // Write CSV header
    fprintf(grad_file, "x,y,grad_x,grad_y,magnitude,valid\n");
    
    int valid_count = 0;
    int total_count = 0;
    
    // Sample gradient field at regular intervals
    for (double y = min_y; y <= max_y; y += resolution) {
        for (double x = min_x; x <= max_x; x += resolution) {
            double grad_x, grad_y, magnitude;
            bool valid = estimate_gradient_from_map(x, y, grad_x, grad_y, magnitude);
            
            fprintf(grad_file, "%.3f,%.3f,%.6f,%.6f,%.6f,%d\n",
                    x, y, grad_x, grad_y, magnitude, valid ? 1 : 0);
            
            if (valid) valid_count++;
            total_count++;
        }
    }
    
    fclose(grad_file);
    printf("Gradient field saved to '%s' (%d/%d valid points)\n", filename.c_str(), valid_count, total_count);
}

// Main loop
int main() {
    printf("=== HISTORY-BASED GRADIENT FOLLOWING WITH EXPLORE/SEEK STATES ===\n");
    printf("Commands: S=START STEERING, Q=QUIT\n");
    printf("================================================================\n\n");

    wb_robot_init();

    // Sensor/emitter setup
    WbDeviceTag imu = wb_robot_get_device("imu"); wb_inertial_unit_enable(imu, TIMESTEP);
    WbDeviceTag gyro = wb_robot_get_device("gyro"); wb_gyro_enable(gyro, TIMESTEP);
    WbDeviceTag gps = wb_robot_get_device("gps"); wb_gps_enable(gps, TIMESTEP);
    WbDeviceTag light_sensor = wb_robot_get_device("light_sensor"); wb_light_sensor_enable(light_sensor, TIMESTEP);
    WbDeviceTag gEmitter = wb_robot_get_device("emitter");
    wb_keyboard_enable(TIMESTEP);

    // --- Log File Setup ---
    FILE *trajectory_log = fopen("logs/history_gradient_trajectory.csv", "w");
    if (!trajectory_log) {
        // Try current directory if logs directory doesn't exist
        trajectory_log = fopen("history_gradient_trajectory.csv", "w");
    }
    if (trajectory_log) {
        fprintf(trajectory_log, "time,x,y,z,yaw,setpoint_x,setpoint_y,setpoint_z,setpoint_yaw,setpoint_speed,current_speed,mode,distance_to_target,light_intensity,gradient_angle,gradient_magnitude\n");
    }

    // --- Algorithm State Initialization ---
    double controls[12] = {0}; 
    double past_time = 0.0; 
    bool search_active = false;
    State current_state = EXPLORING;
    measurement_map.clear(); // Initialize the measurement map
    double est_x = 0.0, est_y = 0.0; 
    double setpoint_yaw = 0.0;

    // Wait for first valid readings to initialize state
    while (wb_robot_step(TIMESTEP) != -1) {
        if (wb_robot_get_time() > 0.0) {
            past_time = wb_robot_get_time();
            const double *rpy = wb_inertial_unit_get_roll_pitch_yaw(imu);
            setpoint_yaw = rpy[2] * 180.0 / M_PI;
            const double *initial_pos = wb_gps_get_values(gps);
            est_x = initial_pos[0]; 
            est_y = initial_pos[1];
            break;
        }
    }

    printf("Blimp initialized. State: EXPLORING. Ready for command.\n");

    while (wb_robot_step(TIMESTEP) != -1) {
        double current_time = wb_robot_get_time();
        const double dt = current_time - past_time;
        past_time = current_time;

        // --- Keyboard Input ---
        int key = wb_keyboard_get_key();
        if (key > 0) {
            if (key == 'S' && !search_active) {
                search_active = true;
                measurement_map.clear();
                printf(">>> 'S' key pressed. Starting autonomous steering with map-based gradient...\n");
            } else if (key == 'Q') {
                printf(">>> 'Q' key pressed. Quitting simulation.\n");
                break;
            }
        }

        // --- Sensor Feedback ---
        const double *rpy = wb_inertial_unit_get_roll_pitch_yaw(imu);
        const double *velocity = wb_gps_get_speed_vector(gps);
        const double *position = wb_gps_get_values(gps);
        double current_signal = wb_light_sensor_get_value(light_sensor);
        double current_yaw_rad = rpy[2];
        double current_yaw_deg = current_yaw_rad * 180.0 / M_PI;
        
        // --- Check proximity to target light source ---
        double distance_to_target = sqrt(pow(est_x - TARGET_X, 2) + pow(est_y - TARGET_Y, 2));
        
        if (distance_to_target <= PROXIMITY_THRESHOLD) {
            printf("\n🎯 SUCCESS! Reached light source at (%.3f, %.3f)\n", TARGET_X, TARGET_Y);
            printf("Final position: (%.3f, %.3f, %.3f)\n", est_x, est_y, position[2]);
            printf("Distance to target: %.3f m (threshold: %.3f m)\n", distance_to_target, PROXIMITY_THRESHOLD);
            printf("Total mission time: %.1f seconds\n", current_time);
            printf("Stopping simulation...\n");
            break;
        }
        
        // --- Update Position Estimate ---
        double forward_speed = (current_state == EXPLORING) ? EXPLORING_SPEED : SEEKING_SPEED;
        
        if (search_active) {
            est_x += forward_speed * cos(current_yaw_rad) * dt;
            est_y += forward_speed * sin(current_yaw_rad) * dt;
            
            // --- Update Measurement Map (only if light improved) ---
            bool map_updated = update_measurement_map(est_x, est_y, current_signal, current_time, current_yaw_rad);
            if (!map_updated) {
                // Uncomment for debugging non-updates
                // printf("No map update: light=%.3f not better at (%.3f,%.3f)\n", current_signal, est_x, est_y);
            }
        }

        // --- State Machine Logic ---
        double grad_angle = -1000; // Default invalid value
        double grad_magnitude = 0.0;
        
        if (search_active && measurement_map.size() >= 3) {
            double grad_x, grad_y;
            bool gradient_found = estimate_gradient_from_map(est_x, est_y, grad_x, grad_y, grad_magnitude);
            
            if (gradient_found) {
                grad_angle = gradient_to_angle(grad_x, grad_y);
            }
            
            if (current_state == EXPLORING) {
                // Exploring: Spiral search pattern
                double yaw_rate = (EXPLORING_SPEED / EXPLORING_RADIUS) * (180.0 / M_PI);
                setpoint_yaw += yaw_rate * dt;
                
                // Check if gradient is strong enough to switch to seeking
                if (gradient_found && grad_magnitude > GRADIENT_THRESHOLD) {
                    printf(">>> MAP GRADIENT DETECTED! Switching to SEEKING mode. Magnitude: %.4f\n", grad_magnitude);
                    current_state = SEEKING;
                    setpoint_yaw = grad_angle;  // Immediately turn toward gradient
                }
            } else if (current_state == SEEKING) {
                // Seeking: Follow gradient direction
                if (gradient_found && grad_magnitude > GRADIENT_THRESHOLD) {
                    setpoint_yaw = grad_angle;
                    printf(">>> FOLLOWING MAP GRADIENT: angle=%.1f°, magnitude=%.4f\n", grad_angle, grad_magnitude);
                } else {
                    printf(">>> MAP GRADIENT LOST! Reverting to EXPLORING mode.\n");
                    current_state = EXPLORING;
                }
            }
        } else if (search_active) {
            // Not enough map data yet - continue exploring
            if (current_state == EXPLORING) {
                double yaw_rate = (EXPLORING_SPEED / EXPLORING_RADIUS) * (180.0 / M_PI);
                setpoint_yaw += yaw_rate * dt;
            }
        } else {
            // Search not active - hover in place
            setpoint_yaw = current_yaw_deg;
            forward_speed = 0;
        }
        
        // Normalize yaw setpoint
        while (setpoint_yaw > 180.0) setpoint_yaw -= 360.0;
        while (setpoint_yaw < -180.0) setpoint_yaw += 360.0;

        // --- Send Commands to Low-Level Controller ---
        if (gEmitter) {
            controls[0] = ALTITUDE_SETPOINT; 
            controls[1] = setpoint_yaw;
            controls[2] = forward_speed; 
            controls[3] = 0.0; 
            controls[4] = 0.0;
            controls[5] = velocity[0]; 
            controls[6] = velocity[1]; 
            controls[7] = velocity[2];
            controls[8] = position[2]; 
            controls[9] = current_yaw_deg;
            controls[10] = wb_gyro_get_values(gyro)[2] * 180.0 / M_PI; 
            controls[11] = dt;
            wb_emitter_send(gEmitter, controls, sizeof(controls));
        }
        
        // --- Comprehensive Trajectory Logging ---
        if (trajectory_log) {
            double current_speed = sqrt(velocity[0]*velocity[0] + velocity[1]*velocity[1]); // Horizontal speed
            const char* mode;
            if (!search_active) {
                mode = "FORWARD";
            } else if (current_state == EXPLORING) {
                mode = "EXPLORING";
            } else if (current_state == SEEKING) {
                mode = "SEEKING";
            } else {
                mode = "UNKNOWN";
            }
            
            // For this gradient-based controller, we don't have explicit X,Y setpoints
            // But we can log the current position and derived setpoints
            double setpoint_x = est_x; // No explicit X setpoint in this controller
            double setpoint_y = est_y; // No explicit Y setpoint in this controller
            double setpoint_z = ALTITUDE_SETPOINT;
            double setpoint_yaw_rad = setpoint_yaw * M_PI / 180.0;
            
            fprintf(trajectory_log, "%.3f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%s,%.3f,%.1f,%.1f,%.4f\n",
                    current_time,
                    est_x, est_y, position[2], // Current x,y,z
                    current_yaw_rad, // Current yaw in radians
                    setpoint_x, setpoint_y, setpoint_z, // Setpoint x,y,z
                    setpoint_yaw_rad, // Setpoint yaw in radians
                    forward_speed, // Current setpoint speed (varies by state)
                    current_speed, // Current actual speed
                    mode, // Controller mode (FORWARD/EXPLORING/SEEKING)
                    distance_to_target, // Distance to light source target
                    current_signal, // Light sensor reading
                    grad_angle, // Gradient direction angle
                    grad_magnitude); // Gradient magnitude
        }

        // --- Status Printing ---
        static double last_print_time = 0;
        if (current_time - last_print_time > 1.0) {
            const char* state_name = (current_state == EXPLORING) ? "EXPLORING" : "SEEKING";
            printf("[T:%.1fs] State:%s | Light:%.1f | Dist2Target:%.2fm | MapSize:%zu | GradMag:%.4f | Yaw:%.1f°\n",
                   current_time, state_name, current_signal, distance_to_target, measurement_map.size(), 
                   grad_magnitude, setpoint_yaw);
            last_print_time = current_time;
        }
    }

    // --- Cleanup ---
    if (trajectory_log) fclose(trajectory_log);
    
    // Write out map data and gradient field for visualization
    if (measurement_map.size() > 0) {
        write_map_data("logs/measurement_map.csv");
        write_map_data("measurement_map.csv"); // Fallback to current directory
        
        // Calculate bounds for gradient field based on trajectory
        double min_x = est_x - 5.0, max_x = est_x + 5.0;
        double min_y = est_y - 5.0, max_y = est_y + 5.0;
        
        // Adjust bounds to include target and map points
        min_x = std::min(min_x, TARGET_X - 2.0);
        max_x = std::max(max_x, TARGET_X + 2.0);
        min_y = std::min(min_y, TARGET_Y - 2.0);
        max_y = std::max(max_y, TARGET_Y + 2.0);
        
        for (const auto& entry : measurement_map) {
            const MeasurementPoint& point = entry.second;
            min_x = std::min(min_x, point.x - 1.0);
            max_x = std::max(max_x, point.x + 1.0);
            min_y = std::min(min_y, point.y - 1.0);
            max_y = std::max(max_y, point.y + 1.0);
        }
        
        write_gradient_field("logs/gradient_field.csv", min_x, max_x, min_y, max_y);
        write_gradient_field("gradient_field.csv", min_x, max_x, min_y, max_y); // Fallback
    }
    
    printf("Map-based gradient following completed. Final map size: %zu locations\n", measurement_map.size());
    printf("Trajectory log saved to 'history_gradient_trajectory.csv'\n");

    wb_robot_cleanup();
    return 0;
}