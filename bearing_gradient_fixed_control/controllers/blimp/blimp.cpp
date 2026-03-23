/****************************************************************************

 blimp_combined_fixed.cpp -- Fixed-Weight Bearing + Gradient Fusion

 Fuses two light-seeking strategies with FIXED weights:
   - Bearing  (16-sensor array)  : 70% weight  -- fast, direct
   - Gradient (spatial map)      : 30% weight  -- spatially aware

 GRADIENT ESTIMATION: Weighted Least Squares (WLS)
   Each neighbour point is weighted by 1/(d² + epsilon), giving closer
   measurements more influence on the fitted plane.  This replaces the
   original unweighted Cramer's-rule fit and improves gradient direction
   accuracy when map coverage is sparse or unevenly distributed along the
   flight path.  The R² quality check uses the weighted form for consistency.

 The gradient weight is zeroed out until the map has enough points
 (MIN_MAP_POINTS) and the gradient magnitude exceeds GRADIENT_THRESHOLD,
 at which point the fixed 70/30 split kicks in permanently.

 Both methods produce a desired heading angle. The fused heading is computed
 as a weighted circular mean to correctly handle angle wrapping.

 MAP BUILDING:
   The spatial measurement map is built passively from the blimp's path.
   No exploration spiral is used -- the blimp always moves forward along
   the fused heading, and map data accumulates naturally.

 FALLBACK:
   If the map is not yet ready (too few points or gradient too weak),
   the controller falls back to pure bearing (weight = 100%).

 COMMANDS:  S = start    Q = quit

******************************************************************************/

#include <webots/emitter.h>
#include <webots/gps.h>
#include <webots/gyro.h>
#include <webots/inertial_unit.h>
#include <webots/keyboard.h>
#include <webots/light_sensor.h>
#include <webots/robot.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

#define TIMESTEP 32
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Sensor array
// ---------------------------------------------------------------------------
static const int NUM_SENSORS = 16;
static const double SENSOR_ANGLES[NUM_SENSORS] = {
      0.0,  22.5,  45.0,  67.5,
     90.0, 112.5, 135.0, 157.5,
    180.0, 202.5, 225.0, 247.5,
    270.0, 292.5, 315.0, 337.5
};

// ---------------------------------------------------------------------------
// Navigation parameters
// ---------------------------------------------------------------------------
static const double ALTITUDE_SETPOINT     = 2.0;   // m
static const double FORWARD_SPEED         = 0.2;   // m/s (constant)
static const double BEARING_SMOOTH_FACTOR = 0.3;   // low-pass on bearing angle
static const double MIN_SENSOR_INTENSITY  = 0.1;   // per-sensor threshold
static const double MIN_TOTAL_LIGHT       = 1.0;   // aggregate threshold

// ---------------------------------------------------------------------------
// Fusion weights (fixed)
// ---------------------------------------------------------------------------
static const double W_BEARING  = 0.5;
static const double W_GRADIENT = 0.5;  // only active once gradient is ready

// ---------------------------------------------------------------------------
// Gradient readiness thresholds
// ---------------------------------------------------------------------------
static const int    MIN_MAP_POINTS        = 3;     // map cells needed before gradient is used
static const double GRADIENT_THRESHOLD    = 0.5;   // minimum gradient magnitude to trust

// ---------------------------------------------------------------------------
// Spatial map parameters
// ---------------------------------------------------------------------------
static const double GRID_RESOLUTION       = 0.2;   // m per cell
static const double MIN_LIGHT_IMPROVEMENT = 0.1;   // hysteresis for map updates
static const double EXPLORE_AMPLITUDE     = 0.0;   // ±0.3m lateral oscillation
static const double EXPLORE_FREQUENCY     = 0.0;   // rad/s

// ---------------------------------------------------------------------------
// Target
// ---------------------------------------------------------------------------
static const double TARGET_X            = 5.1988;
static const double TARGET_Y            = 5.329;
static const double PROXIMITY_THRESHOLD = 2.0;     // m

// ---------------------------------------------------------------------------
// Spatial map types
// ---------------------------------------------------------------------------
struct MeasurementPoint {
    double x, y;
    double light_intensity;
    double timestamp;
    double yaw;
};

struct GridCell {
    int gx, gy;
    bool operator<(const GridCell& o) const {
        return (gx != o.gx) ? gx < o.gx : gy < o.gy;
    }
};

static std::map<GridCell, MeasurementPoint> measurement_map;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static double normalize_angle(double a) {
    while (a >  180.0) a -= 360.0;
    while (a < -180.0) a += 360.0;
    return a;
}

static GridCell pos_to_grid(double x, double y) {
    GridCell c;
    c.gx = (int)floor(x / GRID_RESOLUTION);
    c.gy = (int)floor(y / GRID_RESOLUTION);
    return c;
}

// ---------------------------------------------------------------------------
// Map update -- keeps the best (highest) reading per grid cell
// ---------------------------------------------------------------------------
static void update_map(double x, double y, double intensity,
                        double timestamp, double yaw) {
    GridCell cell = pos_to_grid(x, y);
    auto it = measurement_map.find(cell);
    if (it == measurement_map.end()) {
        MeasurementPoint p = {x, y, intensity, timestamp, yaw};
        measurement_map[cell] = p;
    } else if (intensity > it->second.light_intensity + MIN_LIGHT_IMPROVEMENT) {
        MeasurementPoint p = {x, y, intensity, timestamp, yaw};
        measurement_map[cell] = p;
    }
}

// ---------------------------------------------------------------------------
// Weighted least-squares plane fit over nearby map points → gradient (∂z/∂x, ∂z/∂y)
//
// Each neighbour point is weighted by  w_i = 1 / (d_i² + epsilon)  where d_i
// is its distance from the query position (cx, cy).  This gives closer points
// more influence on the fitted plane, improving accuracy when the measurement
// distribution is uneven along the flight path.
//
// The weighted normal equations are:
//   (Aᵀ W A) [a, b, c]ᵀ = Aᵀ W z
// where A = [x  y  1] for each point and W = diag(w_i).
// Expanded into scalar sums this avoids any matrix library dependency.
//
// R² is computed in weighted form so the quality threshold remains meaningful:
//   R² = 1 - Σ w_i(z_i - ẑ_i)² / Σ w_i(z_i - z̄_w)²
//
// Returns false if insufficient data or weighted R² < GRADIENT_THRESHOLD.
// ---------------------------------------------------------------------------
static bool estimate_gradient(double cx, double cy,
                               double& gx_out, double& gy_out,
                               double& magnitude) {
    gx_out = gy_out = magnitude = 0.0;
    if ((int)measurement_map.size() < MIN_MAP_POINTS) return false;

    const double MAX_DIST = 3.0;
    const double MIN_DIST = 0.1;
    const double W_EPS    = 0.01;  // epsilon in weight denominator, avoids 1/0

    // Collect neighbours and compute inverse-distance-squared weights
    std::vector<MeasurementPoint> pts;
    std::vector<double> weights;
    double weight_sum = 0.0;

    for (const auto& entry : measurement_map) {
        const MeasurementPoint& p = entry.second;
        double dx = cx - p.x, dy = cy - p.y;
        double d  = sqrt(dx*dx + dy*dy);
        if (d >= MIN_DIST && d <= MAX_DIST) {
            double w = 1.0 / (d*d + W_EPS);
            pts.push_back(p);
            weights.push_back(w);
            weight_sum += w;
        }
    }
    if ((int)pts.size() < MIN_MAP_POINTS) return false;

    // Normalise weights so they sum to 1
    int n = (int)pts.size();
    for (int i = 0; i < n; i++) weights[i] /= weight_sum;

    // Build weighted normal equations:  (Aᵀ W A) coeff = Aᵀ W z
    // Expanded scalars (identical layout to the original unweighted version,
    // but each term is multiplied by the corresponding weight w_i).
    double sw=0, swx=0, swy=0, swz=0;
    double swxx=0, swyy=0, swxy=0, swxz=0, swyz=0;
    for (int i = 0; i < n; i++) {
        double w = weights[i];
        double x = pts[i].x, y = pts[i].y, z = pts[i].light_intensity;
        sw   += w;
        swx  += w*x;    swy  += w*y;    swz  += w*z;
        swxx += w*x*x;  swyy += w*y*y;  swxy += w*x*y;
        swxz += w*x*z;  swyz += w*y*z;
    }

    // 3×3 weighted normal matrix  M = Aᵀ W A
    double M[3][3] = {
        {swxx, swxy, swx},
        {swxy, swyy, swy},
        {swx,  swy,  sw }
    };
    double R[3] = {swxz, swyz, swz};  // Aᵀ W z

    // Solve via Cramer's rule (no external library required)
    double det = M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1])
               - M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
               + M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]);
    if (fabs(det) < 1e-6) return false;

    double det_a = R[0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1])
                 - M[0][1]*(R[1]*M[2][2] - M[1][2]*R[2])
                 + M[0][2]*(R[1]*M[2][1] - M[1][1]*R[2]);
    double det_b = M[0][0]*(R[1]*M[2][2] - M[1][2]*R[2])
                 - R[0]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
                 + M[0][2]*(M[1][0]*R[2] - R[1]*M[2][0]);

    gx_out    = det_a / det;
    gy_out    = det_b / det;
    magnitude = sqrt(gx_out*gx_out + gy_out*gy_out);

    // Weighted R² quality check
    double c_coeff  = (swz - gx_out*swx - gy_out*swy) / sw;
    double mean_z_w = swz / sw;   // weighted mean of z
    double ss_res = 0, ss_tot = 0;
    for (int i = 0; i < n; i++) {
        double w    = weights[i];
        double z    = pts[i].light_intensity;
        double pred = gx_out*pts[i].x + gy_out*pts[i].y + c_coeff;
        ss_res += w * (z - pred) * (z - pred);
        ss_tot += w * (z - mean_z_w) * (z - mean_z_w);
    }
    double r2 = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;
    if (r2 < GRADIENT_THRESHOLD) {
        printf("Poor weighted plane fit (R²=%.3f), rejecting gradient\n", r2);
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Bearing from 16-sensor array (weighted vector sum → atan2)
// Returns angle in degrees relative to robot frame, or false if no signal
// ---------------------------------------------------------------------------
static bool calculate_bearing(const double* lv,
                               double& angle_out, double& weight_out) {
    double sx = 0, sy = 0, ws = 0;
    for (int i = 0; i < NUM_SENSORS; i++) {
        if (lv[i] > MIN_SENSOR_INTENSITY) {
            double rad = SENSOR_ANGLES[i] * M_PI / 180.0;
            sx += lv[i] * cos(rad);
            sy += lv[i] * sin(rad);
            ws += lv[i];
        }
    }
    weight_out = ws;
    if (ws <= MIN_SENSOR_INTENSITY) { angle_out = 0.0; return false; }
    angle_out = atan2(sy, sx) * 180.0 / M_PI;
    return true;
}

// ---------------------------------------------------------------------------
// Weighted circular mean of two angles (handles wrap-around correctly)
// w1 + w2 should equal 1.0
// ---------------------------------------------------------------------------
static double weighted_circular_mean(double angle1_deg, double w1,
                                     double angle2_deg, double w2) {
    double r1 = angle1_deg * M_PI / 180.0;
    double r2 = angle2_deg * M_PI / 180.0;
    double cx = w1 * cos(r1) + w2 * cos(r2);
    double cy = w1 * sin(r1) + w2 * sin(r2);
    return atan2(cy, cx) * 180.0 / M_PI;
}

// ---------------------------------------------------------------------------
// Map data export for visualization
// ---------------------------------------------------------------------------
static void write_map_data(const char* filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Failed to create map data file: %s\n", filename);
        return;
    }
    
    fprintf(fp, "grid_x,grid_y,x,y,light_intensity,timestamp,yaw\n");
    
    for (const auto& entry : measurement_map) {
        const GridCell& cell = entry.first;
        const MeasurementPoint& pt = entry.second;
        fprintf(fp, "%d,%d,%.6f,%.6f,%.3f,%.3f,%.6f\n",
                cell.gx, cell.gy,
                pt.x, pt.y, pt.light_intensity,
                pt.timestamp, pt.yaw);
    }
    
    fclose(fp);
    printf("Map data saved to '%s' with %zu locations\n", filename, measurement_map.size());
}

static void write_gradient_field(const char* filename,
                                  double min_x, double max_x,
                                  double min_y, double max_y,
                                  double resolution = 0.5) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Failed to create gradient field file: %s\n", filename);
        return;
    }
    
    fprintf(fp, "x,y,grad_x,grad_y,magnitude,valid\n");
    
    int valid_count = 0, total_count = 0;
    
    for (double y = min_y; y <= max_y; y += resolution) {
        for (double x = min_x; x <= max_x; x += resolution) {
            double gx, gy, mag;
            bool valid = estimate_gradient(x, y, gx, gy, mag);
            fprintf(fp, "%.3f,%.3f,%.6f,%.6f,%.6f,%d\n",
                    x, y, gx, gy, mag, valid ? 1 : 0);
            if (valid) valid_count++;
            total_count++;
        }
    }
    
    fclose(fp);
    printf("Gradient field saved to '%s' (%d/%d valid points)\n",
           filename, valid_count, total_count);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    printf("=== FIXED-WEIGHT FUSION: %.0f%% Bearing + %.0f%% Gradient ===\n",
           W_BEARING * 100.0, W_GRADIENT * 100.0);
    printf("Gradient only activates once map has >=%d points and mag >= %.2f\n",
           MIN_MAP_POINTS, GRADIENT_THRESHOLD);
    printf("Commands: S=START  Q=QUIT\n");
    printf("============================================================\n\n");

    wb_robot_init();

    WbDeviceTag imu  = wb_robot_get_device("imu");
    wb_inertial_unit_enable(imu, TIMESTEP);
    WbDeviceTag gyro = wb_robot_get_device("gyro");
    wb_gyro_enable(gyro, TIMESTEP);
    WbDeviceTag gps  = wb_robot_get_device("gps");
    wb_gps_enable(gps, TIMESTEP);

    WbDeviceTag light_sensors[NUM_SENSORS];
    char sname[32];
    for (int i = 0; i < NUM_SENSORS; i++) {
        sprintf(sname, "light_sensor_%d", i);
        light_sensors[i] = wb_robot_get_device(sname);
        if (light_sensors[i] == 0)
            printf("Warning: %s not found\n", sname);
        else
            wb_light_sensor_enable(light_sensors[i], TIMESTEP);
    }

    WbDeviceTag gEmitter = wb_robot_get_device("emitter");
    wb_keyboard_enable(TIMESTEP);

    // --- State ---
    double controls[12]    = {0};
    double past_time       = 0.0;
    bool   search_active   = false;

    double light_values[NUM_SENSORS] = {0};
    double total_light     = 0.0;
    double bearing_angle   = 0.0;   // relative to robot frame (degrees)
    double smooth_bearing  = 0.0;   // low-pass filtered bearing (relative)
    bool   bearing_valid   = false;

    measurement_map.clear();

    // --- Wait for first valid step ---
    while (wb_robot_step(TIMESTEP) != -1) {
        if (wb_robot_get_time() > 0.0) {
            past_time = wb_robot_get_time();
            for (int i = 0; i < NUM_SENSORS; i++) {
                if (light_sensors[i]) {
                    light_values[i] = wb_light_sensor_get_value(light_sensors[i]);
                    total_light += light_values[i];
                }
            }
            double bw;
            bearing_valid = calculate_bearing(light_values, bearing_angle, bw);
            smooth_bearing = bearing_angle;
            break;
        }
    }

    printf("Initialized. Press 'S' to start.\n");

    // Log file
    FILE *log_fp = fopen("logs/fixed_fusion_trajectory.csv", "w");
    if (!log_fp) log_fp = fopen("fixed_fusion_trajectory.csv", "w");
    if (log_fp)
        fprintf(log_fp, "time,x,y,z,yaw_deg,cmd_yaw_deg,forward_speed,"
                        "bearing_angle,bearing_valid,bearing_weight,"
                        "grad_angle,grad_mag,grad_valid,grad_weight,"
                        "map_size,total_light,dist_to_target\n");

    // -----------------------------------------------------------------------
    // Main control loop
    // -----------------------------------------------------------------------
    while (wb_robot_step(TIMESTEP) != -1) {
        double t  = wb_robot_get_time();
        double dt = t - past_time;
        past_time = t;

        // Keyboard
        int key = wb_keyboard_get_key();
        if (key == 'S' && !search_active) {
            search_active = true;
            measurement_map.clear();
            printf(">>> S pressed. Starting fixed-weight fusion.\n");
        } else if (key == 'Q') {
            printf(">>> Q pressed. Quitting.\n");
            break;
        }

        // Sensors
        const double *rpy      = wb_inertial_unit_get_roll_pitch_yaw(imu);
        const double *gyro_v   = wb_gyro_get_values(gyro);
        const double *velocity = wb_gps_get_speed_vector(gps);
        const double *pos      = wb_gps_get_values(gps);
        double current_yaw     = rpy[2] * 180.0 / M_PI;

        // Read all light sensors
        total_light = 0.0;
        for (int i = 0; i < NUM_SENSORS; i++) {
            if (light_sensors[i]) {
                light_values[i] = wb_light_sensor_get_value(light_sensors[i]);
                total_light += light_values[i];
            }
        }

        // --- Bearing estimation ---
        double bw = 0.0;
        bearing_valid = calculate_bearing(light_values, bearing_angle, bw);
        if (bearing_valid) {
            double diff = normalize_angle(bearing_angle - smooth_bearing);
            smooth_bearing = normalize_angle(
                smooth_bearing + BEARING_SMOOTH_FACTOR * diff);
        }
        // Convert relative bearing to absolute world heading
        double abs_bearing_yaw = normalize_angle(current_yaw + smooth_bearing);

        // --- Passive map update with lateral exploration offset ---
        // Add small sinusoidal lateral deviation to build 2D spatial coverage
        // even when flying mostly straight. This ensures the gradient has
        // enough spread in perpendicular directions to fit a meaningful plane.
        if (search_active) {
            double lateral_offset = EXPLORE_AMPLITUDE * sin(t * EXPLORE_FREQUENCY);
            
            // Perpendicular direction to current yaw (right-hand side)
            double perp_x = -sin(current_yaw * M_PI / 180.0);
            double perp_y =  cos(current_yaw * M_PI / 180.0);
            
            // Offset position for map storage (actual blimp position unchanged)
            double map_x = pos[0] + lateral_offset * perp_x;
            double map_y = pos[1] + lateral_offset * perp_y;
            
            update_map(map_x, map_y, total_light, t, rpy[2]);
        }

        // --- Gradient estimation ---
        double gx = 0, gy = 0, grad_mag = 0;
        double grad_angle = 0.0;
        bool   grad_valid = false;
        if (search_active) {
            grad_valid = estimate_gradient(pos[0], pos[1], gx, gy, grad_mag);
            if (grad_valid)
                grad_angle = atan2(gy, gx) * 180.0 / M_PI;
        }

        // Gradient is "ready" only when map is populated and signal is strong
        bool grad_ready = grad_valid
                          && (int)measurement_map.size() >= MIN_MAP_POINTS
                          && grad_mag >= GRADIENT_THRESHOLD;

        // --- Proximity check ---
        double dist = sqrt(pow(pos[0]-TARGET_X,2) + pow(pos[1]-TARGET_Y,2));
        if (dist <= PROXIMITY_THRESHOLD && search_active) {
            printf("\n>>> SUCCESS! Reached target. Pos=(%.3f,%.3f) Dist=%.3fm T=%.1fs\n",
                   pos[0], pos[1], dist, t);
            break;
        }

        // -------------------------------------------------------------------
        // Heading fusion
        // -------------------------------------------------------------------
        double cmd_yaw      = current_yaw;
        double forward_speed = 0.0;
        double w_b = 1.0, w_g = 0.0;   // effective weights this step

        if (!search_active) {
            // IDLE: hold heading
            cmd_yaw       = current_yaw;
            forward_speed = 0.0;

        } else if (!bearing_valid || total_light <= MIN_TOTAL_LIGHT) {
            // No light signal at all -- stop and wait
            cmd_yaw       = current_yaw;
            forward_speed = 0.0;
            printf("[T:%.1fs] No light signal. Hovering.[bearing valid: %i] [total light: %0.0f]\n", t, bearing_valid, total_light);

        } else if (!grad_ready) {
            // Gradient not ready -- pure bearing (100% bearing weight)
            w_b = 1.0; w_g = 0.0;
            cmd_yaw       = abs_bearing_yaw;
            forward_speed = FORWARD_SPEED;

        } else {
            // Both methods ready -- fixed weighted fusion
            w_b = W_BEARING;
            w_g = W_GRADIENT;
            cmd_yaw = weighted_circular_mean(abs_bearing_yaw, w_b,
                                             grad_angle,      w_g);
            forward_speed = FORWARD_SPEED;
        }

        // --- Emit to low-level controller ---
        if (gEmitter) {
            controls[0]  = ALTITUDE_SETPOINT;
            controls[1]  = cmd_yaw;
            controls[2]  = forward_speed;
            controls[3]  = 0.0;
            controls[4]  = 0.0;
            controls[5]  = velocity[0];
            controls[6]  = velocity[1];
            controls[7]  = velocity[2];
            controls[8]  = pos[2];
            controls[9]  = current_yaw;
            controls[10] = gyro_v[2] * 180.0 / M_PI;
            controls[11] = dt;
            wb_emitter_send(gEmitter, controls, sizeof(controls));
        }

        // --- CSV logging ---
        if (log_fp) {
            fprintf(log_fp,
                "%.3f,%.6f,%.6f,%.6f,%.4f,%.4f,%.3f,"
                "%.4f,%d,%.2f,"
                "%.4f,%.4f,%d,%.2f,"
                "%zu,%.3f,%.4f\n",
                t,
                pos[0], pos[1], pos[2],
                current_yaw, cmd_yaw, forward_speed,
                smooth_bearing, (int)bearing_valid, w_b,
                grad_angle, grad_mag, (int)grad_valid, w_g,
                measurement_map.size(), total_light, dist);
        }

        // --- Console status (every 0.5 s) ---
        static double last_print = 0.0;
        if (t - last_print >= 0.5) {
            const char* mode_str = !search_active          ? "IDLE        " :
                                   (!bearing_valid)        ? "NO_SIGNAL   " :
                                   !grad_ready             ? "BEARING_ONLY" :
                                                             "FUSED       ";
            printf("[T:%5.1fs] %s | Light:%6.1f | Bearing:%6.1f° | "
                   "GradAngle:%6.1f° GradMag:%5.3f | "
                   "Weights: B=%.2f G=%.2f | MapSz:%3zu | "
                   "CmdYaw:%6.1f° | Dist:%5.2fm\n",
                   t, mode_str,
                   total_light,
                   bearing_valid ? smooth_bearing : 0.0,
                   grad_valid    ? grad_angle     : 0.0,
                   grad_mag,
                   w_b, w_g,
                   measurement_map.size(),
                   cmd_yaw, dist);
            last_print = t;
        }
    }

    // --- Cleanup and map export ---
    if (log_fp) fclose(log_fp);
    
    // Write map data and gradient field for visualization
    if (measurement_map.size() > 0) {
        write_map_data("logs/fixed_measurement_map.csv");
        write_map_data("fixed_measurement_map.csv");  // Fallback
        
        // Calculate bounds covering trajectory, target, and all map points
        const double *final_pos = wb_gps_get_values(gps);
        double min_x = final_pos[0] - 5.0, max_x = final_pos[0] + 5.0;
        double min_y = final_pos[1] - 5.0, max_y = final_pos[1] + 5.0;
        
        min_x = std::min(min_x, TARGET_X - 2.0);
        max_x = std::max(max_x, TARGET_X + 2.0);
        min_y = std::min(min_y, TARGET_Y - 2.0);
        max_y = std::max(max_y, TARGET_Y + 2.0);
        
        for (const auto& entry : measurement_map) {
            const MeasurementPoint& pt = entry.second;
            min_x = std::min(min_x, pt.x - 1.0);
            max_x = std::max(max_x, pt.x + 1.0);
            min_y = std::min(min_y, pt.y - 1.0);
            max_y = std::max(max_y, pt.y + 1.0);
        }
        
        write_gradient_field("logs/fixed_gradient_field.csv", min_x, max_x, min_y, max_y);
        write_gradient_field("fixed_gradient_field.csv", min_x, max_x, min_y, max_y);  // Fallback
    }
    
    printf("\nFixed-weight fusion completed. Final map size: %zu locations\n", measurement_map.size());

    wb_robot_cleanup();
    return 0;
}