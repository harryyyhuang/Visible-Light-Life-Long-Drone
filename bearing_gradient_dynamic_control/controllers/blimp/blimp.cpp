/****************************************************************************

 blimp_combined_dynamic.cpp -- Dynamic-Weight Bearing + Gradient Fusion

 Fuses two light-seeking strategies with DYNAMIC weights that shift based on:
   (1) Heading disagreement between bearing and gradient (deg)
   (2) Gradient magnitude (confidence in the spatial map signal)

 WEIGHT LOGIC:
   The base gradient weight is computed from two signals multiplied together:

     disagreement_factor = sigmoid-like ramp from 0→1 as |angle_diff|
                           goes from DISAGREE_MIN_DEG to DISAGREE_MAX_DEG

     magnitude_factor    = sigmoid-like ramp from 0→1 as grad_magnitude
                           goes from GRADIENT_WEAK to GRADIENT_STRONG

     raw_gradient_weight = disagreement_factor * magnitude_factor

   This is clamped to [0, MAX_GRADIENT_WEIGHT] and smoothed with a low-pass
   filter to prevent rapid oscillation between the two methods.
   The bearing weight is always (1 - gradient_weight).

 INTERPRETATION:
   - Open space, both agree  → gradient weight ≈ 0   → almost pure bearing
   - Obstacle nearby, signals disagree AND gradient is strong
                             → gradient weight → MAX_GRADIENT_WEIGHT
   - Gradient weak/sparse    → gradient weight ≈ 0 regardless of disagreement
     (prevents bad gradient from overriding a good bearing signal)

 FALLBACK:
   Gradient weight is forced to 0 (pure bearing) until the map has at least
   MIN_MAP_POINTS cells and gradient magnitude exceeds GRADIENT_THRESHOLD.

 MAP BUILDING:
   Built passively from the blimp's forward path. No exploration spiral.

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
// Dynamic weight parameters
// ---------------------------------------------------------------------------

// Disagreement ramp: below MIN → factor=0, above MAX → factor=1, linear between
static const double DISAGREE_MIN_DEG    = 15.0;   // start ramping gradient up
static const double DISAGREE_MAX_DEG    = 60.0;   // full gradient weight from disagreement

// Gradient magnitude ramp: below WEAK → factor=0, above STRONG → factor=1
static const double GRADIENT_WEAK       = 0.5;    // gradient too weak to trust
static const double GRADIENT_STRONG     = 2.0;    // gradient fully trusted

// Clamp: gradient weight never exceeds this even at maximum disagreement
static const double MAX_GRADIENT_WEIGHT = 0.7;

// Low-pass smoothing on the weight itself (prevents oscillation)
static const double WEIGHT_SMOOTH_FACTOR = 0.1;

// ---------------------------------------------------------------------------
// Gradient readiness thresholds
// ---------------------------------------------------------------------------
static const int    MIN_MAP_POINTS     = 3;
static const double GRADIENT_THRESHOLD = 0.5;   // minimum to allow any gradient weight

// ---------------------------------------------------------------------------
// Spatial map parameters
// ---------------------------------------------------------------------------
static const double GRID_RESOLUTION       = 0.2;
static const double MIN_LIGHT_IMPROVEMENT = 0.1;

// ---------------------------------------------------------------------------
// Target
// ---------------------------------------------------------------------------
static const double TARGET_X            = 5.1988;
static const double TARGET_Y            = 5.329;
static const double PROXIMITY_THRESHOLD = 2.0;

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

static double clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Linear ramp from 0 to 1 as x goes from lo to hi
static double linear_ramp(double x, double lo, double hi) {
    if (x <= lo) return 0.0;
    if (x >= hi) return 1.0;
    return (x - lo) / (hi - lo);
}

static GridCell pos_to_grid(double x, double y) {
    GridCell c;
    c.gx = (int)floor(x / GRID_RESOLUTION);
    c.gy = (int)floor(y / GRID_RESOLUTION);
    return c;
}

// ---------------------------------------------------------------------------
// Map update
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
// Least-squares plane fit → gradient
// ---------------------------------------------------------------------------
static bool estimate_gradient(double cx, double cy,
                               double& gx_out, double& gy_out,
                               double& magnitude) {
    gx_out = gy_out = magnitude = 0.0;
    if ((int)measurement_map.size() < MIN_MAP_POINTS) return false;

    const double MAX_DIST = 3.0;
    const double MIN_DIST = 0.1;

    std::vector<MeasurementPoint> pts;
    for (const auto& entry : measurement_map) {
        const MeasurementPoint& p = entry.second;
        double dx = cx - p.x, dy = cy - p.y;
        double d  = sqrt(dx*dx + dy*dy);
        if (d >= MIN_DIST && d <= MAX_DIST) pts.push_back(p);
    }
    if ((int)pts.size() < MIN_MAP_POINTS) return false;

    int n = (int)pts.size();
    double sx=0, sy=0, sz=0, sxx=0, syy=0, sxy=0, sxz=0, syz=0;
    for (const auto& p : pts) {
        sx  += p.x;         sy  += p.y;
        sz  += p.light_intensity;
        sxx += p.x * p.x;  syy += p.y * p.y;
        sxy += p.x * p.y;
        sxz += p.x * p.light_intensity;
        syz += p.y * p.light_intensity;
    }

    double A[3][3] = {
        {sxx, sxy, sx},
        {sxy, syy, sy},
        {sx,  sy,  (double)n}
    };
    double B[3] = {sxz, syz, sz};

    double det = A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
               - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
               + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);
    if (fabs(det) < 1e-6) return false;

    double det_a = B[0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
                 - A[0][1]*(B[1]*A[2][2] - A[1][2]*B[2])
                 + A[0][2]*(B[1]*A[2][1] - A[1][1]*B[2]);
    double det_b = A[0][0]*(B[1]*A[2][2] - A[1][2]*B[2])
                 - B[0]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
                 + A[0][2]*(A[1][0]*B[2] - B[1]*A[2][0]);

    gx_out    = det_a / det;
    gy_out    = det_b / det;
    magnitude = sqrt(gx_out*gx_out + gy_out*gy_out);

    // R² quality check
    double mean_z = sz / n, ss_tot = 0, ss_res = 0;
    double c_coeff = (sz - gx_out*sx - gy_out*sy) / n;
    for (const auto& p : pts) {
        double pred = gx_out*p.x + gy_out*p.y + c_coeff;
        ss_res += (p.light_intensity - pred) * (p.light_intensity - pred);
        ss_tot += (p.light_intensity - mean_z) * (p.light_intensity - mean_z);
    }
    double r2 = (ss_tot > 1e-10) ? (1.0 - ss_res / ss_tot) : 0.0;
    if (r2 < 0.2) return false;

    return true;
}

// ---------------------------------------------------------------------------
// Bearing from 16-sensor array
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
// Weighted circular mean (handles angle wrap correctly)
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
// Main
// ---------------------------------------------------------------------------
int main() {
    printf("=== DYNAMIC-WEIGHT FUSION: Bearing + Gradient ===\n");
    printf("Gradient weight grows with heading disagreement AND gradient strength.\n");
    printf("Max gradient weight = %.0f%%\n", MAX_GRADIENT_WEIGHT * 100.0);
    printf("Disagree ramp: %.0f° → %.0f°\n", DISAGREE_MIN_DEG, DISAGREE_MAX_DEG);
    printf("Magnitude ramp: %.2f → %.2f\n", GRADIENT_WEAK, GRADIENT_STRONG);
    printf("Commands: S=START  Q=QUIT\n");
    printf("=================================================\n\n");

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

    // State
    double controls[12]        = {0};
    double past_time           = 0.0;
    bool   search_active       = false;

    double light_values[NUM_SENSORS] = {0};
    double total_light         = 0.0;
    double bearing_angle       = 0.0;
    double smooth_bearing      = 0.0;
    bool   bearing_valid       = false;

    double smoothed_grad_weight = 0.0;   // low-pass filtered gradient weight

    measurement_map.clear();

    // Wait for first valid step
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
            bearing_valid  = calculate_bearing(light_values, bearing_angle, bw);
            smooth_bearing = bearing_angle;
            break;
        }
    }

    printf("Initialized. Press 'S' to start.\n");

    // Log file
    FILE *log_fp = fopen("logs/dynamic_fusion_trajectory.csv", "w");
    if (!log_fp) log_fp = fopen("dynamic_fusion_trajectory.csv", "w");
    if (log_fp)
        fprintf(log_fp, "time,x,y,z,yaw_deg,cmd_yaw_deg,forward_speed,"
                        "bearing_angle,bearing_valid,bearing_weight,"
                        "grad_angle,grad_mag,grad_valid,grad_weight,"
                        "disagree_factor,magnitude_factor,raw_grad_weight,"
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
            smoothed_grad_weight = 0.0;
            printf(">>> S pressed. Starting dynamic-weight fusion.\n");
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

        // Read sensors
        total_light = 0.0;
        for (int i = 0; i < NUM_SENSORS; i++) {
            if (light_sensors[i]) {
                light_values[i] = wb_light_sensor_get_value(light_sensors[i]);
                total_light += light_values[i];
            }
        }

        // Bearing estimation
        double bw = 0.0;
        bearing_valid = calculate_bearing(light_values, bearing_angle, bw);
        if (bearing_valid) {
            double diff = normalize_angle(bearing_angle - smooth_bearing);
            smooth_bearing = normalize_angle(
                smooth_bearing + BEARING_SMOOTH_FACTOR * diff);
        }
        double abs_bearing_yaw = normalize_angle(current_yaw + smooth_bearing);

        // Passive map update
        if (search_active)
            update_map(pos[0], pos[1], total_light, t, rpy[2]);

        // Gradient estimation
        double gx = 0, gy = 0, grad_mag = 0;
        double grad_angle = 0.0;
        bool   grad_valid = false;
        if (search_active) {
            grad_valid = estimate_gradient(pos[0], pos[1], gx, gy, grad_mag);
            if (grad_valid)
                grad_angle = atan2(gy, gx) * 180.0 / M_PI;
        }

        // Proximity check
        double dist = sqrt(pow(pos[0]-TARGET_X,2) + pow(pos[1]-TARGET_Y,2));
        if (dist <= PROXIMITY_THRESHOLD && search_active) {
            printf("\n>>> SUCCESS! Reached target. Pos=(%.3f,%.3f) Dist=%.3fm T=%.1fs\n",
                   pos[0], pos[1], dist, t);
            break;
        }

        // -------------------------------------------------------------------
        // Dynamic weight calculation
        // -------------------------------------------------------------------
        double disagree_factor  = 0.0;
        double magnitude_factor = 0.0;
        double raw_grad_weight  = 0.0;
        double w_g = 0.0, w_b = 1.0;

        bool grad_ready = grad_valid
                          && (int)measurement_map.size() >= MIN_MAP_POINTS
                          && grad_mag >= GRADIENT_THRESHOLD;

        if (search_active && bearing_valid && grad_ready) {
            // (1) Disagreement factor: how different are the two headings?
            double heading_diff = fabs(normalize_angle(abs_bearing_yaw - grad_angle));
            disagree_factor = linear_ramp(heading_diff,
                                          DISAGREE_MIN_DEG, DISAGREE_MAX_DEG);

            // (2) Magnitude factor: how strong/trustworthy is the gradient?
            magnitude_factor = linear_ramp(grad_mag,
                                            GRADIENT_WEAK, GRADIENT_STRONG);

            // Combined raw gradient weight
            raw_grad_weight = disagree_factor * magnitude_factor * MAX_GRADIENT_WEIGHT;
            raw_grad_weight = clamp(raw_grad_weight, 0.0, MAX_GRADIENT_WEIGHT);

            // Smooth the weight to avoid rapid oscillation
            smoothed_grad_weight += WEIGHT_SMOOTH_FACTOR
                                    * (raw_grad_weight - smoothed_grad_weight);
        } else {
            // Not ready -- drive weight back toward 0 smoothly
            smoothed_grad_weight += WEIGHT_SMOOTH_FACTOR * (0.0 - smoothed_grad_weight);
        }

        w_g = smoothed_grad_weight;
        w_b = 1.0 - w_g;

        // -------------------------------------------------------------------
        // Heading fusion and speed
        // -------------------------------------------------------------------
        double cmd_yaw       = current_yaw;
        double forward_speed = 0.0;

        if (!search_active) {
            cmd_yaw       = current_yaw;
            forward_speed = 0.0;

        } else if (!bearing_valid || total_light <= MIN_TOTAL_LIGHT) {
            cmd_yaw       = current_yaw;
            forward_speed = 0.0;
            printf("[T:%.1fs] No light signal. Hovering.\n", t);

        } else if (w_g < 1e-4) {
            // Effectively pure bearing
            cmd_yaw       = abs_bearing_yaw;
            forward_speed = FORWARD_SPEED;

        } else {
            // Weighted fusion
            cmd_yaw       = weighted_circular_mean(abs_bearing_yaw, w_b,
                                                    grad_angle,     w_g);
            forward_speed = FORWARD_SPEED;
        }

        // Emit to low-level controller
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

        // CSV logging
        if (log_fp) {
            fprintf(log_fp,
                "%.3f,%.6f,%.6f,%.6f,%.4f,%.4f,%.3f,"
                "%.4f,%d,%.4f,"
                "%.4f,%.4f,%d,%.4f,"
                "%.4f,%.4f,%.4f,"
                "%zu,%.3f,%.4f\n",
                t,
                pos[0], pos[1], pos[2],
                current_yaw, cmd_yaw, forward_speed,
                smooth_bearing, (int)bearing_valid, w_b,
                grad_angle, grad_mag, (int)grad_valid, w_g,
                disagree_factor, magnitude_factor, raw_grad_weight,
                measurement_map.size(), total_light, dist);
        }

        // Console status (every 0.5 s)
        static double last_print = 0.0;
        if (t - last_print >= 0.5) {
            printf("[T:%5.1fs] Light:%6.1f | "
                   "Bearing:%6.1f° | GradAngle:%6.1f° GradMag:%5.3f | "
                   "DisagreeFactor:%.2f MagFactor:%.2f | "
                   "Weights B=%.2f G=%.2f | "
                   "MapSz:%3zu | CmdYaw:%6.1f° | Dist:%5.2fm\n",
                   t,
                   total_light,
                   bearing_valid ? smooth_bearing : 0.0,
                   grad_valid    ? grad_angle     : 0.0, grad_mag,
                   disagree_factor, magnitude_factor,
                   w_b, w_g,
                   measurement_map.size(),
                   cmd_yaw, dist);
            last_print = t;
        }
    }

    if (log_fp) fclose(log_fp);
    wb_robot_cleanup();
    return 0;
}
