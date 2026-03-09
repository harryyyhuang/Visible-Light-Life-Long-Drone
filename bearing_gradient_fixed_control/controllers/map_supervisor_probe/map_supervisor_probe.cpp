/****************************************************************************

 map_probe_supervisor.cpp

 Webots Supervisor controller that teleports the blimp through a grid pattern,
 samples all 16 light sensors at each position, and generates a realistic
 measurement map CSV file with individual sensor readings.

 USAGE:
 1. Compile this controller in Webots
 2. Temporarily set this as the blimp's controller in your world file
 3. Set the blimp's supervisor field to TRUE in the world file
 4. Run the simulation
 5. The supervisor will systematically probe the environment
 6. Outputs: precomputed_map.csv (in controller directory or project root)
 7. Switch back to your normal controller and it will auto-load the map

 CONFIGURATION:
 - Adjust constants below for grid resolution and coverage area
 - Probing takes time: ~0.5s per grid point for sensor stabilization

******************************************************************************/

#include <webots/robot.h>
#include <webots/supervisor.h>
#include <webots/light_sensor.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

// Probe grid configuration
static const double GRID_RESOLUTION = 0.2;   // meters between sample points
static const double X_MIN = -5.0;
static const double X_MAX =  8.0;
static const double Y_MIN = -3.0;
static const double Y_MAX =  8.0;
static const double ALTITUDE = 2.0;          // Flight altitude to sample at

// Sensor configuration (must match Blimp.proto)
static const int NUM_LIGHT_SENSORS = 16;

// Stabilization time after teleport
static const int SETTLE_STEPS = 25;          // Number of timesteps to wait

// Safety margin around obstacles (meters)
static const double OBSTACLE_SAFETY_MARGIN = 0.5;  // Stay this far away from obstacles

int main(int argc, char **argv) {
    printf("=== WEBOTS MAP PROBE SUPERVISOR ===\n");
    printf("Systematically sampling light field to generate precomputed map...\n");
    printf("Grid: %.2fm resolution from (%.1f,%.1f) to (%.1f,%.1f)\n",
           GRID_RESOLUTION, X_MIN, Y_MIN, X_MAX, Y_MAX);
    printf("Altitude: %.1fm\n", ALTITUDE);
    printf("==================================================\n\n");

    wb_robot_init();
    int timestep = (int)wb_robot_get_basic_time_step();

    // Get the blimp node (must be defined with DEF in world file)
    WbNodeRef blimp = wb_supervisor_node_get_from_def("blimp_lis");
    if (blimp == NULL) {
        printf("ERROR: Could not find blimp node with DEF 'blimp_lis'\n");
        printf("Make sure your world file has: DEF blimp_lis Blimp { ... }\n");
        wb_robot_cleanup();
        return 1;
    }

    // Get translation field (to teleport blimp)
    WbFieldRef translation_field = wb_supervisor_node_get_field(blimp, "translation");
    if (translation_field == NULL) {
        printf("ERROR: Could not access blimp translation field\n");
        wb_robot_cleanup();
        return 1;
    }

    // Get rotation field (to keep blimp upright)
    WbFieldRef rotation_field = wb_supervisor_node_get_field(blimp, "rotation");

    printf(">>> Starting probe sweep (skipping positions inside obstacles)\n");
    printf(">>> Safety margin: %.2fm around each obstacle\n\n", OBSTACLE_SAFETY_MARGIN);
    
    // Store obstacle information for collision checking
    struct Obstacle {
        const char* def_name;
        double x, y, z;
        double radius;  // For cylinders
        double width, depth, height;  // For boxes
        bool is_cylinder;
    };
    
    std::vector<Obstacle> obstacles;
    
    // Define obstacles (adjust to match your world file)
    // SINGLE OBSTACLE WORLD
    Obstacle pillar_center = {"PILLAR_CENTER", 0.8, 2.0, 1.5, 0.4, 0, 0, 3.0, true};
    obstacles.push_back(pillar_center);
    
    // MULTI-OBSTACLE WORLD (comment out if using single obstacle)
    // Obstacle pillar_a = {"PILLAR_A", -1.9, 0.9, 1.5, 0.25, 0, 0, 3.0, true};
    // Obstacle pillar_b = {"PILLAR_B", -0.9, -0.2, 1.5, 0.25, 0, 0, 3.0, true};
    // Obstacle pillar_c = {"PILLAR_C", 2.3, 2.1, 1.5, 0.25, 0, 0, 3.0, true};
    // Obstacle pillar_d = {"PILLAR_D", 3.6, 4.2, 1.5, 0.25, 0, 0, 3.0, true};
    // Obstacle wall = {"WALL_A", 0.36, 1.68, 1.5, 0, 2.5, 0.2, 3.0, false};  // Box
    // obstacles.push_back(pillar_a);
    // obstacles.push_back(pillar_b);
    // obstacles.push_back(pillar_c);
    // obstacles.push_back(pillar_d);
    // obstacles.push_back(wall);

    // Initialize light sensors
    WbDeviceTag light_sensors[NUM_LIGHT_SENSORS];
    char sensor_name[32];
    for (int i = 0; i < NUM_LIGHT_SENSORS; i++) {
        sprintf(sensor_name, "light_sensor_%d", i);
        light_sensors[i] = wb_robot_get_device(sensor_name);
        if (light_sensors[i] == 0) {
            printf("WARNING: Could not find sensor %s\n", sensor_name);
        } else {
            wb_light_sensor_enable(light_sensors[i], timestep);
        }
    }

    // Initial step to initialize devices
    wb_robot_step(timestep);

    // Open output file
    const char* output_file = "precomputed_map.csv";
    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        printf("ERROR: Could not open output file '%s'\n", output_file);
        wb_robot_cleanup();
        return 1;
    }

    // Write CSV header
    fprintf(fp, "grid_x,grid_y,x,y,total_light,timestamp,yaw");
    for (int i = 0; i < NUM_LIGHT_SENSORS; i++) {
        fprintf(fp, ",sensor_%d", i);
    }
    fprintf(fp, "\n");

    // Calculate total points
    int x_count = (int)ceil((X_MAX - X_MIN) / GRID_RESOLUTION) + 1;
    int y_count = (int)ceil((Y_MAX - Y_MIN) / GRID_RESOLUTION) + 1;
    int total_points = x_count * y_count;

    printf("Starting probe sweep: %d points to sample\n", total_points);
    printf("This will take approximately %.1f seconds...\n\n",
           total_points * SETTLE_STEPS * timestep / 1000.0);

    int point_count = 0;
    int skipped_count = 0;

    // Lambda to check if point (x,y,z) is inside any obstacle (including safety margin)
    auto is_inside_obstacle = [&](double x, double y, double z) -> bool {
        for (const auto& obs : obstacles) {
            if (obs.is_cylinder) {
                // Cylinder collision check (XY distance from center + safety margin)
                double dx = x - obs.x;
                double dy = y - obs.y;
                double dist_xy = sqrt(dx*dx + dy*dy);
                double half_height = obs.height / 2.0;
                
                // Add safety margin to radius
                if (dist_xy < (obs.radius + OBSTACLE_SAFETY_MARGIN) && 
                    fabs(z - obs.z) < (half_height + OBSTACLE_SAFETY_MARGIN)) {
                    return true;  // Inside cylinder (or too close)
                }
            } else {
                // Box collision check (simplified - assumes axis-aligned, with safety margin)
                double dx = fabs(x - obs.x);
                double dy = fabs(y - obs.y);
                double dz = fabs(z - obs.z);
                
                // Add safety margin to box dimensions
                if (dx < (obs.width/2.0 + OBSTACLE_SAFETY_MARGIN) && 
                    dy < (obs.depth/2.0 + OBSTACLE_SAFETY_MARGIN) && 
                    dz < (obs.height/2.0 + OBSTACLE_SAFETY_MARGIN)) {
                    return true;  // Inside box (or too close)
                }
            }
        }
        return false;
    };

    // Sweep through grid
    for (double x = X_MIN; x <= X_MAX; x += GRID_RESOLUTION) {
        for (double y = Y_MIN; y <= Y_MAX; y += GRID_RESOLUTION) {
            // Check if this position is inside an obstacle
            if (is_inside_obstacle(x, y, ALTITUDE)) {
                skipped_count++;
                continue;  // Skip this grid point
            }
            
            point_count++;

            // Teleport blimp to grid position
            double new_position[3] = {x, y, ALTITUDE};
            wb_supervisor_field_set_sf_vec3f(translation_field, new_position);

            // Keep blimp upright (rotation: axis-angle, default up is [0,0,1,0])
            if (rotation_field != NULL) {
                double upright_rotation[4] = {0, 0, 1, 0};
                wb_supervisor_field_set_sf_rotation(rotation_field, upright_rotation);
            }

            // Let sensors stabilize
            for (int settle = 0; settle < SETTLE_STEPS; settle++) {
                wb_robot_step(timestep);
            }

            // Read all 16 light sensors
            double sensor_readings[NUM_LIGHT_SENSORS];
            double total_light = 0.0;
            int valid_sensors = 0;

            for (int i = 0; i < NUM_LIGHT_SENSORS; i++) {
                if (light_sensors[i] != 0) {
                    sensor_readings[i] = wb_light_sensor_get_value(light_sensors[i]);
                    total_light += sensor_readings[i];
                    valid_sensors++;
                } else {
                    sensor_readings[i] = 0.0;
                }
            }

            // Calculate grid coordinates
            int grid_x = (int)round(x / GRID_RESOLUTION);
            int grid_y = (int)round(y / GRID_RESOLUTION);

            // Get current simulation time
            double current_time = wb_robot_get_time();

            // Write to CSV: grid coords, position, total_light, timestamp, yaw, then all 16 sensors
            fprintf(fp, "%d,%d,%.6f,%.6f,%.3f,%.3f,0.0",
                    grid_x, grid_y, x, y, total_light, current_time);

            for (int i = 0; i < NUM_LIGHT_SENSORS; i++) {
                fprintf(fp, ",%.3f", sensor_readings[i]);
            }
            fprintf(fp, "\n");

            // Calculate sensor statistics for progress display
            double max_sensor = 0.0;
            double min_sensor = 1000.0;
            int brightest_idx = -1;

            for (int i = 0; i < NUM_LIGHT_SENSORS; i++) {
                if (sensor_readings[i] > max_sensor) {
                    max_sensor = sensor_readings[i];
                    brightest_idx = i;
                }
                if (sensor_readings[i] < min_sensor) {
                    min_sensor = sensor_readings[i];
                }
            }

            double sensor_delta = max_sensor - min_sensor;

            // Progress indicator
            if (point_count % 10 == 0 || point_count == total_points) {
                double progress = (double)point_count / total_points * 100.0;
                printf("[%d/%d] (%.1f%%) Pos(%.2f,%.2f): total=%.2f max=%.2f@s%d delta=%.2f\n",
                       point_count, total_points, progress,
                       x, y, total_light, max_sensor, brightest_idx, sensor_delta);
            }
        }
    }

    // Close file
    fclose(fp);

    printf("\n==================================================\n");
    printf("✓ Map probing complete!\n");
    printf("✓ Saved %d measurements to '%s'\n", point_count, output_file);
    printf("✓ Skipped %d points inside obstacles\n", skipped_count);
    printf("✓ Total grid points checked: %d\n", point_count + skipped_count);
    
    printf("\nNext steps:\n");
    printf("  1. Stop this simulation\n");
    printf("  2. Change the blimp controller back to your normal controller\n");
    printf("  3. Set supervisor field back to FALSE\n");
    printf("  4. Run simulation - controller will load 'precomputed_map.csv' at startup\n");
    printf("==================================================\n");

    // Keep supervisor alive so user can read the message
    while (wb_robot_step(timestep) != -1) {
        // Just idle
    }

    wb_robot_cleanup();
    return 0;
}