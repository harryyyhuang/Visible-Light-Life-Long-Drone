#ifndef ANGLE_CALCULATOR_H
#define ANGLE_CALCULATOR_H

#include <Arduino.h>
#include "adc_reader.h"

// Automatically adapt to the number of sensors defined in adc_reader.h
// const int NUM_SENSORS = NUM_PINS;

// Define sensor angles in degrees (modify these to match your physical setup)
// Sensor 0: 0°, Sensor 1: 120°, Sensor 2: 240° for a triangular array
// Or 0°, 90°, 180°, 270° for a square array, etc.
// Update this array to match your actual sensor positions
const float SENSOR_ANGLES[15] = {
    315.0,    // Sensor 0 at 0 degrees
    292.5,   // Sensor 1 at 45 degrees  
    270.0,   // Sensor 2 at 90 degrees
    247.5,  // Sensor 3 at 135 degrees
    225.0,  // Sensor 4 at 180 degrees
    202.5,  // Sensor 5 at 225 degrees
    157.5,  // Sensor 6 at 270 degrees
    135.0,  // Sensor 7 at 315 degrees
    // Add more angles as needed for additional sensors
    // #if NUM_SENSORS > 8
    112.5,   // Sensor 8
    90.0,   // Sensor 9
    67.5,  // Sensor 10
    45.0,  // Sensor 11
    22.5,  // Sensor 12
    0.0,  // Sensor 13
    337.5,  // Sensor 14
    // #endif
};

class AngleCalculator {
public:
    // Constructor
    AngleCalculator() : initialized(false) {}

    // Destructor
    ~AngleCalculator() {}

    /**
     * @brief Initializes the angle calculator.
     * @return True if initialization was successful.
     */
    bool init() {
        // Verify that we have the right number of sensor angles
        if (sizeof(SENSOR_ANGLES)/sizeof(SENSOR_ANGLES[0]) < 15) {
            Serial.println("Error: Not enough sensor angles defined for NUM_SENSORS");
            return false;
        }
        
        initialized = true;
        return true;
    }

    /**
     * @brief Calculates weighted bearing angle using SNR values as weights.
     * @param snr_values Array of SNR values for each sensor
     * @param result_angle Pointer to store the calculated bearing angle in degrees
     * @param total_weight Pointer to store the total weight (sum of SNR values)
     * @return True if calculation was successful
     */
    bool calculate_weighted_angle_snr(float* mag_values, float* snr_values, float* result_angle, float* total_weight) {
        if (!initialized) return false;
        float sum_x = 0.0f;  // Sum of weighted cosines
        float sum_y = 0.0f;  // Sum of weighted sines
        float weight_sum = 0.0f;
        float snr_sum = 0.0f;

        float snr_values_copy[15];

        for (int i = 0; i< 15; i++) {
            snr_sum += snr_values[i];
        }

        for (int i = 0; i< 15; i++) {
            snr_values_copy[i] = (snr_sum > 0.0f) ? (snr_values[i] / snr_sum) : 0.0f;
        }

        // Calculate weighted vector sum
        for (int i = 0; i < 15; i++) {
            if (mag_values[i] > 0.0f) {  // Only include sensors with positive weights
                float angle_rad = SENSOR_ANGLES[i] * PI / 180.0f;  // Convert to radians
                sum_x += mag_values[i] * cos(angle_rad) * snr_values_copy[i];
                sum_y += mag_values[i] * sin(angle_rad) * snr_values_copy[i];
                weight_sum += mag_values[i];
            }
        }
        
        *total_weight = weight_sum;
        
        // If total weight is zero, we can't calculate a meaningful angle
        if (weight_sum <= 0.0f) {
            *result_angle = 0.0f;
            return false;
        }
        
        // Calculate the resulting angle using atan2
        float result_rad = atan2(sum_y, sum_x);
        
        // Convert to degrees and normalize to 0-360 range
        *result_angle = result_rad * 180.0f / PI;
        if (*result_angle < 0.0f) {
            *result_angle += 360.0f;
        }
        
        return true;
        // return calculate_weighted_angle_internal(snr_values, result_angle, total_weight);
    }

    /**
     * @brief Calculates weighted bearing angle using magnitude values as weights.
     * @param magnitude_values Array of magnitude values for each sensor
     * @param result_angle Pointer to store the calculated bearing angle in degrees
     * @param total_weight Pointer to store the total weight (sum of magnitude values)
     * @return True if calculation was successful
     */
    bool calculate_weighted_angle_magnitude(float* magnitude_values, float* result_angle, float* total_weight) {
        if (!initialized) return false;
        
        return calculate_weighted_angle_internal(magnitude_values, result_angle, total_weight);
    }

    /**
     * @brief Calculates weighted bearing angle using normalized weights.
     * This version normalizes the weights first to prevent numerical issues.
     * @param weight_values Array of weight values for each sensor
     * @param result_angle Pointer to store the calculated bearing angle in degrees
     * @param total_weight Pointer to store the total normalized weight
     * @return True if calculation was successful
     */
    bool calculate_weighted_angle_normalized(float* weight_values, float* result_angle, float* total_weight) {
        if (!initialized) return false;
        
        // Find max weight for normalization
        float max_weight = 0.0f;
        for (int i = 0; i < 15; i++) {
            if (weight_values[i] > max_weight) {
                max_weight = weight_values[i];
            }
        }
        
        // If all weights are zero, return false
        if (max_weight <= 0.0f) {
            *result_angle = 0.0f;
            *total_weight = 0.0f;
            return false;
        }
        
        // Normalize weights
        float normalized_weights[15];
        for (int i = 0; i < 15; i++) {
            normalized_weights[i] = weight_values[i] / max_weight;
        }
        
        return calculate_weighted_angle_internal(normalized_weights, result_angle, total_weight);
    }

    /**
     * @brief Get the sensor angle for a specific sensor index.
     * @param sensor_index The index of the sensor (0 to 15-1)
     * @return The angle in degrees, or -1.0 if index is invalid
     */
    float get_sensor_angle(int sensor_index) {
        if (sensor_index < 0 || sensor_index >= 15) {
            return -1.0f;
        }
        return SENSOR_ANGLES[sensor_index];
    }

    /**
     * @brief Get the number of sensors being used.
     * @return The number of sensors
     */
    int get_num_sensors() {
        return 15;
    }

private:
    bool initialized;

    /**
     * @brief Internal function to calculate weighted bearing angle using vector addition.
     * Uses the formula: angle = atan2(sum(w_i * sin(θ_i)), sum(w_i * cos(θ_i)))
     * @param weights Array of weight values for each sensor
     * @param result_angle Pointer to store the calculated bearing angle in degrees
     * @param total_weight Pointer to store the total weight
     * @return True if calculation was successful
     */
    bool calculate_weighted_angle_internal(float* weights, float* result_angle, float* total_weight) {
        float sum_x = 0.0f;  // Sum of weighted cosines
        float sum_y = 0.0f;  // Sum of weighted sines
        float weight_sum = 0.0f;
        
        // Calculate weighted vector sum
        for (int i = 0; i < 15; i++) {
            if (weights[i] > 0.0f) {  // Only include sensors with positive weights
                float angle_rad = SENSOR_ANGLES[i] * PI / 180.0f;  // Convert to radians
                sum_x += weights[i] * cos(angle_rad);
                sum_y += weights[i] * sin(angle_rad);
                weight_sum += weights[i];
            }
        }
        
        *total_weight = weight_sum;
        
        // If total weight is zero, we can't calculate a meaningful angle
        if (weight_sum <= 0.0f) {
            *result_angle = 0.0f;
            return false;
        }
        
        // Calculate the resulting angle using atan2
        float result_rad = atan2(sum_y, sum_x);
        
        // Convert to degrees and normalize to 0-360 range
        *result_angle = result_rad * 180.0f / PI;
        if (*result_angle < 0.0f) {
            *result_angle += 360.0f;
        }
        
        return true;
    }
};

#endif
