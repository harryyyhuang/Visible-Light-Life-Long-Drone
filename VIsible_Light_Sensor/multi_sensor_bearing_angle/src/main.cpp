#include <Arduino.h>
#include <ADC.h>
#include <ADC_util.h>
#include <Wire.h>
#include <TeensyThreads.h> // Include the TeensyThreads library

#include "adc_reader.h"      // For reading ADC values
#include "fft_analyzer.h"    // The new FFT-based analyzer
#include "angle_calculator.h" // For weighted angle calculation


// Add I2C data structure after global variables
struct TeensyI2cData_t {  // Changed from I2CData
    float sensor_max_snr;
    float bearing_angle_mag;
    uint32_t timestamp;
    uint8_t data_valid;
} i2c_data;


volatile bool i2c_data_ready = false;

// Add I2C configuration after other defines
#define I2C_SLAVE_ADDRESS 0x08  // I2C slave address for this Teensy

// --- FFT Analysis Configuration ---
#define SAMPLING_RATE 1600.0f // Desired sampling rate
#define TARGET_FREQ 150.0f      // The frequency we want to analyze

// ADC object
ADC* adc = new ADC();

// --- Global Variables & Shared Resources ---
const int NUM_SENSORS = 15; // Must match PINS_TO_READ in adc_reader.h

// A single, shared circular buffer for all sensors.
uint16_t sample_buffers[NUM_SENSORS][FFT_SIZE];
float local_fft_buffer[NUM_SENSORS][FFT_SIZE];
float fft_output_buffer[NUM_SENSORS][FFT_SIZE / 2];


// A mutex to protect access to the shared sample buffers and related variables.
std::mutex buffer_lock;

// `volatile` is crucial for variables shared between threads.
volatile int sample_write_index = 0; // The index where the next sample will be written.
volatile bool fft_data_ready = false; // Flag to signal the FFT thread.
volatile float last_measured_sampling_rate = SAMPLING_RATE;

// Create an array of FFTAnalyzer instances, one for each sensor.
FFTAnalyzer analyzers[NUM_SENSORS];

// Create an angle calculator instance
AngleCalculator angle_calc;



// --- Thread Functions ---
void onI2CRequest() {
    Wire.write((uint8_t*)&i2c_data, sizeof(i2c_data));
}
// Thread 1: High-priority data acquisition
void data_collection_thread() {
    const unsigned long sample_period_micros = 1000000.0f / SAMPLING_RATE;
    unsigned long next_sample_micros = micros();

    // Variables for real-time sample rate calculation. These are local to the thread.
    int hop_samples_collected = 0;
    unsigned long hop_start_micros = micros();
    bool buffer_filled = false; // To track if the main FFT_SIZE buffer has been filled at least once

    for (;;) {
        // Busy-wait for precise sample timing. This is the highest priority.
        while (micros() < next_sample_micros) {}
        next_sample_micros += sample_period_micros;

        // --- CRITICAL SECTION: Read all ADC pins directly into sample_buffers ---
        buffer_lock.lock();

        // Read all ADC pins and store directly in sample_buffers at current index
        read_all_pins_to_buffers(adc, sample_buffers, sample_write_index);
        sample_write_index = (sample_write_index + 1) % FFT_SIZE;
        hop_samples_collected++;

        if (!buffer_filled && hop_samples_collected >= FFT_SIZE) {
            buffer_filled = true; // The buffer has been filled once
        }

        // Check if it's time to trigger an FFT analysis.
        if (buffer_filled && hop_samples_collected >= FFT_HOP_SIZE) {
            unsigned long elapsed_micros = micros() - hop_start_micros;
            
            // Calculate the actual sampling rate for this hop.
            last_measured_sampling_rate = (float)hop_samples_collected * 1000000.0f / elapsed_micros;
            
            // Signal the processing thread that data is ready.
            fft_data_ready = true;

            // Reset for the next hop.
            hop_samples_collected = 0;
            hop_start_micros = micros();
        }

        buffer_lock.unlock();
        // --- END CRITICAL SECTION ---
    }
}

// Thread 2: Lower-priority FFT processing and serial output
void fft_processing_thread() {
    // By declaring these buffers as `static`, they are allocated in global memory,
    // not on the thread's limited stack. This prevents a stack overflow crash.

    bool should_run_fft = false;
    float current_sample_rate; // Local variable for the rate used in one FFT cycle
    
    for (;;) {

        // --- CRITICAL SECTION: Check the flag and copy data if ready ---
        buffer_lock.lock();
        should_run_fft = fft_data_ready;
        if (should_run_fft) {
            // Copy all the data we need for the analysis inside this single critical section.
            current_sample_rate = last_measured_sampling_rate;
            for(int i = 0; i < NUM_SENSORS; i++) {
                // Copy the circular buffer data for each sensor into the local FFT buffer.
                analyzers[i].copy_from_circular_buffer(sample_buffers[i], sample_write_index, local_fft_buffer[i]);
            }
            // analyzers[0].copy_from_circular_buffer(sample_buffers, sample_write_index, local_fft_buffer);
            
            // Reset the flag immediately after copying, so the data thread can continue.
            fft_data_ready = false;
        }
        buffer_lock.unlock();
        // --- END CRITICAL SECTION ---

        if (!should_run_fft) {
            threads.yield(); // Give up CPU time if there's no work to do.
            continue;
        }

        // --- Perform FFT Analysis (on local data, no lock needed) ---
        // Update the analyzer with the sample rate that was current when the data was copied.
        for (int i = 0; i < NUM_SENSORS; i++) {
            analyzers[i].set_sample_rate(current_sample_rate);
        }
        // analyzers[0].set_sample_rate(current_sample_rate);

        // Perform the FFT and get the full magnitude spectrum.
        for (int i = 0; i < NUM_SENSORS; i++) {
            analyzers[i].get_full_spectrum(local_fft_buffer[i], fft_output_buffer[i]);
        }

        // --- Analyze 150Hz signal and Print Data for each sensor ---
        Serial.print("DATA,");
        Serial.print(millis());
        
        // Arrays to store the SNR and magnitude values for angle calculation
        float snr_values[NUM_SENSORS];
        float magnitude_values[NUM_SENSORS];
        float max_snr_value = 0.0f;
        
        for (int i = 0; i < NUM_SENSORS; i++) {
            float peak_mag, snr;
            analyzers[i].analyze_150hz_smoothed(fft_output_buffer[i], &peak_mag, &snr);
            
            // Store values for angle calculation
            snr_values[i] = snr;
            magnitude_values[i] = peak_mag;
            
            Serial.print(",");
            Serial.print(peak_mag, 4);
            Serial.print(",");
            Serial.print(snr, 4);

            if (snr > max_snr_value) {
                max_snr_value = snr; // Track the maximum SNR value
            }
        }
            // Store first sensor SNR for I2C transmission
        i2c_data.sensor_max_snr = magnitude_values[13];
        i2c_data.timestamp = millis();
        
        // --- Calculate Weighted Bearing Angles ---
        float bearing_angle_snr, bearing_angle_mag;
        float total_weight_snr, total_weight_mag;
        
        // Calculate bearing using SNR weights
        bool snr_valid = angle_calc.calculate_weighted_angle_snr(magnitude_values, snr_values, &bearing_angle_snr, &total_weight_snr);
        
        // Calculate bearing using magnitude weights
        bool mag_valid = angle_calc.calculate_weighted_angle_magnitude(magnitude_values, &bearing_angle_mag, &total_weight_mag);
        
        // Add bearing data to output
        Serial.print(",");
        if (snr_valid) {
            Serial.print(bearing_angle_snr, 2);
        } else {
            Serial.print("NaN");
        }
        Serial.print(",");
        Serial.print(total_weight_snr, 4);
        Serial.print(",");
        if (mag_valid) {
            Serial.print(bearing_angle_mag, 2);
        } else {
            Serial.print("NaN");
        }
        Serial.print(",");
        Serial.print(total_weight_mag, 4);

        // Update I2C data with bearing angle from magnitude weighting
        if (mag_valid) {
            i2c_data.bearing_angle_mag = bearing_angle_mag;
            i2c_data.data_valid = 1;
        } else {
            i2c_data.bearing_angle_mag = 0.0f;
            i2c_data.data_valid = 0;
        }
        i2c_data_ready = true;

        Serial.println();
    }
}


// put function declarations here:
int myFunction(int, int);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  unsigned long serial_wait_start = millis();
  while (!Serial && (millis() - serial_wait_start < 4000)) {}

  Serial.println("Teensy 4.1 Real-Time ADC Streamer with FFT Analysis (Mutex)");
  Serial.println("-----------------------------------------------------------");

  // --- ADC Configuration ---
  // adc->adc0->setAveraging(0);
  adc->adc0->setResolution(12);
  // adc->adc0->setConversionSpeed(ADC_CONVERSION_SPEED::VERY_HIGH_SPEED);
  // adc->adc0->setSamplingSpeed(ADC_SAMPLING_SPEED::VERY_HIGH_SPEED);
  
  // adc->adc1->setAveraging(0);
  adc->adc1->setResolution(12);
  // adc->adc1->setConversionSpeed(ADC_CONVERSION_SPEED::VERY_HIGH_SPEED);
  // adc->adc1->setSamplingSpeed(ADC_SAMPLING_SPEED::VERY_HIGH_SPEED);

}

void loop() {
  for (int i = 0; i < NUM_SENSORS; i++) {
        if (!analyzers[i].init(SAMPLING_RATE)) {
            Serial.print("FFT Analyzer for sensor ");
            Serial.print(i);
            Serial.println(" failed to initialize!");
            while(1); // Halt execution
        }
    }

    // --- Initialize Angle Calculator ---
    if (!angle_calc.init()) {
        Serial.println("Angle calculator failed to initialize!");
        while(1); // Halt execution
    }
    Serial.print("Angle calculator initialized for ");
    Serial.print(angle_calc.get_num_sensors());
    Serial.println(" sensors");

    // --- Disable all analog pins before use ---
    for (int i = 0; i < NUM_PINS; i++) {
        pinMode(PINS_TO_READ[i], INPUT_DISABLE);
    }
    
    // --- I2C Configuration ---
    Wire.begin(I2C_SLAVE_ADDRESS);  // Initialize as I2C slave at address 0x08
    Wire.onRequest(onI2CRequest);   // Set up the callback
    Serial.println("I2C initialized as slave at address 0x08");
    Serial.println("Configuration complete. Starting threads...");

    // --- Start Threads ---
    threads.addThread(data_collection_thread);
    threads.addThread(fft_processing_thread);

    // The main thread will now continuously yield to the scheduler,
    // allowing the data collection and FFT threads to run. This is the
    // standard and most robust way to run threads.
    while(1) {
        threads.yield();
    }
}

// put function definitions here:
int myFunction(int x, int y) {
  return x + y;
}