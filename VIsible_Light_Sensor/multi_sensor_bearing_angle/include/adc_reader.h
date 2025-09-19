#ifndef ADC_READER_H
#define ADC_READER_H

#include <Arduino.h>
#include <ADC.h>
#include "fft_analyzer.h"

// --- CONFIGURATION ---
// Set to 1 to use the optimized, hybrid synchronized/sequential reading.
// Set to 0 to use the simple, purely sequential reading.
#define USE_SYNC_READ 1

// --- PIN DEFINITIONS ---
// The pins to be read (excluding A9/pin 23), in the order specified.
const int PINS_TO_READ[] = {
    14, 15, 16, 17, 20, 21, 22, // A0, A1, A2, A3, A6, A7, A8
    24, 25, 26, 27, 40, 39, 38, 41  // A10, A11, A12, A13, A16, A15, A14, A17
};
const int NUM_PINS = sizeof(PINS_TO_READ) / sizeof(PINS_TO_READ[0]);

// For hybrid reading, we define which pins can be paired for synchronized reading.
// These are pins that are on ADC0 and ADC1 respectively.
const int ADC0_PINS_FOR_SYNC[6] = {14, 15, 16, 17, 20, 21}; // A0, A1, A2, A3, A6, A7
const int ADC1_PINS_FOR_SYNC[6] = {26, 27, 40, 39, 38, 41}; // A12, A13, A16, A15, A14, A17

// These are the remaining pins that need to be read sequentially.
const int ADC0_PINS_REMAINING[3] = {22, 24, 25}; // A8, A10, A11


/**
 * @brief Reads all 15 specified analog pins into a destination array.
 * 
 * This function uses a hybrid approach if USE_SYNC_READ is 1:
 * 1. It reads 6 pairs of pins simultaneously using both ADCs.
 * 2. It reads the remaining 3 pins sequentially using ADC0.
 * 
 * If USE_SYNC_READ is 0, it reads all 15 pins sequentially.
 * 
 * @param adc A pointer to the ADC object.
 * @param destination_array A pointer to a uint16_t array of at least 15 elements to store the results.
 */
void read_all_pins(ADC* adc, uint16_t* destination_array) {

#if USE_SYNC_READ
    // --- Hybrid Synchronized + Sequential Read ---
    ADC::Sync_result result;
    
    // Read 6 pairs synchronously
    for (int i = 0; i < 6; i++) {
        result = adc->analogSynchronizedRead(ADC0_PINS_FOR_SYNC[i], ADC1_PINS_FOR_SYNC[i]);
        // Place results in correct array positions
        destination_array[i] = result.result_adc0;           // A0-A7 sequence
        destination_array[i + 9] = result.result_adc1;       // A12-A17 sequence
    }

    // Read the 3 remaining ADC0 pins sequentially
    destination_array[6] = adc->analogRead(ADC0_PINS_REMAINING[0]); // A8 (pin 22)
    destination_array[7] = adc->analogRead(ADC0_PINS_REMAINING[1]); // A10 (pin 24)
    destination_array[8] = adc->analogRead(ADC0_PINS_REMAINING[2]); // A11 (pin 25)

#else
    // --- Purely Sequential Read ---
    for (int i = 0; i < NUM_PINS; i++) {
        destination_array[i] = adc->analogRead(PINS_TO_READ[i]);
    }
#endif
}

/**
 * @brief Reads all 15 specified analog pins into a 2D sample buffer array.
 * 
 * This function reads all pins and stores them directly into the specified
 * index of all sensor buffers in one operation.
 * 
 * @param adc A pointer to the ADC object.
 * @param sample_buffers A 2D array [NUM_SENSORS][FFT_SIZE] to store the results.
 * @param sample_index The index where to store the current sample in each buffer.
 */
void read_all_pins_to_buffers(ADC* adc, uint16_t sample_buffers[NUM_PINS][FFT_SIZE], int sample_index) {
    
#if USE_SYNC_READ
    // --- Hybrid Synchronized + Sequential Read ---
    ADC::Sync_result result;
    
    // Read 6 pairs synchronously
    for (int i = 0; i < 6; i++) {
        result = adc->analogSynchronizedRead(ADC0_PINS_FOR_SYNC[i], ADC1_PINS_FOR_SYNC[i]);
        // Place results in correct array positions
        sample_buffers[i][sample_index] = result.result_adc0;           // A0-A7 sequence
        sample_buffers[i + 9][sample_index] = result.result_adc1;       // A12-A17 sequence
    }

    // Read the 3 remaining ADC0 pins sequentially
    sample_buffers[6][sample_index] = adc->analogRead(ADC0_PINS_REMAINING[0]); // A8 (pin 22)
    sample_buffers[7][sample_index] = adc->analogRead(ADC0_PINS_REMAINING[1]); // A10 (pin 24)
    sample_buffers[8][sample_index] = adc->analogRead(ADC0_PINS_REMAINING[2]); // A11 (pin 25)

#else
    // --- Purely Sequential Read ---
    for (int i = 0; i < NUM_PINS; i++) {
        destination_array[i] = adc->analogRead(PINS_TO_READ[i]);
    }
#endif
}


#endif // ADC_READER_H
