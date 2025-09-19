#ifndef FFT_ANALYZER_H
#define FFT_ANALYZER_H

#include <Arduino.h>
#include <arm_math.h> // Use ARM CMSIS-DSP library

// --- FFT Configuration ---
// The size of the FFT. Must be a power of 2.
#define FFT_SIZE 1024
// The hop size for the FFT, defining the overlap between successive FFTs.
// A hop size of FFT_SIZE/4 means 75% overlap.
#define FFT_HOP_SIZE (FFT_SIZE)

// The maximum frequency to analyze and stream in the debug output.
#define MAX_ANALYSIS_FREQ_HZ 2000 // We want to see up to 6kHz to capture our 5kHz signal

// --- Debugging Configuration ---
// Set to 1 to print the full FFT spectrum to the Serial monitor for debugging.
// Set to 0 for normal operation.
#define FFT_DEBUG_STREAM 1

class FFTAnalyzer {
public:
    // Default constructor
    FFTAnalyzer() : initialized(false), sample_rate(0.0) {}

    // Destructor
    ~FFTAnalyzer() {}

    /**
     * @brief Initializes the FFT analyzer using the ARM CMSIS-DSP library.
     * @param sample_rate The rate at which the ADC samples are collected.
     * @return True if initialization was successful, false otherwise.
     */
    bool init(float sample_rate) {
        this->sample_rate = sample_rate;
        if (arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE) != ARM_MATH_SUCCESS) {
            initialized = false;
            return false;
        }
        // Generate Hamming window coefficients
        for (int i = 0; i < FFT_SIZE; i++) {
            hamming_window[i] = 0.54f - 0.46f * cosf(2.0f * PI * i / (FFT_SIZE - 1));
        }
        initialized = true;
        return true;
    }

    /**
     * @brief Updates the sample rate used for frequency calculations.
     * @param new_sample_rate The new sample rate in Hz.
     */
    void set_sample_rate(float new_sample_rate) {
        this->sample_rate = new_sample_rate;
    }

    /**
     * @brief Copies and reorders data from a circular buffer into a linear buffer.
     * @param circular_buffer The source circular buffer.
     * @param start_index The index of the oldest sample in the circular buffer.
     * @param linear_buffer The destination linear buffer.
     */
    void copy_from_circular_buffer(float* circular_buffer, int start_index, float* linear_buffer) {
        if (!initialized) return;

        int dest_idx = 0;
        for (int i = start_index; i < FFT_SIZE; i++) {
            linear_buffer[dest_idx++] = circular_buffer[i];
        }
        for (int i = 0; i < start_index; i++) {
            linear_buffer[dest_idx++] = circular_buffer[i];
        }
    }
    void copy_from_circular_buffer(uint16_t* circular_buffer, int start_index, float* linear_buffer) {
        if (!initialized) return;

        int dest_idx = 0;
        for (int i = start_index; i < FFT_SIZE; i++) {
            linear_buffer[dest_idx++] = circular_buffer[i];
        }
        for (int i = 0; i < start_index; i++) {
            linear_buffer[dest_idx++] = circular_buffer[i];
        }
    }

    /**
     * @brief Performs a full FFT on the input data and stores the magnitude spectrum in the output buffer.
     * @param input_buffer A pointer to an array of `FFT_SIZE` floating-point samples.
     * @param output_buffer A pointer to an array of `FFT_SIZE / 2` elements to store the magnitude spectrum.
     */
    void get_full_spectrum(float* input_buffer, float* output_buffer) {
        if (!initialized) return;

        // 1. Calculate the mean (DC offset) and remove it.
        float mean = 0.0f;
        arm_mean_f32(input_buffer, FFT_SIZE, &mean);
        arm_offset_f32(input_buffer, -mean, input_buffer, FFT_SIZE);

        // 2. Apply a Hamming window function.
        arm_mult_f32(input_buffer, hamming_window, input_buffer, FFT_SIZE);

        // 3. Compute the RFFT. The output is packed into the input_buffer.
        arm_rfft_fast_f32(&fft_instance, input_buffer, output_buffer, 0);

        // 4. Calculate the magnitude of the complex FFT output.
        // The output of arm_rfft_fast_f32 is [R(0), R(N/2), R(1), I(1), ...]
        // We calculate the magnitude for the first (N/2) bins and store it in the user-provided output_buffer.
        arm_cmplx_mag_f32(output_buffer, output_buffer, FFT_SIZE / 2);
    }

    /**
     * @brief Calculates the magnitude of a specific target frequency from a full spectrum.
     * @param fft_magnitudes A pointer to an array of `FFT_SIZE / 2` magnitude values.
     * @param target_freq The frequency (in Hz) to find the magnitude of.
     * @return The magnitude of the target frequency.
     */
    float get_magnitude_at_freq(float* fft_magnitudes, float target_freq) {
        if (!initialized) return 0.0f;

        float frequency_resolution = sample_rate / FFT_SIZE;
        int target_bin = (int)round(target_freq / frequency_resolution);

        if (target_bin >= 0 && target_bin < (FFT_SIZE / 2)) {
            return fft_magnitudes[target_bin];
        }
        return 0.0f;
    }

    /**
     * @brief Calculates the average magnitude of the noise floor, excluding the target frequency.
     * @param fft_magnitudes A pointer to an array of `FFT_SIZE / 2` magnitude values.
     * @param target_freq The frequency (in Hz) of the signal to exclude from the noise calculation.
     * @return The average magnitude of the noise.
     */
    float get_average_noise(float* fft_magnitudes, float target_freq) {
        if (!initialized) return 0.0f;

        float frequency_resolution = sample_rate / FFT_SIZE;
        int target_bin = (int)round(target_freq / frequency_resolution);
        
        // Define a window around the target bin to exclude from noise calculation.
        // This prevents the signal's energy spread from contaminating the noise estimate.
        const int exclusion_window = 5; // Exclude target_bin +/- 5 bins.
        int start_exclude = target_bin - exclusion_window;
        int end_exclude = target_bin + exclusion_window;

        float noise_sum = 0.0f;
        int noise_bins = 0;

        // Iterate through the magnitude bins, skipping the DC component (bin 0) and the signal window.
        for (int i = 1; i < (FFT_SIZE / 2); i++) {
            if (i < start_exclude || i > end_exclude) {
                noise_sum += fft_magnitudes[i];
                noise_bins++;
            }
        }

        if (noise_bins > 0) {
            return noise_sum / noise_bins;
        }
        return 0.0f;
    }

    /**
     * @brief Finds the peak frequency and magnitude in the spectrum, ignoring the DC component.
     * @param fft_magnitudes A pointer to an array of `FFT_SIZE / 2` magnitude values.
     * @param peak_freq A pointer to a float where the peak frequency (in Hz) will be stored.
     * @param peak_mag A pointer to a float where the peak magnitude will be stored.
     */
    void find_peak_frequency(float* fft_magnitudes, float* peak_freq, float* peak_mag) {
        if (!initialized) return;

        float max_mag = 0.0f;
        int max_bin = 0;

        // Start from bin 1 to ignore the DC offset.
        for (int i = 1; i < (FFT_SIZE / 2); i++) {
            if (fft_magnitudes[i] > max_mag) {
                max_mag = fft_magnitudes[i];
                max_bin = i;
            }
        }

        float frequency_resolution = sample_rate / FFT_SIZE;
        *peak_freq = max_bin * frequency_resolution;
        *peak_mag = max_mag;
    }

    /**
     * @brief Analyzes the spectrum around 150Hz target frequency and calculates SNR.
     * @param fft_magnitudes A pointer to an array of `FFT_SIZE / 2` magnitude values.
     * @param peak_mag A pointer to a float where the peak magnitude around 150Hz will be stored.
     * @param snr A pointer to a float where the SNR (peak/noise ratio) will be stored.
     */
    void analyze_150hz_signal(float* fft_magnitudes, float* peak_mag, float* snr) {
        if (!initialized) return;

        const float target_freq = 150.0f; // Target frequency in Hz
        const float search_window = 1.0f; // Search within +/- 1Hz of target
        
        float frequency_resolution = sample_rate / FFT_SIZE;
        int target_bin = (int)round(target_freq / frequency_resolution);
        int window_bins = (int)round(search_window / frequency_resolution);
        
        // Define search range around 150Hz
        int search_start = max(1, target_bin - window_bins); // Start from bin 1 to avoid DC
        int search_end = min((FFT_SIZE / 2) - 1, target_bin + window_bins);
        
        // Single loop to find peak and accumulate noise
        float max_mag = 0.0f;
        int peak_bin = target_bin;
        float noise_sum = 0.0f;
        int noise_bins = 0;
        
        
        // Single pass through the entire spectrum
        for (int i = 1; i < (FFT_SIZE / 2); i++) {
            // Check if this bin is in the search window for peak finding
            if (i >= search_start && i <= search_end) {
                if (fft_magnitudes[i] > max_mag) {
                    max_mag = fft_magnitudes[i];
                    peak_bin = i;
                }
            }
            
            // Add to noise calculation (will be corrected after peak is found)
            // noise_sum += fft_magnitudes[i];
            // noise_bins++;
            noise_sum += fft_magnitudes[i];
            noise_bins++;
        }
        
        *peak_mag = max_mag;
        
        const int noise_exclusion_window = 10; // ±5 bins around each frequency
    
        // Define frequencies to exclude (fundamental + harmonics)
        float excluded_freqs[] = {150.0f, 300.0f, 450.0f, 600.0f};
        int num_excluded = sizeof(excluded_freqs) / sizeof(excluded_freqs[0]);
        
        // Subtract each exclusion zone
        for (int freq_idx = 0; freq_idx < num_excluded; freq_idx++) {
            int exclude_center = (int)round(excluded_freqs[freq_idx] / frequency_resolution);
            int exclude_start = max(1, exclude_center - noise_exclusion_window);
            int exclude_end = min((FFT_SIZE / 2) - 1, exclude_center + noise_exclusion_window);
            
            // Subtract this exclusion zone from noise
            for (int i = exclude_start; i <= exclude_end; i++) {
                noise_sum -= fft_magnitudes[i];
                noise_bins--;
            }
        }
        
        float noise_avg = (noise_bins > 0) ? (noise_sum / noise_bins) : 1.0f;
        
        // Calculate SNR as ratio of peak magnitude to average noise
        *snr = (noise_avg > 0.0f) ? (max_mag / noise_avg) : 0.0f;
    }

    /**
     * @brief Analyzes 150Hz signal with stable magnitude-based weighting metric.
     * @param fft_magnitudes A pointer to an array of `FFT_SIZE / 2` magnitude values.
     * @param peak_mag A pointer to a float where the peak magnitude around 150Hz will be stored.
     * @param weight_metric A pointer to a float where the weighting metric will be stored.
     */
    void analyze_150hz_stable(float* fft_magnitudes, float* peak_mag, float* weight_metric) {
        if (!initialized) return;

        const float target_freq = 150.0f; // Target frequency in Hz
        const float search_window = 2.0f; // Search within +/- 2Hz of target
        
        float frequency_resolution = sample_rate / FFT_SIZE;
        int target_bin = (int)round(target_freq / frequency_resolution);
        int window_bins = (int)round(search_window / frequency_resolution);
        
        // Define search range around 150Hz
        int search_start = max(1, target_bin - window_bins);
        int search_end = min((FFT_SIZE / 2) - 1, target_bin + window_bins);
        
        // Find peak within the search window
        float max_mag = 0.0f;
        int peak_bin = target_bin;
        
        for (int i = search_start; i <= search_end; i++) {
            if (fft_magnitudes[i] > max_mag) {
                max_mag = fft_magnitudes[i];
                peak_bin = i;
            }
        }
        
        *peak_mag = max_mag;
        
        // Calculate stable baseline noise from remote frequency regions
        // Use 50-100Hz and 250-350Hz regions (far from 150Hz)
        float baseline_noise = 0.0f;
        int baseline_bins = 0;
        
        // Low frequency baseline (50-100Hz)
        int low_start = max(1, (int)round(50.0f / frequency_resolution));
        int low_end = min((FFT_SIZE / 2) - 1, (int)round(100.0f / frequency_resolution));
        for (int i = low_start; i <= low_end; i++) {
            baseline_noise += fft_magnitudes[i];
            baseline_bins++;
        }
        
        // High frequency baseline (250-350Hz)
        int high_start = max(1, (int)round(250.0f / frequency_resolution));
        int high_end = min((FFT_SIZE / 2) - 1, (int)round(350.0f / frequency_resolution));
        for (int i = high_start; i <= high_end; i++) {
            baseline_noise += fft_magnitudes[i];
            baseline_bins++;
        }
        
        float avg_baseline = (baseline_bins > 0) ? (baseline_noise / baseline_bins) : 1.0f;
        
        // Use magnitude normalized by baseline as weight metric
        // This is more stable than SNR but still accounts for sensor differences
        *weight_metric = (avg_baseline > 0.0f) ? (max_mag / avg_baseline) : 0.0f;
    }

    // Static variables for temporal smoothing (one per instance)
    mutable float smoothed_snr = 0.0f;
    mutable float smoothed_magnitude = 0.0f;
    mutable bool smoothing_initialized = false;

    /**
     * @brief Analyzes 150Hz signal with temporal smoothing for stable metrics.
     * @param fft_magnitudes A pointer to an array of `FFT_SIZE / 2` magnitude values.
     * @param peak_mag A pointer to a float where the smoothed peak magnitude will be stored.
     * @param snr A pointer to a float where the smoothed SNR will be stored.
     */
    void analyze_150hz_smoothed(float* fft_magnitudes, float* peak_mag, float* snr) {
        if (!initialized) return;

        // Get raw values using existing function
        float raw_mag, raw_snr;
        analyze_150hz_signal(fft_magnitudes, &raw_mag, &raw_snr);
        
        // Exponential smoothing factor (0.1 = heavy smoothing, 0.9 = light smoothing)
        const float alpha = 0.5f;
        
        if (!smoothing_initialized) {
            smoothed_magnitude = raw_mag;
            smoothed_snr = raw_snr;
            smoothing_initialized = true;
        } else {
            // Exponential moving average
            smoothed_magnitude = alpha * raw_mag + (1.0f - alpha) * smoothed_magnitude;
            smoothed_snr = alpha * raw_snr + (1.0f - alpha) * smoothed_snr;
        }
        
        *peak_mag = smoothed_magnitude;
        *snr = smoothed_snr;
    }

private:
    arm_rfft_fast_instance_f32 fft_instance;
    float sample_rate;
    bool initialized;
    float hamming_window[FFT_SIZE];
};

#endif // FFT_ANALYZER_H
